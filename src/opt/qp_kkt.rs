//! Gemeinsam assembliertes, musterfestes KKT-System mit wählbarem Faktorisierungsbackend.
use super::{OptError, QpKktBackend, QpKktStats};
use crate::{
    Real,
    linalg::{DynLdlt, DynMatrix, DynVector},
    sparse::{CsrMatrix, Ordering, SparseLdl, SparseLdlSymbolic},
};
use alloc::{collections::BTreeSet, vec, vec::Vec};

pub(super) struct KktSystem<T> {
    n: usize,
    p: usize,
    base: Vec<T>,
    matrix: CsrMatrix<T>,
    grows: Vec<Vec<(usize, T)>>,
    mode: QpKktBackend,
    analysis: Option<SparseLdlSymbolic>,
    sparse: Option<SparseLdl<T>>,
    dense: Option<DynLdlt<T>>,
}
impl<T: Real> KktSystem<T> {
    pub fn new(
        q: &[T],
        a: &[T],
        g: &[T],
        n: usize,
        p: usize,
        m: usize,
        delta: T,
        mode: QpKktBackend,
        stats: &mut QpKktStats<T>,
    ) -> Result<Self, OptError> {
        let dim = n + p;
        let mut pattern = vec![BTreeSet::new(); dim];
        for (i, row) in pattern.iter_mut().enumerate() {
            row.insert(i);
        }
        for i in 0..n {
            for j in 0..i {
                if q[i * n + j] != T::ZERO {
                    pattern[i].insert(j);
                }
            }
        }
        let grows: Vec<Vec<(usize, T)>> = (0..m)
            .map(|r| {
                (0..n)
                    .filter_map(|i| {
                        if g[r * n + i] != T::ZERO {
                            Some((i, g[r * n + i]))
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();
        for row in &grows {
            for &(i, _) in row {
                for &(j, _) in row {
                    if j <= i {
                        pattern[i].insert(j);
                    }
                }
            }
        }
        for r in 0..p {
            for j in 0..n {
                if a[r * n + j] != T::ZERO {
                    pattern[n + r].insert(j);
                }
            }
        }
        let mut ptr = vec![0];
        let mut idx = Vec::new();
        let mut base = Vec::new();
        for (i, row) in pattern.iter().enumerate() {
            for &j in row {
                idx.push(j);
                base.push(if i < n {
                    q[i * n + j] + if i == j { delta } else { T::ZERO }
                } else if j < n {
                    a[(i - n) * n + j]
                } else {
                    -delta
                });
            }
            ptr.push(idx.len());
        }
        let matrix = CsrMatrix::from_parts(dim, dim, ptr, idx, base.clone()).map_err(|error| {
            if error == crate::sparse::SparseError::NotFinite {
                OptError::NotFinite
            } else {
                OptError::InvalidInput
            }
        })?;
        let mode = if mode == QpKktBackend::Auto {
            let structural_nnz = matrix
                .row_ptr()
                .windows(2)
                .map(|w| 2 * (w[1] - w[0]) - 1)
                .sum::<usize>();
            if structural_nnz as f64 <= (dim as f64) * (dim as f64) * 0.25 {
                QpKktBackend::Sparse
            } else {
                QpKktBackend::Dense
            }
        } else {
            mode
        };
        stats.selected_backend = Some(mode);
        let analysis = if mode == QpKktBackend::Sparse {
            stats.symbolic_analyses += 1;
            Some(
                SparseLdlSymbolic::analyze(&matrix, Ordering::MinimumDegree)
                    .map_err(|_| OptError::InvalidInput)?,
            )
        } else {
            None
        };
        Ok(Self {
            n,
            p,
            base,
            matrix,
            grows,
            mode,
            analysis,
            sparse: None,
            dense: None,
        })
    }
    fn dense(&mut self, stats: &mut QpKktStats<T>) -> bool {
        if self.dense.is_some() {
            return true;
        }
        stats.dense_factorizations += 1;
        if self.mode == QpKktBackend::Sparse {
            stats.dense_fallbacks += 1;
        }
        let dim = self.n + self.p;
        let mut matrix = DynMatrix::zeros(dim, dim);
        for i in 0..dim {
            for (j, v) in self.matrix.row(i) {
                matrix[(i, j)] = v;
                matrix[(j, i)] = v;
            }
        }
        self.dense = matrix.ldlt().ok();
        self.dense.is_some()
    }
    pub fn factor(&mut self, s: &[T], z: &[T], stats: &mut QpKktStats<T>) -> bool {
        self.dense = None;
        self.matrix.values_mut().copy_from_slice(&self.base);
        for (r, row) in self.grows.iter().enumerate() {
            let w = z[r] / s[r];
            for &(i, gi) in row {
                for &(j, gj) in row {
                    if j <= i {
                        let start = self.matrix.row_ptr()[i];
                        let end = self.matrix.row_ptr()[i + 1];
                        let Ok(slot) = self.matrix.col_idx()[start..end].binary_search(&j) else {
                            return false;
                        };
                        self.matrix.values_mut()[start + slot] += (gi * w) * gj;
                    }
                }
            }
        }
        if self.matrix.values().iter().any(|v| !v.is_finite()) {
            return false;
        }
        if self.mode == QpKktBackend::Dense {
            return self.dense(stats);
        }
        stats.sparse_factorizations += 1;
        let success = if let Some(f) = &mut self.sparse {
            stats.sparse_refactorizations += 1;
            f.refactor(&self.matrix).is_ok()
        } else {
            self.sparse = self
                .analysis
                .as_ref()
                .and_then(|a| a.factorize(&self.matrix).ok());
            self.sparse.is_some()
        };
        if success
            && self
                .sparse
                .as_ref()
                .is_some_and(|f| f.inertia() == (self.n, self.p))
        {
            true
        } else {
            self.dense(stats)
        }
    }
    // Normweiser Rückwärtsfehler, mit global skalierter Matrix und skaliertem Nenner.
    fn error(&self, x: &[T], b: &[T]) -> Option<T> {
        let dim = self.n + self.p;
        let scale = self
            .matrix
            .values()
            .iter()
            .fold(T::ZERO, |s, v| s.max(v.abs()));
        if scale == T::ZERO {
            return None;
        }
        let mut r: Vec<T> = b.iter().map(|v| -*v / scale).collect();
        let mut rownorm = vec![T::ZERO; dim];
        for i in 0..dim {
            for (j, v) in self.matrix.row(i) {
                let v = v / scale;
                r[i] = v.mul_add(x[j], r[i]);
                rownorm[i] += v.abs();
                if i != j {
                    r[j] = v.mul_add(x[i], r[j]);
                    rownorm[j] += v.abs();
                }
            }
        }
        if r.iter().any(|v| !v.is_finite()) || x.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let xn = x.iter().fold(T::ZERO, |s, v| s.max(v.abs()));
        let bn = b.iter().fold(T::ZERO, |s, v| s.max((*v / scale).abs()));
        let rn = r.iter().fold(T::ZERO, |s, v| s.max(v.abs()));
        let an = rownorm.iter().fold(T::ZERO, |s, v| s.max(*v));
        let vscale = xn.max(bn);
        let e = if vscale == T::ZERO {
            T::ZERO
        } else {
            (rn / vscale) / (an * (xn / vscale) + bn / vscale)
        };
        if e.is_finite() && e <= T::from_f64(64.0) * T::from_usize(dim) * T::EPSILON {
            Some(e)
        } else {
            None
        }
    }
    pub fn solve(&mut self, b: &[T], stats: &mut QpKktStats<T>) -> Option<Vec<T>> {
        if self.dense.is_none() {
            if let Some(x) = self.sparse.as_ref().and_then(|f| f.solve(b).ok()) {
                if let Some(e) = self.error(&x, b) {
                    stats.max_linear_backward_error = stats.max_linear_backward_error.max(e);
                    return Some(x);
                }
            }
            if !self.dense(stats) {
                return None;
            }
        }
        let x = self
            .dense
            .as_ref()?
            .solve(&DynVector::new(b.to_vec()))
            .ok()?;
        let x = x.as_slice().to_vec();
        let e = self.error(&x, b)?;
        stats.max_linear_backward_error = stats.max_linear_backward_error.max(e);
        Some(x)
    }
}
