//! Geliehene Matrixzugriffe und Sparse-Presolve ohne dichte Zwischenmatrix.
use crate::{Real, sparse::CsrMatrix};
use alloc::{collections::BTreeMap, vec, vec::Vec};

#[derive(Clone, Copy)]
pub(super) enum QpMatrix<'a, T> {
    Dense {
        data: &'a [T],
        rows: usize,
        cols: usize,
    },
    Csr(&'a CsrMatrix<T>),
}
impl<'a, T: Real> QpMatrix<'a, T> {
    pub fn rows(self) -> usize {
        match self {
            Self::Dense { rows, .. } => rows,
            Self::Csr(a) => a.rows(),
        }
    }
    pub fn cols(self) -> usize {
        match self {
            Self::Dense { cols, .. } => cols,
            Self::Csr(a) => a.cols(),
        }
    }
    pub fn values(self) -> &'a [T] {
        match self {
            Self::Dense { data, .. } => data,
            Self::Csr(a) => a.values(),
        }
    }
    pub fn row(self, i: usize, mut visit: impl FnMut(usize, T)) {
        match self {
            Self::Dense { data, cols, .. } => {
                for (j, &v) in data[i * cols..(i + 1) * cols].iter().enumerate() {
                    if v != T::ZERO {
                        visit(j, v);
                    }
                }
            }
            Self::Csr(a) => {
                for (j, v) in a.row(i) {
                    visit(j, v);
                }
            }
        }
    }
    pub fn get(self, i: usize, j: usize) -> T {
        match self {
            Self::Dense { data, cols, .. } => data[i * cols + j],
            Self::Csr(a) => {
                let start = a.row_ptr()[i];
                let end = a.row_ptr()[i + 1];
                match a.col_idx()[start..end].binary_search(&j) {
                    Ok(p) => a.values()[start + p],
                    Err(_) => T::ZERO,
                }
            }
        }
    }
    pub fn mul(self, x: &[T], y: &mut [T]) {
        for (i, out) in y.iter_mut().enumerate() {
            let mut sum = T::ZERO;
            self.row(i, |j, v| sum += v * x[j]);
            *out = sum;
        }
    }
    pub fn mul_t(self, x: &[T], y: &mut [T]) {
        y.fill(T::ZERO);
        for (i, &xi) in x.iter().enumerate() {
            self.row(i, |j, v| y[j] += v * xi);
        }
    }
    pub fn selected_rows(self, keep: &[usize]) -> CsrMatrix<T> {
        let mut ptr = vec![0];
        let mut idx = Vec::new();
        let mut values = Vec::new();
        for &i in keep {
            self.row(i, |j, v| {
                idx.push(j);
                values.push(v);
            });
            ptr.push(idx.len());
        }
        CsrMatrix::from_parts(keep.len(), self.cols(), ptr, idx, values)
            .expect("selected rows preserve CSR invariants")
    }
}
// Partielle Zeilenpivotierung auf Sparse-Zeilen; speichert nur tatsächliches Fill-in.
pub(super) fn independent_rows<T: Real>(a: QpMatrix<'_, T>, b: &[T]) -> Result<Vec<usize>, ()> {
    let p = a.rows();
    let n = a.cols();
    if p == 0 {
        return Ok(Vec::new());
    }
    let mut rows = vec![BTreeMap::new(); p];
    for (i, row) in rows.iter_mut().enumerate() {
        a.row(i, |j, v| {
            if v != T::ZERO {
                row.insert(j, v);
            }
        });
    }
    let mut rhs = b.to_vec();
    let mut idx: Vec<usize> = (0..p).collect();
    let scale = a.values().iter().fold(T::ZERO, |s, v| s.max(v.abs()));
    let tol = T::from_f64(p.max(n) as f64 * 10.0) * T::EPSILON * scale.max(T::MIN_POSITIVE);
    let mut r = 0;
    for col in 0..n {
        if r == p {
            break;
        }
        let mut best = r;
        for i in r + 1..p {
            if rows[i].get(&col).copied().unwrap_or(T::ZERO).abs()
                > rows[best].get(&col).copied().unwrap_or(T::ZERO).abs()
            {
                best = i;
            }
        }
        let pivot = rows[best].get(&col).copied().unwrap_or(T::ZERO);
        if pivot.abs() <= tol {
            continue;
        }
        rows.swap(best, r);
        rhs.swap(best, r);
        idx.swap(best, r);
        let tail: Vec<(usize, T)> = rows[r].range(col + 1..).map(|(&j, &v)| (j, v)).collect();
        for i in r + 1..p {
            let Some(value) = rows[i].remove(&col) else {
                continue;
            };
            let f = value / pivot;
            for &(j, v) in &tail {
                let updated = rows[i].get(&j).copied().unwrap_or(T::ZERO) - f * v;
                if !updated.is_finite() {
                    return Err(());
                }
                if updated == T::ZERO {
                    rows[i].remove(&j);
                } else {
                    rows[i].insert(j, updated);
                }
            }
            let v = rhs[r];
            rhs[i] -= f * v;
            if !rhs[i].is_finite() {
                return Err(());
            }
        }
        r += 1;
    }
    let bnorm = b.iter().fold(T::ZERO, |s, v| s.max(v.abs()));
    let btol = (tol * (T::ONE + bnorm) + T::EPSILON * bnorm).max(tol);
    if (r < p && !btol.is_finite()) || rhs[r..].iter().any(|v| v.abs() > btol) {
        return Err(());
    }
    let mut keep = idx[..r].to_vec();
    keep.sort_unstable();
    Ok(keep)
}
