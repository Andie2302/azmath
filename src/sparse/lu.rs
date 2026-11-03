//! Direkter Löser für allgemeine (nichtsymmetrische) dünn besetzte Matrizen: LU mit Zeilenpivotisierung.

use super::{
    CooMatrix, CsrMatrix, Ordering, minimum_degree_ordering, permute_symmetric, rcm_ordering,
};
use crate::Real;
use crate::linalg::LinalgError;
use alloc::vec;
use alloc::vec::Vec;

/// `R·P·A·Pᵀ = L·U` einer quadratischen dünn besetzten Matrix: `P` eine Umordnung zur Fill-in-Verringerung (symmetrisch auf
/// Zeilen und Spalten, bestimmt aus dem Muster von `A + Aᵀ`), `R` die Zeilenvertauschung der Pivotisierung (größter Betrag der
/// Spalte, „partial pivoting“). Spaltenweise links-schauendes Verfahren nach Gilbert und Peierls in einer einfachen Form
/// (Aufwand `O(n² + Flops)`).
///
/// ```
/// use azmath::sparse::{CooMatrix, CsrMatrix, Ordering, SparseLu};
///
/// // [[0, 2], [1, 1]] braucht Zeilenvertauschung
/// let mut coo = CooMatrix::new(2, 2);
/// coo.push(0, 1, 2.0f64).unwrap();
/// coo.push(1, 0, 1.0).unwrap();
/// coo.push(1, 1, 1.0).unwrap();
/// let lu = SparseLu::new(&CsrMatrix::from_coo(&coo), Ordering::Natural).unwrap();
/// let x = lu.solve(&[4.0, 3.0]).unwrap();
/// assert!((x[0] - 1.0).abs() < 1e-14 && (x[1] - 2.0).abs() < 1e-14);
/// ```
#[derive(Debug, Clone)]
pub struct SparseLu<T> {
    n: usize,
    /// `perm[neu] = alt`
    perm: Vec<usize>,
    /// `prow[k]` = Zeile (im umgeordneten System), die im Schritt `k` Pivotzeile wurde
    prow: Vec<usize>,
    /// `pinv[i]` = Schritt, in dem Zeile `i` Pivotzeile wurde
    pinv: Vec<usize>,
    /// `L`-Spalten ohne Diagonale: `(Zeile im umgeordneten System, Wert)`
    lcols: Vec<Vec<(usize, T)>>,
    /// `U`-Spalten ohne Diagonale: `(Schritt k < j, Wert)`
    ucols: Vec<Vec<(usize, T)>>,
    udiag: Vec<T>,
}

fn symmetric_pattern<T: Real>(a: &CsrMatrix<T>) -> CsrMatrix<T> {
    let n = a.rows();
    let mut coo = CooMatrix::with_capacity(n, n, 2 * a.nnz());
    for i in 0..n {
        for (j, _) in a.row(i) {
            let _ = coo.push(i, j, T::ONE);
            let _ = coo.push(j, i, T::ONE);
        }
    }
    CsrMatrix::from_coo(&coo)
}

impl<T: Real> SparseLu<T> {
    /// Faktorisiert `a` (quadratisch). Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NotFinite`],
    /// [`LinalgError::Singular`] (kein von Null verschiedener Pivot in einer Spalte).
    pub fn new(a: &CsrMatrix<T>, ordering: Ordering) -> Result<Self, LinalgError> {
        let n = a.rows();
        if a.cols() != n {
            return Err(LinalgError::DimensionMismatch);
        }
        if a.values().iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        let perm: Vec<usize> = match ordering {
            Ordering::Natural => (0..n).collect(),
            Ordering::ReverseCuthillMcKee => rcm_ordering(&symmetric_pattern(a)),
            Ordering::MinimumDegree => minimum_degree_ordering(&symmetric_pattern(a)),
        };
        let c = if ordering == Ordering::Natural {
            a.clone()
        } else {
            permute_symmetric(a, &perm)
        };
        // spaltenweise Kopie
        let mut cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        for r in 0..n {
            for (j, v) in c.row(r) {
                cols[j].push((r, v));
            }
        }
        let none = usize::MAX;
        let mut pinv = vec![none; n];
        let mut prow = vec![0usize; n];
        let mut lcols: Vec<Vec<(usize, T)>> = Vec::with_capacity(n);
        let mut ucols: Vec<Vec<(usize, T)>> = Vec::with_capacity(n);
        let mut udiag = Vec::with_capacity(n);
        let mut x = vec![T::ZERO; n];
        for j in 0..n {
            for &(i, v) in &cols[j] {
                x[i] = v;
            }
            let mut ucol = Vec::new();
            for k in 0..j {
                let ukj = x[prow[k]];
                if ukj != T::ZERO {
                    ucol.push((k, ukj));
                    for &(i, l) in &lcols[k] {
                        x[i] -= l * ukj;
                    }
                }
                x[prow[k]] = T::ZERO;
            }
            // Pivot unter den noch nicht gewählten Zeilen
            let mut p = none;
            let mut best = T::ZERO;
            for i in 0..n {
                if pinv[i] == none && x[i].abs() > best {
                    best = x[i].abs();
                    p = i;
                }
            }
            if p == none {
                return Err(LinalgError::Singular);
            }
            let piv = x[p];
            let mut lcol = Vec::new();
            for i in 0..n {
                if pinv[i] == none && i != p && x[i] != T::ZERO {
                    lcol.push((i, x[i] / piv));
                }
                if pinv[i] == none {
                    x[i] = T::ZERO;
                }
            }
            pinv[p] = j;
            prow[j] = p;
            lcols.push(lcol);
            ucols.push(ucol);
            udiag.push(piv);
        }
        Ok(Self {
            n,
            perm,
            prow,
            pinv,
            lcols,
            ucols,
            udiag,
        })
    }

    /// Dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Einträge von `L` ohne die Einsen der Diagonale.
    pub fn nnz_l(&self) -> usize {
        self.lcols.iter().map(Vec::len).sum()
    }

    /// Einträge von `U` einschließlich der Diagonale.
    pub fn nnz_u(&self) -> usize {
        self.ucols.iter().map(Vec::len).sum::<usize>() + self.n
    }

    /// `ln|det A|`.
    pub fn log_abs_determinant(&self) -> T {
        self.udiag.iter().fold(T::ZERO, |s, v| s + v.abs().ln())
    }

    /// Vorzeichen von `det A` (`+1` oder `−1`).
    pub fn determinant_sign(&self) -> T {
        let mut neg = self.udiag.iter().filter(|v| **v < T::ZERO).count();
        // Parität der Zeilenvertauschung R (Zyklenzählung von prow)
        let mut seen = vec![false; self.n];
        for s in 0..self.n {
            if !seen[s] {
                let mut len = 0;
                let mut k = s;
                while !seen[k] {
                    seen[k] = true;
                    k = self.prow[k];
                    len += 1;
                }
                if len % 2 == 0 {
                    neg += 1;
                }
            }
        }
        if neg % 2 == 0 { T::ONE } else { -T::ONE }
    }

    /// Löst `A·x = b`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, LinalgError> {
        let n = self.n;
        if b.len() != n {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut w: Vec<T> = (0..n).map(|i| b[self.perm[i]]).collect();
        let mut z = vec![T::ZERO; n];
        for k in 0..n {
            z[k] = w[self.prow[k]];
            for &(i, l) in &self.lcols[k] {
                w[i] -= l * z[k];
            }
        }
        for j in (0..n).rev() {
            z[j] /= self.udiag[j];
            let zj = z[j];
            for &(k, u) in &self.ucols[j] {
                z[k] -= u * zj;
            }
        }
        let mut x = vec![T::ZERO; n];
        for i in 0..n {
            x[self.perm[i]] = z[i];
        }
        Ok(x)
    }

    /// Löst `Aᵀ·x = b`.
    pub fn solve_transposed(&self, b: &[T]) -> Result<Vec<T>, LinalgError> {
        let n = self.n;
        if b.len() != n {
            return Err(LinalgError::DimensionMismatch);
        }
        // C = Pa·A·Paᵀ  ⇒  Cᵀ = Pa·Aᵀ·Paᵀ: gleiche Umordnung wie in `solve`
        let c: Vec<T> = (0..n).map(|i| b[self.perm[i]]).collect();
        // Uᵀ u = c
        let mut u = vec![T::ZERO; n];
        for j in 0..n {
            let mut s = c[j];
            for &(k, ukj) in &self.ucols[j] {
                s -= ukj * u[k];
            }
            u[j] = s / self.udiag[j];
        }
        // Lᵀ w = u mit w_k = y[prow[k]]
        let mut w = vec![T::ZERO; n];
        for k in (0..n).rev() {
            let mut s = u[k];
            for &(i, l) in &self.lcols[k] {
                s -= l * w[self.pinv[i]];
            }
            w[k] = s;
        }
        let mut y = vec![T::ZERO; n];
        for k in 0..n {
            y[self.prow[k]] = w[k];
        }
        let mut x = vec![T::ZERO; n];
        for i in 0..n {
            x[self.perm[i]] = y[i];
        }
        Ok(x)
    }
}
