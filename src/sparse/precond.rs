#![allow(clippy::needless_range_loop)]

//! Vorkonditionierer für CSR-Matrizen: unvollständige Zerlegungen ILU(0) und IC(0) sowie SSOR.

use super::CsrMatrix;
use crate::Real;
use crate::linalg::LinalgError;
use crate::linalg::iterative::Preconditioner;
use alloc::vec;
use alloc::vec::Vec;

fn square(a: &CsrMatrix<impl Real>) -> Result<usize, LinalgError> {
    if a.rows() != a.cols() {
        return Err(LinalgError::DimensionMismatch);
    }
    Ok(a.rows())
}

/// Position des Eintrags `(i, j)` in den CSR-Arrays (Binärsuche in der Zeile).
fn find(row_ptr: &[usize], col_idx: &[usize], i: usize, j: usize) -> Option<usize> {
    let (a, b) = (row_ptr[i], row_ptr[i + 1]);
    col_idx[a..b].binary_search(&j).ok().map(|k| a + k)
}

/// **ILU(0)**: unvollständige LU-Zerlegung ohne zusätzlichen Fill-in. `L` (Einheitsdiagonale) und `U` haben genau das
/// Muster von `A`; `M = L·U ≈ A`. Für nichtsymmetrische Matrizen mit GMRES oder BiCGSTAB. Bei tridiagonalem `A` ist die
/// Zerlegung exakt.
///
/// Fehler beim Aufbau: [`LinalgError::DimensionMismatch`] (nicht quadratisch), [`LinalgError::Singular`] (fehlende
/// Diagonale oder Pivot 0), [`LinalgError::NotFinite`].
#[derive(Debug, Clone, PartialEq)]
pub struct Ilu0<T> {
    lu: CsrMatrix<T>,
    diag: Vec<usize>,
}

impl<T: Real> Ilu0<T> {
    /// Berechnet die Zerlegung von `a`.
    pub fn new(a: &CsrMatrix<T>) -> Result<Self, LinalgError> {
        let n = square(a)?;
        let mut lu = a.clone();
        let (rp, ci) = (a.row_ptr().to_vec(), a.col_idx().to_vec());
        let mut diag = vec![0usize; n];
        for i in 0..n {
            diag[i] = find(&rp, &ci, i, i).ok_or(LinalgError::Singular)?;
        }
        let v = lu.values_mut();
        if v.iter().any(|x| x.is_nan() || x.is_infinite()) {
            return Err(LinalgError::NotFinite);
        }
        let scale = v.iter().fold(T::ZERO, |m, x| m.max(x.abs()));
        let tol = T::from_f64(n as f64) * T::EPSILON * scale;
        for i in 0..n {
            for p in rp[i]..diag[i] {
                let k = ci[p];
                let pivot = v[diag[k]];
                if pivot.abs() <= tol {
                    return Err(LinalgError::Singular);
                }
                let lik = v[p] / pivot;
                v[p] = lik;
                // Zeile k, Spalten j > k, nur wenn (i, j) im Muster liegt
                for q in (diag[k] + 1)..rp[k + 1] {
                    if let Some(pos) = find(&rp, &ci, i, ci[q]) {
                        let ukj = v[q];
                        v[pos] -= lik * ukj;
                    }
                }
            }
            if v[diag[i]].abs() <= tol {
                return Err(LinalgError::Singular);
            }
        }
        Ok(Self { lu, diag })
    }
}

impl<T: Real> Preconditioner<T> for Ilu0<T> {
    fn apply(&self, r: &[T], z: &mut [T]) {
        let (rp, ci, v) = (self.lu.row_ptr(), self.lu.col_idx(), self.lu.values());
        let n = self.diag.len();
        for i in 0..n {
            let mut s = r[i];
            for p in rp[i]..self.diag[i] {
                s -= v[p] * z[ci[p]];
            }
            z[i] = s;
        }
        for i in (0..n).rev() {
            let mut s = z[i];
            for p in (self.diag[i] + 1)..rp[i + 1] {
                s -= v[p] * z[ci[p]];
            }
            z[i] = s / v[self.diag[i]];
        }
    }
}

/// **IC(0)**: unvollständige Cholesky-Zerlegung ohne zusätzlichen Fill-in für symmetrisch positiv definite Matrizen,
/// `M = L·Lᵀ ≈ A` mit dem unteren Dreiecksmuster von `A`. Vorkonditionierer für `pcg`. Es wird nur die untere Hälfte
/// von `A` gelesen (Symmetrie wird nicht geprüft).
///
/// Fehler beim Aufbau: [`LinalgError::DimensionMismatch`], [`LinalgError::NotPositiveDefinite`] (fehlende Diagonale oder
/// nicht positiver Pivot – IC(0) kann auch bei SPD-Matrizen scheitern), [`LinalgError::NotFinite`].
#[derive(Debug, Clone, PartialEq)]
pub struct Ic0<T> {
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<T>,
}

impl<T: Real> Ic0<T> {
    /// Berechnet den Faktor `L` aus dem unteren Dreieck von `a`.
    pub fn new(a: &CsrMatrix<T>) -> Result<Self, LinalgError> {
        let n = square(a)?;
        let mut row_ptr = vec![0usize];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            for (j, v) in a.row(i) {
                if v.is_nan() || v.is_infinite() {
                    return Err(LinalgError::NotFinite);
                }
                if j <= i {
                    col_idx.push(j);
                    values.push(v);
                }
            }
            row_ptr.push(col_idx.len());
        }
        for i in 0..n {
            let (start, end) = (row_ptr[i], row_ptr[i + 1]);
            if end == start || col_idx[end - 1] != i {
                return Err(LinalgError::NotPositiveDefinite);
            }
            for p in start..end {
                let j = col_idx[p];
                // s = a_ij − Σ_{k<j} l_ik·l_jk (gemeinsame Spalten der Zeilen i und j)
                let mut s = values[p];
                let (mut x, mut y) = (start, row_ptr[j]);
                let (xe, ye) = (p, if j < i { row_ptr[j + 1] - 1 } else { p });
                while x < xe && y < ye {
                    match col_idx[x].cmp(&col_idx[y]) {
                        core::cmp::Ordering::Less => x += 1,
                        core::cmp::Ordering::Greater => y += 1,
                        core::cmp::Ordering::Equal => {
                            s -= values[x] * values[y];
                            x += 1;
                            y += 1;
                        }
                    }
                }
                if j < i {
                    values[p] = s / values[row_ptr[j + 1] - 1];
                } else {
                    if s.is_nan() || s <= T::ZERO {
                        return Err(LinalgError::NotPositiveDefinite);
                    }
                    values[p] = s.sqrt();
                }
            }
        }
        Ok(Self {
            row_ptr,
            col_idx,
            values,
        })
    }
}

impl<T: Real> Preconditioner<T> for Ic0<T> {
    fn apply(&self, r: &[T], z: &mut [T]) {
        let n = self.row_ptr.len() - 1;
        // L·y = r
        for i in 0..n {
            let end = self.row_ptr[i + 1];
            let mut s = r[i];
            for p in self.row_ptr[i]..end - 1 {
                s -= self.values[p] * z[self.col_idx[p]];
            }
            z[i] = s / self.values[end - 1];
        }
        // Lᵀ·z = y
        for i in (0..n).rev() {
            let end = self.row_ptr[i + 1];
            z[i] /= self.values[end - 1];
            let zi = z[i];
            for p in self.row_ptr[i]..end - 1 {
                z[self.col_idx[p]] -= self.values[p] * zi;
            }
        }
    }
}

/// **SSOR** (symmetrisches Successive Over-Relaxation) mit Relaxationsparameter `ω ∈ (0, 2)`: braucht keine Zerlegung,
/// nur `A` selbst. `M⁻¹ = ω(2−ω)·(D+ωU)⁻¹·D·(D+ωL)⁻¹` mit der Zerlegung `A = L + D + U`. Für symmetrisch positiv
/// definite `A` ist `M` SPD (geeignet für `pcg`); `ω = 1` ist symmetrisches Gauß-Seidel.
///
/// Fehler beim Aufbau: [`LinalgError::DimensionMismatch`], [`LinalgError::Singular`] (Diagonale fehlt oder ist 0),
/// [`LinalgError::Breakdown`] (`ω` nicht in `(0, 2)`).
#[derive(Debug, Clone, PartialEq)]
pub struct Ssor<T> {
    a: CsrMatrix<T>,
    diag: Vec<T>,
    omega: T,
}

impl<T: Real> Ssor<T> {
    /// Baut den Vorkonditionierer aus `a` und `omega`.
    pub fn new(a: &CsrMatrix<T>, omega: T) -> Result<Self, LinalgError> {
        let n = square(a)?;
        if omega.is_nan() || omega <= T::ZERO || omega >= T::from_f64(2.0) {
            return Err(LinalgError::Breakdown);
        }
        let diag = a.diagonal();
        for i in 0..n {
            if diag[i] == T::ZERO || diag[i].is_nan() || diag[i].is_infinite() {
                return Err(LinalgError::Singular);
            }
        }
        Ok(Self {
            a: a.clone(),
            diag,
            omega,
        })
    }
}

impl<T: Real> Preconditioner<T> for Ssor<T> {
    fn apply(&self, r: &[T], z: &mut [T]) {
        let n = self.diag.len();
        let w = self.omega;
        // (D + ωL)·y = r
        for i in 0..n {
            let mut s = r[i];
            for (j, v) in self.a.row(i) {
                if j < i {
                    s -= w * v * z[j];
                }
            }
            z[i] = s / self.diag[i];
        }
        // y ← D·y
        for i in 0..n {
            z[i] *= self.diag[i];
        }
        // (D + ωU)·z = y
        for i in (0..n).rev() {
            let mut s = z[i];
            for (j, v) in self.a.row(i) {
                if j > i {
                    s -= w * v * z[j];
                }
            }
            z[i] = s / self.diag[i];
        }
        let scale = w * (T::from_f64(2.0) - w);
        for v in z.iter_mut() {
            *v *= scale;
        }
    }
}
