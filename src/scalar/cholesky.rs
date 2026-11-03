//! Cholesky-Zerlegung hermitescher positiv definiter Matrizen für beliebige [`Scalar`].

use super::Scalar;
use crate::Real;
use crate::linalg::LinalgError;
use alloc::vec;
use alloc::vec::Vec;

/// `A = L·Lᴴ` mit unterer Dreiecksmatrix `L` (reelle positive Diagonale).
#[derive(Debug, Clone)]
pub struct ScalarCholesky<T: Scalar> {
    n: usize,
    l: Vec<T>,
}

impl<T: Scalar> ScalarCholesky<T> {
    /// Zerlegt die hermitesche `n×n`-Matrix `a` (zeilenweise; nur die untere Hälfte wird gelesen, die obere dient der
    /// Symmetrieprüfung). Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NotFinite`],
    /// [`LinalgError::NotSymmetric`] (nicht hermitesch), [`LinalgError::NotPositiveDefinite`].
    pub fn new(n: usize, a: &[T]) -> Result<Self, LinalgError> {
        if a.len() != n * n {
            return Err(LinalgError::DimensionMismatch);
        }
        if a.iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        let mut scale = <T::Real as Real>::ZERO;
        for v in a {
            scale = scale.max(v.modulus());
        }
        let sym_tol = <T::Real as Real>::EPSILON * <T::Real as Real>::from_f64(100.0) * scale;
        for i in 0..n {
            for j in 0..i {
                if (a[i * n + j] - a[j * n + i].conj()).modulus() > sym_tol {
                    return Err(LinalgError::NotSymmetric);
                }
            }
            if a[i * n + i].im().abs() > sym_tol {
                return Err(LinalgError::NotSymmetric);
            }
        }
        let mut l = vec![T::ZERO; n * n];
        for j in 0..n {
            let mut d = a[j * n + j].re();
            for k in 0..j {
                d -= l[j * n + k].modulus_sq();
            }
            if !(d > <T::Real as Real>::ZERO) {
                return Err(LinalgError::NotPositiveDefinite);
            }
            let dj = d.sqrt();
            l[j * n + j] = T::from_real(dj);
            for i in j + 1..n {
                let mut s = a[i * n + j];
                for k in 0..j {
                    s -= l[i * n + k] * l[j * n + k].conj();
                }
                l[i * n + j] = s / T::from_real(dj);
            }
        }
        Ok(Self { n, l })
    }

    /// Die Dreiecksmatrix `L` (zeilenweise).
    pub fn l(&self) -> &[T] {
        &self.l
    }

    /// `ln det A`.
    pub fn log_determinant(&self) -> T::Real {
        let mut s = <T::Real as Real>::ZERO;
        for i in 0..self.n {
            s += self.l[i * self.n + i].re().ln();
        }
        s * <T::Real as Real>::from_f64(2.0)
    }

    /// Löst `A·x = b`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, LinalgError> {
        let n = self.n;
        if b.len() != n {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut y = b.to_vec();
        for i in 0..n {
            for k in 0..i {
                let t = self.l[i * n + k] * y[k];
                y[i] -= t;
            }
            y[i] /= self.l[i * n + i];
        }
        for i in (0..n).rev() {
            for k in i + 1..n {
                let t = self.l[k * n + i].conj() * y[k];
                y[i] -= t;
            }
            y[i] /= self.l[i * n + i].conj();
        }
        Ok(y)
    }
}
