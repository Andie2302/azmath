//! LU-Zerlegung mit Teilpivotisierung für beliebige [`Scalar`] (reell oder komplex).

use super::Scalar;
use crate::Real;
use crate::linalg::LinalgError;
use alloc::vec;
use alloc::vec::Vec;

/// `P·A = L·U` einer quadratischen Matrix (zeilenweise gespeichert), Pivot nach größtem Betrag.
#[derive(Debug, Clone)]
pub struct ScalarLu<T: Scalar> {
    n: usize,
    lu: Vec<T>,
    piv: Vec<usize>,
    sign: bool,
}

impl<T: Scalar> ScalarLu<T> {
    /// Zerlegt die `n×n`-Matrix `a` (zeilenweise, Länge `n·n`).
    ///
    /// Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NotFinite`], [`LinalgError::Singular`] (Pivot 0).
    pub fn new(n: usize, a: &[T]) -> Result<Self, LinalgError> {
        if a.len() != n * n {
            return Err(LinalgError::DimensionMismatch);
        }
        if a.iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        let mut lu = a.to_vec();
        let mut piv: Vec<usize> = (0..n).collect();
        let mut sign = false;
        for k in 0..n {
            let mut p = k;
            let mut best = lu[k * n + k].modulus();
            for i in k + 1..n {
                let m = lu[i * n + k].modulus();
                if m > best {
                    best = m;
                    p = i;
                }
            }
            if best == <T::Real as Real>::ZERO {
                return Err(LinalgError::Singular);
            }
            if p != k {
                for j in 0..n {
                    lu.swap(k * n + j, p * n + j);
                }
                piv.swap(k, p);
                sign = !sign;
            }
            let d = lu[k * n + k];
            for i in k + 1..n {
                let f = lu[i * n + k] / d;
                lu[i * n + k] = f;
                for j in k + 1..n {
                    let u = lu[k * n + j];
                    lu[i * n + j] -= f * u;
                }
            }
        }
        Ok(Self { n, lu, piv, sign })
    }

    /// Dimension `n`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Löst `A·x = b`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, LinalgError> {
        let n = self.n;
        if b.len() != n {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut x: Vec<T> = (0..n).map(|i| b[self.piv[i]]).collect();
        for i in 0..n {
            for j in 0..i {
                let l = self.lu[i * n + j];
                let xj = x[j];
                x[i] -= l * xj;
            }
        }
        for i in (0..n).rev() {
            for j in i + 1..n {
                let u = self.lu[i * n + j];
                let xj = x[j];
                x[i] -= u * xj;
            }
            x[i] /= self.lu[i * n + i];
        }
        Ok(x)
    }

    /// Determinante.
    pub fn determinant(&self) -> T {
        let mut d = if self.sign { -T::ONE } else { T::ONE };
        for i in 0..self.n {
            d *= self.lu[i * self.n + i];
        }
        d
    }

    /// Inverse (zeilenweise).
    pub fn inverse(&self) -> Vec<T> {
        let n = self.n;
        let mut inv = vec![T::ZERO; n * n];
        let mut e = vec![T::ZERO; n];
        for j in 0..n {
            e.iter_mut().for_each(|v| *v = T::ZERO);
            e[j] = T::ONE;
            if let Ok(col) = self.solve(&e) {
                for i in 0..n {
                    inv[i * n + j] = col[i];
                }
            }
        }
        inv
    }
}
