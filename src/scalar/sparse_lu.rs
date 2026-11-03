//! Direkter Löser für dünn besetzte Matrizen über [`Scalar`] (reell oder komplex): LU mit Zeilenpivotisierung.

use super::{Scalar, ScalarCsr};
use crate::Real;
use crate::linalg::LinalgError;
use crate::sparse::{CooMatrix, CsrMatrix, Ordering, minimum_degree_ordering, rcm_ordering};
use alloc::vec;
use alloc::vec::Vec;

/// `R·P·A·Pᵀ = L·U` einer quadratischen [`ScalarCsr`]-Matrix; Aufbau wie bei [`SparseLu`](crate::sparse::SparseLu)
/// (Umordnung aus dem Muster von `A + Aᵀ`, Pivot nach größtem Betrag), aber für beliebige [`Scalar`], insbesondere komplexe Matrizen.
///
/// ```
/// use azmath::complex::Complex;
/// use azmath::scalar::{ScalarCsr, ScalarSparseLu};
/// use azmath::sparse::Ordering;
///
/// let a = ScalarCsr::from_triplets(2, 2, &[
///     (0, 0, Complex::new(0.0, 1.0)), (0, 1, Complex::new(1.0, 0.0)), (1, 1, Complex::new(2.0, 0.0)),
/// ]).unwrap();
/// let lu = ScalarSparseLu::new(&a, Ordering::Natural).unwrap();
/// let x = lu.solve(&[Complex::new(1.0, 1.0), Complex::new(4.0, 0.0)]).unwrap();
/// // x₁ = 2,  i·x₀ + x₁ = 1 + i  ⇒  x₀ = (−1 + i)/i = 1 + i
/// assert!((x[1] - Complex::new(2.0, 0.0)).abs() < 1e-14);
/// assert!((x[0] - Complex::new(1.0, 1.0)).abs() < 1e-14);
/// ```
#[derive(Debug, Clone)]
pub struct ScalarSparseLu<T: Scalar> {
    n: usize,
    perm: Vec<usize>,
    prow: Vec<usize>,
    pinv: Vec<usize>,
    lcols: Vec<Vec<(usize, T)>>,
    ucols: Vec<Vec<(usize, T)>>,
    udiag: Vec<T>,
}

impl<T: Scalar> ScalarSparseLu<T> {
    /// Faktorisiert `a`. Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NotFinite`], [`LinalgError::Singular`].
    pub fn new(a: &ScalarCsr<T>, ordering: Ordering) -> Result<Self, LinalgError> {
        let n = a.rows();
        if a.cols() != n {
            return Err(LinalgError::DimensionMismatch);
        }
        let entries = a.entries();
        if entries.iter().any(|&(_, _, v)| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        let perm: Vec<usize> = if ordering == Ordering::Natural {
            (0..n).collect()
        } else {
            let one = <T::Real as Real>::ONE;
            let mut coo = CooMatrix::<T::Real>::with_capacity(n, n, 2 * entries.len());
            for &(i, j, _) in &entries {
                let _ = coo.push(i, j, one);
                let _ = coo.push(j, i, one);
            }
            let pat = CsrMatrix::from_coo(&coo);
            match ordering {
                Ordering::ReverseCuthillMcKee => rcm_ordering(&pat),
                _ => minimum_degree_ordering(&pat),
            }
        };
        let mut inv = vec![0usize; n];
        for (new, &old) in perm.iter().enumerate() {
            inv[old] = new;
        }
        let mut cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
        for &(i, j, v) in &entries {
            cols[inv[j]].push((inv[i], v));
        }
        let none = usize::MAX;
        let rz = <T::Real as Real>::ZERO;
        let mut pinv = vec![none; n];
        let mut prow = vec![0usize; n];
        let mut lcols: Vec<Vec<(usize, T)>> = Vec::with_capacity(n);
        let mut ucols: Vec<Vec<(usize, T)>> = Vec::with_capacity(n);
        let mut udiag = Vec::with_capacity(n);
        let mut x = vec![T::ZERO; n];
        for j in 0..n {
            for &(i, v) in &cols[j] {
                x[i] += v;
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
            let mut p = none;
            let mut best = rz;
            for i in 0..n {
                if pinv[i] == none && x[i].modulus() > best {
                    best = x[i].modulus();
                    p = i;
                }
            }
            if p == none {
                return Err(LinalgError::Singular);
            }
            let piv = x[p];
            let mut lcol = Vec::new();
            for i in 0..n {
                if pinv[i] == none {
                    if i != p && x[i] != T::ZERO {
                        lcol.push((i, x[i] / piv));
                    }
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

    /// Einträge von `L` (ohne Einsen der Diagonale).
    pub fn nnz_l(&self) -> usize {
        self.lcols.iter().map(Vec::len).sum()
    }

    /// Einträge von `U` einschließlich der Diagonale.
    pub fn nnz_u(&self) -> usize {
        self.ucols.iter().map(Vec::len).sum::<usize>() + self.n
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

    /// Löst `Aᵀ·x = b` (einfach transponiert, **nicht** konjugiert).
    pub fn solve_transposed(&self, b: &[T]) -> Result<Vec<T>, LinalgError> {
        let n = self.n;
        if b.len() != n {
            return Err(LinalgError::DimensionMismatch);
        }
        let c: Vec<T> = (0..n).map(|i| b[self.perm[i]]).collect();
        let mut u = vec![T::ZERO; n];
        for j in 0..n {
            let mut s = c[j];
            for &(k, ukj) in &self.ucols[j] {
                s -= ukj * u[k];
            }
            u[j] = s / self.udiag[j];
        }
        let mut w = vec![T::ZERO; n];
        for k in (0..n).rev() {
            let mut s = u[k];
            for &(i, l) in &self.lcols[k] {
                s -= l * w[self.pinv[i]];
            }
            w[k] = s;
        }
        let mut x = vec![T::ZERO; n];
        for k in 0..n {
            x[self.perm[self.prow[k]]] = w[k];
        }
        Ok(x)
    }

    /// Löst `Aᴴ·x = b` (konjugiert transponiert): `Aᴴx = b ⇔ Aᵀ·conj(x) = conj(b)`.
    pub fn solve_adjoint(&self, b: &[T]) -> Result<Vec<T>, LinalgError> {
        let cb: Vec<T> = b.iter().map(|v| v.conj()).collect();
        let x = self.solve_transposed(&cb)?;
        Ok(x.into_iter().map(|v| v.conj()).collect())
    }
}
