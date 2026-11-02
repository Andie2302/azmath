use super::super::{LinalgError, Matrix};
use crate::Real;

/// Ein Vorkonditionierer `M ≈ A`: berechnet `z = M⁻¹·r`.
///
/// Ein guter Vorkonditionierer ist billig anzuwenden und macht `M⁻¹·A` besser konditioniert,
/// wodurch das Verfahren mit weniger Iterationen konvergiert.
pub trait Preconditioner<T: Real> {
    /// Setzt `z = M⁻¹·r`. `r` und `z` haben gleiche Länge.
    fn apply(&self, r: &[T], z: &mut [T]);
}

impl<T: Real, M: Preconditioner<T> + ?Sized> Preconditioner<T> for &M {
    fn apply(&self, r: &[T], z: &mut [T]) {
        (**self).apply(r, z)
    }
}

/// Der triviale Vorkonditionierer `M = I` (keine Vorkonditionierung).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Identity;

impl<T: Real> Preconditioner<T> for Identity {
    fn apply(&self, r: &[T], z: &mut [T]) {
        z.copy_from_slice(r);
    }
}

/// Jacobi-Vorkonditionierer `M = diag(A)` für eine `N × N`-Matrix.
///
/// Hilft vor allem bei Matrizen mit stark unterschiedlich großen Diagonaleinträgen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Jacobi<T, const N: usize> {
    inv_diag: [T; N],
}

impl<T: Real, const N: usize> Jacobi<T, N> {
    /// Baut den Vorkonditionierer aus der Diagonale von `a`.
    ///
    /// Fehler: [`LinalgError::Singular`], wenn ein Diagonalelement 0 oder NaN ist.
    pub fn new(a: &Matrix<T, N, N>) -> Result<Self, LinalgError> {
        let mut inv_diag = [T::ZERO; N];
        for (i, inv) in inv_diag.iter_mut().enumerate() {
            let d = a.data[i][i];
            if d == T::ZERO || d.is_nan() {
                return Err(LinalgError::Singular);
            }
            *inv = T::ONE / d;
        }
        Ok(Self { inv_diag })
    }
}

impl<T: Real, const N: usize> Preconditioner<T> for Jacobi<T, N> {
    fn apply(&self, r: &[T], z: &mut [T]) {
        for ((zi, ri), d) in z.iter_mut().zip(r).zip(&self.inv_diag) {
            *zi = *ri * *d;
        }
    }
}

/// Jacobi-Vorkonditionierer für dynamisch große Matrizen (Feature `alloc`).
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq)]
pub struct DynJacobi<T> {
    inv_diag: alloc::vec::Vec<T>,
}

#[cfg(feature = "alloc")]
impl<T: Real> DynJacobi<T> {
    /// Baut den Vorkonditionierer aus der Diagonale von `a`.
    ///
    /// Fehler: [`LinalgError::DimensionMismatch`] (nicht quadratisch),
    /// [`LinalgError::Singular`] (Diagonalelement 0 oder NaN).
    pub fn new(a: &super::super::DynMatrix<T>) -> Result<Self, LinalgError> {
        if a.rows() != a.cols() {
            return Err(LinalgError::DimensionMismatch);
        }
        let n = a.rows();
        let mut inv_diag = alloc::vec::Vec::with_capacity(n);
        for i in 0..n {
            let d = a[(i, i)];
            if d == T::ZERO || d.is_nan() {
                return Err(LinalgError::Singular);
            }
            inv_diag.push(T::ONE / d);
        }
        Ok(Self { inv_diag })
    }
}

#[cfg(feature = "alloc")]
impl<T: Real> Preconditioner<T> for DynJacobi<T> {
    fn apply(&self, r: &[T], z: &mut [T]) {
        for ((zi, ri), d) in z.iter_mut().zip(r).zip(&self.inv_diag) {
            *zi = *ri * *d;
        }
    }
}
