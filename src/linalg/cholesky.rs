use super::kernel::{
    cholesky_decompose, cholesky_determinant, cholesky_solve, cholesky_solve_checked,
};
use super::{LinalgError, Matrix, Vector};
use crate::Real;

/// Cholesky-Zerlegung `A = L·Lᵀ` einer symmetrisch positiv definiten `N x N`-Matrix.
///
/// Etwa doppelt so schnell wie die LU-Zerlegung und ohne Pivotisierung stabil. Es wird die
/// untere Dreiecksmatrix verwendet; die Symmetrie wird geprüft.
///
/// Hinweis: Die Zerlegung benötigt `sqrt`. Mit dem Backend `micromath` ist die Genauigkeit
/// deshalb so gering wie dessen Näherung; für genaue Ergebnisse `std` oder `libm` verwenden.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cholesky<T, const N: usize> {
    l: Matrix<T, N, N>,
}

impl<T: Real, const N: usize> Cholesky<T, N> {
    /// Berechnet die Zerlegung von `a`.
    ///
    /// Fehler: [`LinalgError::NotSymmetric`], [`LinalgError::NotPositiveDefinite`],
    /// [`LinalgError::NotFinite`].
    pub fn new(a: &Matrix<T, N, N>) -> Result<Self, LinalgError> {
        let mut l = *a;
        cholesky_decompose(l.data.as_flattened_mut(), N)?;
        Ok(Self { l })
    }

    /// Die untere Dreiecksmatrix `L`.
    pub fn l(&self) -> &Matrix<T, N, N> {
        &self.l
    }

    /// Löst `A·x = b` ohne Fehlerprüfung; siehe [`Self::try_solve`] für Fehlerdiagnosen.
    pub fn solve(&self, b: &Vector<T, N>) -> Vector<T, N> {
        let mut x = Vector::<T, N>::zeros();
        cholesky_solve(self.l.data.as_flattened(), N, &b.data, &mut x.data);
        x
    }

    /// Löst `A·x = b` mit Prüfung der rechten Seite und beider Substitutionen.
    ///
    /// [`LinalgError::NotFinite`] bei NaN/Unendlich in `b` oder nichtendlichen Rechenergebnissen.
    pub fn try_solve(&self, b: &Vector<T, N>) -> Result<Vector<T, N>, LinalgError> {
        let mut x = Vector::<T, N>::zeros();
        cholesky_solve_checked(self.l.data.as_flattened(), N, &b.data, &mut x.data)?;
        Ok(x)
    }

    /// Determinante von `A`.
    pub fn determinant(&self) -> T {
        cholesky_determinant(self.l.data.as_flattened(), N)
    }

    /// Inverse von `A`.
    pub fn inverse(&self) -> Matrix<T, N, N> {
        let mut inv = Matrix::<T, N, N>::zeros();
        for j in 0..N {
            let mut e = Vector::<T, N>::zeros();
            e.data[j] = T::ONE;
            let col = self.solve(&e);
            for i in 0..N {
                inv.data[i][j] = col.data[i];
            }
        }
        inv
    }
}
