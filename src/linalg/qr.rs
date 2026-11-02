use super::kernel::{qr_apply_q, qr_apply_qt, qr_decompose, qr_residual_norm, qr_solve};
use super::{LinalgError, Matrix, Vector};
use crate::Real;

/// QR-Zerlegung `A = Q·R` einer `M x N`-Matrix (`M >= N`) mit Householder-Spiegelungen.
///
/// Löst auch überbestimmte Systeme im Sinne der kleinsten Quadrate (`min ‖A·x − b‖₂`),
/// zum Beispiel für Ausgleichsgeraden. Für quadratische Matrizen ist es ein stabiler,
/// aber etwas langsamerer Ersatz für die LU-Zerlegung.
///
/// Hinweis: Die Zerlegung benötigt `sqrt`. Mit dem Backend `micromath` ist die Genauigkeit
/// deshalb so gering wie dessen Näherung; für genaue Ergebnisse `std` oder `libm` verwenden.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QrDecomposition<T, const M: usize, const N: usize> {
    qr: Matrix<T, M, N>,
    tau: [T; N],
}

impl<T: Real, const M: usize, const N: usize> QrDecomposition<T, M, N> {
    /// Berechnet die Zerlegung von `a`.
    ///
    /// Fehler: [`LinalgError::DimensionMismatch`] bei `M < N`, [`LinalgError::Singular`] bei
    /// Rangdefizit (die Spalten sind linear abhängig), [`LinalgError::NotFinite`].
    pub fn new(a: &Matrix<T, M, N>) -> Result<Self, LinalgError> {
        if M < N {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut qr = *a;
        let mut tau = [T::ZERO; N];
        qr_decompose(qr.data.as_flattened_mut(), M, N, &mut tau)?;
        Ok(Self { qr, tau })
    }

    /// Löst `min ‖A·x − b‖₂` (für quadratische `A` genau `A·x = b`).
    pub fn solve(&self, b: &Vector<T, M>) -> Vector<T, N> {
        let mut y = b.data;
        let mut x = Vector::<T, N>::zeros();
        qr_solve(
            self.qr.data.as_flattened(),
            M,
            N,
            &self.tau,
            &mut y,
            &mut x.data,
        );
        x
    }

    /// Euklidische Norm des Residuums `‖A·x − b‖₂` der Kleinste-Quadrate-Lösung.
    pub fn residual_norm(&self, b: &Vector<T, M>) -> T {
        let mut y = b.data;
        qr_apply_qt(self.qr.data.as_flattened(), M, N, &self.tau, &mut y);
        qr_residual_norm(&y, N)
    }

    /// Die obere Dreiecksmatrix `R` (`N x N`).
    pub fn r(&self) -> Matrix<T, N, N> {
        Matrix::<T, N, N>::from_fn(|i, j| if j >= i { self.qr.data[i][j] } else { T::ZERO })
    }

    /// Die orthogonale Matrix `Q` (`M x M`).
    pub fn q(&self) -> Matrix<T, M, M> {
        let mut q = Matrix::<T, M, M>::zeros();
        for j in 0..M {
            let mut col = [T::ZERO; M];
            col[j] = T::ONE;
            qr_apply_q(self.qr.data.as_flattened(), M, N, &self.tau, &mut col);
            for (row, &v) in q.data.iter_mut().zip(col.iter()) {
                row[j] = v;
            }
        }
        q
    }
}
