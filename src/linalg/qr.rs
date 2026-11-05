use super::kernel::{
    qr_apply_q, qr_apply_qt, qr_decompose, qr_residual_norm, qr_solve, qr_solve_checked,
};
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
    ///
    /// Ohne Prüfung der rechten Seite oder des Ergebnisses; für Fehlerdiagnosen siehe [`Self::try_solve`].
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

    /// Wie [`Self::solve`], aber mit Fehlerprüfung.
    ///
    /// [`LinalgError::NotFinite`] bei NaN/Unendlich in `b`, nichtendlichen
    /// Zwischenergebnissen oder einer nicht darstellbaren Lösung.
    pub fn try_solve(&self, b: &Vector<T, M>) -> Result<Vector<T, N>, LinalgError> {
        let mut y = b.data;
        let mut x = Vector::<T, N>::zeros();
        qr_solve_checked(
            self.qr.data.as_flattened(),
            M,
            N,
            &self.tau,
            &mut y,
            &mut x.data,
        )?;
        Ok(x)
    }

    /// Euklidische Norm des Residuums `‖A·x − b‖₂` der Kleinste-Quadrate-Lösung.
    /// Für Fehlerdiagnosen siehe [`Self::try_residual_norm`].
    pub fn residual_norm(&self, b: &Vector<T, M>) -> T {
        let mut y = b.data;
        qr_apply_qt(self.qr.data.as_flattened(), M, N, &self.tau, &mut y);
        qr_residual_norm(&y, N)
    }

    /// Geprüfte Variante von [`Self::residual_norm`].
    ///
    /// [`LinalgError::NotFinite`] bei NaN/Unendlich in `b` oder nichtendlichen
    /// Zwischenergebnissen bzw. einer nicht darstellbaren Residuumsnorm.
    pub fn try_residual_norm(&self, b: &Vector<T, M>) -> Result<T, LinalgError> {
        if b.data.iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        let mut y = b.data;
        qr_apply_qt(self.qr.data.as_flattened(), M, N, &self.tau, &mut y);
        let norm = qr_residual_norm(&y, N);
        if y.iter().any(|v| !v.is_finite()) || !norm.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        Ok(norm)
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

    /// Das dünne `Q` (`M × N`, erste `N` Spalten der vollen orthogonalen Matrix); stabile Alternative zu
    /// Gram-Schmidt.
    pub fn thin_q(&self) -> Matrix<T, M, N> {
        let mut q = Matrix::<T, M, N>::zeros();
        for j in 0..N {
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
