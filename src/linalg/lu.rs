use super::kernel::{lu_decompose, lu_determinant, lu_solve};
use super::{LinalgError, Matrix, Vector};
use crate::Real;

/// LU-Zerlegung `P·A = L·U` einer `N x N`-Matrix mit Teilpivotisierung.
///
/// Einmal berechnet, lassen sich damit beliebig viele rechte Seiten günstig lösen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LuDecomposition<T, const N: usize> {
    lu: Matrix<T, N, N>,
    original: Matrix<T, N, N>,
    perm: [usize; N],
    odd: bool,
}

impl<T: Real, const N: usize> LuDecomposition<T, N> {
    /// Berechnet die Zerlegung von `a`.
    ///
    /// Fehler: [`LinalgError::Singular`] bei (numerisch) singulärer Matrix,
    /// [`LinalgError::NotFinite`] bei NaN/unendlichen Einträgen oder Überlauf der Zerlegung.
    pub fn new(a: &Matrix<T, N, N>) -> Result<Self, LinalgError> {
        let mut lu = *a;
        let mut perm = [0usize; N];
        let odd = lu_decompose(lu.data.as_flattened_mut(), N, &mut perm)?;
        Ok(Self {
            lu,
            original: *a,
            perm,
            odd,
        })
    }

    /// Löst `A·x = b`.
    pub fn solve(&self, b: &Vector<T, N>) -> Vector<T, N> {
        let mut x = Vector::<T, N>::zeros();
        lu_solve(
            self.lu.data.as_flattened(),
            N,
            &self.perm,
            &b.data,
            &mut x.data,
        );
        x
    }

    /// Schätzt `1/κ∞(A)` ohne explizite Inverse (Hager mit alternierendem Testvektor).
    /// Werte nahe 0 zeigen hohe Empfindlichkeit; dies ist keine zertifizierte Schranke.
    /// Leere Matrizen ergeben 1. Globale Skalierung wird intern herausgerechnet.
    pub fn rcond_inf(&self) -> Result<T, LinalgError> {
        super::lu_refine::rcond(
            self.original.data.as_flattened(),
            self.lu.data.as_flattened(),
            N,
            &self.perm,
            &mut [T::ZERO; N],
            &mut [T::ZERO; N],
            &mut [T::ZERO; N],
            &mut [T::ZERO; N],
        )
    }

    /// Verbessert eine vorhandene Lösung mit kompensierten Residuen und wiederverwendeter LU.
    /// Die ursprüngliche Matrix wird in der Zerlegung gespeichert. Beste Lösung bleibt erhalten.
    /// Toleranz/Limit/Stagnation werden im Bericht unterschieden. Gleiche Rechengenauigkeit wie T;
    /// keine Garantie auf Verbesserung oder kleinen Vorwärtsfehler. Nicht darstellbare
    /// Zwischenwerte ergeben `NotFinite`, ungültige Toleranzen `InvalidInput`.
    pub fn refine(
        &self,
        b: &Vector<T, N>,
        x: &mut Vector<T, N>,
        opts: super::RefinementOptions<T>,
    ) -> Result<super::RefinementResult<T>, LinalgError> {
        super::lu_refine::refine(
            self.original.data.as_flattened(),
            self.lu.data.as_flattened(),
            N,
            &self.perm,
            &b.data,
            &mut x.data,
            opts,
            &mut [T::ZERO; N],
            &mut [T::ZERO; N],
            &mut [T::ZERO; N],
            &mut [T::ZERO; N],
        )
    }

    /// Löst das System und verfeinert anschließend die Lösung; siehe [`Self::refine`].
    pub fn solve_refined(
        &self,
        b: &Vector<T, N>,
        opts: super::RefinementOptions<T>,
    ) -> Result<(Vector<T, N>, super::RefinementResult<T>), LinalgError> {
        let mut x = self.solve(b);
        let report = self.refine(b, &mut x, opts)?;
        Ok((x, report))
    }

    /// Determinante von `A`.
    pub fn determinant(&self) -> T {
        lu_determinant(self.lu.data.as_flattened(), N, self.odd)
    }

    /// Inverse von `A` (löst `A·X = I` spaltenweise).
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

    /// Die Zeilenpermutation: Zeile `i` von `P·A` ist Zeile `permutation()[i]` von `A`.
    pub fn permutation(&self) -> &[usize; N] {
        &self.perm
    }
}
