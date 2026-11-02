use super::super::{LinalgError, Matrix, Vector};
use super::MAX_SWEEPS;
use super::kernel::jacobi_eigen;
use super::power::{EigenResult, inverse_iteration, power_iteration};
use crate::LuDecomposition;
use crate::iterative::IterOptions;
use azmath_core::Real;

/// Spektralzerlegung `A = V·diag(λ)·Vᵀ` einer symmetrischen `N × N`-Matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SymmetricEigen<T, const N: usize> {
    /// Eigenwerte, **aufsteigend** sortiert.
    pub values: Vector<T, N>,
    /// Orthonormale Eigenvektoren als **Spalten**; Spalte `j` gehört zu `values[j]`.
    pub vectors: Matrix<T, N, N>,
    /// Anzahl der benötigten Jacobi-Sweeps.
    pub sweeps: usize,
}

impl<T: Real, const N: usize> SymmetricEigen<T, N> {
    /// Rekonstruiert `V·diag(λ)·Vᵀ` (zur Probe).
    pub fn reconstruct(&self) -> Matrix<T, N, N> {
        Matrix::from_fn(|i, j| {
            let mut s = T::ZERO;
            for k in 0..N {
                s += self.vectors[(i, k)] * self.values[k] * self.vectors[(j, k)];
            }
            s
        })
    }
}

fn start_vector<T: Real, const N: usize>() -> [T; N] {
    let mut x = [T::ZERO; N];
    super::fill_start_vector(&mut x);
    x
}

impl<T: Real, const N: usize> Matrix<T, N, N> {
    /// Vollständige Spektralzerlegung einer **symmetrischen** Matrix mit dem zyklischen
    /// Jacobi-Verfahren: alle Eigenwerte (aufsteigend) und orthonormale Eigenvektoren.
    ///
    /// Das Verfahren ist sehr genau (auch kleine Eigenwerte bekommen hohe relative Genauigkeit), aber
    /// mit `O(n³)` pro Sweep langsamer als QR-basierte Verfahren; es lohnt für kleine bis mittlere
    /// Matrizen. Es nutzt `sqrt` (siehe Hinweis zu `micromath`).
    ///
    /// Fehler: [`LinalgError::NotSymmetric`], [`LinalgError::NotFinite`],
    /// [`LinalgError::NoConvergence`].
    ///
    /// ```
    /// # #[cfg(any(feature = "std", feature = "libm"))] {
    /// use azmath_linalg::Matrix;
    ///
    /// let a = Matrix::new([[2.0f64, 1.0], [1.0, 2.0]]);
    /// let e = a.symmetric_eigen().unwrap();
    /// assert!((e.values[0] - 1.0).abs() < 1e-12 && (e.values[1] - 3.0).abs() < 1e-12);
    /// # }
    /// ```
    pub fn symmetric_eigen(&self) -> Result<SymmetricEigen<T, N>, LinalgError> {
        let mut a = *self;
        let mut v = Matrix::<T, N, N>::zeros();
        let sweeps = jacobi_eigen(
            a.data.as_flattened_mut(),
            v.data.as_flattened_mut(),
            N,
            MAX_SWEEPS,
        )?;
        Ok(SymmetricEigen {
            values: Vector::from_fn_diag(&a),
            vectors: v,
            sweeps,
        })
    }

    /// Betragsgrößter Eigenwert und zugehöriger (normierter) Eigenvektor per Potenzmethode, mit einem
    /// festen Pseudozufalls-Startvektor. Rückgabe: `(Eigenwert, Eigenvektor, Iterationen)`. Für einen
    /// eigenen Startvektor oder matrixfrei: [`power_iteration`].
    pub fn dominant_eigenpair(
        &self,
        opts: IterOptions<T>,
    ) -> Result<(T, Vector<T, N>, usize), LinalgError> {
        let mut x = start_vector::<T, N>();
        let mut work = [T::ZERO; N];
        let EigenResult {
            eigenvalue,
            iterations,
        } = power_iteration(self, &mut x, opts, &mut work)?;
        Ok((eigenvalue, Vector::new(x), iterations))
    }

    /// Eigenwert, der `shift` am nächsten liegt, samt Eigenvektor, per inverser Iteration (LU von
    /// `A − shift·I`). Rückgabe: `(Eigenwert, Eigenvektor, Iterationen)`.
    ///
    /// Fehler zusätzlich: [`LinalgError::Singular`], wenn `shift` genau auf einen Eigenwert fällt.
    pub fn eigenpair_near(
        &self,
        shift: T,
        opts: IterOptions<T>,
    ) -> Result<(T, Vector<T, N>, usize), LinalgError> {
        let mut shifted = *self;
        for i in 0..N {
            shifted.data[i][i] -= shift;
        }
        let lu = LuDecomposition::new(&shifted)?;
        let mut x = start_vector::<T, N>();
        let mut work = [T::ZERO; N];
        let solve = |b: &[T], out: &mut [T]| {
            let mut rhs = [T::ZERO; N];
            rhs.copy_from_slice(b);
            out.copy_from_slice(lu.solve(&Vector::new(rhs)).as_slice());
        };
        let EigenResult {
            eigenvalue,
            iterations,
        } = inverse_iteration(solve, shift, &mut x, opts, &mut work)?;
        Ok((eigenvalue, Vector::new(x), iterations))
    }
}

impl<T: Real, const N: usize> Vector<T, N> {
    /// Diagonale einer Matrix als Vektor.
    fn from_fn_diag(m: &Matrix<T, N, N>) -> Self {
        Vector::new(core::array::from_fn(|i| m.data[i][i]))
    }
}
