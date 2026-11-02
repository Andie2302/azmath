//! Dynamisch große Eigenwert- und SVD-Berechnungen (Feature `alloc`).

use alloc::vec;
use alloc::vec::Vec;

use super::super::iterative::IterOptions;
use super::super::{DynMatrix, DynVector, LinalgError};
use super::MAX_SWEEPS;
use super::kernel::{jacobi_eigen, jacobi_svd};
use super::power::{EigenResult, inverse_iteration, power_iteration};
use azmath_core::Real;

/// Spektralzerlegung einer symmetrischen dynamischen Matrix, siehe
/// [`SymmetricEigen`](super::SymmetricEigen).
#[derive(Debug, Clone, PartialEq)]
pub struct DynSymmetricEigen<T> {
    /// Eigenwerte, aufsteigend sortiert.
    pub values: DynVector<T>,
    /// Orthonormale Eigenvektoren als Spalten.
    pub vectors: DynMatrix<T>,
    /// Anzahl der benötigten Jacobi-Sweeps.
    pub sweeps: usize,
}

/// Dünne SVD einer dynamischen `m × n`-Matrix mit `m ≥ n`, siehe [`Svd`](super::Svd).
#[derive(Debug, Clone, PartialEq)]
pub struct DynSvd<T> {
    /// Linke Singulärvektoren als Spalten (`m × n`; Nullspalte bei Singulärwert 0).
    pub u: DynMatrix<T>,
    /// Singulärwerte, absteigend sortiert.
    pub s: DynVector<T>,
    /// Rechte Singulärvektoren als Spalten (`n × n`).
    pub v: DynMatrix<T>,
    /// Anzahl der benötigten Jacobi-Sweeps.
    pub sweeps: usize,
}

impl<T: Real> DynSvd<T> {
    fn dims(&self) -> (usize, usize) {
        (self.u.rows(), self.u.cols())
    }

    /// Standard-Schwelle für den numerischen Rang: `max(m, n)·ε·σ_max`.
    pub fn default_tolerance(&self) -> T {
        let (m, n) = self.dims();
        let smax = if n == 0 { T::ZERO } else { self.s[0] };
        T::from_f64(m.max(n) as f64) * T::EPSILON * smax
    }

    /// Anzahl der Singulärwerte größer als `tol`.
    pub fn rank_with_tolerance(&self, tol: T) -> usize {
        self.s.as_slice().iter().filter(|&&v| v > tol).count()
    }

    /// Numerischer Rang.
    pub fn rank(&self) -> usize {
        self.rank_with_tolerance(self.default_tolerance())
    }

    /// Konditionszahl `σ_max/σ_min` (`∞` bei Rangmangel).
    pub fn condition_number(&self) -> T {
        let n = self.s.len();
        if n == 0 {
            return T::ZERO;
        }
        if self.s[n - 1] == T::ZERO {
            T::INFINITY
        } else {
            self.s[0] / self.s[n - 1]
        }
    }

    /// Rekonstruiert `U·diag(s)·Vᵀ`.
    pub fn reconstruct(&self) -> DynMatrix<T> {
        let (m, n) = self.dims();
        let mut out = DynMatrix::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::ZERO;
                for k in 0..n {
                    sum += self.u[(i, k)] * self.s[k] * self.v[(j, k)];
                }
                out[(i, j)] = sum;
            }
        }
        out
    }

    /// Minimale-Norm-Lösung von `min ‖A·x − b‖` (`x = A⁺·b`); Fehler, wenn die Länge von `b` nicht `m`
    /// ist.
    pub fn solve(&self, b: &DynVector<T>) -> Result<DynVector<T>, LinalgError> {
        self.solve_with_tolerance(b, self.default_tolerance())
    }

    /// Wie [`solve`](Self::solve) mit eigener Schwelle `tol` für die Singulärwerte.
    pub fn solve_with_tolerance(
        &self,
        b: &DynVector<T>,
        tol: T,
    ) -> Result<DynVector<T>, LinalgError> {
        let (m, n) = self.dims();
        if b.len() != m {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut x = DynVector::zeros(n);
        for k in 0..n {
            if self.s[k] > tol {
                let mut utb = T::ZERO;
                for i in 0..m {
                    utb += self.u[(i, k)] * b[i];
                }
                let coeff = utb / self.s[k];
                for j in 0..n {
                    x[j] += coeff * self.v[(j, k)];
                }
            }
        }
        Ok(x)
    }

    /// Pseudoinverse `A⁺` (`n × m`).
    pub fn pseudo_inverse(&self) -> DynMatrix<T> {
        let (m, n) = self.dims();
        let tol = self.default_tolerance();
        let mut out = DynMatrix::zeros(n, m);
        for j in 0..n {
            for i in 0..m {
                let mut sum = T::ZERO;
                for k in 0..n {
                    if self.s[k] > tol {
                        sum += self.v[(j, k)] * self.u[(i, k)] / self.s[k];
                    }
                }
                out[(j, i)] = sum;
            }
        }
        out
    }
}

fn start_vector<T: Real>(n: usize) -> Vec<T> {
    let mut x = vec![T::ZERO; n];
    super::fill_start_vector(&mut x);
    x
}

impl<T: Real> DynMatrix<T> {
    /// Vollständige Spektralzerlegung einer symmetrischen quadratischen Matrix
    /// (zyklisches Jacobi-Verfahren), siehe [`Matrix::symmetric_eigen`](crate::Matrix).
    ///
    /// Fehler: [`LinalgError::DimensionMismatch`] (nicht quadratisch), [`LinalgError::NotSymmetric`],
    /// [`LinalgError::NotFinite`], [`LinalgError::NoConvergence`].
    pub fn symmetric_eigen(&self) -> Result<DynSymmetricEigen<T>, LinalgError> {
        let n = self.rows();
        if n != self.cols() {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut a = self.as_slice().to_vec();
        let mut v = vec![T::ZERO; n * n];
        let sweeps = jacobi_eigen(&mut a, &mut v, n, MAX_SWEEPS)?;
        let values = DynVector::new((0..n).map(|i| a[i * n + i]).collect());
        Ok(DynSymmetricEigen {
            values,
            vectors: DynMatrix::new(n, n, v)?,
            sweeps,
        })
    }

    /// Singulärwertzerlegung (einseitiges Jacobi-Verfahren); erfordert `rows ≥ cols`.
    ///
    /// Fehler: [`LinalgError::DimensionMismatch`] bei `rows < cols`, [`LinalgError::NotFinite`],
    /// [`LinalgError::NoConvergence`].
    pub fn svd(&self) -> Result<DynSvd<T>, LinalgError> {
        let (m, n) = (self.rows(), self.cols());
        if m < n {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut u = self.as_slice().to_vec();
        let mut s = vec![T::ZERO; n];
        let mut v = vec![T::ZERO; n * n];
        let sweeps = jacobi_svd(&mut u, &mut s, &mut v, m, n, MAX_SWEEPS)?;
        Ok(DynSvd {
            u: DynMatrix::new(m, n, u)?,
            s: DynVector::new(s),
            v: DynMatrix::new(n, n, v)?,
            sweeps,
        })
    }

    /// Betragsgrößter Eigenwert und Eigenvektor per Potenzmethode. Rückgabe:
    /// `(Eigenwert, Eigenvektor, Iterationen)`.
    pub fn dominant_eigenpair(
        &self,
        opts: IterOptions<T>,
    ) -> Result<(T, DynVector<T>, usize), LinalgError> {
        let n = self.rows();
        let mut x = start_vector::<T>(n);
        let mut work = vec![T::ZERO; n];
        let EigenResult {
            eigenvalue,
            iterations,
        } = power_iteration(self, &mut x, opts, &mut work)?;
        Ok((eigenvalue, DynVector::new(x), iterations))
    }

    /// Eigenwert nahe `shift` samt Eigenvektor per inverser Iteration (LU von `A − shift·I`).
    /// Rückgabe: `(Eigenwert, Eigenvektor, Iterationen)`.
    pub fn eigenpair_near(
        &self,
        shift: T,
        opts: IterOptions<T>,
    ) -> Result<(T, DynVector<T>, usize), LinalgError> {
        let n = self.rows();
        if n != self.cols() {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut shifted = self.clone();
        for i in 0..n {
            shifted[(i, i)] -= shift;
        }
        let lu = shifted.lu()?;
        let mut x = start_vector::<T>(n);
        let mut work = vec![T::ZERO; n];
        let solve = |b: &[T], out: &mut [T]| {
            let y = lu
                .solve(&DynVector::new(b.to_vec()))
                .expect("Längen stimmen überein");
            out.copy_from_slice(y.as_slice());
        };
        let EigenResult {
            eigenvalue,
            iterations,
        } = inverse_iteration(solve, shift, &mut x, opts, &mut work)?;
        Ok((eigenvalue, DynVector::new(x), iterations))
    }
}
