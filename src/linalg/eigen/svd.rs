use super::super::{LinalgError, Matrix, Vector};
use super::MAX_SWEEPS;
use super::kernel::jacobi_svd;
use crate::Real;

/// Singulärwertzerlegung `A = U·diag(s)·Vᵀ` einer `M × N`-Matrix mit `M ≥ N` (dünne SVD).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Svd<T, const M: usize, const N: usize> {
    /// Linke Singulärvektoren als Spalten (`M × N`, orthonormal). Zu Singulärwert 0 gehört eine
    /// Nullspalte (die Richtung ist dann nicht festgelegt).
    pub u: Matrix<T, M, N>,
    /// Singulärwerte, **absteigend** sortiert und nicht negativ.
    pub s: Vector<T, N>,
    /// Rechte Singulärvektoren als Spalten (`N × N`, orthogonal).
    pub v: Matrix<T, N, N>,
    /// Anzahl der benötigten Jacobi-Sweeps.
    pub sweeps: usize,
}

impl<T: Real, const M: usize, const N: usize> Svd<T, M, N> {
    /// Standard-Schwelle für den numerischen Rang: `max(M, N)·ε·σ_max`.
    pub fn default_tolerance(&self) -> T {
        let smax = if N == 0 { T::ZERO } else { self.s[0] };
        T::from_f64(M.max(N) as f64) * T::EPSILON * smax
    }

    /// Anzahl der Singulärwerte größer als `tol`.
    pub fn rank_with_tolerance(&self, tol: T) -> usize {
        (0..N).filter(|&i| self.s[i] > tol).count()
    }

    /// Numerischer Rang (Singulärwerte oberhalb von [`default_tolerance`](Self::default_tolerance)).
    pub fn rank(&self) -> usize {
        self.rank_with_tolerance(self.default_tolerance())
    }

    /// Konditionszahl `σ_max/σ_min` (`∞` bei Rangmangel, 0 für eine Nullmatrix mit `N = 0`).
    pub fn condition_number(&self) -> T {
        if N == 0 {
            return T::ZERO;
        }
        if self.s[N - 1] == T::ZERO {
            T::INFINITY
        } else {
            self.s[0] / self.s[N - 1]
        }
    }

    /// Rekonstruiert `U·diag(s)·Vᵀ` (zur Probe).
    pub fn reconstruct(&self) -> Matrix<T, M, N> {
        Matrix::from_fn(|i, j| {
            let mut sum = T::ZERO;
            for k in 0..N {
                sum += self.u[(i, k)] * self.s[k] * self.v[(j, k)];
            }
            sum
        })
    }

    /// Lösung `x` mit kleinster Norm unter allen Kleinste-Quadrate-Lösungen von `min ‖A·x − b‖`
    /// (`x = A⁺·b`). Funktioniert auch bei **Rangmangel**, wo QR scheitert; Singulärwerte unterhalb von
    /// [`default_tolerance`](Self::default_tolerance) werden als 0 behandelt.
    pub fn solve(&self, b: &Vector<T, M>) -> Vector<T, N> {
        self.solve_with_tolerance(b, self.default_tolerance())
    }

    /// Wie [`solve`](Self::solve) mit eigener Schwelle `tol` für die Singulärwerte.
    pub fn solve_with_tolerance(&self, b: &Vector<T, M>, tol: T) -> Vector<T, N> {
        let mut x = Vector::<T, N>::zeros();
        for k in 0..N {
            if self.s[k] > tol {
                let mut utb = T::ZERO;
                for i in 0..M {
                    utb += self.u[(i, k)] * b[i];
                }
                let coeff = utb / self.s[k];
                for j in 0..N {
                    x[j] += coeff * self.v[(j, k)];
                }
            }
        }
        x
    }

    /// Pseudoinverse `A⁺ = V·diag(1/s)·Uᵀ` (`N × M`), Singulärwerte unter der Standardschwelle zählen
    /// als 0.
    pub fn pseudo_inverse(&self) -> Matrix<T, N, M> {
        let tol = self.default_tolerance();
        Matrix::from_fn(|j, i| {
            let mut sum = T::ZERO;
            for k in 0..N {
                if self.s[k] > tol {
                    sum += self.v[(j, k)] * self.u[(i, k)] / self.s[k];
                }
            }
            sum
        })
    }
}

impl<T: Real, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Singulärwertzerlegung per einseitigem Jacobi-Verfahren (Hestenes). Erfordert `M ≥ N` (für
    /// `M < N` die Transponierte zerlegen). Sehr genaue Singulärwerte, auch kleine.
    ///
    /// Fehler: [`LinalgError::DimensionMismatch`] bei `M < N`, [`LinalgError::NotFinite`],
    /// [`LinalgError::NoConvergence`].
    ///
    /// ```
    /// # #[cfg(any(feature = "std", feature = "libm"))] {
    /// use azmath::linalg::Matrix;
    ///
    /// let a = Matrix::new([[3.0f64, 0.0], [0.0, -2.0], [0.0, 0.0]]);
    /// let svd = a.svd().unwrap();
    /// assert!((svd.s[0] - 3.0).abs() < 1e-14 && (svd.s[1] - 2.0).abs() < 1e-14);
    /// assert_eq!(svd.rank(), 2);
    /// # }
    /// ```
    pub fn svd(&self) -> Result<Svd<T, M, N>, LinalgError> {
        if M < N {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut u = *self;
        let mut s = Vector::<T, N>::zeros();
        let mut v = Matrix::<T, N, N>::zeros();
        let sweeps = jacobi_svd(
            u.data.as_flattened_mut(),
            s.as_mut_slice(),
            v.data.as_flattened_mut(),
            M,
            N,
            MAX_SWEEPS,
        )?;
        Ok(Svd { u, s, v, sweeps })
    }
}
