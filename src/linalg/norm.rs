//! Matrixnormen: Frobenius-, 1-, ∞- und 2-Norm (letztere über die SVD).
//!
//! | Methode | Norm | Aufwand |
//! |---|---|---|
//! | `norm_frobenius` | `√Σ aᵢⱼ²` (überlaufsicher skaliert) | `O(mn)` |
//! | `norm_1` | größte Spaltensumme `maxⱼ Σᵢ |aᵢⱼ|` | `O(mn)` |
//! | `norm_inf` | größte Zeilensumme `maxᵢ Σⱼ |aᵢⱼ|` | `O(mn)` |
//! | `norm_2` | größter Singulärwert `σ_max` | Jacobi-SVD, deutlich teurer |
//!
//! Die Maximumnorm ist `max_abs`. NaN in der Matrix ergibt bei den drei einfachen Normen NaN.

use super::{LinalgError, Matrix, Vector};
use crate::Real;

/// Maximum, das NaN weitergibt (`Real::max` würde NaN verschlucken).
fn max_nan<T: Real>(a: T, b: T) -> T {
    if a.is_nan() || b.is_nan() {
        T::NAN
    } else {
        a.max(b)
    }
}

/// Frobenius-Norm eines Slices; skaliert mit dem größten Betrag gegen Über- und Unterlauf.
pub(crate) fn frobenius<T: Real>(a: &[T]) -> T {
    let mut m = T::ZERO;
    for &x in a {
        m = max_nan(m, x.abs());
    }
    if m.is_nan() || m.is_infinite() || m == T::ZERO {
        return m;
    }
    let mut s = T::ZERO;
    for &x in a {
        let r = x / m;
        s += r * r;
    }
    m * s.sqrt()
}

/// Größte Spaltensumme einer zeilenweise gespeicherten `rows × cols`-Matrix.
pub(crate) fn norm_1<T: Real>(a: &[T], rows: usize, cols: usize) -> T {
    let mut best = T::ZERO;
    for j in 0..cols {
        let mut s = T::ZERO;
        for i in 0..rows {
            s += a[i * cols + j].abs();
        }
        best = max_nan(best, s);
    }
    best
}

/// Größte Zeilensumme einer zeilenweise gespeicherten `rows × cols`-Matrix.
pub(crate) fn norm_inf<T: Real>(a: &[T], rows: usize, cols: usize) -> T {
    let mut best = T::ZERO;
    for i in 0..rows {
        let mut s = T::ZERO;
        for j in 0..cols {
            s += a[i * cols + j].abs();
        }
        best = max_nan(best, s);
    }
    best
}

impl<T: Real, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Frobenius-Norm `√Σ aᵢⱼ²`, überlaufsicher.
    ///
    /// ```
    /// use azmath::linalg::Matrix;
    /// let a = Matrix::new([[3.0f64, 0.0], [0.0, 4.0]]);
    /// assert_eq!(a.norm_frobenius(), 5.0);
    /// ```
    pub fn norm_frobenius(&self) -> T {
        frobenius(self.data.as_flattened())
    }

    /// 1-Norm: größte Spaltensumme.
    pub fn norm_1(&self) -> T {
        norm_1(self.data.as_flattened(), M, N)
    }

    /// ∞-Norm: größte Zeilensumme.
    pub fn norm_inf(&self) -> T {
        norm_inf(self.data.as_flattened(), M, N)
    }

    /// Spektralnorm (2-Norm) = größter Singulärwert, per Jacobi-SVD; für `M < N` wird die Transponierte zerlegt.
    ///
    /// Fehler wie bei [`svd`](Matrix::svd): [`LinalgError::NotFinite`], [`LinalgError::NoConvergence`].
    ///
    /// ```
    /// # #[cfg(any(feature = "std", feature = "libm"))] {
    /// use azmath::linalg::Matrix;
    /// let a = Matrix::new([[3.0f64, 0.0], [0.0, -2.0], [0.0, 0.0]]);
    /// assert!((a.norm_2().unwrap() - 3.0).abs() < 1e-14);
    /// # }
    /// ```
    pub fn norm_2(&self) -> Result<T, LinalgError> {
        if M == 0 || N == 0 {
            return Ok(T::ZERO);
        }
        let svd = if M >= N {
            self.svd()?
        } else {
            // svd verlangt M ≥ N; σ(A) = σ(Aᵀ)
            return Ok(self.transpose().svd()?.s[0]);
        };
        Ok(svd.s[0])
    }
}

impl<T: Real, const N: usize> Vector<T, N> {
    /// 1-Norm `Σ |xᵢ|`.
    pub fn norm_1(&self) -> T {
        self.as_slice().iter().fold(T::ZERO, |s, x| s + x.abs())
    }
}

#[cfg(feature = "alloc")]
mod dynamic {
    use super::{frobenius, norm_1, norm_inf};
    use crate::Real;
    use crate::linalg::{DynMatrix, DynVector, LinalgError};

    impl<T: Real> DynMatrix<T> {
        /// Frobenius-Norm `√Σ aᵢⱼ²`, überlaufsicher.
        pub fn norm_frobenius(&self) -> T {
            frobenius(self.as_slice())
        }

        /// 1-Norm: größte Spaltensumme.
        pub fn norm_1(&self) -> T {
            norm_1(self.as_slice(), self.rows(), self.cols())
        }

        /// ∞-Norm: größte Zeilensumme.
        pub fn norm_inf(&self) -> T {
            norm_inf(self.as_slice(), self.rows(), self.cols())
        }

        /// Spektralnorm (2-Norm) = größter Singulärwert; für `rows < cols` wird die Transponierte zerlegt.
        pub fn norm_2(&self) -> Result<T, LinalgError> {
            if self.rows() == 0 || self.cols() == 0 {
                return Ok(T::ZERO);
            }
            let svd = if self.rows() >= self.cols() {
                self.svd()?
            } else {
                self.transpose().svd()?
            };
            Ok(svd.s[0])
        }
    }

    impl<T: Real> DynVector<T> {
        /// 1-Norm `Σ |xᵢ|`.
        pub fn norm_1(&self) -> T {
            self.as_slice().iter().fold(T::ZERO, |s, x| s + x.abs())
        }
    }
}
