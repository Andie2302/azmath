//! Orthonormalisierung und Polarzerlegung.
//!
//! | Funktion | Ergebnis |
//! |---|---|
//! | `orthonormalize_mgs` | `A = Q·R` mit modifiziertem Gram-Schmidt (optional mit Reorthogonalisierung); meldet Rangmangel |
//! | `QrDecomposition::thin_q` (in `qr.rs`) | dünnes `Q` (`M × N`) aus Householder-QR, stabil auch bei Rangmangel |
//! | `polar` | `A = U·P` mit orthonormalen Spalten `U` und symmetrisch positiv semidefinitem `P` (über die SVD) |
//! | `nearest_orthogonal` | die orthogonale Matrix, die `A` in der Frobenius-Norm am nächsten liegt (`U` der Polarzerlegung) |
//! | `nearest_rotation` | wie oben, aber mit `det = +1` (Drehung statt Spiegelung) |
//! | `orthogonality_error` | `‖QᵀQ − I‖_F` |

use super::eigen::Svd;
use super::{LinalgError, Matrix};
use crate::Real;

fn finite<T: Real>(x: T) -> bool {
    !x.is_nan() && !x.is_infinite()
}

/// Modifiziertes Gram-Schmidt auf zeilenweise gespeicherten `m × n`-Daten (`m ≥ n`).
/// `q` enthält vorab eine Kopie von `a` und danach die orthonormalen Spalten; `r` (`n × n`) wird gefüllt.
pub(crate) fn mgs<T: Real>(
    q: &mut [T],
    r: &mut [T],
    m: usize,
    n: usize,
    reorthogonalize: bool,
) -> Result<(), LinalgError> {
    if m < n {
        return Err(LinalgError::DimensionMismatch);
    }
    if q.iter().any(|&x| !finite(x)) {
        return Err(LinalgError::NotFinite);
    }
    r.fill(T::ZERO);
    let tol = T::from_f64(n as f64) * T::EPSILON;
    for j in 0..n {
        let mut norm0 = T::ZERO;
        for i in 0..m {
            norm0 = norm0.hypot(q[i * n + j]);
        }
        let passes = if reorthogonalize { 2 } else { 1 };
        for _ in 0..passes {
            for k in 0..j {
                let mut c = T::ZERO;
                for i in 0..m {
                    c += q[i * n + k] * q[i * n + j];
                }
                r[k * n + j] += c;
                for i in 0..m {
                    let qk = q[i * n + k];
                    q[i * n + j] -= c * qk;
                }
            }
        }
        let mut rjj = T::ZERO;
        for i in 0..m {
            rjj = rjj.hypot(q[i * n + j]);
        }
        if rjj == T::ZERO || rjj <= tol * norm0 {
            return Err(LinalgError::Singular);
        }
        for i in 0..m {
            q[i * n + j] /= rjj;
        }
        r[j * n + j] = rjj;
    }
    Ok(())
}

/// `‖QᵀQ − I‖_F` für eine zeilenweise `m × n`-Matrix (ohne Allokation, überlaufsicher skaliert).
pub(crate) fn orthogonality_error<T: Real>(q: &[T], m: usize, n: usize) -> T {
    let entry = |j: usize, k: usize| {
        let mut c = T::ZERO;
        for i in 0..m {
            c += q[i * n + j] * q[i * n + k];
        }
        if j == k { c - T::ONE } else { c }
    };
    let mut scale = T::ZERO;
    for j in 0..n {
        for k in 0..n {
            let e = entry(j, k).abs();
            if e.is_nan() {
                return T::NAN;
            }
            scale = scale.max(e);
        }
    }
    if scale == T::ZERO || scale.is_infinite() {
        return scale;
    }
    let mut sum = T::ZERO;
    for j in 0..n {
        for k in 0..n {
            let e = entry(j, k) / scale;
            sum += e * e;
        }
    }
    scale * sum.sqrt()
}

/// Polarfaktoren aus einer SVD `A = Us·diag(s)·Vᵀ` (`us`: `m × n`, `v`: `n × n`, zeilenweise):
/// `u = Us·D·Vᵀ`, `p = V·diag(s)·Vᵀ`. Mit `flip_last` wird die letzte Singulärrichtung gespiegelt
/// (`D = diag(1, …, 1, −1)`), um `det(u) = +1` zu erzwingen.
pub(crate) fn polar_from_svd<T: Real>(
    (us, s, v): (&[T], &[T], &[T]),
    (m, n): (usize, usize),
    flip_last: bool,
    (u, p): (&mut [T], &mut [T]),
) {
    for i in 0..m {
        for j in 0..n {
            let mut acc = T::ZERO;
            for k in 0..n {
                let d = if flip_last && k + 1 == n {
                    -T::ONE
                } else {
                    T::ONE
                };
                acc += us[i * n + k] * d * v[j * n + k];
            }
            u[i * n + j] = acc;
        }
    }
    for i in 0..n {
        for j in 0..n {
            let mut acc = T::ZERO;
            for k in 0..n {
                acc += v[i * n + k] * s[k] * v[j * n + k];
            }
            p[i * n + j] = acc;
        }
    }
}

/// Ergebnis von [`Matrix::orthonormalize_mgs`]: `A = Q·R`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orthonormalized<T, const M: usize, const N: usize> {
    /// Matrix mit orthonormalen Spalten (`M × N`).
    pub q: Matrix<T, M, N>,
    /// Obere Dreiecksmatrix mit positiver Diagonale (`N × N`).
    pub r: Matrix<T, N, N>,
}

/// Polarzerlegung `A = U·P`, siehe [`Matrix::polar`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Polar<T, const M: usize, const N: usize> {
    /// Matrix mit orthonormalen Spalten (`M × N`).
    pub u: Matrix<T, M, N>,
    /// Symmetrisch positiv semidefinite Matrix (`N × N`).
    pub p: Matrix<T, N, N>,
}

impl<T: Real, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Orthonormalisiert die Spalten mit **modifiziertem Gram-Schmidt**: `A = Q·R`.
    ///
    /// Mit `reorthogonalize` läuft pro Spalte ein zweiter Durchgang („twice is enough“); das hält `Q` auch
    /// bei fast abhängigen Spalten auf Rundungsfehlerniveau orthogonal. Bei Rangmangel (Restnorm einer Spalte
    /// `≤ N·ε·‖aⱼ‖`) kommt [`LinalgError::Singular`]; für diesen Fall oder höchste Stabilität
    /// [`QrDecomposition::thin_q`](crate::linalg::QrDecomposition::thin_q) verwenden.
    ///
    /// Fehler: [`LinalgError::DimensionMismatch`] (`M < N`), [`LinalgError::NotFinite`], [`LinalgError::Singular`].
    ///
    /// ```
    /// # #[cfg(any(feature = "std", feature = "libm"))] {
    /// use azmath::linalg::Matrix;
    ///
    /// let a = Matrix::new([[1.0f64, 1.0], [0.0, 1.0], [0.0, 0.0]]);
    /// let o = a.orthonormalize_mgs(true).unwrap();
    /// assert!(o.q.orthogonality_error() < 1e-15);
    /// # }
    /// ```
    pub fn orthonormalize_mgs(
        &self,
        reorthogonalize: bool,
    ) -> Result<Orthonormalized<T, M, N>, LinalgError> {
        let mut q = *self;
        let mut r = Matrix::<T, N, N>::zeros();
        mgs(
            q.data.as_flattened_mut(),
            r.data.as_flattened_mut(),
            M,
            N,
            reorthogonalize,
        )?;
        Ok(Orthonormalized { q, r })
    }

    /// Orthogonalitätsfehler `‖QᵀQ − I‖_F` der Spalten (0 für orthonormale Spalten).
    pub fn orthogonality_error(&self) -> T {
        orthogonality_error(self.data.as_flattened(), M, N)
    }

    /// Polarzerlegung `A = U·P` über die SVD: `U` hat orthonormale Spalten, `P = V·Σ·Vᵀ` ist symmetrisch
    /// positiv semidefinit. `U` ist bei vollem Spaltenrang eindeutig.
    ///
    /// Fehler: [`LinalgError::DimensionMismatch`] (`M < N`), [`LinalgError::NotFinite`],
    /// [`LinalgError::NoConvergence`], [`LinalgError::Singular`] bei Rangmangel (`U` wäre nicht eindeutig).
    pub fn polar(&self) -> Result<Polar<T, M, N>, LinalgError> {
        let svd = self.checked_svd()?;
        Ok(self.polar_with(&svd, false))
    }

    /// SVD mit Rangprüfung (Voraussetzung für eine eindeutige Polarzerlegung).
    fn checked_svd(&self) -> Result<Svd<T, M, N>, LinalgError> {
        let svd = self.svd()?;
        if N > 0 && svd.s[N - 1] <= svd.default_tolerance() {
            return Err(LinalgError::Singular);
        }
        Ok(svd)
    }

    fn polar_with(&self, svd: &Svd<T, M, N>, flip_last: bool) -> Polar<T, M, N> {
        let mut u = Matrix::<T, M, N>::zeros();
        let mut p = Matrix::<T, N, N>::zeros();
        polar_from_svd(
            (
                svd.u.data.as_flattened(),
                svd.s.as_slice(),
                svd.v.data.as_flattened(),
            ),
            (M, N),
            flip_last,
            (u.data.as_flattened_mut(), p.data.as_flattened_mut()),
        );
        Polar { u, p }
    }
}

impl<T: Real, const N: usize> Matrix<T, N, N> {
    /// Die orthogonale Matrix, die `A` in der Frobenius-Norm am nächsten liegt (Faktor `U` der Polarzerlegung).
    /// Eine Spiegelung bleibt eine Spiegelung (`det = −1`); für Drehungen siehe [`nearest_rotation`](Self::nearest_rotation).
    pub fn nearest_orthogonal(&self) -> Result<Matrix<T, N, N>, LinalgError> {
        Ok(self.polar()?.u)
    }

    /// Die Drehmatrix (`det = +1`), die `A` in der Frobenius-Norm am nächsten liegt: wie
    /// [`nearest_orthogonal`](Self::nearest_orthogonal), bei `det(U) < 0` wird die Richtung des kleinsten
    /// Singulärwerts gespiegelt (Standardverfahren zur Re-Orthonormalisierung von Rotationsmatrizen).
    pub fn nearest_rotation(&self) -> Result<Matrix<T, N, N>, LinalgError> {
        let svd = self.checked_svd()?;
        let u = self.polar_with(&svd, false).u;
        if u.determinant() >= T::ZERO {
            Ok(u)
        } else {
            Ok(self.polar_with(&svd, true).u)
        }
    }
}

#[cfg(feature = "alloc")]
mod dynamic {
    use super::{mgs, orthogonality_error, polar_from_svd};
    use crate::Real;
    use crate::linalg::{DynMatrix, LinalgError};
    use alloc::vec;

    /// Ergebnis von [`DynMatrix::orthonormalize_mgs`]: `A = Q·R`.
    #[derive(Debug, Clone, PartialEq)]
    pub struct DynOrthonormalized<T> {
        /// Matrix mit orthonormalen Spalten (`m × n`).
        pub q: DynMatrix<T>,
        /// Obere Dreiecksmatrix mit positiver Diagonale (`n × n`).
        pub r: DynMatrix<T>,
    }

    /// Polarzerlegung `A = U·P` dynamischer Größe, siehe [`DynMatrix::polar`].
    #[derive(Debug, Clone, PartialEq)]
    pub struct DynPolar<T> {
        /// Matrix mit orthonormalen Spalten (`m × n`).
        pub u: DynMatrix<T>,
        /// Symmetrisch positiv semidefinite Matrix (`n × n`).
        pub p: DynMatrix<T>,
    }

    impl<T: Real> DynMatrix<T> {
        /// Modifiziertes Gram-Schmidt, siehe [`Matrix::orthonormalize_mgs`](crate::linalg::Matrix::orthonormalize_mgs).
        pub fn orthonormalize_mgs(
            &self,
            reorthogonalize: bool,
        ) -> Result<DynOrthonormalized<T>, LinalgError> {
            let (m, n) = (self.rows(), self.cols());
            let mut q = self.as_slice().to_vec();
            let mut r = vec![T::ZERO; n * n];
            mgs(&mut q, &mut r, m, n, reorthogonalize)?;
            Ok(DynOrthonormalized {
                q: DynMatrix::new(m, n, q)?,
                r: DynMatrix::new(n, n, r)?,
            })
        }

        /// Orthogonalitätsfehler `‖QᵀQ − I‖_F` der Spalten.
        pub fn orthogonality_error(&self) -> T {
            orthogonality_error(self.as_slice(), self.rows(), self.cols())
        }

        fn checked_svd(&self) -> Result<crate::linalg::eigen::DynSvd<T>, LinalgError> {
            let svd = self.svd()?;
            let n = self.cols();
            if n > 0 && svd.s[n - 1] <= svd.default_tolerance() {
                return Err(LinalgError::Singular);
            }
            Ok(svd)
        }

        fn polar_with(
            &self,
            svd: &crate::linalg::eigen::DynSvd<T>,
            flip_last: bool,
        ) -> Result<DynPolar<T>, LinalgError> {
            let (m, n) = (self.rows(), self.cols());
            let mut u = vec![T::ZERO; m * n];
            let mut p = vec![T::ZERO; n * n];
            polar_from_svd(
                (svd.u.as_slice(), svd.s.as_slice(), svd.v.as_slice()),
                (m, n),
                flip_last,
                (&mut u, &mut p),
            );
            Ok(DynPolar {
                u: DynMatrix::new(m, n, u)?,
                p: DynMatrix::new(n, n, p)?,
            })
        }

        /// Polarzerlegung `A = U·P`, siehe [`Matrix::polar`](crate::linalg::Matrix::polar).
        pub fn polar(&self) -> Result<DynPolar<T>, LinalgError> {
            let svd = self.checked_svd()?;
            self.polar_with(&svd, false)
        }

        /// Nächste orthogonale Matrix (quadratisch), siehe [`Matrix::nearest_orthogonal`](crate::linalg::Matrix::nearest_orthogonal).
        pub fn nearest_orthogonal(&self) -> Result<DynMatrix<T>, LinalgError> {
            if self.rows() != self.cols() {
                return Err(LinalgError::DimensionMismatch);
            }
            Ok(self.polar()?.u)
        }

        /// Nächste Drehmatrix (`det = +1`), siehe [`Matrix::nearest_rotation`](crate::linalg::Matrix::nearest_rotation).
        pub fn nearest_rotation(&self) -> Result<DynMatrix<T>, LinalgError> {
            if self.rows() != self.cols() {
                return Err(LinalgError::DimensionMismatch);
            }
            let svd = self.checked_svd()?;
            let u = self.polar_with(&svd, false)?.u;
            if u.determinant()? >= T::ZERO {
                Ok(u)
            } else {
                Ok(self.polar_with(&svd, true)?.u)
            }
        }
    }
}

#[cfg(feature = "alloc")]
pub use dynamic::{DynOrthonormalized, DynPolar};
