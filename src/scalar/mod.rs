//! Skalar-Trait [`Scalar`] für reelle **und** komplexe Zahlen (Entwurf `docs/design/0002`).
//!
//! `Scalar` verlangt nur Körperarithmetik und Konjugation; Beträge, Normen und Toleranzen sind immer reell
//! ([`Scalar::Real`]). Implementiert für `f32`, `f64` und [`Complex`]. [`Real`] bleibt unverändert.
//!
//! Darauf bauen (Feature `alloc`) auf: `ScalarLu`, `ScalarQr` (kleinste Quadrate), `ScalarCholesky` (hermitesch),
//! `complex_eigenvalues`/`hermitian_eigenvalues`/`complex_eigenvector`, `ScalarCsr` und `gmres_scalar`.

use crate::Real;
use crate::complex::Complex;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[cfg(feature = "alloc")]
mod cholesky;
#[cfg(feature = "alloc")]
mod eigen;
#[cfg(feature = "alloc")]
mod lu;
#[cfg(feature = "alloc")]
mod qr;
#[cfg(feature = "alloc")]
mod sparse;
#[cfg(feature = "alloc")]
pub use cholesky::ScalarCholesky;
#[cfg(feature = "alloc")]
pub use eigen::{complex_eigenvalues, complex_eigenvector, hermitian_eigenvalues};
#[cfg(feature = "alloc")]
pub use lu::ScalarLu;
#[cfg(feature = "alloc")]
pub use qr::ScalarQr;
#[cfg(feature = "alloc")]
pub use sparse::{GmresScalarResult, ScalarCsr, gmres_scalar};

/// Zahlentyp für lineare Algebra: reell oder komplex.
pub trait Scalar:
    Copy
    + PartialEq
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// Zugehöriger reeller Typ (für Beträge, Normen, Toleranzen).
    type Real: Real;
    /// Die Zahl 0.
    const ZERO: Self;
    /// Die Zahl 1.
    const ONE: Self;
    /// Aus einer reellen Zahl.
    fn from_real(r: Self::Real) -> Self;
    /// Realteil.
    fn re(self) -> Self::Real;
    /// Imaginärteil (bei reellen Typen 0).
    fn im(self) -> Self::Real;
    /// Konjugierte (bei reellen Typen die Zahl selbst).
    fn conj(self) -> Self;
    /// Betrag `|z|`.
    fn modulus(self) -> Self::Real;
    /// Betragsquadrat `|z|²`.
    fn modulus_sq(self) -> Self::Real;
    /// `false` bei NaN oder unendlichen Anteilen.
    fn is_finite(self) -> bool;
}

macro_rules! impl_scalar_real {
    ($t:ty) => {
        impl Scalar for $t {
            type Real = $t;
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            fn from_real(r: $t) -> Self {
                r
            }
            fn re(self) -> $t {
                self
            }
            fn im(self) -> $t {
                0.0
            }
            fn conj(self) -> Self {
                self
            }
            fn modulus(self) -> $t {
                if self < 0.0 { -self } else { self }
            }
            fn modulus_sq(self) -> $t {
                self * self
            }
            fn is_finite(self) -> bool {
                !(self.is_nan() || self.is_infinite())
            }
        }
    };
}
impl_scalar_real!(f32);
impl_scalar_real!(f64);

impl<T: Real> Scalar for Complex<T> {
    type Real = T;
    const ZERO: Self = Complex::ZERO;
    const ONE: Self = Complex::ONE;
    fn from_real(r: T) -> Self {
        Complex::from_real(r)
    }
    fn re(self) -> T {
        self.re
    }
    fn im(self) -> T {
        self.im
    }
    fn conj(self) -> Self {
        Complex::conj(self)
    }
    fn modulus(self) -> T {
        self.abs()
    }
    fn modulus_sq(self) -> T {
        self.norm_sqr()
    }
    fn is_finite(self) -> bool {
        !(self.re.is_nan() || self.re.is_infinite() || self.im.is_nan() || self.im.is_infinite())
    }
}
