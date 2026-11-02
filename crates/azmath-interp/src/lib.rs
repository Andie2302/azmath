//! Interpolation: Stützwerte `(xᵢ, yᵢ)` durch Funktionen verbinden.
//!
//! | Typ / Funktion | Eigenschaft |
//! |---|---|
//! | [`linear`] | stückweise linear, überall stetig; die einfachste und sicherste Wahl |
//! | [`CubicSpline`] | zweimal stetig differenzierbar, glatt; Randbedingungen natürlich, eingespannt, not-a-knot, periodisch |
//! | [`Pchip`] | stückweise kubisch, **monoton** (kein Überschwingen), nur einmal stetig differenzierbar |
//! | [`lagrange`] | ein Polynom durch alle Punkte; nur für wenige, gut verteilte Punkte |
//!
//! Die Stützstellen `x` müssen **streng aufsteigend** sein. Alle Verfahren arbeiten auf Slices und
//! brauchen keinen Allokator: [`CubicSpline`] und [`Pchip`] **leihen** sich die Stützwerte und
//! speichern die berechneten Koeffizienten in einem vom Aufrufer übergebenen Slice. Mit dem Feature
//! `alloc` gibt es die besitzenden Varianten [`CubicSplineBuf`] und [`PchipBuf`].
//!
//! **Auswertung außerhalb** der Stützstellen (Extrapolation) setzt das letzte Polynom bzw. die
//! letzte Gerade fort; das ist selten sinnvoll (Splines divergieren schnell). `contains(x)` prüft
//! den Bereich.
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath_interp::{cubic_spline_work_len, BoundaryCondition, CubicSpline};
//!
//! // Punkte von y = x²
//! let x = [0.0f64, 1.0, 2.0, 3.0, 4.0];
//! let y = [0.0f64, 1.0, 4.0, 9.0, 16.0];
//! let (mut m, mut work) = ([0.0f64; 5], [0.0f64; cubic_spline_work_len(5)]);
//! let s = CubicSpline::new(&x, &y, BoundaryCondition::NotAKnot, &mut m, &mut work).unwrap();
//! // not-a-knot reproduziert Polynome bis Grad 3 exakt
//! assert!((s.value(2.5) - 6.25).abs() < 1e-12);
//! assert!((s.derivative(2.5) - 5.0).abs() < 1e-12);
//! assert!((s.integral(0.0, 4.0) - 64.0 / 3.0).abs() < 1e-12);
//! # }
//! ```
//!
//! Hinweis zu `micromath`: Die Verfahren selbst nutzen nur `+ - * /`, sind also unabhängig vom
//! Backend; die Tests laufen dennoch nur mit `std` und `libm`.

#![no_std]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod cubic;
mod lagrange;
mod linear;
mod pchip;

#[cfg(feature = "alloc")]
mod owned;

pub use cubic::{BoundaryCondition, CubicSpline, cubic_spline_work_len};
pub use lagrange::lagrange;
pub use linear::{linear, locate};
pub use pchip::Pchip;

#[cfg(feature = "alloc")]
pub use owned::{CubicSplineBuf, PchipBuf};

use core::fmt;

use azmath_core::Real;
use azmath_linalg::LinalgError;

/// Fehler der Interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InterpError {
    /// Ungültige Eingabe: weniger als 2 Punkte, verschieden lange Slices, zu kurzer Arbeitsspeicher
    /// oder für das Verfahren zu wenige Punkte.
    InvalidInput,
    /// Die Stützstellen `x` sind nicht streng aufsteigend (oder doppelt vorhanden).
    NotIncreasing,
    /// NaN oder Unendlich in den Stützstellen oder -werten.
    NotFinite,
    /// Periodische Randbedingung, aber `y[0] != y[n−1]`.
    NotPeriodic,
    /// Das zugrunde liegende Gleichungssystem war nicht lösbar.
    Singular,
}

impl fmt::Display for InterpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            InterpError::InvalidInput => "ungültige Eingabe oder zu wenig Arbeitsspeicher",
            InterpError::NotIncreasing => "Stützstellen nicht streng aufsteigend",
            InterpError::NotFinite => "NaN oder unendlicher Wert",
            InterpError::NotPeriodic => "y[0] und y[n-1] sind für periodische Splines verschieden",
            InterpError::Singular => "Gleichungssystem nicht lösbar",
        })
    }
}

impl core::error::Error for InterpError {}

impl From<LinalgError> for InterpError {
    fn from(e: LinalgError) -> Self {
        match e {
            LinalgError::NotFinite => InterpError::NotFinite,
            LinalgError::DimensionMismatch => InterpError::InvalidInput,
            _ => InterpError::Singular,
        }
    }
}

pub(crate) fn finite<T: Real>(x: T) -> bool {
    !(x.is_nan() || x.is_infinite())
}

/// Prüft Längen, Endlichkeit und strenge Monotonie der Stützstellen.
pub(crate) fn check_nodes<T: Real>(x: &[T], y: &[T]) -> Result<(), InterpError> {
    if x.len() != y.len() || x.len() < 2 {
        return Err(InterpError::InvalidInput);
    }
    if x.iter().chain(y.iter()).any(|v| !finite(*v)) {
        return Err(InterpError::NotFinite);
    }
    if x.windows(2).any(|w| w[1] <= w[0]) {
        return Err(InterpError::NotIncreasing);
    }
    Ok(())
}
