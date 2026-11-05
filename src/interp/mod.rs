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
//! use azmath::interp::{cubic_spline_work_len, BoundaryCondition, CubicSpline};
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
//!
//! Nichtendliche Abfragen werden von [`linear`] und [`lagrange`] als Fehler zurückgegeben;
//! die direkten Spline-Auswertungen und -Integrale liefern dafür `NaN`. Konstruktoren lehnen
//! nicht endlich darstellbare Steigungen bzw. zweite Ableitungen mit [`InterpError::NotFinite`] ab.
//! Auch bei endlichen Eingaben bleiben Rundungsfehler und schlecht konditionierte Daten möglich.

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

use crate::Real;
use crate::linalg::LinalgError;

/// Fehler der Interpolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum InterpError {
    /// Ungültige Eingabe: weniger als 2 Punkte, verschieden lange Slices, zu kurzer Arbeitsspeicher
    /// oder für das Verfahren zu wenige Punkte.
    InvalidInput,
    /// Die Stützstellen `x` sind nicht streng aufsteigend (oder doppelt vorhanden).
    NotIncreasing,
    /// Nichtendliche Eingabe oder ein nicht endlich darstellbares Ergebnis bzw. eine solche Ableitung.
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

/// Anteil von q im Intervall, auch bei überlaufenden Koordinatendifferenzen.
pub(crate) fn fraction<T: Real>(a: T, b: T, q: T) -> T {
    let (num, den) = (q - a, b - a);
    if finite(num) && finite(den) {
        num / den
    } else {
        let s = a.abs().max(b.abs()).max(q.abs());
        (q / s - a / s) / (b / s - a / s)
    }
}

pub(crate) fn slope<T: Real>(a: T, b: T, ya: T, yb: T) -> T {
    let (num, den) = (yb - ya, b - a);
    if finite(num) && finite(den) {
        num / den
    } else if den.is_infinite() {
        (yb * T::from_f64(0.5) - ya * T::from_f64(0.5))
            / (b * T::from_f64(0.5) - a * T::from_f64(0.5))
    } else {
        yb / den - ya / den
    }
}

pub(crate) fn blend<T: Real>(a: T, b: T, t: T) -> T {
    if a == b {
        return a;
    }
    if t == T::ZERO {
        return a;
    }
    if t == T::ONE {
        return b;
    }
    if t >= T::ZERO && t <= T::ONE {
        if (a < T::ZERO) == (b < T::ZERO) {
            a + t * (b - a)
        } else {
            (T::ONE - t) * a + t * b
        }
    } else {
        let proposed = a + t * (b - a);
        if finite(proposed) {
            proposed
        } else {
            let s = a.abs().max(b.abs());
            (a / s + t * (b / s - a / s)) * s
        }
    }
}

pub(crate) fn line_value<T: Real>(a: T, b: T, ya: T, yb: T, q: T) -> T {
    if ya == yb {
        return ya;
    }
    let t = fraction(a, b, q);
    if finite(t) {
        blend(ya, yb, t)
    } else {
        ya + width_product(a, q, slope(a, b, ya, yb), T::ONE)
    }
}

pub(crate) fn weighted<T: Real>(values: &[T], weights: &[T]) -> T {
    let mut sum = crate::integrate::WeightedSum::new();
    for (&v, &w) in values.iter().zip(weights) {
        if sum.add(v, w).is_err() {
            return T::NAN;
        }
    }
    sum.value()
}

/// Produkt einer Breite, eines Wertes und eines Faktors ohne vermeidbaren Zwischenüberlauf.
pub(crate) fn width_product<T: Real>(a: T, b: T, value: T, factor: T) -> T {
    if value == T::ZERO || factor == T::ZERO {
        return T::ZERO;
    }
    let h = b - a;
    if finite(h) {
        crate::integrate::mul3(h, value, factor)
    } else {
        crate::integrate::mul3(
            crate::integrate::half_width(a, b),
            value,
            factor * T::from_f64(2.0),
        )
    }
}

/// Zwei Gauß-Punkte integrieren ein kubisches Teilstück exakt, ohne Stammfunktionssubtraktion.
pub(crate) fn piece_integral<T: Real>(a: T, b: T, f: impl Fn(T) -> T) -> T {
    if a == b {
        return T::ZERO;
    }
    let offset = T::from_f64(0.288_675_134_594_812_87);
    let half = T::from_f64(0.5);
    let left = crate::integrate::point(a, b, half - offset);
    let right = crate::integrate::point(a, b, half + offset);
    let mean = weighted(&[f(left), f(right)], &[half, half]);
    crate::integrate::integral_scale(mean, a, b)
}
