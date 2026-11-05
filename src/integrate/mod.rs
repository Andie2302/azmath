//! Numerische Integration: `∫ₐᵇ f(x) dx`.
//!
//! | Funktion | Art | Eigenschaft |
//! |---|---|---|
//! | [`trapezoid`] | feste Regel | exakt für Geraden, Fehler ∝ h² |
//! | [`simpson`] | feste Regel | exakt bis Grad 3, Fehler ∝ h⁴ |
//! | [`gauss_legendre`] | feste Regel | `k` Punkte pro Teilintervall, exakt bis Grad `2k−1` |
//! | [`gauss_rule`] | Gauß-Regeln beliebiger Ordnung (Legendre, Laguerre, Hermite, Jacobi, Tschebyscheff) |
//! | [`tanh_sinh`], [`integrate_semi_infinite`], [`integrate_infinite`] | Endpunktsingularitäten, unendliche Intervalle |
//! | [`tensor_gauss`] | Kubatur über Quader (Tensorprodukt) |
//! | [`adaptive`] | adaptiv | Gauß-Kronrod (G7/K15) mit Fehlerschätzung; die erste Wahl |
//! | [`trapezoid_samples`], [`simpson_samples`], [`trapezoid_xy`] | Messdaten | Integral aus Stützwerten |
//!
//! Alle Verfahren sind generisch über [`Real`] und brauchen keinen Allokator. Die Funktion wird
//! als Closure `FnMut(T) -> T` übergeben.
//!
//! **Adaptiv ist meist die richtige Wahl:** [`adaptive`] teilt das Intervall nur dort, wo die
//! Funktion schwierig ist, wertet die Funktion nie an den Intervallenden aus (Randsingularitäten
//! wie `√x` oder `ln x` bei 0 sind daher erlaubt) und liefert eine Fehlerschätzung mit.
//! Unendliche Grenzen: [`integrate_semi_infinite`], [`integrate_infinite`].
//!
//! Hinweis zu `micromath`: Mit der bewusst ungenauen Näherung (z. B. bei `sin`, `exp`, `sqrt`
//! in `f`) lassen sich strenge Toleranzen unter Umständen nicht erreichen; für verlässliche
//! Ergebnisse `std` oder `libm` verwenden.
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::integrate::{adaptive, IntegrateOptions};
//! use azmath::Real;
//!
//! // ∫₀^π sin(x) dx = 2
//! let r = adaptive(|x: f64| Real::sin(x), 0.0, core::f64::consts::PI, IntegrateOptions::default())
//!     .unwrap();
//! assert!((r.value - 2.0).abs() < 1e-13);
//! # }
//! ```

mod adaptive;
mod fixed;
mod gauss;
mod improper;
mod quadrature;

pub use adaptive::adaptive;
pub use fixed::{simpson, simpson_samples, trapezoid, trapezoid_samples, trapezoid_xy};
pub use gauss::gauss_legendre;
pub use improper::{integrate_infinite, integrate_semi_infinite, tanh_sinh};
pub use quadrature::{
    Family, gauss_apply, gauss_quad_work_len, gauss_rule, tensor_gauss, tensor_gauss_work_len,
};

use core::fmt;

use crate::Real;

/// Einstellungen für das adaptive Verfahren [`adaptive`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntegrateOptions<T> {
    /// Absolute Toleranz für den Fehler (endlich und nichtnegativ).
    pub abs_tol: T,
    /// Relative Toleranz (bezogen auf den Betrag des Integrals; endlich und nichtnegativ).
    pub rel_tol: T,
    /// Maximale Anzahl gleichzeitiger Teilintervalle (mindestens 1, höchstens 256; größere Werte werden
    /// gekappt).
    pub max_intervals: usize,
    /// Maximale Anzahl an Funktionsauswertungen (für ein nichtleeres Intervall mindestens 15).
    pub max_evals: usize,
}

impl<T: Real> IntegrateOptions<T> {
    /// Erzeugt Einstellungen mit den gegebenen Toleranzen; höchstens 200 Teilintervalle und
    /// 200 000 Auswertungen.
    pub fn new(abs_tol: T, rel_tol: T) -> Self {
        Self {
            abs_tol,
            rel_tol,
            max_intervals: 200,
            max_evals: 200_000,
        }
    }
}

impl<T: Real> Default for IntegrateOptions<T> {
    /// `abs_tol = ε`, `rel_tol = ε^0,75` (etwa `1.8e-12` bei `f64`), 200 Teilintervalle,
    /// 200 000 Auswertungen.
    ///
    /// Der Fehler wird pessimistisch geschätzt (Differenz zweier Formeln); der tatsächliche Fehler
    /// ist bei glatten Funktionen meist deutlich kleiner als die Toleranz.
    fn default() -> Self {
        Self {
            abs_tol: T::EPSILON,
            rel_tol: T::EPSILON.powf(T::from_f64(0.75)),
            max_intervals: 200,
            max_evals: 200_000,
        }
    }
}

/// Ergebnis des adaptiven Verfahrens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IntegralResult<T> {
    /// Der Wert des Integrals.
    pub value: T,
    /// Schätzung des absoluten Fehlers (Summe der Differenzen zwischen den beiden Teilregeln).
    ///
    /// Bei glatten Funktionen ist das eine vorsichtige Schätzung; bei Singularitäten am Rand
    /// kann der wirkliche Fehler sie um einen kleinen Faktor übersteigen. Es ist keine strenge
    /// Schranke.
    pub error_estimate: T,
    /// Anzahl der Funktionsauswertungen.
    pub evaluations: usize,
}

/// Fehler der numerischen Integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IntegrateError {
    /// Ungültige Eingabe: z. B. 0 Teilintervalle, ungerade Zahl bei Simpson, unzulässige
    /// Ordnung bei Gauß-Legendre oder zu kurze bzw. verschieden lange Datenreihen.
    InvalidInput,
    /// NaN oder unendlicher Wert in den Grenzen, in `f` oder im Ergebnis.
    NotFinite,
    /// Die geforderte Genauigkeit wurde mit `max_intervals`/`max_evals` nicht erreicht (zum
    /// Beispiel bei nicht integrierbaren Singularitäten wie `1/x` bei 0).
    NoConvergence,
}

impl fmt::Display for IntegrateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            IntegrateError::InvalidInput => "ungültige Eingabe",
            IntegrateError::NotFinite => "NaN oder unendlicher Wert aufgetreten",
            IntegrateError::NoConvergence => "geforderte Genauigkeit nicht erreicht",
        })
    }
}

impl core::error::Error for IntegrateError {}

/// `true`, wenn `x` weder NaN noch unendlich ist.
pub(crate) fn finite<T: Real>(x: T) -> bool {
    !(x.is_nan() || x.is_infinite())
}

/// `i` als `T`.
pub(crate) fn from_usize<T: Real>(i: usize) -> T {
    T::from_f64(i as f64)
}

/// Punkt im Intervall ohne überlaufende Differenz bei verschiedenen Vorzeichen.
pub(crate) fn point<T: Real>(a: T, b: T, t: T) -> T {
    if (a < T::ZERO) == (b < T::ZERO) {
        a + (b - a) * t
    } else {
        (T::ONE - t) * a + t * b
    }
}

pub(crate) fn half_width<T: Real>(a: T, b: T) -> T {
    let half = T::from_f64(0.5);
    let width = b - a;
    if finite(width) {
        width * half
    } else {
        b * half - a * half
    }
}

/// Kleinsten und größten Faktor zuerst multiplizieren, um unnötigen Wertebereichsverlust zu vermeiden.
pub(crate) fn mul3<T: Real>(mut a: T, mut b: T, mut c: T) -> T {
    if a.abs() > b.abs() {
        core::mem::swap(&mut a, &mut b);
    }
    if b.abs() > c.abs() {
        core::mem::swap(&mut b, &mut c);
    }
    if a.abs() > b.abs() {
        core::mem::swap(&mut a, &mut b);
    }
    (a * c) * b
}

pub(crate) fn integral_scale<T: Real>(mean: T, a: T, b: T) -> T {
    let width = b - a;
    if finite(width) {
        mean * width
    } else {
        mul3(mean, half_width(a, b), T::from_f64(2.0))
    }
}

/// Skalierte, kompensierte gewichtete Summe; die Gewichte werden vor der Rückskalierung angewendet.
pub(crate) struct WeightedSum<T> {
    scale: T,
    sum: T,
    correction: T,
}
impl<T: Real> WeightedSum<T> {
    pub(crate) fn new() -> Self {
        Self {
            scale: T::ZERO,
            sum: T::ZERO,
            correction: T::ZERO,
        }
    }
    pub(crate) fn add(&mut self, value: T, weight: T) -> Result<(), IntegrateError> {
        if !finite(value) || !finite(weight) {
            return Err(IntegrateError::NotFinite);
        }
        if value.abs() > self.scale {
            let ratio = self.scale / value.abs();
            self.sum *= ratio;
            self.correction *= ratio;
            self.scale = value.abs();
        }
        if self.scale != T::ZERO {
            let term = (value / self.scale) * weight - self.correction;
            let next = self.sum + term;
            self.correction = (next - self.sum) - term;
            self.sum = next;
        }
        Ok(())
    }
    pub(crate) fn value(&self) -> T {
        self.scale * self.sum
    }
}

pub(crate) fn checked<T: Real>(value: T) -> Result<T, IntegrateError> {
    if finite(value) {
        Ok(value)
    } else {
        Err(IntegrateError::NotFinite)
    }
}
