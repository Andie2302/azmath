//! Nichtlineare Gleichungssysteme und nichtlineare Ausgleichsrechnung.
//!
//! | Funktion | Problem | Verfahren |
//! |---|---|---|
//! | [`newton_system`] | `F(x) = 0` mit `n` Gleichungen, `n` Unbekannten | Newton mit Schrittweitensteuerung |
//! | [`broyden`] | `F(x) = 0` | Broyden (Rang-1-Update der inversen Jacobi-Matrix), nur eine Jacobi-Matrix am Start |
//! | [`gauss_newton`] | `min ‖r(x)‖₂` mit `m ≥ n` Residuen | Gauß-Newton (QR) mit Schrittweitensteuerung |
//! | [`levenberg_marquardt`] | `min ‖r(x)‖₂` | Levenberg-Marquardt, robust auch bei schlechtem Startwert und Rangmangel |
//! | [`bfgs`], [`lbfgs`] | `min f(x)` mit Gradient | Quasi-Newton (dicht bzw. mit begrenztem Speicher), starke Wolfe-Schrittweite |
//! | [`line_search_wolfe`] | Schrittweite entlang einer Abstiegsrichtung | starke Wolfe-Bedingungen |
//!
//! Zu jedem Verfahren gibt es eine Variante `*_fd`, die die Jacobi-Matrix selbst per
//! Vorwärtsdifferenzen bestimmt ([`numerical_jacobian`]); eine analytische Jacobi-Matrix ist
//! schneller und genauer.
//!
//! # Schnittstelle
//!
//! Die Funktionen arbeiten **auf Slices** und brauchen keinen Allokator: Der Arbeitsspeicher wird
//! als Slice übergeben (Größe über `*_work_len`, als `const fn` auch für Arrays nutzbar). Damit
//! passen sie gleichermaßen zu festen Größen und zu zur Laufzeit bekannten Datenmengen, wie sie
//! bei Kurvenanpassungen üblich sind.
//!
//! - `residual(x, r)`: schreibt die `m` Residuen `r(x)` zu den `n` Parametern `x`.
//! - `jacobian(x, j)`: schreibt die `m × n` Jacobi-Matrix **zeilenweise** (`j[i*n + k] = ∂rᵢ/∂xₖ`).
//! - `x`: Startwert beim Aufruf, Lösung beim Rückgabe (auch bei Fehlern der letzte Stand).
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::optimize::{levenberg_marquardt_fd, levenberg_marquardt_work_len, OptimizeOptions};
//! use azmath::Real;
//!
//! // Kurvenanpassung y = a·exp(b·t) an exakte Daten (a = 2, b = −1)
//! let t = [0.0, 0.5, 1.0, 1.5, 2.0];
//! let y = t.map(|t| 2.0 * Real::exp(-t));
//! let residual = |p: &[f64], r: &mut [f64]| {
//!     for i in 0..5 {
//!         r[i] = p[0] * Real::exp(p[1] * t[i]) - y[i];
//!     }
//! };
//! let mut p = [1.0, 0.0];
//! let mut work = [0.0; levenberg_marquardt_work_len(5, 2)];
//! // Jacobi-Matrix per Differenzen: `levenberg_marquardt_fd`
//! levenberg_marquardt_fd(residual, &mut p, 5, OptimizeOptions::default(), &mut work).unwrap();
//! assert!((p[0] - 2.0).abs() < 1e-7 && (p[1] + 1.0).abs() < 1e-7);
//! # }
//! ```
//!
//! # Hinweise
//!
//! - Mit einer Differenzen-Jacobi-Matrix (`*_fd`) rauscht der Gradient mit etwa `√ε`; die Verfahren
//!   enden dann meist über [`Termination::Cost`] oder [`Termination::Step`] statt über den Gradienten.
//! - Alle Verfahren sind **lokal**: Sie finden eine Lösung nahe am Startwert, nicht unbedingt die
//!   global beste. Bei schlechtem Startwert ist Levenberg-Marquardt die robustere Wahl.
//! - Levenberg-Marquardt löst die gedämpften Normalengleichungen per Cholesky (die Konditionszahl
//!   wird dabei quadriert), Gauß-Newton und Newton verwenden QR.
//! - Mit `micromath` (bewusst ungenau, die Normen brauchen `sqrt`) werden strenge Toleranzen
//!   unter Umständen nie erreicht; für verlässliche Ergebnisse `std` oder `libm` verwenden.

mod bfgs;
mod broyden;
mod gauss_newton;
mod jacobian;
mod linesearch;
mod lm;

pub use bfgs::{MinimizeResult, bfgs, bfgs_work_len, lbfgs, lbfgs_work_len};
pub use broyden::{broyden, broyden_fd, broyden_work_len};
pub use gauss_newton::{
    gauss_newton, gauss_newton_fd, gauss_newton_work_len, newton_system, newton_system_fd,
    newton_work_len,
};
pub use jacobian::{numerical_jacobian, numerical_jacobian_work_len};
pub use linesearch::{LineSearchError, LineSearchResult, WolfeOptions, line_search_wolfe};
pub use lm::{levenberg_marquardt, levenberg_marquardt_fd, levenberg_marquardt_work_len};

use core::fmt;

use crate::Real;

/// Einstellungen für die Verfahren in diesem Modul.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimizeOptions<T> {
    /// Abbruch, wenn der Schritt klein ist: `‖Δx‖ ≤ tol_x · (‖x‖ + tol_x)`.
    pub tol_x: T,
    /// Abbruch, wenn das Residuum klein ist: `‖r‖ ≤ tol_f`.
    pub tol_f: T,
    /// Abbruch, wenn der Gradient klein ist: `‖Jᵀr‖∞ ≤ tol_g · (1 + ‖r‖)`.
    pub tol_g: T,
    /// Maximale Anzahl an Iterationen.
    pub max_iter: usize,
}

impl<T: Real> OptimizeOptions<T> {
    /// Setzt alle Toleranzen auf den Wert `tol` (bei `tol_f` absolut, sonst wie oben beschrieben).
    pub fn with_tolerance(tol: T) -> Self {
        Self {
            tol_x: tol,
            tol_f: tol,
            tol_g: tol,
            max_iter: 200,
        }
    }
}

impl<T: Real> Default for OptimizeOptions<T> {
    /// `tol_x = tol_g = ε^0,75` (etwa `1.8e-12` bei `f64`), `tol_f = ε`, höchstens 200 Iterationen.
    fn default() -> Self {
        let tol = T::EPSILON.powf(T::from_f64(0.75));
        Self {
            tol_x: tol,
            tol_f: T::EPSILON,
            tol_g: tol,
            max_iter: 200,
        }
    }
}

/// Weshalb ein Verfahren erfolgreich beendet wurde.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    /// Das Residuum ist (praktisch) 0: `‖r‖ ≤ tol_f`.
    Residual,
    /// Der Schritt ist klein geworden: `‖Δx‖ ≤ tol_x · (‖x‖ + tol_x)`.
    Step,
    /// Der Gradient `Jᵀr` ist (praktisch) 0, ein stationärer Punkt ist erreicht.
    Gradient,
    /// Das Modell sagt keine im Rahmen der Rechengenauigkeit überprüfbare Verbesserung mehr
    /// voraus (vorhergesagte Abnahme von `‖r‖²` höchstens `16·ε·‖r‖²`). Das ist der übliche
    /// Abbruch bei Daten mit Restfehler oder bei Jacobi-Matrizen aus Differenzen, wo der
    /// Gradient wegen Rundungsrauschen nie genau 0 wird.
    Cost,
}

/// Ergebnis eines erfolgreichen Verfahrens (die Lösung steht im übergebenen `x`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimizeResult<T> {
    /// Anzahl der ausgeführten Iterationen.
    pub iterations: usize,
    /// Euklidische Norm des Residuums `‖r(x)‖` an der Lösung.
    pub residual_norm: T,
    /// Abbruchgrund.
    pub termination: Termination,
}

/// Fehler der Verfahren in diesem Modul.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OptimizeError {
    /// Ungültige Größen: `x` leer, `m < n`, `x` hat nicht `n` Elemente oder der
    /// Arbeitsspeicher ist kürzer als `*_work_len`.
    InvalidInput,
    /// NaN oder Unendlich in Residuum, Jacobi-Matrix oder Startwert.
    NotFinite,
    /// Die Jacobi-Matrix hat Rangmangel (Newton, Gauß-Newton); Levenberg-Marquardt ist dagegen
    /// robust.
    SingularJacobian,
    /// Kein Fortschritt mehr möglich oder `max_iter` erreicht. `x` enthält den letzten Stand.
    NoConvergence,
}

impl fmt::Display for OptimizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            OptimizeError::InvalidInput => "ungültige Größen oder zu wenig Arbeitsspeicher",
            OptimizeError::NotFinite => "NaN oder unendlicher Wert aufgetreten",
            OptimizeError::SingularJacobian => "Jacobi-Matrix hat Rangmangel",
            OptimizeError::NoConvergence => "kein Fortschritt mehr oder Iterationslimit erreicht",
        })
    }
}

impl core::error::Error for OptimizeError {}

pub(crate) fn finite<T: Real>(x: T) -> bool {
    !(x.is_nan() || x.is_infinite())
}

/// Skalarprodukt zweier gleich langer Slices.
pub(crate) fn dot<T: Real>(a: &[T], b: &[T]) -> T {
    let mut s = T::ZERO;
    for (x, y) in a.iter().zip(b) {
        s += *x * *y;
    }
    s
}

/// Euklidische Norm.
pub(crate) fn norm<T: Real>(a: &[T]) -> T {
    dot(a, a).sqrt()
}

/// Woher die Jacobi-Matrix kommt: vom Aufrufer (analytisch) oder per Vorwärtsdifferenzen.
pub(crate) enum Jac<J> {
    Analytic(J),
    Numeric,
}

/// Platzhalter-Typ für `Jac::Numeric` (wird nie aufgerufen).
pub(crate) type NoJacobian<T> = fn(&[T], &mut [T]);

/// Berechnet die Jacobi-Matrix `out` an der Stelle `x` (mit bereits ausgewertetem `r = r(x)`).
/// `fd` ist Arbeitsspeicher der Länge `m + n` für die Differenzenvariante.
pub(crate) fn eval_jacobian<T: Real, R: FnMut(&[T], &mut [T]), J: FnMut(&[T], &mut [T])>(
    residual: &mut R,
    jac: &mut Jac<J>,
    x: &[T],
    r: &[T],
    out: &mut [T],
    fd: &mut [T],
) {
    match jac {
        Jac::Analytic(j) => j(x, out),
        Jac::Numeric => numerical_jacobian(&mut *residual, x, r, out, fd),
    }
}
