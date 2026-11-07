//! Gewöhnliche Differentialgleichungen (Anfangswertprobleme): `y' = f(t, y)`, `y(t₀) = y₀`.
//!
//! | Funktion | Art | Ordnung | Geeignet für |
//! |---|---|---|---|
//! | [`euler`] | explizit, feste Schrittweite | 1 | Demonstrationen, grobe Abschätzungen |
//! | [`rk4`] | explizit, feste Schrittweite | 4 | glatte Probleme mit bekannter Schrittweite |
//! | [`rk45`] | explizit, **adaptiv** (Dormand-Prince) | 5(4) | die erste Wahl für nicht steife Probleme |
//! | [`gauss_legendre4`] | implizit, feste Schrittweite, symplektisch bei Hamilton-Systemen | 4 | auch nicht separierbare Hamilton-Systeme |
//! | [`implicit_euler`] | implizit, feste Schrittweite | 1 | **steife** Probleme, sehr robust |
//! | [`implicit_trapezoid`] | implizit, feste Schrittweite | 2 | steife Probleme, genauer |
//! | [`verlet`], [`verlet_separable`] | explizit, **symplektisch**, feste Schrittweite | 2 | Hamilton-Systeme und `q'' = a(t, q)`; beschränkter Energiefehler |
//!
//! Die Funktionen arbeiten **auf Slices** und brauchen keinen Allokator (wie
//! [`optimize`](crate::optimize)): Der Zustand `y` (`n` Elemente) wird **in-place** fortgeschrieben,
//! der Arbeitsspeicher als Slice übergeben (Größe über `*_work_len`, als `const fn` auch für
//! Arrays nutzbar).
//!
//! - `f(t, y, dy)` schreibt die rechte Seite `y' = f(t, y)` nach `dy`.
//! - `observer(t, y)` wird mit dem Anfangszustand und nach jedem (akzeptierten) Schritt
//!   aufgerufen, um den Verlauf aufzuzeichnen, ohne Speicher zu belegen. Wer ihn nicht braucht,
//!   übergibt `|_, _| {}`.
//! - Die Integration kann auch **rückwärts** laufen (`t1 < t0`).
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::ode::{rk45, rk45_work_len, Rk45Options};
//! use azmath::Real;
//!
//! // Harmonischer Oszillator y'' = −y als System: (y, v)' = (v, −y), y(0) = 0, v(0) = 1
//! let f = |_t: f64, y: &[f64], dy: &mut [f64]| {
//!     dy[0] = y[1];
//!     dy[1] = -y[0];
//! };
//! let mut y = [0.0, 1.0];
//! let mut work = [0.0; rk45_work_len(2)];
//! let t_end = core::f64::consts::PI / 2.0;
//! rk45(f, 0.0, t_end, &mut y, Rk45Options::default(), &mut work, |_, _| {}).unwrap();
//! // y(π/2) = sin(π/2) = 1, v(π/2) = cos(π/2) = 0
//! assert!((y[0] - 1.0).abs() < 1e-5 && y[1].abs() < 1e-5);
//! # }
//! ```
//!
//! **Steife Probleme:** Explizite Verfahren (Euler, RK4, RK45) brauchen bei steifen Gleichungen
//! (sehr schnell abklingende Anteile) winzige Schritte, um stabil zu bleiben. Dort sind die impliziten
//! Verfahren im Vorteil, die in jedem Schritt mit dem Newton-Verfahren aus
//! [`optimize`](crate::optimize) ein nichtlineares Gleichungssystem lösen.
//!
//! Hinweis zu `micromath`: Mit der bewusst ungenauen Näherung (z. B. bei `sin`, `exp`, `sqrt`
//! in `f`) lassen sich strenge Toleranzen unter Umständen nicht erreichen; für verlässliche
//! Ergebnisse `std` oder `libm` verwenden.

mod adams;
mod adaptive;
mod bdf;
#[cfg(feature = "alloc")]
mod bdf_adaptive;
#[cfg(feature = "alloc")]
mod bvp;
#[cfg(feature = "alloc")]
mod dae;
mod events;
mod explicit;
mod gauss_legendre;
mod implicit;
mod radau;
mod rk45;
mod rosenbrock;
#[cfg(feature = "alloc")]
mod seulex;
mod verlet;

pub use adams::{adams_bashforth_moulton, adams_work_len};
pub use adaptive::{
    AdaptiveOptions, AdaptiveResult, adaptive_work_len, radau5_adaptive, radau5_adaptive_fd,
    ros2_adaptive, ros2_adaptive_fd,
};
pub use bdf::{bdf, bdf_fd, bdf_work_len};
#[cfg(feature = "alloc")]
pub use bdf_adaptive::{BdfAdaptiveOptions, BdfAdaptiveResult, bdf_adaptive, bdf_adaptive_fd};
#[cfg(feature = "alloc")]
pub use bvp::{BvpResult, bvp_shooting, bvp_trapezoid};
#[cfg(feature = "alloc")]
pub use dae::dae_bdf;
pub use events::{
    Event, EventDirection, event_work_len, hermite_interpolate, rk4_event, rk45_event,
    rk45_event_work_len,
};
pub use explicit::{euler, euler_work_len, rk4, rk4_step, rk4_work_len, try_rk4_step};
pub use gauss_legendre::{gauss_legendre4, gauss_legendre4_fd, gauss_legendre4_work_len};
pub use implicit::{
    implicit_euler, implicit_euler_fd, implicit_trapezoid, implicit_trapezoid_fd, implicit_work_len,
};
pub use radau::{radau_work_len, radau5, radau5_fd};
pub use rk45::{Rk45Options, Rk45Result, rk45, rk45_work_len};
pub use rosenbrock::{ros2, ros2_fd, ros2_work_len};
#[cfg(feature = "alloc")]
pub use seulex::{SeulexOptions, SeulexResult, seulex, seulex_fd};
pub use verlet::{verlet, verlet_separable, verlet_work_len};

use core::fmt;

use crate::Real;
use crate::optimize::OptimizeError;

/// Fehler der ODE-Löser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OdeError {
    /// Ungültige Eingabe: leerer Zustand, 0 Schritte oder zu wenig Arbeitsspeicher.
    InvalidInput,
    /// NaN oder Unendlich in den Zeitgrenzen, im Zustand oder in `f`.
    NotFinite,
    /// Die Schrittweite wurde zu klein, um noch fortzuschreiten (zum Beispiel bei einer
    /// Singularität oder sehr steifem Problem für ein explizites Verfahren).
    StepSizeTooSmall,
    /// Das Limit `max_steps` wurde erreicht, bevor `t1` erreicht war.
    MaxStepsExceeded,
    /// Das Newton-Verfahren eines impliziten Schritts ist nicht konvergiert (zu große
    /// Schrittweite, singuläre Jacobi-Matrix oder ungeeignete Funktion).
    NewtonFailed,
    /// Die Nullstelle der Ereignisfunktion konnte im Schritt nicht lokalisiert werden.
    EventNotLocated,
}

impl fmt::Display for OdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            OdeError::InvalidInput => "ungültige Eingabe oder zu wenig Arbeitsspeicher",
            OdeError::NotFinite => "NaN oder unendlicher Wert aufgetreten",
            OdeError::StepSizeTooSmall => "Schrittweite zu klein",
            OdeError::MaxStepsExceeded => "maximale Schrittzahl erreicht",
            OdeError::NewtonFailed => "Newton-Verfahren im impliziten Schritt nicht konvergiert",
            OdeError::EventNotLocated => "Ereignis im Schritt nicht lokalisierbar",
        })
    }
}

impl core::error::Error for OdeError {}

impl From<OptimizeError> for OdeError {
    fn from(e: OptimizeError) -> Self {
        match e {
            OptimizeError::NotFinite => OdeError::NotFinite,
            OptimizeError::InvalidInput => OdeError::InvalidInput,
            _ => OdeError::NewtonFailed,
        }
    }
}

pub(crate) fn finite<T: Real>(x: T) -> bool {
    !(x.is_nan() || x.is_infinite())
}

pub(crate) fn all_finite<T: Real>(v: &[T]) -> bool {
    v.iter().all(|x| finite(*x))
}

/// `i` als `T`.
pub(crate) fn from_usize<T: Real>(i: usize) -> T {
    T::from_f64(i as f64)
}

/// Zeitraster direkt aus den Grenzen bilden; nicht darstellbare Schritte nicht vortäuschen.
pub(crate) fn fixed_times<T: Real>(
    t0: T,
    t1: T,
    i: usize,
    steps: usize,
) -> Result<(T, T, T), OdeError> {
    let point = |index| {
        if index == steps {
            return t1;
        }
        let fraction = from_usize::<T>(index) / from_usize::<T>(steps);
        if (t0 < T::ZERO) == (t1 < T::ZERO) {
            t0 + (t1 - t0) * fraction
        } else {
            (T::ONE - fraction) * t0 + fraction * t1
        }
    };
    let (t, next) = (point(i), point(i + 1));
    let h = next - t;
    if !finite(t) || !finite(next) || !finite(h) {
        return Err(OdeError::NotFinite);
    }
    if next == t {
        return Err(OdeError::StepSizeTooSmall);
    }
    Ok((t, next, h))
}

/// Gewichtet normierte Stufen erst nach der Summation rückskalieren.
pub(crate) fn weighted_update<T: Real>(
    base: T,
    h: T,
    weights: &[f64],
    value: impl Fn(usize) -> T,
) -> T {
    let mut scale = T::ZERO;
    for (i, w) in weights.iter().enumerate() {
        if *w != 0.0 {
            let v = value(i);
            if !finite(v) {
                return T::NAN;
            }
            scale = scale.max(v.abs());
        }
    }
    if scale == T::ZERO {
        return base;
    }
    let (mut sum, mut correction) = (T::ZERO, T::ZERO);
    for (i, w) in weights.iter().enumerate() {
        if *w != 0.0 {
            let term = T::from_f64(*w) * (value(i) / scale) - correction;
            let next = sum + term;
            correction = (next - sum) - term;
            sum = next;
        }
    }
    let (mut a, mut b, mut c) = (h, scale, sum);
    if a.abs() > b.abs() {
        core::mem::swap(&mut a, &mut b);
    }
    if b.abs() > c.abs() {
        core::mem::swap(&mut b, &mut c);
    }
    if a.abs() > b.abs() {
        core::mem::swap(&mut a, &mut b);
    }
    base + (a * c) * b
}
