//! Adaptive Schrittweite für die impliziten Einschrittverfahren [`radau5`] und [`ros2`] durch **Schrittverdopplung**.
//!
//! In jedem Versuch wird ein Schritt `h` mit zwei Schritten `h/2` verglichen. Der Fehler des genaueren Ergebnisses wird als
//! `‖y₂ − y₁‖ / (2ᵖ − 1)` geschätzt (`p` = Ordnung des Verfahrens) und in der gewichteten RMS-Norm mit der Skala
//! `atol + rtol·|y|` gemessen. Scheitert Newton in einem Schritt, wird die Schrittweite verkleinert statt abzubrechen.
//!
//! BDF mit variabler Schrittweite ist damit nicht abgedeckt (Mehrschritt-Historie); für steife Probleme mit Genauigkeitssteuerung
//! genügt [`radau5_adaptive`].

use super::{
    OdeError, all_finite, finite, radau_work_len, radau5, radau5_fd, ros2, ros2_fd, ros2_work_len,
};
use crate::Real;
use crate::optimize::OptimizeOptions;

/// Benötigter Arbeitsspeicher der `*_adaptive`-Funktionen bei `n` Zustandsgrößen.
pub const fn adaptive_work_len(n: usize) -> usize {
    let (a, b) = (radau_work_len(n), ros2_work_len(n));
    3 * n + if a > b { a } else { b }
}

/// Einstellungen der adaptiven Verfahren.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveOptions<T> {
    /// Relative Toleranz pro Komponente.
    pub rtol: T,
    /// Absolute Toleranz pro Komponente.
    pub atol: T,
    /// Anfangsschrittweite; `0` bedeutet: ein Tausendstel des Intervalls.
    pub h_init: T,
    /// Größte erlaubte Schrittweite; `0` bedeutet: unbegrenzt.
    pub h_max: T,
    /// Höchstzahl der Schrittversuche (akzeptierte und abgelehnte).
    pub max_steps: usize,
    /// Einstellungen für Newton (nur [`radau5_adaptive`]).
    pub newton: OptimizeOptions<T>,
}

impl<T: Real> Default for AdaptiveOptions<T> {
    /// `rtol = 1e-6`, `atol = 1e-9`, 100 000 Schrittversuche.
    fn default() -> Self {
        Self {
            rtol: T::from_f64(1.0e-6),
            atol: T::from_f64(1.0e-9),
            h_init: T::ZERO,
            h_max: T::ZERO,
            max_steps: 100_000,
            newton: OptimizeOptions::default(),
        }
    }
}

impl<T: Real> AdaptiveOptions<T> {
    /// Setzt `rtol` und `atol`, der Rest wie in `default()`.
    pub fn with_tolerance(rtol: T, atol: T) -> Self {
        Self {
            rtol,
            atol,
            ..Self::default()
        }
    }
}

/// Statistik eines adaptiven Laufs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveResult<T> {
    /// Anzahl der akzeptierten Schritte.
    pub accepted_steps: usize,
    /// Anzahl der abgelehnten Schritte (Fehler zu groß oder Newton gescheitert).
    pub rejected_steps: usize,
    /// Zuletzt verwendete Schrittweite (brauchbar als `h_init` einer Fortsetzung).
    pub last_step: T,
}

#[allow(clippy::too_many_arguments)]
fn adaptive_core<T: Real>(
    mut one: impl FnMut(T, T, &mut [T], &mut [T]) -> Result<(), OdeError>,
    order: usize,
    t0: T,
    t1: T,
    y: &mut [T],
    opts: AdaptiveOptions<T>,
    work: &mut [T],
    mut observer: impl FnMut(T, &[T]),
) -> Result<AdaptiveResult<T>, OdeError> {
    let n = y.len();
    if n == 0 || work.len() < adaptive_work_len(n) {
        return Err(OdeError::InvalidInput);
    }
    if !finite(t0) || !finite(t1) || !all_finite(y) {
        return Err(OdeError::NotFinite);
    }
    observer(t0, y);
    if t0 == t1 {
        return Ok(AdaptiveResult {
            accepted_steps: 0,
            rejected_steps: 0,
            last_step: T::ZERO,
        });
    }
    let (y_start, rest) = work.split_at_mut(n);
    let (y_full, rest) = rest.split_at_mut(n);
    let (y_half, inner) = rest.split_at_mut(n);

    let dir = if t1 > t0 { T::ONE } else { -T::ONE };
    let span = (t1 - t0).abs();
    let mut h = if opts.h_init > T::ZERO {
        opts.h_init
    } else {
        span * T::from_f64(1.0e-3)
    };
    let h_max = if opts.h_max > T::ZERO {
        opts.h_max
    } else {
        span
    };
    h = h.min(h_max).min(span);

    let denom = T::from_f64((1u64 << order) as f64 - 1.0);
    let exponent = T::from_f64(-1.0 / (order as f64 + 1.0));
    let (safety, min_f, max_f) = (T::from_f64(0.9), T::from_f64(0.2), T::from_f64(5.0));
    let half = T::from_f64(0.5);

    let mut t = t0;
    let (mut accepted, mut rejected, mut attempts) = (0usize, 0usize, 0usize);
    let mut last_step = h;
    while dir * (t1 - t) > T::ZERO {
        if attempts >= opts.max_steps {
            return Err(OdeError::MaxStepsExceeded);
        }
        attempts += 1;
        let remaining = dir * (t1 - t);
        let mut step = h;
        if step * T::from_f64(1.0001) >= remaining {
            step = remaining;
        }
        let hh = dir * step;
        if t + hh == t {
            return Err(OdeError::StepSizeTooSmall);
        }
        y_start.copy_from_slice(y);
        y_full.copy_from_slice(y_start);
        y_half.copy_from_slice(y_start);
        let tried = one(t, t + hh, y_full, inner)
            .and_then(|_| one(t, t + hh * half, y_half, inner))
            .and_then(|_| one(t + hh * half, t + hh, y_half, inner));
        match tried {
            Ok(()) => {}
            Err(OdeError::NewtonFailed) => {
                rejected += 1;
                h = step * T::from_f64(0.25);
                continue;
            }
            Err(e) => return Err(e),
        }
        if !all_finite(y_half) || !all_finite(y_full) {
            rejected += 1;
            h = step * T::from_f64(0.25);
            continue;
        }
        let mut s = T::ZERO;
        for i in 0..n {
            let e = (y_half[i] - y_full[i]) / denom;
            let q = e / (opts.atol + opts.rtol * y_half[i].abs().max(y_start[i].abs()));
            s += q * q;
        }
        let err = (s / T::from_usize(n)).sqrt();
        let factor = if err <= T::ZERO {
            max_f
        } else {
            (safety * err.powf(exponent)).max(min_f).min(max_f)
        };
        if err <= T::ONE {
            y.copy_from_slice(y_half);
            t += hh;
            accepted += 1;
            last_step = step;
            observer(t, y);
            h = (step * factor).min(h_max);
        } else {
            rejected += 1;
            h = step * factor.min(safety);
        }
    }
    Ok(AdaptiveResult {
        accepted_steps: accepted,
        rejected_steps: rejected,
        last_step,
    })
}

/// Radau IIA (Ordnung 5) mit Schrittweitensteuerung durch Schrittverdopplung. `jacobian(t, y, j)` schreibt `∂f/∂y`
/// zeilenweise. `work`: mindestens [`adaptive_work_len`]`(n)`. `observer(t, y)` nach jedem akzeptierten Schritt.
///
/// Fehler: [`OdeError::InvalidInput`], [`OdeError::NotFinite`], [`OdeError::StepSizeTooSmall`], [`OdeError::MaxStepsExceeded`].
#[allow(clippy::too_many_arguments)]
pub fn radau5_adaptive<T: Real>(
    mut f: impl FnMut(T, &[T], &mut [T]),
    mut jacobian: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    opts: AdaptiveOptions<T>,
    work: &mut [T],
    observer: impl FnMut(T, &[T]),
) -> Result<AdaptiveResult<T>, OdeError> {
    let newton = opts.newton;
    adaptive_core(
        |a, b, y, w| radau5(&mut f, &mut jacobian, a, b, y, 1, newton, w, |_, _| {}),
        5,
        t0,
        t1,
        y,
        opts,
        work,
        observer,
    )
}

/// Wie [`radau5_adaptive`], mit Jacobi-Matrix per Vorwärtsdifferenzen.
pub fn radau5_adaptive_fd<T: Real>(
    mut f: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    opts: AdaptiveOptions<T>,
    work: &mut [T],
    observer: impl FnMut(T, &[T]),
) -> Result<AdaptiveResult<T>, OdeError> {
    let newton = opts.newton;
    adaptive_core(
        |a, b, y, w| radau5_fd(&mut f, a, b, y, 1, newton, w, |_, _| {}),
        5,
        t0,
        t1,
        y,
        opts,
        work,
        observer,
    )
}

/// Rosenbrock-Verfahren [`ros2`] (Ordnung 2) mit Schrittweitensteuerung durch Schrittverdopplung.
#[allow(clippy::too_many_arguments)]
pub fn ros2_adaptive<T: Real>(
    mut f: impl FnMut(T, &[T], &mut [T]),
    mut jacobian: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    opts: AdaptiveOptions<T>,
    work: &mut [T],
    observer: impl FnMut(T, &[T]),
) -> Result<AdaptiveResult<T>, OdeError> {
    adaptive_core(
        |a, b, y, w| ros2(&mut f, &mut jacobian, a, b, y, 1, w, |_, _| {}),
        2,
        t0,
        t1,
        y,
        opts,
        work,
        observer,
    )
}

/// Wie [`ros2_adaptive`], mit Jacobi-Matrix per Vorwärtsdifferenzen.
pub fn ros2_adaptive_fd<T: Real>(
    mut f: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    opts: AdaptiveOptions<T>,
    work: &mut [T],
    observer: impl FnMut(T, &[T]),
) -> Result<AdaptiveResult<T>, OdeError> {
    adaptive_core(
        |a, b, y, w| ros2_fd(&mut f, a, b, y, 1, w, |_, _| {}),
        2,
        t0,
        t1,
        y,
        opts,
        work,
        observer,
    )
}
