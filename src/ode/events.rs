use super::{
    OdeError, Rk45Options, all_finite, finite, from_usize, rk4_step, rk4_work_len, rk45,
    rk45_work_len,
};
use crate::Real;
use crate::roots::{RootError, RootOptions, brent};

/// Kubische Hermite-Interpolation („dichte Ausgabe“) zwischen zwei Lösungspunkten: aus den Zuständen `y0`, `y1` und den
/// Ableitungen `f0 = f(t0, y0)`, `f1 = f(t0 + h, y1)` ergibt sich der Zustand bei `t0 + θ·h` (`θ ∈ [0, 1]`) in `out`.
/// Der Interpolationsfehler ist von der Ordnung `h⁴`, passt also zu Runge-Kutta-Verfahren der Ordnung 4.
///
/// ```
/// use azmath::ode::hermite_interpolate;
///
/// // y = t² auf [0, 1]: y0 = 0, y1 = 1, f0 = 0, f1 = 2 -> exakt bei jedem θ
/// let mut out = [0.0f64];
/// hermite_interpolate(&[0.0], &[0.0], &[1.0], &[2.0], 1.0, 0.5, &mut out);
/// assert!((out[0] - 0.25).abs() < 1e-15);
/// ```
pub fn hermite_interpolate<T: Real>(
    y0: &[T],
    f0: &[T],
    y1: &[T],
    f1: &[T],
    h: T,
    theta: T,
    out: &mut [T],
) {
    let t2 = theta * theta;
    let t3 = t2 * theta;
    let two = T::from_f64(2.0);
    let three = T::from_f64(3.0);
    let h00 = two * t3 - three * t2 + T::ONE;
    let h10 = t3 - two * t2 + theta;
    let h01 = -two * t3 + three * t2;
    let h11 = t3 - t2;
    for k in 0..out.len() {
        out[k] = h00 * y0[k] + h10 * h * f0[k] + h01 * y1[k] + h11 * h * f1[k];
    }
}

/// In welcher Richtung ein Vorzeichenwechsel der Ereignisfunktion `g` zählt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventDirection {
    /// Nur von negativ nach nicht negativ.
    Rising,
    /// Nur von positiv nach nicht positiv.
    Falling,
    /// In beiden Richtungen.
    Either,
}

/// Ein gefundenes Ereignis `g(t, y) = 0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Event<T> {
    /// Zeitpunkt des Ereignisses.
    pub t: T,
    /// Wert von `g` dort (nahe 0).
    pub g_value: T,
    /// Zahl der vollständig ausgeführten Schritte vor dem Schritt, der das Ereignis enthält.
    pub steps_before: usize,
}

/// Benötigter Arbeitsspeicher für [`rk4_event`] bei `n` Zustandsgrößen.
pub const fn event_work_len(n: usize) -> usize {
    // RK4 (5n) + Zustand vor dem Schritt, f an beiden Enden, Interpolationspuffer (4n)
    rk4_work_len(n) + 4 * n
}

/// Integriert mit dem klassischen Runge-Kutta-Verfahren (`steps` gleich große Schritte von `t0` nach `t1`) und bricht
/// beim **ersten Nulldurchgang** der Ereignisfunktion `g(t, y)` ab.
///
/// Innerhalb des Schritts mit Vorzeichenwechsel wird die Nullstelle auf der Hermite-Interpolierenden
/// ([`hermite_interpolate`]) mit dem Brent-Verfahren ([`brent`]) auf `tol` genau bestimmt. Bei einem Ereignis steht in
/// `y` der **Zustand zum Ereigniszeitpunkt**, sonst der Endzustand bei `t1` und die Rückgabe ist `None`. Typische
/// Anwendungen: Aufprall eines Balls (`g = Höhe`), Maximum einer Größe (`g = Ableitung`, Richtung `Falling`).
///
/// - Ein Vorzeichenwechsel zählt nur zwischen den Schrittenden; zwei Nulldurchgänge innerhalb **eines** Schritts bleiben
///   unbemerkt (Schrittzahl erhöhen).
/// - `work`: mindestens [`event_work_len`]`(n)`.
///
/// Fehler: [`OdeError::InvalidInput`], [`OdeError::NotFinite`], [`OdeError::EventNotLocated`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::ode::{rk4_event, event_work_len, EventDirection};
///
/// // freier Fall: h' = v, v' = −9,81; wann ist h = 0? (Start in 10 m Höhe)
/// let mut y = [10.0, 0.0];
/// let mut work = [0.0; event_work_len(2)];
/// let ev = rk4_event(
///     |_t, y: &[f64], dy: &mut [f64]| { dy[0] = y[1]; dy[1] = -9.81; },
///     |_t, y: &[f64]| y[0],
///     0.0, 5.0, &mut y, 50, EventDirection::Falling, 1e-12, &mut work,
/// ).unwrap().unwrap();
/// assert!((ev.t - (2.0f64 * 10.0 / 9.81).sqrt()).abs() < 1e-8);
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn rk4_event<T: Real>(
    mut f: impl FnMut(T, &[T], &mut [T]),
    mut g: impl FnMut(T, &[T]) -> T,
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    direction: EventDirection,
    tol: T,
    work: &mut [T],
) -> Result<Option<Event<T>>, OdeError> {
    let n = y.len();
    if n == 0 || steps == 0 || work.len() < event_work_len(n) {
        return Err(OdeError::InvalidInput);
    }
    if !finite(t0) || !finite(t1) || !all_finite(y) {
        return Err(OdeError::NotFinite);
    }
    let (rk_work, rest) = work.split_at_mut(rk4_work_len(n));
    let (y_prev, rest) = rest.split_at_mut(n);
    let (f0, rest) = rest.split_at_mut(n);
    let (f1, rest) = rest.split_at_mut(n);
    let tmp = &mut rest[..n];

    let h = (t1 - t0) / from_usize::<T>(steps);
    let mut g_prev = g(t0, y);
    if !finite(g_prev) {
        return Err(OdeError::NotFinite);
    }
    for i in 0..steps {
        let t = t0 + h * from_usize::<T>(i);
        y_prev.copy_from_slice(y);
        f(t, y_prev, f0);
        rk4_step(&mut f, t, y, h, rk_work);
        if !all_finite(y) {
            return Err(OdeError::NotFinite);
        }
        let t_next = t0 + h * from_usize::<T>(i + 1);
        let g_new = g(t_next, y);
        if !finite(g_new) {
            return Err(OdeError::NotFinite);
        }
        let rising = g_prev < T::ZERO && g_new >= T::ZERO;
        let falling = g_prev > T::ZERO && g_new <= T::ZERO;
        let hit = match direction {
            EventDirection::Rising => rising,
            EventDirection::Falling => falling,
            EventDirection::Either => rising || falling,
        };
        if hit {
            f(t_next, y, f1);
            // Nullstelle von φ(θ) = g(t + θh, y(θ)) auf [0, 1]
            let phi = |theta: T| {
                hermite_interpolate(y_prev, f0, y, f1, h, theta, tmp);
                g(t + theta * h, tmp)
            };
            let root =
                brent(phi, T::ZERO, T::ONE, RootOptions::new(tol, 200)).map_err(|e| match e {
                    RootError::NotFinite => OdeError::NotFinite,
                    _ => OdeError::EventNotLocated,
                })?;
            hermite_interpolate(y_prev, f0, y, f1, h, root.root, tmp);
            y.copy_from_slice(tmp);
            return Ok(Some(Event {
                t: t + root.root * h,
                g_value: root.f_value,
                steps_before: i,
            }));
        }
        g_prev = g_new;
    }
    Ok(None)
}

/// Benötigter Arbeitsspeicher für [`rk45_event`] bei `n` Zustandsgrößen.
pub const fn rk45_event_work_len(n: usize) -> usize {
    rk45_work_len(n) + 2 * n
}

/// Wie [`rk4_event`], aber mit dem adaptiven Verfahren [`rk45`].
///
/// Das Intervall `[t0, t1]` wird in `segments` gleich lange Abschnitte geteilt; jeder wird mit `rk45` integriert und an
/// den Abschnittsgrenzen auf einen Vorzeichenwechsel von `g` geprüft (zwei Nulldurchgänge innerhalb **eines** Abschnitts
/// bleiben unbemerkt). Die Nullstelle wird mit [`brent`] auf `t ↦ g(t, y(t))` bestimmt, wobei `y(t)` jeweils durch
/// erneute Integration vom Abschnittsanfang entsteht (kein Interpolationsfehler). Bei einem Ereignis steht in `y` der
/// Zustand zum Ereigniszeitpunkt, `steps_before` ist der Index des Abschnitts. Sonst steht in `y` der Endzustand und die
/// Rückgabe ist `None`.
///
/// `work`: mindestens [`rk45_event_work_len`]`(n)`.
///
/// Fehler: [`OdeError::InvalidInput`], [`OdeError::NotFinite`], [`OdeError::EventNotLocated`] sowie die Fehler von `rk45`.
#[allow(clippy::too_many_arguments)]
pub fn rk45_event<T: Real>(
    mut f: impl FnMut(T, &[T], &mut [T]),
    mut g: impl FnMut(T, &[T]) -> T,
    t0: T,
    t1: T,
    y: &mut [T],
    opts: Rk45Options<T>,
    direction: EventDirection,
    tol: T,
    segments: usize,
    work: &mut [T],
) -> Result<Option<Event<T>>, OdeError> {
    let n = y.len();
    if n == 0 || segments == 0 || work.len() < rk45_event_work_len(n) {
        return Err(OdeError::InvalidInput);
    }
    if !finite(t0) || !finite(t1) || !all_finite(y) {
        return Err(OdeError::NotFinite);
    }
    let (rk_work, rest) = work.split_at_mut(rk45_work_len(n));
    let (y_a, rest) = rest.split_at_mut(n);
    let y_trial = &mut rest[..n];

    let h = (t1 - t0) / from_usize::<T>(segments);
    let mut g_prev = g(t0, y);
    if !finite(g_prev) {
        return Err(OdeError::NotFinite);
    }
    for i in 0..segments {
        let ta = t0 + h * from_usize::<T>(i);
        let tb = if i + 1 == segments {
            t1
        } else {
            t0 + h * from_usize::<T>(i + 1)
        };
        y_a.copy_from_slice(y);
        rk45(&mut f, ta, tb, y, opts, rk_work, |_, _| {})?;
        let g_new = g(tb, y);
        if !finite(g_new) {
            return Err(OdeError::NotFinite);
        }
        let rising = g_prev < T::ZERO && g_new >= T::ZERO;
        let falling = g_prev > T::ZERO && g_new <= T::ZERO;
        let hit = match direction {
            EventDirection::Rising => rising,
            EventDirection::Falling => falling,
            EventDirection::Either => rising || falling,
        };
        if hit {
            let mut failure: Option<OdeError> = None;
            let root = {
                let phi = |s: T| {
                    y_trial.copy_from_slice(y_a);
                    match rk45(&mut f, ta, s, y_trial, opts, rk_work, |_, _| {}) {
                        Ok(_) => g(s, y_trial),
                        Err(e) => {
                            failure = Some(e);
                            T::NAN
                        }
                    }
                };
                brent(phi, ta, tb, RootOptions::new(tol, 200))
            };
            if let Some(e) = failure {
                return Err(e);
            }
            let root = root.map_err(|e| match e {
                RootError::NotFinite => OdeError::NotFinite,
                _ => OdeError::EventNotLocated,
            })?;
            y.copy_from_slice(y_a);
            rk45(&mut f, ta, root.root, y, opts, rk_work, |_, _| {})?;
            return Ok(Some(Event {
                t: root.root,
                g_value: root.f_value,
                steps_before: i,
            }));
        }
        g_prev = g_new;
    }
    Ok(None)
}
