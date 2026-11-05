use super::{OdeError, all_finite, finite, fixed_times, weighted_update};
use crate::Real;

/// Benötigter Arbeitsspeicher für [`euler`] bei `n` Zustandsgrößen.
pub const fn euler_work_len(n: usize) -> usize {
    n
}

/// Benötigter Arbeitsspeicher für [`rk4`] und [`rk4_step`] bei `n` Zustandsgrößen.
pub const fn rk4_work_len(n: usize) -> usize {
    5 * n
}

fn check_fixed<T: Real>(
    t0: T,
    t1: T,
    y: &[T],
    steps: usize,
    work_len: usize,
    need: usize,
) -> Result<(), OdeError> {
    if y.is_empty() || steps == 0 || work_len < need {
        return Err(OdeError::InvalidInput);
    }
    if !finite(t0) || !finite(t1) || !all_finite(y) {
        return Err(OdeError::NotFinite);
    }
    Ok(())
}

/// Explizites Euler-Verfahren mit `steps` gleich großen Schritten von `t0` nach `t1`
/// (Ordnung 1, `y ← y + h·f(t, y)`).
///
/// - `y`: Anfangszustand und Ergebnis (in-place), `work`: mindestens [`euler_work_len`]`(n)`.
/// - `observer(t, y)` wird mit dem Anfangszustand und nach jedem Schritt aufgerufen.
///
/// Fehler: [`OdeError::InvalidInput`], [`OdeError::NotFinite`], [`OdeError::StepSizeTooSmall`].
/// Ein leeres Zeitintervall wertet `f` nicht aus; nicht darstellbare Zeitfortschritte sind Fehler.
pub fn euler<T: Real>(
    mut f: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    work: &mut [T],
    mut observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError> {
    let n = y.len();
    check_fixed(t0, t1, y, steps, work.len(), euler_work_len(n))?;
    let k = &mut work[..n];
    observer(t0, y);
    if t0 == t1 {
        return Ok(());
    }
    for i in 0..steps {
        let (t, next, h) = fixed_times(t0, t1, i, steps)?;
        f(t, y, k);
        if !all_finite(k) {
            return Err(OdeError::NotFinite);
        }
        for j in 0..n {
            k[j] = y[j] + h * k[j];
        }
        if !all_finite(k) {
            return Err(OdeError::NotFinite);
        }
        y.copy_from_slice(k);
        observer(next, y);
    }
    Ok(())
}

/// Ein einzelner Schritt des klassischen Runge-Kutta-Verfahrens (Ordnung 4) der Größe `h` von
/// `t` aus; `y` wird in-place fortgeschrieben. `work` hat mindestens [`rk4_work_len`]`(n)`
/// Elemente. Nützlich, um die Schrittschleife selbst zu steuern.
///
/// Bei numerischen Fehlern wird `y` auf NaN gesetzt; [`try_rk4_step`] meldet den Fehler
/// explizit und lässt den bisherigen Zustand unverändert.
///
/// # Panics
///
/// Wenn `work` zu kurz ist.
pub fn rk4_step<T: Real, F: FnMut(T, &[T], &mut [T])>(
    f: &mut F,
    t: T,
    y: &mut [T],
    h: T,
    work: &mut [T],
) {
    assert!(
        work.len() >= rk4_work_len(y.len()),
        "rk4_step: Arbeitsspeicher zu kurz"
    );
    if try_rk4_step(f, t, y, h, work).is_err() {
        y.fill(T::NAN);
    }
}

/// Geprüfter RK4-Einzelschritt. Nichtendliche Stufen werden vor dem nächsten Callback erkannt.
/// Bei Fehler bleibt `y` unverändert; siehe [`OdeError`].
pub fn try_rk4_step<T: Real, F: FnMut(T, &[T], &mut [T])>(
    f: &mut F,
    t: T,
    y: &mut [T],
    h: T,
    work: &mut [T],
) -> Result<(), OdeError> {
    let n = y.len();
    if n == 0 || work.len() < rk4_work_len(n) {
        return Err(OdeError::InvalidInput);
    }
    if !finite(t) || !finite(h) || !all_finite(y) {
        return Err(OdeError::NotFinite);
    }
    if h == T::ZERO {
        return Ok(());
    }
    let next = t + h;
    if !finite(next) {
        return Err(OdeError::NotFinite);
    }
    if next == t {
        return Err(OdeError::StepSizeTooSmall);
    }
    let h = next - t;
    if !finite(h) {
        return Err(OdeError::NotFinite);
    }
    let (k1, rest) = work.split_at_mut(n);
    let (k2, rest) = rest.split_at_mut(n);
    let (k3, rest) = rest.split_at_mut(n);
    let (k4, rest) = rest.split_at_mut(n);
    let tmp = &mut rest[..n];
    let half = h * T::from_f64(0.5);
    f(t, y, k1);
    if !all_finite(k1) {
        return Err(OdeError::NotFinite);
    }
    for i in 0..n {
        tmp[i] = y[i] + half * k1[i];
    }
    if !all_finite(tmp) {
        return Err(OdeError::NotFinite);
    }
    f(t + half, tmp, k2);
    if !all_finite(k2) {
        return Err(OdeError::NotFinite);
    }
    for i in 0..n {
        tmp[i] = y[i] + half * k2[i];
    }
    if !all_finite(tmp) {
        return Err(OdeError::NotFinite);
    }
    f(t + half, tmp, k3);
    if !all_finite(k3) {
        return Err(OdeError::NotFinite);
    }
    for i in 0..n {
        tmp[i] = y[i] + h * k3[i];
    }
    if !all_finite(tmp) {
        return Err(OdeError::NotFinite);
    }
    f(next, tmp, k4);
    if !all_finite(k4) {
        return Err(OdeError::NotFinite);
    }
    for i in 0..n {
        let values = [k1[i], k2[i], k3[i], k4[i]];
        tmp[i] = weighted_update(
            y[i],
            h,
            &[1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0],
            |j| values[j],
        );
    }
    if !all_finite(tmp) {
        return Err(OdeError::NotFinite);
    }
    y.copy_from_slice(tmp);
    Ok(())
}

/// Klassisches Runge-Kutta-Verfahren (Ordnung 4) mit `steps` gleich großen Schritten von `t0` nach
/// `t1`.
///
/// Der Fehler fällt mit `h⁴`: Halbiert man die Schrittweite, wird er etwa 16-mal kleiner. Ist die
/// passende Schrittweite nicht bekannt, ist [`rk45`](super::rk45) bequemer.
///
/// - `y`: Anfangszustand und Ergebnis (in-place), `work`: mindestens [`rk4_work_len`]`(n)`.
/// - `observer(t, y)` wird mit dem Anfangszustand und nach jedem Schritt aufgerufen.
///
/// Fehler: [`OdeError::InvalidInput`], [`OdeError::NotFinite`], [`OdeError::StepSizeTooSmall`].
/// Ein leeres Zeitintervall wertet `f` nicht aus; nicht darstellbare Zeitfortschritte sind Fehler.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::ode::{rk4, rk4_work_len};
///
/// // y' = −y, y(0) = 1  ->  y(1) = e⁻¹
/// let mut y = [1.0];
/// let mut work = [0.0; rk4_work_len(1)];
/// rk4(|_t, y: &[f64], dy: &mut [f64]| dy[0] = -y[0], 0.0, 1.0, &mut y, 20, &mut work, |_, _| {})
///     .unwrap();
/// assert!((y[0] - (-1.0f64).exp()).abs() < 1e-7);
/// # }
/// ```
pub fn rk4<T: Real>(
    mut f: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    work: &mut [T],
    mut observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError> {
    let n = y.len();
    check_fixed(t0, t1, y, steps, work.len(), rk4_work_len(n))?;
    observer(t0, y);
    if t0 == t1 {
        return Ok(());
    }
    for i in 0..steps {
        let (t, next, h) = fixed_times(t0, t1, i, steps)?;
        try_rk4_step(&mut f, t, y, h, work)?;
        observer(next, y);
    }
    Ok(())
}
