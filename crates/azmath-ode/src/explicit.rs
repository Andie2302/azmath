use super::{OdeError, all_finite, finite, from_usize};
use azmath_core::Real;

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
/// Fehler: [`OdeError::InvalidInput`], [`OdeError::NotFinite`].
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
    let h = (t1 - t0) / from_usize::<T>(steps);
    let k = &mut work[..n];
    observer(t0, y);
    for i in 0..steps {
        let t = t0 + h * from_usize::<T>(i);
        f(t, y, k);
        for (yi, ki) in y.iter_mut().zip(k.iter()) {
            *yi += h * *ki;
        }
        if !all_finite(y) {
            return Err(OdeError::NotFinite);
        }
        observer(t0 + h * from_usize::<T>(i + 1), y);
    }
    Ok(())
}

/// Ein einzelner Schritt des klassischen Runge-Kutta-Verfahrens (Ordnung 4) der Größe `h` von
/// `t` aus; `y` wird in-place fortgeschrieben. `work` hat mindestens [`rk4_work_len`]`(n)`
/// Elemente. Nützlich, um die Schrittschleife selbst zu steuern.
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
    let n = y.len();
    assert!(
        work.len() >= rk4_work_len(n),
        "rk4_step: Arbeitsspeicher zu kurz"
    );
    let (k1, rest) = work.split_at_mut(n);
    let (k2, rest) = rest.split_at_mut(n);
    let (k3, rest) = rest.split_at_mut(n);
    let (k4, rest) = rest.split_at_mut(n);
    let tmp = &mut rest[..n];
    let half = T::from_f64(0.5) * h;

    f(t, y, k1);
    for i in 0..n {
        tmp[i] = y[i] + half * k1[i];
    }
    f(t + half, tmp, k2);
    for i in 0..n {
        tmp[i] = y[i] + half * k2[i];
    }
    f(t + half, tmp, k3);
    for i in 0..n {
        tmp[i] = y[i] + h * k3[i];
    }
    f(t + h, tmp, k4);

    let sixth = h / T::from_f64(6.0);
    let two = T::from_f64(2.0);
    for i in 0..n {
        y[i] += sixth * (k1[i] + two * k2[i] + two * k3[i] + k4[i]);
    }
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
/// Fehler: [`OdeError::InvalidInput`], [`OdeError::NotFinite`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath_ode::{rk4, rk4_work_len};
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
    let h = (t1 - t0) / from_usize::<T>(steps);
    observer(t0, y);
    for i in 0..steps {
        let t = t0 + h * from_usize::<T>(i);
        rk4_step(&mut f, t, y, h, work);
        if !all_finite(y) {
            return Err(OdeError::NotFinite);
        }
        observer(t0 + h * from_usize::<T>(i + 1), y);
    }
    Ok(())
}
