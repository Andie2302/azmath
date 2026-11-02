use super::{OdeError, all_finite, finite, from_usize, rk4_step, rk4_work_len};
use crate::Real;

/// Höchste unterstützte Ordnung der Adams-Verfahren.
const MAX_ORDER: usize = 4;

/// Benötigter Arbeitsspeicher für [`adams_bashforth_moulton`] bei `n` Zustandsgrößen.
pub const fn adams_work_len(n: usize) -> usize {
    // RK4-Anlaufrechnung (5n) + Verlauf von f (4n) + Prädiktor (n) + f am Prädiktor (n)
    rk4_work_len(n) + MAX_ORDER * n + 2 * n
}

/// Koeffizienten des Adams-Bashforth-Prädiktors (Ordnung `p`), neuester Wert zuerst, bezogen auf den gemeinsamen Nenner.
fn bashforth(p: usize) -> (&'static [f64], f64) {
    match p {
        1 => (&[1.0], 1.0),
        2 => (&[3.0, -1.0], 2.0),
        3 => (&[23.0, -16.0, 5.0], 12.0),
        _ => (&[55.0, -59.0, 37.0, -9.0], 24.0),
    }
}

/// Koeffizienten des Adams-Moulton-Korrektors (Ordnung `p`): zuerst der Wert an `tₙ₊₁`, dann `tₙ`, `tₙ₋₁`, …
fn moulton(p: usize) -> (&'static [f64], f64) {
    match p {
        1 => (&[1.0], 1.0),
        2 => (&[1.0, 1.0], 2.0),
        3 => (&[5.0, 8.0, -1.0], 12.0),
        _ => (&[9.0, 19.0, -5.0, 1.0], 24.0),
    }
}

/// Adams-Bashforth-Moulton-Prädiktor-Korrektor-Verfahren (PECE) der Ordnung `order ∈ 1..=4` mit `steps` gleich großen
/// Schritten von `t0` nach `t1`.
///
/// Jeder Schritt sagt `yₙ₊₁` mit Adams-Bashforth voraus, wertet `f` dort aus und korrigiert mit Adams-Moulton; danach wird `f`
/// am korrigierten Wert noch einmal ausgewertet. Das braucht **zwei** Auswertungen von `f` pro Schritt (Runge-Kutta
/// gleicher Ordnung vier), nutzt aber die zurückliegenden Werte weiter. Es ist ein **explizites** Verfahren und daher für
/// steife Probleme ungeeignet (dort [`radau5`](super::radau5), [`bdf`](super::bdf) oder [`ros2`](super::ros2)).
///
/// Die ersten `order − 1` Schritte rechnet das klassische Runge-Kutta-Verfahren ([`rk4_step`]) an.
///
/// - `y`: Anfangszustand und Ergebnis (in-place), `work`: mindestens [`adams_work_len`]`(n)`.
/// - `observer(t, y)` wird mit dem Anfangszustand und nach jedem Schritt aufgerufen.
///
/// Fehler: [`OdeError::InvalidInput`] (auch bei `order` außerhalb von `1..=4`), [`OdeError::NotFinite`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::ode::{adams_bashforth_moulton, adams_work_len};
///
/// // y' = −y, y(0) = 1  ->  y(1) = e⁻¹
/// let mut y = [1.0];
/// let mut work = [0.0; adams_work_len(1)];
/// adams_bashforth_moulton(
///     |_t, y: &[f64], dy: &mut [f64]| dy[0] = -y[0],
///     0.0, 1.0, &mut y, 4, 40, &mut work, |_, _| {},
/// ).unwrap();
/// assert!((y[0] - (-1.0f64).exp()).abs() < 1e-7);
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn adams_bashforth_moulton<T: Real>(
    mut f: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    order: usize,
    steps: usize,
    work: &mut [T],
    mut observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError> {
    let n = y.len();
    if n == 0 || steps == 0 || order == 0 || order > MAX_ORDER || work.len() < adams_work_len(n) {
        return Err(OdeError::InvalidInput);
    }
    if !finite(t0) || !finite(t1) || !all_finite(y) {
        return Err(OdeError::NotFinite);
    }
    let (rk_work, rest) = work.split_at_mut(rk4_work_len(n));
    let (hist, rest) = rest.split_at_mut(MAX_ORDER * n);
    let (pred, rest) = rest.split_at_mut(n);
    let f_pred = &mut rest[..n];

    let h = (t1 - t0) / from_usize::<T>(steps);
    let (b, b_den) = bashforth(order);
    let (c, c_den) = moulton(order);
    let eval_hist =
        |f: &mut dyn FnMut(T, &[T], &mut [T]), t: T, y: &[T], dst: &mut [T]| f(t, y, dst);

    observer(t0, y);
    eval_hist(&mut f, t0, y, &mut hist[..n]);
    for i in 0..steps {
        let t = t0 + h * from_usize::<T>(i);
        let t_next = t0 + h * from_usize::<T>(i + 1);
        if i + 1 < order {
            // Anlaufrechnung mit RK4
            rk4_step(&mut f, t, y, h, rk_work);
        } else {
            // Prädiktor (hist[j] = f an tₙ₋ⱼ)
            for k in 0..n {
                let mut s = T::ZERO;
                for (j, &bj) in b.iter().enumerate() {
                    s += T::from_f64(bj) * hist[j * n + k];
                }
                pred[k] = y[k] + h * s / T::from_f64(b_den);
            }
            f(t_next, pred, f_pred);
            // Korrektor
            for k in 0..n {
                let mut s = T::from_f64(c[0]) * f_pred[k];
                for (j, &cj) in c.iter().enumerate().skip(1) {
                    s += T::from_f64(cj) * hist[(j - 1) * n + k];
                }
                y[k] += h * s / T::from_f64(c_den);
            }
        }
        if !all_finite(y) {
            return Err(OdeError::NotFinite);
        }
        // Verlauf nach hinten schieben, neuen Wert vorn eintragen
        hist.copy_within(0..(MAX_ORDER - 1) * n, n);
        eval_hist(&mut f, t_next, y, &mut hist[..n]);
        observer(t_next, y);
    }
    Ok(())
}
