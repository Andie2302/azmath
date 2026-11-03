//! Differential-algebraische Gleichungen in voll impliziter Form `F(t, y, y') = 0` (Feature `alloc`).

use super::{OdeError, all_finite, finite};
use crate::Real;
use crate::optimize::{OptimizeOptions, newton_system_fd, newton_work_len};
use alloc::vec;
use alloc::vec::Vec;

/// BDF-Verfahren der Ordnung 1 oder 2 mit fester Schrittweite für `F(t, y, y') = 0` (Index 1; `F_{y'}` darf singulär sein,
/// `F_y' + γF_y` muss regulär sein). In jedem Schritt wird `F(tₙ₊₁, y, (αy + …)/h) = 0` für `y` mit Newton
/// (Differenzenquotienten) gelöst. **`y0` muss konsistent sein**, also die algebraischen Gleichungen erfüllen; `observer(t, y)` wird
/// am Anfang und nach jedem Schritt aufgerufen.
///
/// - `f(t, y, yp, res)` schreibt das Residuum `F(t, y, yp)` nach `res` (`n` Werte).
/// - `order`: 1 (implizites Euler-Verfahren) oder 2 (BDF2, der erste Schritt läuft mit BDF1).
///
/// Fehler: [`OdeError::InvalidInput`], [`OdeError::NotFinite`], [`OdeError::NewtonFailed`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::ode::dae_bdf;
/// use azmath::optimize::OptimizeOptions;
///
/// // y₁' = −y₁ + y₂,  0 = y₂ − 2·y₁  ⇒  y₁' = y₁, y₁(0) = 1
/// let mut y = [1.0f64, 2.0];
/// dae_bdf(
///     |_t, y: &[f64], yp: &[f64], r: &mut [f64]| { r[0] = yp[0] + y[0] - y[1]; r[1] = y[1] - 2.0 * y[0]; },
///     0.0, 1.0, &mut y, 400, 2, OptimizeOptions::default(), |_, _| {},
/// ).unwrap();
/// assert!((y[0] - 1f64.exp()).abs() < 1e-4);
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn dae_bdf<T: Real>(
    mut f: impl FnMut(T, &[T], &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    order: usize,
    newton: OptimizeOptions<T>,
    mut observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError> {
    let n = y.len();
    if n == 0 || steps == 0 || !(1..=2).contains(&order) {
        return Err(OdeError::InvalidInput);
    }
    if !finite(t0) || !finite(t1) || !all_finite(y) {
        return Err(OdeError::NotFinite);
    }
    let h = (t1 - t0) / T::from_usize(steps);
    let mut prev: Vec<T> = y.to_vec();
    let mut cur: Vec<T> = y.to_vec();
    let mut yp = vec![T::ZERO; n];
    let mut nw = vec![T::ZERO; newton_work_len(n)];
    observer(t0, y);
    let (two, three, four) = (T::from_f64(2.0), T::from_f64(3.0), T::from_f64(4.0));
    for k in 0..steps {
        let t = t0 + h * T::from_usize(k + 1);
        let use_bdf2 = order == 2 && k > 0;
        let mut x = cur.clone();
        {
            let (cur_ref, prev_ref) = (&cur, &prev);
            let residual = |x: &[T], out: &mut [T]| {
                for i in 0..n {
                    yp[i] = if use_bdf2 {
                        (three * x[i] - four * cur_ref[i] + prev_ref[i]) / (two * h)
                    } else {
                        (x[i] - cur_ref[i]) / h
                    };
                }
                f(t, x, &yp, out);
            };
            newton_system_fd(residual, &mut x, newton, &mut nw)?;
        }
        prev = cur;
        cur = x;
        observer(t, &cur);
    }
    y.copy_from_slice(&cur);
    Ok(())
}
