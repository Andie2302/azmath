//! Randwertprobleme `y' = f(t, y)`, `g(y(a), y(b)) = 0` (Feature `alloc`): Einfachschießen und Trapezkollokation.

use super::{OdeError, Rk45Options, rk45, rk45_work_len};
use crate::Real;
use crate::optimize::{OptimizeOptions, newton_system_fd, newton_work_len};
use alloc::vec;
use alloc::vec::Vec;

/// Lösung eines Randwertproblems auf einem Gitter.
#[derive(Debug, Clone, PartialEq)]
pub struct BvpResult<T> {
    /// Gitterpunkte `t₀ < … < t_N` (aufsteigend, auch bei `t1 < t0` in Integrationsrichtung).
    pub t: Vec<T>,
    /// Lösung, `n` Werte je Gitterpunkt (zeilenweise: `y[k·n + i]`).
    pub y: Vec<T>,
    /// Iterationen des Newton-Verfahrens.
    pub iterations: usize,
}

/// Einfachschießverfahren: Gesucht ist der Anfangswert `s = y(t0)`, für den die Randbedingungen `bc(y(t0), y(t1), res) = 0`
/// (`n` Gleichungen) gelten; `y(t1)` entsteht mit [`rk45`], die Nullstelle von `s ↦ bc(…)` mit Newton (Differenzenquotienten).
/// Die Lösung wird auf `points + 1` gleich weit entfernten Punkten ausgegeben.
///
/// Gut für gutartige Probleme; bei stark instabilen Anfangswertproblemen ist [`bvp_trapezoid`] robuster.
///
/// Fehler: [`OdeError::InvalidInput`], [`OdeError::NotFinite`], [`OdeError::NewtonFailed`] sowie Fehler von `rk45`.
#[allow(clippy::too_many_arguments)]
pub fn bvp_shooting<T: Real>(
    mut f: impl FnMut(T, &[T], &mut [T]),
    mut bc: impl FnMut(&[T], &[T], &mut [T]),
    t0: T,
    t1: T,
    y0_guess: &[T],
    points: usize,
    rk: Rk45Options<T>,
    newton: OptimizeOptions<T>,
) -> Result<BvpResult<T>, OdeError> {
    let n = y0_guess.len();
    if n == 0 || points == 0 {
        return Err(OdeError::InvalidInput);
    }
    let mut s = y0_guess.to_vec();
    let mut rk_work = vec![T::ZERO; rk45_work_len(n)];
    let mut nw = vec![T::ZERO; newton_work_len(n)];
    let res = {
        let residual = |s: &[T], out: &mut [T]| {
            let mut y = s.to_vec();
            match rk45(&mut f, t0, t1, &mut y, rk, &mut rk_work, |_, _| {}) {
                Ok(_) => bc(s, &y, out),
                Err(_) => out.iter_mut().for_each(|v| *v = T::NAN),
            }
        };
        newton_system_fd(residual, &mut s, newton, &mut nw)?
    };
    let h = (t1 - t0) / T::from_usize(points);
    let mut t = Vec::with_capacity(points + 1);
    let mut y = Vec::with_capacity((points + 1) * n);
    let mut cur = s.clone();
    t.push(t0);
    y.extend_from_slice(&cur);
    for k in 1..=points {
        let (ta, tb) = (t0 + h * T::from_usize(k - 1), t0 + h * T::from_usize(k));
        rk45(&mut f, ta, tb, &mut cur, rk, &mut rk_work, |_, _| {})?;
        t.push(tb);
        y.extend_from_slice(&cur);
    }
    Ok(BvpResult {
        t,
        y,
        iterations: res.iterations,
    })
}

/// Trapezkollokation (Differenzenverfahren 2. Ordnung): Auf `intervals` gleichen Teilintervallen gelten
/// `yₖ₊₁ − yₖ = h/2·(f(tₖ, yₖ) + f(tₖ₊₁, yₖ₊₁))` und `bc(y₀, y_N) = 0`; das gekoppelte System aus `n·(intervals + 1)` Gleichungen
/// wird mit Newton (Differenzenquotienten, dicht) gelöst. `guess(t, y)` liefert die Startnäherung. Robust auch bei instabilen
/// Anfangswertproblemen; Kosten wachsen schnell mit `n·intervals` (dichte Jacobi-Matrix, für einige hundert Unbekannte gedacht).
///
/// Fehler: [`OdeError::InvalidInput`], [`OdeError::NotFinite`], [`OdeError::NewtonFailed`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::ode::bvp_trapezoid;
/// use azmath::optimize::OptimizeOptions;
///
/// // y'' = −y, y(0) = 0, y(π/2) = 1  ⇒  y = sin t; als System (y, y')
/// let r = bvp_trapezoid(
///     |_t, y: &[f64], dy: &mut [f64]| { dy[0] = y[1]; dy[1] = -y[0]; },
///     |ya: &[f64], yb: &[f64], res: &mut [f64]| { res[0] = ya[0]; res[1] = yb[0] - 1.0; },
///     0.0, core::f64::consts::FRAC_PI_2, 40, 2,
///     |_t, y: &mut [f64]| { y[0] = 0.0; y[1] = 0.0; },
///     OptimizeOptions::default(),
/// ).unwrap();
/// let mid = &r.y[20 * 2..20 * 2 + 2];
/// assert!((mid[0] - (core::f64::consts::FRAC_PI_4).sin()).abs() < 1e-3);
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn bvp_trapezoid<T: Real>(
    mut f: impl FnMut(T, &[T], &mut [T]),
    mut bc: impl FnMut(&[T], &[T], &mut [T]),
    t0: T,
    t1: T,
    intervals: usize,
    n: usize,
    guess: impl Fn(T, &mut [T]),
    newton: OptimizeOptions<T>,
) -> Result<BvpResult<T>, OdeError> {
    if n == 0 || intervals == 0 {
        return Err(OdeError::InvalidInput);
    }
    let total = n * (intervals + 1);
    let h = (t1 - t0) / T::from_usize(intervals);
    let ts: Vec<T> = (0..=intervals).map(|k| t0 + h * T::from_usize(k)).collect();
    let mut x = vec![T::ZERO; total];
    for (k, &t) in ts.iter().enumerate() {
        guess(t, &mut x[k * n..(k + 1) * n]);
    }
    let half = T::from_f64(0.5);
    let mut fa = vec![T::ZERO; n];
    let mut fb = vec![T::ZERO; n];
    let mut nw = vec![T::ZERO; newton_work_len(total)];
    let res = {
        let residual = |x: &[T], out: &mut [T]| {
            for k in 0..intervals {
                let (ya, yb) = (&x[k * n..(k + 1) * n], &x[(k + 1) * n..(k + 2) * n]);
                f(ts[k], ya, &mut fa);
                f(ts[k + 1], yb, &mut fb);
                for i in 0..n {
                    out[k * n + i] = yb[i] - ya[i] - half * h * (fa[i] + fb[i]);
                }
            }
            bc(&x[..n], &x[intervals * n..], &mut out[intervals * n..]);
        };
        newton_system_fd(residual, &mut x, newton, &mut nw)?
    };
    Ok(BvpResult {
        t: ts,
        y: x,
        iterations: res.iterations,
    })
}
