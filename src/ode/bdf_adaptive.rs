//! BDF-Verfahren mit variabler Schrittweite **und** variabler Ordnung (1 bis 5), Feature `alloc`.
//!
//! Die Formel wird in jedem Schritt aus der Lagrange-Interpolierenden durch `tₙ₊₁` und die letzten `p` Lösungspunkte auf dem
//! tatsächlichen (ungleichmäßigen) Gitter abgeleitet. Fehlerschätzer und Ordnungssteuerung nutzen dividierte Differenzen der
//! Lösungspunkte: `LTE_k ≈ |h|^{k+1}·k!·‖[tₙ₊₁ … tₙ₋ₖ] y‖` mit der gewichteten RMS-Norm der Skala `atol + rtol·|y|`.

use super::implicit::{Jac, NoJacobian};
use super::{OdeError, all_finite, finite};
use crate::Real;
use crate::optimize::{OptimizeOptions, newton_system, newton_system_fd, newton_work_len};
use alloc::vec;
use alloc::vec::Vec;

/// Einstellungen für [`bdf_adaptive`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BdfAdaptiveOptions<T> {
    /// Relative Toleranz pro Komponente.
    pub rtol: T,
    /// Absolute Toleranz pro Komponente.
    pub atol: T,
    /// Anfangsschrittweite; `0` bedeutet: ein Tausendstel des Intervalls.
    pub h_init: T,
    /// Größte erlaubte Schrittweite; `0` bedeutet: unbegrenzt.
    pub h_max: T,
    /// Höchste verwendete Ordnung (1 bis 5).
    pub max_order: usize,
    /// Höchstzahl der Schrittversuche (akzeptierte und abgelehnte).
    pub max_steps: usize,
    /// Einstellungen für Newton in jedem Schritt.
    pub newton: OptimizeOptions<T>,
}

impl<T: Real> Default for BdfAdaptiveOptions<T> {
    /// `rtol = 1e-6`, `atol = 1e-9`, Ordnung bis 5, 100 000 Schrittversuche.
    fn default() -> Self {
        Self {
            rtol: T::from_f64(1.0e-6),
            atol: T::from_f64(1.0e-9),
            h_init: T::ZERO,
            h_max: T::ZERO,
            max_order: 5,
            max_steps: 100_000,
            newton: OptimizeOptions::default(),
        }
    }
}

impl<T: Real> BdfAdaptiveOptions<T> {
    /// Setzt `rtol` und `atol`, der Rest wie in `default()`.
    pub fn with_tolerance(rtol: T, atol: T) -> Self {
        Self {
            rtol,
            atol,
            ..Self::default()
        }
    }
}

/// Statistik eines Laufs von [`bdf_adaptive`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BdfAdaptiveResult<T> {
    /// Akzeptierte Schritte.
    pub accepted_steps: usize,
    /// Abgelehnte Schritte (Fehler zu groß oder Newton gescheitert).
    pub rejected_steps: usize,
    /// Zuletzt verwendete Schrittweite.
    pub last_step: T,
    /// Zuletzt verwendete Ordnung.
    pub last_order: usize,
}

/// Gewichte `c₀ … c_p` mit `y'(x₀) ≈ Σ cⱼ·yⱼ` aus dem Interpolationspolynom durch die Knoten `x₀ … x_p` (Ableitung im Knoten `x₀`).
fn derivative_weights<T: Real>(x: &[T]) -> Vec<T> {
    let p = x.len() - 1;
    let mut c = vec![T::ZERO; p + 1];
    for j in 1..=p {
        c[0] += T::ONE / (x[0] - x[j]);
    }
    for j in 1..=p {
        let mut w = T::ONE / (x[j] - x[0]);
        for m in 1..=p {
            if m != j {
                w = w * (x[0] - x[m]) / (x[j] - x[m]);
            }
        }
        c[j] = w;
    }
    c
}

/// Lagrange-Extrapolation der Werte `ys` (an den Knoten `xs`) nach `x`.
fn lagrange<T: Real>(xs: &[T], ys: &[&[T]], x: T, out: &mut [T]) {
    out.iter_mut().for_each(|v| *v = T::ZERO);
    for j in 0..xs.len() {
        let mut w = T::ONE;
        for m in 0..xs.len() {
            if m != j {
                w = w * (x - xs[m]) / (xs[j] - xs[m]);
            }
        }
        for (o, v) in out.iter_mut().zip(ys[j]) {
            *o += w * *v;
        }
    }
}

/// Dividierte Differenz höchster Ordnung `[x₀ … x_k] y` (vektorwertig).
fn divided_difference<T: Real>(xs: &[T], ys: &[&[T]]) -> Vec<T> {
    let m = xs.len();
    let n = ys[0].len();
    let mut table: Vec<Vec<T>> = ys.iter().map(|v| v.to_vec()).collect();
    for level in 1..m {
        for i in 0..m - level {
            let dx = xs[i + level] - xs[i];
            for k in 0..n {
                table[i][k] = (table[i + 1][k] - table[i][k]) / dx;
            }
        }
    }
    table.swap_remove(0)
}

fn weighted_norm<T: Real>(v: &[T], y_new: &[T], y_old: &[T], rtol: T, atol: T) -> T {
    let mut s = T::ZERO;
    for i in 0..v.len() {
        let q = v[i] / (atol + rtol * y_new[i].abs().max(y_old[i].abs()));
        s += q * q;
    }
    (s / T::from_usize(v.len())).sqrt()
}

/// `LTE_k` aus den letzten `k + 2` Punkten des Verlaufs.
fn lte<T: Real>(ts: &[T], ys: &[Vec<T>], k: usize, h: T, rtol: T, atol: T) -> Option<T> {
    let len = ts.len();
    if len < k + 2 {
        return None;
    }
    let lo = len - (k + 2);
    let xs: Vec<T> = ts[lo..].iter().rev().copied().collect();
    let vs: Vec<&[T]> = ys[lo..].iter().rev().map(|v| v.as_slice()).collect();
    let dd = divided_difference(&xs, &vs);
    let mut fact = T::ONE;
    for i in 2..=k {
        fact *= T::from_usize(i);
    }
    let scale = h.abs().powi(k as i32 + 1) * fact;
    let scaled: Vec<T> = dd.iter().map(|v| *v * scale).collect();
    Some(weighted_norm(&scaled, &vs[0], &vs[1], rtol, atol))
}

#[allow(clippy::too_many_arguments)]
fn bdf_adaptive_core<T, F, J>(
    mut f: F,
    mut jac: Jac<J>,
    t0: T,
    t1: T,
    y: &mut [T],
    opts: BdfAdaptiveOptions<T>,
    mut observer: impl FnMut(T, &[T]),
) -> Result<BdfAdaptiveResult<T>, OdeError>
where
    T: Real,
    F: FnMut(T, &[T], &mut [T]),
    J: FnMut(T, &[T], &mut [T]),
{
    let n = y.len();
    if n == 0 || opts.max_order == 0 || opts.max_order > 5 {
        return Err(OdeError::InvalidInput);
    }
    if !finite(t0) || !finite(t1) || !all_finite(y) {
        return Err(OdeError::NotFinite);
    }
    observer(t0, y);
    if t0 == t1 {
        return Ok(BdfAdaptiveResult {
            accepted_steps: 0,
            rejected_steps: 0,
            last_step: T::ZERO,
            last_order: 1,
        });
    }
    let dir = if t1 > t0 { T::ONE } else { -T::ONE };
    let span = (t1 - t0).abs();
    let h_max = if opts.h_max > T::ZERO {
        opts.h_max
    } else {
        span
    };
    let mut h = if opts.h_init > T::ZERO {
        opts.h_init
    } else {
        span * T::from_f64(1.0e-3)
    }
    .min(h_max);

    let mut ts: Vec<T> = vec![t0];
    let mut ys: Vec<Vec<T>> = vec![y.to_vec()];
    let mut f0 = vec![T::ZERO; n];
    f(t0, y, &mut f0);
    if !all_finite(&f0) {
        return Err(OdeError::NotFinite);
    }
    let mut order = 1usize;
    let mut steps_at_order = 0usize;
    let mut nw = vec![T::ZERO; newton_work_len(n)];
    let mut ftmp = vec![T::ZERO; n];
    let (mut accepted, mut rejected, mut attempts) = (0usize, 0usize, 0usize);
    let mut last_step = h;
    let min_f = T::from_f64(0.2);
    let tol = opts.rtol.max(opts.atol);
    let _ = tol;

    let mut t = t0;
    while dir * (t1 - t) > T::ZERO {
        if attempts >= opts.max_steps {
            return Err(OdeError::MaxStepsExceeded);
        }
        attempts += 1;
        let remaining = dir * (t1 - t);
        let mut step = h.min(h_max);
        if step * T::from_f64(1.0001) >= remaining {
            step = remaining;
        }
        let hh = dir * step;
        if t + hh == t {
            return Err(OdeError::StepSizeTooSmall);
        }
        let len = ts.len();
        let p = order
            .min(len.max(1))
            .min(if len >= 2 { len - 1 } else { 1 })
            .max(1);
        let tn1 = t + hh;
        // Knoten: tₙ₊₁, tₙ, tₙ₋₁, …
        let mut xs = vec![tn1];
        for j in 0..p {
            xs.push(ts[len - 1 - j]);
        }
        let c = derivative_weights(&xs);
        // Startnäherung
        let mut x = vec![T::ZERO; n];
        if len >= p + 1 {
            let nodes: Vec<T> = (0..=p).map(|j| ts[len - 1 - j]).collect();
            let vals: Vec<&[T]> = (0..=p).map(|j| ys[len - 1 - j].as_slice()).collect();
            lagrange(&nodes, &vals, tn1, &mut x);
        } else {
            for i in 0..n {
                x[i] = ys[len - 1][i] + hh * f0[i];
            }
        }
        let hist: Vec<&[T]> = (0..p).map(|j| ys[len - 1 - j].as_slice()).collect();
        let c0 = c[0];
        let solved = {
            let g = |z: &[T], out: &mut [T]| {
                f(tn1, z, &mut ftmp);
                for k in 0..n {
                    let mut s = c0 * z[k];
                    for j in 0..p {
                        s += c[j + 1] * hist[j][k];
                    }
                    out[k] = s - ftmp[k];
                }
            };
            match &mut jac {
                Jac::Analytic(jf) => {
                    let jg = |z: &[T], out: &mut [T]| {
                        jf(tn1, z, out);
                        for v in out.iter_mut() {
                            *v = -*v;
                        }
                        for k in 0..n {
                            out[k * n + k] += c0;
                        }
                    };
                    newton_system(g, jg, &mut x, opts.newton, &mut nw)
                }
                Jac::Numeric => newton_system_fd(g, &mut x, opts.newton, &mut nw),
            }
        };
        if solved.is_err() || !all_finite(&x) {
            rejected += 1;
            h = step * T::from_f64(0.25);
            continue;
        }
        // Fehlerschätzung der Ordnung p
        let mut tt = ts.clone();
        let mut yy = ys.clone();
        tt.push(tn1);
        yy.push(x.clone());
        let err = match lte(&tt, &yy, p, step, opts.rtol, opts.atol) {
            Some(e) => e,
            None => {
                // Anlauf: Euler-Prädiktor
                let d: Vec<T> = (0..n)
                    .map(|i| (x[i] - ys[len - 1][i] - hh * f0[i]) * T::from_f64(0.5))
                    .collect();
                weighted_norm(&d, &x, &ys[len - 1], opts.rtol, opts.atol)
            }
        };
        if !(err <= T::ONE) {
            rejected += 1;
            let fac = if err.is_finite() {
                (T::from_f64(0.9) * (T::ONE / err).powf(T::ONE / T::from_usize(p + 1))).max(min_f)
            } else {
                min_f
            };
            h = step * fac.min(T::from_f64(0.9));
            // nach Ablehnung zunächst Ordnung halten
            steps_at_order = 0;
            continue;
        }
        // akzeptiert
        accepted += 1;
        last_step = step;
        t = tn1;
        ts.push(tn1);
        ys.push(x.clone());
        if ts.len() > 8 {
            ts.remove(0);
            ys.remove(0);
        }
        y.copy_from_slice(&x);
        f(t, y, &mut f0);
        if !all_finite(&f0) {
            return Err(OdeError::NotFinite);
        }
        observer(t, y);
        steps_at_order += 1;
        // Ordnungs- und Schrittweitenwahl
        let fac_for = |e: T, k: usize| -> T {
            if e <= T::ZERO {
                T::from_f64(2.0)
            } else {
                (T::from_f64(0.5) / e).powf(T::ONE / T::from_usize(k + 1))
            }
        };
        let mut best_k = p;
        let mut best_fac = fac_for(err, p);
        if steps_at_order > p {
            let lenn = ts.len();
            let mut candidates: Vec<usize> = Vec::new();
            if p > 1 {
                candidates.push(p - 1);
            }
            if p < opts.max_order {
                candidates.push(p + 1);
            }
            for k in candidates {
                if lenn >= k + 2 {
                    if let Some(e) = lte(&ts, &ys, k, step, opts.rtol, opts.atol) {
                        let fk = fac_for(e, k);
                        if fk > best_fac * T::from_f64(1.1) {
                            best_fac = fk;
                            best_k = k;
                        }
                    }
                }
            }
        }
        if best_k != p {
            steps_at_order = 0;
        }
        order = best_k;
        let fac = best_fac.max(min_f).min(T::from_f64(2.0));
        h = (step * fac).min(h_max);
    }
    Ok(BdfAdaptiveResult {
        accepted_steps: accepted,
        rejected_steps: rejected,
        last_step,
        last_order: order,
    })
}

/// BDF mit Schrittweiten- **und** Ordnungssteuerung (Ordnung 1 bis `opts.max_order ≤ 5`) für steife Probleme. `jacobian(t, y, j)`
/// schreibt `∂f/∂y` zeilenweise; ohne Ableitung: [`bdf_adaptive_fd`]. Rückwärtsintegration (`t1 < t0`) ist möglich.
/// `observer(t, y)` wird am Anfang und nach jedem akzeptierten Schritt aufgerufen.
///
/// Fehler: [`OdeError::InvalidInput`], [`OdeError::NotFinite`], [`OdeError::StepSizeTooSmall`], [`OdeError::MaxStepsExceeded`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::ode::{bdf_adaptive, BdfAdaptiveOptions};
///
/// // steif: y' = −1000·(y − cos t)
/// let mut y = [1.0f64];
/// let r = bdf_adaptive(
///     |t, y: &[f64], dy: &mut [f64]| dy[0] = -1000.0 * (y[0] - t.cos()),
///     |_t, _y: &[f64], j: &mut [f64]| j[0] = -1000.0,
///     0.0, 2.0, &mut y, BdfAdaptiveOptions::with_tolerance(1e-7, 1e-10), |_, _| {},
/// ).unwrap();
/// assert!(r.accepted_steps < 1000);
/// let exact = (1.0e6 * 2f64.cos() + 1000.0 * 2f64.sin()) / (1.0e6 + 1.0);
/// assert!((y[0] - exact).abs() < 1e-4);
/// # }
/// ```
pub fn bdf_adaptive<T: Real>(
    f: impl FnMut(T, &[T], &mut [T]),
    jacobian: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    opts: BdfAdaptiveOptions<T>,
    observer: impl FnMut(T, &[T]),
) -> Result<BdfAdaptiveResult<T>, OdeError> {
    bdf_adaptive_core(f, Jac::Analytic(jacobian), t0, t1, y, opts, observer)
}

/// Wie [`bdf_adaptive`], bestimmt die Jacobi-Matrix aber selbst per Vorwärtsdifferenzen.
pub fn bdf_adaptive_fd<T: Real>(
    f: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    opts: BdfAdaptiveOptions<T>,
    observer: impl FnMut(T, &[T]),
) -> Result<BdfAdaptiveResult<T>, OdeError> {
    bdf_adaptive_core(f, Jac::Numeric::<NoJacobian<T>>, t0, t1, y, opts, observer)
}
