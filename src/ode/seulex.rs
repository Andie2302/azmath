//! Extrapoliertes linear-implizites Euler-Verfahren (SEULEX-Typ) für steife Probleme, Feature `alloc`.
//!
//! Das linear-implizite Euler-Verfahren `(I − h·J)·Δ = h·f(y)` besitzt eine asymptotische Entwicklung in `h`; die Extrapolation
//! (Aitken-Neville mit der harmonischen Folge `nⱼ = 1, 2, 3, …`) hebt die Ordnung auf `k` mit `k` Spalten. Das Verfahren ist die
//! praktische Alternative zu hochgradigen Rosenbrock-Verfahren (RODAS): L-stabil in der ersten Spalte, genau, ohne Newton-Iteration
//! (pro Schritt nur LU-Zerlegungen) und mit eingebauter Fehlerschätzung `‖T_{k,k} − T_{k,k−1}‖`.

use super::implicit::{Jac, NoJacobian};
use super::{OdeError, all_finite, finite};
use crate::Real;
use crate::linalg::{DynMatrix, DynVector};
use alloc::vec;
use alloc::vec::Vec;

/// Einstellungen für [`seulex`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeulexOptions<T> {
    /// Relative Toleranz pro Komponente.
    pub rtol: T,
    /// Absolute Toleranz pro Komponente.
    pub atol: T,
    /// Anfangsschrittweite; `0` bedeutet: ein Hundertstel des Intervalls.
    pub h_init: T,
    /// Größte erlaubte Schrittweite; `0` bedeutet: unbegrenzt.
    pub h_max: T,
    /// Zahl der Extrapolationsspalten `k` (Ordnung `k`, `2 ≤ k ≤ 8`).
    pub columns: usize,
    /// Höchstzahl der Schrittversuche.
    pub max_steps: usize,
}

impl<T: Real> Default for SeulexOptions<T> {
    /// `rtol = 1e-6`, `atol = 1e-9`, 5 Spalten, 100 000 Schrittversuche.
    fn default() -> Self {
        Self {
            rtol: T::from_f64(1.0e-6),
            atol: T::from_f64(1.0e-9),
            h_init: T::ZERO,
            h_max: T::ZERO,
            columns: 5,
            max_steps: 100_000,
        }
    }
}

impl<T: Real> SeulexOptions<T> {
    /// Setzt `rtol` und `atol`, der Rest wie in `default()`.
    pub fn with_tolerance(rtol: T, atol: T) -> Self {
        Self {
            rtol,
            atol,
            ..Self::default()
        }
    }
}

/// Statistik eines Laufs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeulexResult<T> {
    /// Akzeptierte Schritte.
    pub accepted_steps: usize,
    /// Abgelehnte Schritte.
    pub rejected_steps: usize,
    /// Zuletzt verwendete Schrittweite.
    pub last_step: T,
}

#[allow(clippy::too_many_arguments)]
fn seulex_core<T, F, J>(
    mut f: F,
    mut jac: Jac<J>,
    t0: T,
    t1: T,
    y: &mut [T],
    opts: SeulexOptions<T>,
    mut observer: impl FnMut(T, &[T]),
) -> Result<SeulexResult<T>, OdeError>
where
    T: Real,
    F: FnMut(T, &[T], &mut [T]),
    J: FnMut(T, &[T], &mut [T]),
{
    let n = y.len();
    let k = opts.columns;
    if n == 0 || !(2..=8).contains(&k) {
        return Err(OdeError::InvalidInput);
    }
    if !finite(t0) || !finite(t1) || !all_finite(y) {
        return Err(OdeError::NotFinite);
    }
    observer(t0, y);
    if t0 == t1 {
        return Ok(SeulexResult {
            accepted_steps: 0,
            rejected_steps: 0,
            last_step: T::ZERO,
        });
    }
    let m = n + 1;
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
        span * T::from_f64(0.01)
    }
    .min(h_max);
    let seq: Vec<usize> = (1..=k).collect();
    let sqrt_eps = T::EPSILON.sqrt();
    let exponent = -T::ONE / T::from_usize(k);
    let (mut accepted, mut rejected, mut attempts) = (0usize, 0usize, 0usize);
    let mut last_step = h;
    let mut t = t0;
    let mut f0 = vec![T::ZERO; n];
    let mut f1 = vec![T::ZERO; n];
    let mut jn = vec![T::ZERO; n * n];

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
        // Jacobi-Matrix des erweiterten Systems (y, t) einmal pro Schritt
        f(t, y, &mut f0);
        if !all_finite(&f0) {
            return Err(OdeError::NotFinite);
        }
        match &mut jac {
            Jac::Analytic(jf) => jf(t, y, &mut jn),
            Jac::Numeric => {
                let mut yp = y.to_vec();
                for c in 0..n {
                    let d = sqrt_eps * T::ONE.max(y[c].abs());
                    yp[c] = y[c] + d;
                    f(t, &yp, &mut f1);
                    for r in 0..n {
                        jn[r * n + c] = (f1[r] - f0[r]) / d;
                    }
                    yp[c] = y[c];
                }
            }
        }
        let delta = sqrt_eps * T::ONE.max(t.abs());
        f(t + delta, y, &mut f1);
        let ft: Vec<T> = (0..n).map(|r| (f1[r] - f0[r]) / delta).collect();
        let jhat = |r: usize, c: usize| -> T {
            if r < n && c < n {
                jn[r * n + c]
            } else if r < n && c == n {
                ft[r]
            } else {
                T::ZERO
            }
        };
        // Extrapolationstabelle
        let mut table: Vec<Vec<Vec<T>>> = Vec::with_capacity(k);
        let mut failed = false;
        'columns: for j in 0..k {
            let nj = seq[j];
            let hj = hh / T::from_usize(nj);
            let mut a = DynMatrix::zeros(m, m);
            for r in 0..m {
                for c in 0..m {
                    a[(r, c)] = if r == c { T::ONE } else { T::ZERO } - hj * jhat(r, c);
                }
            }
            let lu = match a.lu() {
                Ok(l) => l,
                Err(_) => {
                    failed = true;
                    break 'columns;
                }
            };
            let mut state: Vec<T> = y.to_vec();
            state.push(t);
            let mut fv = vec![T::ZERO; n];
            for _ in 0..nj {
                f(state[n], &state[..n], &mut fv);
                let mut rhs = DynVector::zeros(m);
                for r in 0..n {
                    rhs[r] = hj * fv[r];
                }
                rhs[n] = hj;
                let d = match lu.solve(&rhs) {
                    Ok(d) => d,
                    Err(_) => {
                        failed = true;
                        break 'columns;
                    }
                };
                for r in 0..m {
                    state[r] += d[r];
                }
            }
            if !all_finite(&state) {
                failed = true;
                break 'columns;
            }
            let mut row: Vec<Vec<T>> = Vec::with_capacity(j + 1);
            row.push(state[..n].to_vec());
            for l in 0..j {
                let ratio = T::from_usize(seq[j]) / T::from_usize(seq[j - 1 - l]);
                let prev = &table[j - 1][l];
                let cur = &row[l];
                let next: Vec<T> = (0..n)
                    .map(|i| cur[i] + (cur[i] - prev[i]) / (ratio - T::ONE))
                    .collect();
                row.push(next);
            }
            table.push(row);
        }
        if failed {
            rejected += 1;
            h = step * T::from_f64(0.25);
            continue;
        }
        let best = &table[k - 1][k - 1];
        let lower = &table[k - 1][k - 2];
        let mut s = T::ZERO;
        for i in 0..n {
            let sc = opts.atol + opts.rtol * best[i].abs().max(y[i].abs());
            let e = (best[i] - lower[i]) / sc;
            s += e * e;
        }
        let err = (s / T::from_usize(n)).sqrt();
        let fac = if err <= T::ZERO {
            T::from_f64(5.0)
        } else if err.is_finite() {
            (T::from_f64(0.9) * err.powf(exponent))
                .max(T::from_f64(0.2))
                .min(T::from_f64(5.0))
        } else {
            T::from_f64(0.2)
        };
        if err <= T::ONE {
            y.copy_from_slice(best);
            t += hh;
            accepted += 1;
            last_step = step;
            observer(t, y);
            h = (step * fac).min(h_max);
        } else {
            rejected += 1;
            h = step * fac.min(T::from_f64(0.9));
        }
    }
    Ok(SeulexResult {
        accepted_steps: accepted,
        rejected_steps: rejected,
        last_step,
    })
}

/// Extrapoliertes linear-implizites Euler-Verfahren mit Schrittweitensteuerung für steife Probleme.
/// `jacobian(t, y, j)` schreibt `∂f/∂y` zeilenweise; ohne Ableitung: [`seulex_fd`]. Rückwärtsintegration möglich.
/// `observer(t, y)` wird am Anfang und nach jedem akzeptierten Schritt aufgerufen.
///
/// Fehler: [`OdeError::InvalidInput`] (auch `columns` außerhalb von `2..=8`), [`OdeError::NotFinite`],
/// [`OdeError::StepSizeTooSmall`], [`OdeError::MaxStepsExceeded`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::ode::{seulex, SeulexOptions};
///
/// let mut y = [1.0f64];
/// seulex(
///     |_t, y: &[f64], dy: &mut [f64]| dy[0] = -y[0],
///     |_t, _y: &[f64], j: &mut [f64]| j[0] = -1.0,
///     0.0, 1.0, &mut y, SeulexOptions::with_tolerance(1e-8, 1e-11), |_, _| {},
/// ).unwrap();
/// assert!((y[0] - (-1.0f64).exp()).abs() < 1e-6);
/// # }
/// ```
pub fn seulex<T: Real>(
    f: impl FnMut(T, &[T], &mut [T]),
    jacobian: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    opts: SeulexOptions<T>,
    observer: impl FnMut(T, &[T]),
) -> Result<SeulexResult<T>, OdeError> {
    seulex_core(f, Jac::Analytic(jacobian), t0, t1, y, opts, observer)
}

/// Wie [`seulex`], bestimmt die Jacobi-Matrix aber selbst per Vorwärtsdifferenzen.
pub fn seulex_fd<T: Real>(
    f: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    opts: SeulexOptions<T>,
    observer: impl FnMut(T, &[T]),
) -> Result<SeulexResult<T>, OdeError> {
    seulex_core(f, Jac::Numeric::<NoJacobian<T>>, t0, t1, y, opts, observer)
}
