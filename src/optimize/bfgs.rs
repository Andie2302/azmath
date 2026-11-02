use super::linesearch::{WolfeOptions, line_search_wolfe};
use super::{OptimizeError, OptimizeOptions, Termination, dot, finite, norm};
use crate::Real;

/// Ergebnis der Minimierungsverfahren (die Lösung steht im übergebenen `x`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinimizeResult<T> {
    /// Anzahl der Iterationen.
    pub iterations: usize,
    /// Funktionswert `f(x)` an der Lösung.
    pub f_value: T,
    /// Größtbetrag der Gradientenkomponenten `‖∇f(x)‖∞` an der Lösung.
    pub grad_norm: T,
    /// Zahl der Auswertungen von `f` (jeweils mit Gradient).
    pub evaluations: usize,
    /// Abbruchgrund.
    pub termination: Termination,
}

/// Arbeitsspeicher von [`bfgs`]: `n² + 6n` Elemente.
pub const fn bfgs_work_len(n: usize) -> usize {
    n * n + 6 * n
}

/// Arbeitsspeicher von [`lbfgs`] mit `m` gespeicherten Paaren: `2·m·n + 2·m + 4·n` Elemente.
pub const fn lbfgs_work_len(n: usize, m: usize) -> usize {
    2 * m * n + 2 * m + 4 * n
}

fn inf_norm<T: Real>(a: &[T]) -> T {
    a.iter().fold(T::ZERO, |m, &v| m.max(v.abs()))
}

/// Prüft Startwert und Funktion, gibt `f(x)` zurück und füllt `g`.
fn start<T: Real>(
    f: &mut impl FnMut(&[T]) -> T,
    grad: &mut impl FnMut(&[T], &mut [T]),
    x: &[T],
    g: &mut [T],
) -> Result<T, OptimizeError> {
    if x.iter().any(|&v| !finite(v)) {
        return Err(OptimizeError::NotFinite);
    }
    let f0 = f(x);
    grad(x, g);
    if !finite(f0) || g.iter().any(|&v| !finite(v)) {
        return Err(OptimizeError::NotFinite);
    }
    Ok(f0)
}

/// Gemeinsame Abbruchprüfung nach einem Schritt. Rückgabe: `Some(grund)` bei Erfolg.
fn converged<T: Real>(
    opts: &OptimizeOptions<T>,
    g: &[T],
    x: &[T],
    step: T,
    f_old: T,
    f_new: T,
) -> Option<Termination> {
    if inf_norm(g) <= opts.tol_g * (T::ONE + f_new.abs()) {
        return Some(Termination::Gradient);
    }
    if step <= opts.tol_x * (norm(x) + opts.tol_x) {
        return Some(Termination::Step);
    }
    if f_old - f_new <= opts.tol_f * T::ONE.max(f_new.abs()) {
        return Some(Termination::Cost);
    }
    None
}

/// BFGS-Verfahren (Quasi-Newton) für `min f(x)`: baut aus Gradientendifferenzen eine Näherung der inversen
/// Hessematrix auf und sucht entlang `d = −H·∇f` mit starker Wolfe-Schrittweite
/// ([`line_search_wolfe`]). Superlinear konvergent, ohne zweite Ableitungen.
///
/// Speicherbedarf `O(n²)`; für große `n` ist [`lbfgs`] besser.
///
/// - `f(x)` liefert den Funktionswert, `grad(x, g)` schreibt den Gradienten.
/// - `x`: Startwert und Ergebnis. `work`: mindestens [`bfgs_work_len`]`(n)` Elemente.
/// - Abbruch (`opts`): `‖∇f‖∞ ≤ tol_g·(1 + |f|)`, kleiner Schritt (`tol_x`) oder keine messbare Abnahme von
///   `f` (`tol_f`, relativ). Sonst [`OptimizeError::NoConvergence`] nach `max_iter` Iterationen.
///
/// Fehler: [`OptimizeError::InvalidInput`], [`OptimizeError::NotFinite`], [`OptimizeError::NoConvergence`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::optimize::{bfgs, bfgs_work_len, OptimizeOptions};
///
/// // Rosenbrock: Minimum bei (1, 1)
/// let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
/// let grad = |x: &[f64], g: &mut [f64]| {
///     g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
///     g[1] = 200.0 * (x[1] - x[0] * x[0]);
/// };
/// let mut x = [-1.2, 1.0];
/// let mut work = [0.0; bfgs_work_len(2)];
/// bfgs(f, grad, &mut x, OptimizeOptions::default(), &mut work).unwrap();
/// assert!((x[0] - 1.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6);
/// # }
/// ```
pub fn bfgs<T: Real>(
    mut f: impl FnMut(&[T]) -> T,
    mut grad: impl FnMut(&[T], &mut [T]),
    x: &mut [T],
    opts: OptimizeOptions<T>,
    work: &mut [T],
) -> Result<MinimizeResult<T>, OptimizeError> {
    let n = x.len();
    if n == 0 || work.len() < bfgs_work_len(n) {
        return Err(OptimizeError::InvalidInput);
    }
    let (h, rest) = work.split_at_mut(n * n);
    let (g, rest) = rest.split_at_mut(n);
    let (g_new, rest) = rest.split_at_mut(n);
    let (d, rest) = rest.split_at_mut(n);
    let (x_new, rest) = rest.split_at_mut(n);
    let (y, rest) = rest.split_at_mut(n);
    let hy = &mut rest[..n];
    let wolfe = WolfeOptions::default();

    let mut fx = start(&mut f, &mut grad, x, g)?;
    let mut evals = 1;
    let reset = |h: &mut [T]| {
        h.fill(T::ZERO);
        for i in 0..n {
            h[i * n + i] = T::ONE;
        }
    };
    reset(h);
    if inf_norm(g) <= opts.tol_g * (T::ONE + fx.abs()) {
        return Ok(MinimizeResult {
            iterations: 0,
            f_value: fx,
            grad_norm: inf_norm(g),
            evaluations: evals,
            termination: Termination::Gradient,
        });
    }
    let mut first = true;
    let mut fresh = true; // H ist gerade die Einheitsmatrix
    for iter in 1..=opts.max_iter {
        for i in 0..n {
            d[i] = -dot(&h[i * n..(i + 1) * n], g);
        }
        if dot(g, d) >= T::ZERO {
            reset(h);
            fresh = true;
            for i in 0..n {
                d[i] = -g[i];
            }
        }
        let alpha0 = if first {
            T::ONE.min(T::ONE / norm(g))
        } else {
            T::ONE
        };
        let mut ls =
            line_search_wolfe(&mut f, &mut grad, x, fx, g, d, alpha0, &wolfe, x_new, g_new);
        if ls.is_err() && !fresh {
            // Matrix verdächtig: auf steepest descent zurücksetzen und noch einmal versuchen
            reset(h);
            fresh = true;
            for i in 0..n {
                d[i] = -g[i];
            }
            ls = line_search_wolfe(&mut f, &mut grad, x, fx, g, d, alpha0, &wolfe, x_new, g_new);
        }
        let ls = match ls {
            Ok(r) => r,
            Err(super::linesearch::LineSearchError::NotFinite) => {
                return Err(OptimizeError::NotFinite);
            }
            Err(_) => return Err(OptimizeError::NoConvergence),
        };
        evals += ls.evaluations;
        // s = x_new − x (in d), y = g_new − g
        let mut step = T::ZERO;
        for i in 0..n {
            d[i] = x_new[i] - x[i];
            y[i] = g_new[i] - g[i];
            step = step.hypot(d[i]);
        }
        let sy = dot(d, y);
        let f_old = fx;
        fx = ls.f_value;
        x.copy_from_slice(x_new);
        g.copy_from_slice(g_new);
        if sy > T::EPSILON * step * norm(y) {
            if first {
                // Anfangsskalierung H₀ = (sᵀy / yᵀy)·I
                let scale = sy / dot(y, y);
                for v in h.iter_mut() {
                    *v *= scale;
                }
            }
            for i in 0..n {
                hy[i] = dot(&h[i * n..(i + 1) * n], y);
            }
            let c = (sy + dot(y, hy)) / (sy * sy);
            for i in 0..n {
                for j in 0..n {
                    h[i * n + j] += c * d[i] * d[j] - (hy[i] * d[j] + d[i] * hy[j]) / sy;
                }
            }
            fresh = false;
        }
        first = false;
        if let Some(termination) = converged(&opts, g, x, step, f_old, fx) {
            return Ok(MinimizeResult {
                iterations: iter,
                f_value: fx,
                grad_norm: inf_norm(g),
                evaluations: evals,
                termination,
            });
        }
    }
    Err(OptimizeError::NoConvergence)
}

/// L-BFGS-Verfahren (Quasi-Newton mit begrenztem Speicher) für `min f(x)`: speichert nur die letzten `m`
/// Paare `(s, y)` und berechnet `H·∇f` mit der Zweischleifen-Rekursion. Speicher `O(m·n)` statt `O(n²)`.
///
/// Parameter wie bei [`bfgs`]; `m` (typisch 5 bis 20) ist die Zahl der gespeicherten Paare,
/// `work` mindestens [`lbfgs_work_len`]`(n, m)` Elemente.
///
/// Fehler: [`OptimizeError::InvalidInput`] (auch bei `m = 0`), [`OptimizeError::NotFinite`],
/// [`OptimizeError::NoConvergence`].
pub fn lbfgs<T: Real>(
    mut f: impl FnMut(&[T]) -> T,
    mut grad: impl FnMut(&[T], &mut [T]),
    x: &mut [T],
    m: usize,
    opts: OptimizeOptions<T>,
    work: &mut [T],
) -> Result<MinimizeResult<T>, OptimizeError> {
    let n = x.len();
    if n == 0 || m == 0 || work.len() < lbfgs_work_len(n, m) {
        return Err(OptimizeError::InvalidInput);
    }
    let (s_all, rest) = work.split_at_mut(m * n);
    let (y_all, rest) = rest.split_at_mut(m * n);
    let (rho, rest) = rest.split_at_mut(m);
    let (alpha, rest) = rest.split_at_mut(m);
    let (g, rest) = rest.split_at_mut(n);
    let (g_new, rest) = rest.split_at_mut(n);
    let (d, rest) = rest.split_at_mut(n);
    let x_new = &mut rest[..n];
    let wolfe = WolfeOptions::default();

    let mut fx = start(&mut f, &mut grad, x, g)?;
    let mut evals = 1;
    if inf_norm(g) <= opts.tol_g * (T::ONE + fx.abs()) {
        return Ok(MinimizeResult {
            iterations: 0,
            f_value: fx,
            grad_norm: inf_norm(g),
            evaluations: evals,
            termination: Termination::Gradient,
        });
    }
    let (mut count, mut head) = (0usize, 0usize); // head: Platz für das nächste Paar
    for iter in 1..=opts.max_iter {
        // Zweischleifen-Rekursion: d = −H·g
        d.copy_from_slice(g);
        for k in 0..count {
            let i = (head + m - 1 - k) % m; // neuestes zuerst
            let a = rho[i] * dot(&s_all[i * n..(i + 1) * n], d);
            alpha[i] = a;
            for j in 0..n {
                d[j] -= a * y_all[i * n + j];
            }
        }
        if count > 0 {
            let newest = (head + m - 1) % m;
            let gamma = T::ONE
                / (rho[newest]
                    * dot(
                        &y_all[newest * n..(newest + 1) * n],
                        &y_all[newest * n..(newest + 1) * n],
                    ));
            for v in d.iter_mut() {
                *v *= gamma;
            }
        }
        for k in (0..count).rev() {
            let i = (head + m - 1 - k) % m;
            let b = rho[i] * dot(&y_all[i * n..(i + 1) * n], d);
            let c = alpha[i] - b;
            for j in 0..n {
                d[j] += c * s_all[i * n + j];
            }
        }
        for v in d.iter_mut() {
            *v = -*v;
        }
        if dot(g, d) >= T::ZERO {
            count = 0; // Gedächtnis verwerfen, steepest descent
            for j in 0..n {
                d[j] = -g[j];
            }
        }
        let alpha0 = if count == 0 {
            T::ONE.min(T::ONE / norm(g))
        } else {
            T::ONE
        };
        let mut ls =
            line_search_wolfe(&mut f, &mut grad, x, fx, g, d, alpha0, &wolfe, x_new, g_new);
        if ls.is_err() && count > 0 {
            count = 0;
            for j in 0..n {
                d[j] = -g[j];
            }
            ls = line_search_wolfe(
                &mut f,
                &mut grad,
                x,
                fx,
                g,
                d,
                T::ONE.min(T::ONE / norm(g)),
                &wolfe,
                x_new,
                g_new,
            );
        }
        let ls = match ls {
            Ok(r) => r,
            Err(super::linesearch::LineSearchError::NotFinite) => {
                return Err(OptimizeError::NotFinite);
            }
            Err(_) => return Err(OptimizeError::NoConvergence),
        };
        evals += ls.evaluations;
        // neues Paar in den Ring schreiben
        let mut step = T::ZERO;
        let (mut sy, mut sn, mut yn) = (T::ZERO, T::ZERO, T::ZERO);
        for j in 0..n {
            let s = x_new[j] - x[j];
            let y = g_new[j] - g[j];
            s_all[head * n + j] = s;
            y_all[head * n + j] = y;
            sy += s * y;
            sn += s * s;
            yn += y * y;
        }
        step = step.max(sn.sqrt());
        let f_old = fx;
        fx = ls.f_value;
        x.copy_from_slice(x_new);
        g.copy_from_slice(g_new);
        if sy > T::EPSILON * sn.sqrt() * yn.sqrt() {
            rho[head] = T::ONE / sy;
            head = (head + 1) % m;
            count = (count + 1).min(m);
        }
        if let Some(termination) = converged(&opts, g, x, step, f_old, fx) {
            return Ok(MinimizeResult {
                iterations: iter,
                f_value: fx,
                grad_norm: inf_norm(g),
                evaluations: evals,
                termination,
            });
        }
    }
    Err(OptimizeError::NoConvergence)
}
