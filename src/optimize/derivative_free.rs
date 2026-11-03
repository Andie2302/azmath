#![allow(clippy::needless_range_loop)]

use super::{OptimizeError, Termination, finite};
use crate::Real;

/// Einstellungen für die eindimensionalen Minimierer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinOptions<T> {
    /// Relative Toleranz der Lage des Minimums. Nicht kleiner als etwa `√ε` wählen: nahe dem Minimum ist `f` flach und die Lage
    /// nur auf diese Genauigkeit bestimmbar.
    pub tol: T,
    /// Höchstzahl der Iterationen.
    pub max_iter: usize,
}

impl<T: Real> Default for MinOptions<T> {
    /// `tol = √ε`, höchstens 500 Iterationen.
    fn default() -> Self {
        Self {
            tol: T::EPSILON.sqrt(),
            max_iter: 500,
        }
    }
}

/// Ergebnis der eindimensionalen Minimierer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Min1dResult<T> {
    /// Lage des Minimums.
    pub x: T,
    /// Funktionswert dort.
    pub f_value: T,
    /// Anzahl der Iterationen.
    pub iterations: usize,
}

/// Minimum einer **unimodalen** Funktion auf `[a, b]` nach dem **Verfahren des Goldenen Schnitts**: ableitungsfrei, sicher,
/// lineare Konvergenz (Intervall × 0,618 je Auswertung). Bei mehreren lokalen Minima wird irgendeines gefunden.
///
/// Fehler: [`OptimizeError::InvalidInput`] (`a == b`), [`OptimizeError::NotFinite`], [`OptimizeError::NoConvergence`].
///
/// ```
/// use azmath::optimize::{golden_section, MinOptions};
///
/// let r = golden_section(|x: f64| (x - 2.0).powi(2) + 1.0, 0.0, 5.0, MinOptions::default()).unwrap();
/// assert!((r.x - 2.0).abs() < 1e-6 && (r.f_value - 1.0).abs() < 1e-10);
/// ```
pub fn golden_section<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: MinOptions<T>,
) -> Result<Min1dResult<T>, OptimizeError> {
    if !finite(a) || !finite(b) {
        return Err(OptimizeError::NotFinite);
    }
    if a == b {
        return Err(OptimizeError::InvalidInput);
    }
    let gr = T::from_f64(0.618_033_988_749_894_9);
    let (mut lo, mut hi) = if a < b { (a, b) } else { (b, a) };
    let mut c = hi - gr * (hi - lo);
    let mut d = lo + gr * (hi - lo);
    let (mut fc, mut fd) = (f(c), f(d));
    if !finite(fc) || !finite(fd) {
        return Err(OptimizeError::NotFinite);
    }
    for iter in 1..=opts.max_iter {
        if (hi - lo).abs() <= opts.tol * T::ONE.max(c.abs() + d.abs()) {
            let (x, fx) = if fc < fd { (c, fc) } else { (d, fd) };
            return Ok(Min1dResult {
                x,
                f_value: fx,
                iterations: iter,
            });
        }
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - gr * (hi - lo);
            fc = f(c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + gr * (hi - lo);
            fd = f(d);
        }
        if !finite(fc) || !finite(fd) {
            return Err(OptimizeError::NotFinite);
        }
    }
    Err(OptimizeError::NoConvergence)
}

/// Minimum auf `[a, b]` nach **Brent**: kombiniert parabolische Interpolation (schnell bei glatten Funktionen, superlineare
/// Konvergenz) mit dem Goldenen Schnitt als Absicherung. Die erste Wahl für eindimensionale ableitungsfreie Minimierung.
///
/// Fehler wie bei [`golden_section`].
///
/// ```
/// use azmath::optimize::{brent_min, MinOptions};
///
/// let r = brent_min(|x: f64| x.cos(), 2.0, 4.0, MinOptions::default()).unwrap();
/// assert!((r.x - core::f64::consts::PI).abs() < 1e-7);
/// ```
pub fn brent_min<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: MinOptions<T>,
) -> Result<Min1dResult<T>, OptimizeError> {
    if !finite(a) || !finite(b) {
        return Err(OptimizeError::NotFinite);
    }
    if a == b {
        return Err(OptimizeError::InvalidInput);
    }
    let cgold = T::from_f64(0.381_966_011_250_105_1);
    let zeps = T::EPSILON * T::from_f64(1e-3);
    let half = T::from_f64(0.5);
    let sign = |m: T, s: T| if s >= T::ZERO { m.abs() } else { -m.abs() };
    let (mut lo, mut hi) = if a < b { (a, b) } else { (b, a) };
    let mut x = lo + cgold * (hi - lo);
    let (mut w, mut v) = (x, x);
    let mut fx = f(x);
    if !finite(fx) {
        return Err(OptimizeError::NotFinite);
    }
    let (mut fw, mut fv) = (fx, fx);
    let (mut d, mut e) = (T::ZERO, T::ZERO);
    for iter in 1..=opts.max_iter {
        let xm = half * (lo + hi);
        let tol1 = opts.tol * x.abs() + zeps;
        let tol2 = T::from_f64(2.0) * tol1;
        if (x - xm).abs() <= tol2 - half * (hi - lo) {
            return Ok(Min1dResult {
                x,
                f_value: fx,
                iterations: iter,
            });
        }
        let mut golden = true;
        if e.abs() > tol1 {
            // parabolischer Schritt
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = T::from_f64(2.0) * (q - r);
            if q > T::ZERO {
                p = -p;
            }
            q = q.abs();
            let etemp = e;
            e = d;
            let reject =
                p.abs() >= (half * q * etemp).abs() || p <= q * (lo - x) || p >= q * (hi - x);
            if !reject {
                d = p / q;
                let u = x + d;
                if u - lo < tol2 || hi - u < tol2 {
                    d = sign(tol1, xm - x);
                }
                golden = false;
            }
        }
        if golden {
            e = if x >= xm { lo - x } else { hi - x };
            d = cgold * e;
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else {
            x + sign(tol1, d)
        };
        let fu = f(u);
        if !finite(fu) {
            return Err(OptimizeError::NotFinite);
        }
        if fu <= fx {
            if u >= x {
                lo = x;
            } else {
                hi = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        } else {
            if u < x {
                lo = u;
            } else {
                hi = u;
            }
            if fu <= fw || w == x {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    Err(OptimizeError::NoConvergence)
}

/// Einstellungen für [`nelder_mead`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NelderMeadOptions<T> {
    /// Abbruch, wenn die Kantenlänge des Simplex `≤ tol_x·(1 + ‖x‖∞)` ist …
    pub tol_x: T,
    /// … und die Funktionswerte an den Ecken sich um höchstens `tol_f·(1 + |f|)` unterscheiden.
    pub tol_f: T,
    /// Höchstzahl der Iterationen.
    pub max_iter: usize,
    /// Größe des Anfangssimplex relativ zu `max(1, |xᵢ|)`.
    pub initial_step: T,
}

impl<T: Real> Default for NelderMeadOptions<T> {
    /// `tol_x = tol_f = √ε`, 5000 Iterationen, Anfangsschritt 0,1.
    fn default() -> Self {
        let t = T::EPSILON.sqrt();
        Self {
            tol_x: t,
            tol_f: t,
            max_iter: 5000,
            initial_step: T::from_f64(0.1),
        }
    }
}

/// Ergebnis von [`nelder_mead`] (die Lösung steht im übergebenen `x`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NelderMeadResult<T> {
    /// Anzahl der Iterationen.
    pub iterations: usize,
    /// Zahl der Funktionsauswertungen.
    pub evaluations: usize,
    /// Funktionswert an der Lösung.
    pub f_value: T,
    /// Abbruchgrund ([`Termination::Step`]: Simplex klein geworden).
    pub termination: Termination,
}

/// Benötigter Arbeitsspeicher für [`nelder_mead`] bei `n` Variablen: `(n+1)·n + (n+1) + 3·n`.
pub const fn nelder_mead_work_len(n: usize) -> usize {
    (n + 1) * n + (n + 1) + 3 * n
}

/// **Nelder-Mead-Simplexverfahren** zur ableitungsfreien Minimierung von `f: ℝⁿ → ℝ` (Reflexion, Expansion, Kontraktion,
/// Schrumpfung mit den Standardkoeffizienten 1, 2, ½, ½). Robust gegen verrauschte oder nicht glatte Zielfunktionen, aber
/// langsam (lineare Konvergenz) und nur für kleine `n` (bis etwa 10) sinnvoll; es gibt keine Konvergenzgarantie. Mit
/// Ableitungen sind [`bfgs`](super::bfgs) und [`lbfgs`](super::lbfgs) deutlich schneller.
///
/// - `x`: Startwert und Ergebnis. `work`: mindestens [`nelder_mead_work_len`]`(n)` Elemente.
///
/// Fehler: [`OptimizeError::InvalidInput`], [`OptimizeError::NotFinite`], [`OptimizeError::NoConvergence`] (`x` enthält dann die
/// beste Ecke).
///
/// ```
/// use azmath::optimize::{nelder_mead, nelder_mead_work_len, NelderMeadOptions};
///
/// // Rosenbrock ohne Gradient
/// let f = |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2);
/// let mut x = [-1.2, 1.0];
/// let mut work = [0.0; nelder_mead_work_len(2)];
/// nelder_mead(f, &mut x, NelderMeadOptions::default(), &mut work).unwrap();
/// assert!((x[0] - 1.0).abs() < 1e-4 && (x[1] - 1.0).abs() < 1e-4);
/// ```
pub fn nelder_mead<T: Real>(
    mut f: impl FnMut(&[T]) -> T,
    x: &mut [T],
    opts: NelderMeadOptions<T>,
    work: &mut [T],
) -> Result<NelderMeadResult<T>, OptimizeError> {
    let n = x.len();
    if n == 0 || work.len() < nelder_mead_work_len(n) {
        return Err(OptimizeError::InvalidInput);
    }
    if x.iter().any(|&v| !finite(v)) {
        return Err(OptimizeError::NotFinite);
    }
    let (simplex, rest) = work.split_at_mut((n + 1) * n);
    let (fv, rest) = rest.split_at_mut(n + 1);
    let (centroid, rest) = rest.split_at_mut(n);
    let (xr, rest) = rest.split_at_mut(n);
    let xt = &mut rest[..n];

    // Anfangssimplex: x₀ und x₀ + hᵢ·eᵢ
    simplex[..n].copy_from_slice(x);
    for i in 0..n {
        let row = (i + 1) * n;
        simplex[row..row + n].copy_from_slice(x);
        simplex[row + i] += opts.initial_step * T::ONE.max(x[i].abs());
    }
    let mut evals = 0usize;
    for i in 0..=n {
        fv[i] = f(&simplex[i * n..(i + 1) * n]);
        evals += 1;
        if !finite(fv[i]) {
            return Err(OptimizeError::NotFinite);
        }
    }
    let half = T::from_f64(0.5);
    let two = T::from_f64(2.0);

    for iter in 0..opts.max_iter {
        // Ecken aufsteigend nach Funktionswert sortieren (Einfügesortierung, tauscht ganze Zeilen)
        for i in 1..=n {
            let mut j = i;
            while j > 0 && fv[j] < fv[j - 1] {
                fv.swap(j, j - 1);
                for k in 0..n {
                    simplex.swap(j * n + k, (j - 1) * n + k);
                }
                j -= 1;
            }
        }
        // Abbruchprüfung
        let mut size = T::ZERO;
        for i in 1..=n {
            for k in 0..n {
                size = size.max((simplex[i * n + k] - simplex[k]).abs());
            }
        }
        let xnorm = simplex[..n].iter().fold(T::ZERO, |m, v| m.max(v.abs()));
        if size <= opts.tol_x * (T::ONE + xnorm)
            && (fv[n] - fv[0]) <= opts.tol_f * (T::ONE + fv[0].abs())
        {
            x.copy_from_slice(&simplex[..n]);
            return Ok(NelderMeadResult {
                iterations: iter,
                evaluations: evals,
                f_value: fv[0],
                termination: Termination::Step,
            });
        }
        // Schwerpunkt der besten n Ecken
        for k in 0..n {
            let mut s = T::ZERO;
            for i in 0..n {
                s += simplex[i * n + k];
            }
            centroid[k] = s / T::from_usize(n);
        }
        let worst = n * n;
        for k in 0..n {
            xr[k] = centroid[k] + (centroid[k] - simplex[worst + k]);
        }
        let fr = f(xr);
        evals += 1;
        if !finite(fr) {
            return Err(OptimizeError::NotFinite);
        }
        let replace = |simplex: &mut [T], fv: &mut [T], point: &[T], value: T| {
            simplex[worst..worst + n].copy_from_slice(point);
            fv[n] = value;
        };
        if fv[0] <= fr && fr < fv[n - 1] {
            replace(simplex, fv, xr, fr);
        } else if fr < fv[0] {
            for k in 0..n {
                xt[k] = centroid[k] + two * (xr[k] - centroid[k]);
            }
            let fe = f(xt);
            evals += 1;
            if fe < fr {
                replace(simplex, fv, xt, fe);
            } else {
                replace(simplex, fv, xr, fr);
            }
        } else {
            // Kontraktion
            let outside = fr < fv[n];
            for k in 0..n {
                let target = if outside { xr[k] } else { simplex[worst + k] };
                xt[k] = centroid[k] + half * (target - centroid[k]);
            }
            let fc = f(xt);
            evals += 1;
            let accept = if outside { fc <= fr } else { fc < fv[n] };
            if accept {
                replace(simplex, fv, xt, fc);
            } else {
                // Schrumpfung zur besten Ecke
                for i in 1..=n {
                    for k in 0..n {
                        let best = simplex[k];
                        simplex[i * n + k] = best + half * (simplex[i * n + k] - best);
                    }
                    fv[i] = f(&simplex[i * n..(i + 1) * n]);
                    evals += 1;
                }
            }
        }
        if fv.iter().any(|v| !finite(*v)) {
            return Err(OptimizeError::NotFinite);
        }
    }
    // beste Ecke zurückgeben
    let mut best = 0;
    for i in 1..=n {
        if fv[i] < fv[best] {
            best = i;
        }
    }
    x.copy_from_slice(&simplex[best * n..(best + 1) * n]);
    Err(OptimizeError::NoConvergence)
}
