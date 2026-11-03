#![allow(clippy::needless_range_loop)]

//! Das Verfahren von Jenkins und Traub (CPOLY) für die Nullstellen eines Polynoms.

use super::{PolyRootsResult, RootError, RootOptions};
use crate::Real;
use crate::complex::Complex;

type C<T> = Complex<T>;

/// Benötigter Arbeitsspeicher (in `Complex`-Elementen) für [`jenkins_traub`] und [`jenkins_traub_complex`] bei Grad `degree`.
pub const fn jenkins_traub_work_len(degree: usize) -> usize {
    3 * (degree + 1)
}

fn is_zero<T: Real>(z: C<T>) -> bool {
    z.re == T::ZERO && z.im == T::ZERO
}

/// Horner-Auswertung; `p` mit absteigenden Potenzen (`p[0]` gehört zu `z^m`).
fn horner<T: Real>(p: &[C<T>], z: C<T>) -> C<T> {
    let mut acc = C::ZERO;
    for &c in p {
        acc = acc * z + c;
    }
    acc
}

/// Untere Schranke `β` für den kleinsten Betrag einer Nullstelle (Cauchy-Polynom, grob per Newton).
fn lower_bound<T: Real>(p: &[C<T>]) -> T {
    let m = p.len() - 1;
    let a0 = p[m].abs();
    let m_t = T::from_f64(m as f64);
    // Startwert oberhalb der positiven Nullstelle von Σ_{k≥1} |a_k| x^k − |a_0|
    let mut x = T::ZERO;
    for k in 1..=m {
        let ak = p[m - k].abs();
        if ak > T::ZERO {
            let cand = (m_t * a0 / ak).powf(T::ONE / T::from_f64(k as f64));
            x = x.max(cand);
        }
    }
    if x <= T::ZERO || x.is_infinite() || x.is_nan() {
        return T::ONE;
    }
    for _ in 0..100 {
        let (mut f, mut df) = (-a0, T::ZERO);
        for k in 1..=m {
            let ak = p[m - k].abs();
            let xk1 = x.powi(k as i32 - 1);
            f += ak * xk1 * x;
            df += T::from_f64(k as f64) * ak * xk1;
        }
        if df == T::ZERO {
            break;
        }
        let nx = x - f / df;
        let done = (x - nx).abs() <= T::from_f64(0.005) * nx.abs();
        x = nx;
        if done || x <= T::ZERO {
            break;
        }
    }
    if x > T::ZERO && !x.is_infinite() {
        x
    } else {
        T::ONE
    }
}

/// Ein Schritt der H-Polynom-Folge. Mit `shift = Some(s)`: `H ← (H − (H(s)/P(s))·P)/(z − s)`; ohne Shift:
/// `H ← (H − (H(0)/P(0))·P)/z`. `h` hat `m` Koeffizienten (Grad `m − 1`), `p` hat `m + 1`; `n` ist Zwischenspeicher
/// (`m + 1`). Rückgabe `false`, wenn `P(s)` bzw. `P(0)` verschwindet (dann ist `s` bzw. 0 selbst eine Nullstelle).
fn h_step<T: Real>(p: &[C<T>], h: &mut [C<T>], n: &mut [C<T>], shift: Option<C<T>>) -> bool {
    let m = p.len() - 1;
    let (ps, hs) = match shift {
        Some(s) => (horner(p, s), horner(&h[..m], s)),
        None => (p[m], h[m - 1]),
    };
    if is_zero(ps) {
        return false;
    }
    let t = hs / ps;
    n[0] = -(t * p[0]);
    for k in 1..=m {
        n[k] = h[k - 1] - t * p[k];
    }
    match shift {
        Some(s) => {
            // Division durch (z − s) per Horner-Schema
            h[0] = n[0];
            for k in 1..m {
                h[k] = n[k] + s * h[k - 1];
            }
        }
        None => {
            // Division durch z: letzte Komponente (Konstante, = 0) weglassen
            h[..m].copy_from_slice(&n[..m]);
        }
    }
    true
}

/// Sucht eine Nullstelle des Polynoms `p` (Grad `m ≥ 2`, absteigend); `h` und `n` sind Arbeitsspeicher.
fn find_root<T: Real>(
    p: &[C<T>],
    h: &mut [C<T>],
    n: &mut [C<T>],
    opts: &RootOptions<T>,
    iterations: &mut usize,
) -> Result<C<T>, RootError> {
    let m = p.len() - 1;
    let beta = lower_bound(p);
    let deg = |k: usize| T::from_f64((m - k) as f64);
    let magnitude = |s: C<T>| -> T {
        // Σ |p_k|·|s|^(m−k): Maß für die Rundungsfehler der Auswertung
        let r = s.abs();
        p.iter().fold(T::ZERO, |acc, c| acc * r + c.abs())
    };
    let deg_pi = T::PI / T::from_f64(180.0);
    // Annahmeschwelle: Auswertungsfehler des Horner-Schemas ist etwa 2·m·ε·Σ|pₖ||s|ᵏ
    let accept = opts.tol.max(T::from_f64(4.0 * m as f64) * T::EPSILON);
    for attempt in 0..6 {
        // H₀ = P′
        for k in 0..m {
            h[k] = p[k].scale(deg(k));
        }
        // Stufe 1 (ohne Shift)
        for _ in 0..5 {
            if !h_step(p, h, n, None) {
                return Ok(C::ZERO);
            }
        }
        // Stufe 2 (fester Shift auf dem Kreis mit Radius β, Winkel um 94° gedreht je Versuch)
        let angle = (T::from_f64(49.0) + T::from_f64(94.0) * T::from_f64(attempt as f64)) * deg_pi;
        let mut s = C::from_polar(beta, angle);
        for _ in 0..10 {
            *iterations += 1;
            if !h_step(p, h, n, Some(s)) {
                return Ok(s);
            }
        }
        // Stufe 3 (variabler Shift)
        let step = |h: &[C<T>], s: C<T>| -> Option<C<T>> {
            let hs = horner(&h[..m], s) / h[0];
            if is_zero(hs) {
                return None;
            }
            Some(s - horner(p, s) / hs)
        };
        let Some(first) = step(h, s) else { continue };
        s = first;
        for _ in 0..opts.max_iter.min(60) {
            *iterations += 1;
            let ps = horner(p, s);
            if !ps.is_finite() {
                break;
            }
            if ps.abs() <= accept * magnitude(s) {
                return Ok(s);
            }
            if !h_step(p, h, n, Some(s)) {
                return Ok(s);
            }
            let Some(next) = step(h, s) else { break };
            s = next;
            if !s.is_finite() {
                break;
            }
        }
    }
    // Rückfall: Laguerre-Iteration von 0 aus (sehr robust, wenn die Shift-Folge nicht konvergiert)
    laguerre(p, iterations)
}

/// Laguerre-Verfahren für eine Nullstelle von `p` (absteigende Koeffizienten, Grad `m ≥ 2`), Start bei 0.
fn laguerre<T: Real>(p: &[C<T>], iterations: &mut usize) -> Result<C<T>, RootError> {
    let m = p.len() - 1;
    let m_t = T::from_f64(m as f64);
    let frac = [0.0, 0.5, 0.25, 0.75, 0.13, 0.38, 0.62, 0.88, 1.0];
    let mut x = C::ZERO;
    for iter in 1..=80usize {
        *iterations += 1;
        let mut b = p[0];
        let mut err = b.abs();
        let (mut d, mut f) = (C::ZERO, C::ZERO);
        let abx = x.abs();
        for j in 1..=m {
            f = x * f + d;
            d = x * d + b;
            b = x * b + p[j];
            err = b.abs() + abx * err;
        }
        err *= T::EPSILON;
        if b.abs() <= err {
            return Ok(x);
        }
        let g = d / b;
        let g2 = g * g;
        let h = g2 - (f / b).scale(T::from_f64(2.0));
        let sq = (((h.scale(m_t)) - g2).scale(m_t - T::ONE)).sqrt();
        let gp = g + sq;
        let gm = g - sq;
        let denom = if gp.abs() < gm.abs() { gm } else { gp };
        let dx = if denom.abs() > T::ZERO {
            C::from_real(m_t) / denom
        } else {
            C::from_polar(T::ONE + abx, T::from_f64(iter as f64))
        };
        let x1 = x - dx;
        if !x1.is_finite() {
            return Err(RootError::NoConvergence);
        }
        if x1.re == x.re && x1.im == x.im {
            return Ok(x);
        }
        x = if iter % 10 != 0 {
            x1
        } else {
            x - dx.scale(T::from_f64(frac[(iter / 10).min(8)]))
        };
    }
    Err(RootError::NoConvergence)
}

fn solve<T: Real>(
    coef: impl Fn(usize) -> C<T>,
    len: usize,
    roots: &mut [C<T>],
    work: &mut [C<T>],
    opts: RootOptions<T>,
) -> Result<PolyRootsResult<T>, RootError> {
    if (0..len).any(|k| !coef(k).is_finite()) {
        return Err(RootError::NotFinite);
    }
    let Some(hi) = (0..len).rev().find(|&k| !is_zero(coef(k))) else {
        return Err(RootError::InvalidInput);
    };
    let lo = (0..len).find(|&k| !is_zero(coef(k))).unwrap_or(0);
    let d = hi - lo;
    if roots.len() < hi || work.len() < jenkins_traub_work_len(d) {
        return Err(RootError::InvalidInput);
    }
    for r in roots[..lo].iter_mut() {
        *r = C::ZERO;
    }
    let mut result = PolyRootsResult {
        degree: hi,
        iterations: 0,
        last_correction: T::ZERO,
    };
    if d == 0 {
        return Ok(result);
    }
    let (p_all, rest) = work.split_at_mut(d + 1);
    let (h_all, n_all) = rest.split_at_mut(d + 1);
    // absteigende Koeffizienten des Polynoms ohne Nullstellen bei 0
    for k in 0..=d {
        p_all[k] = coef(hi - k);
    }
    let mut m = d;
    while m > 1 {
        let root = find_root(
            &p_all[..=m],
            &mut h_all[..m],
            &mut n_all[..=m],
            &opts,
            &mut result.iterations,
        )?;
        roots[lo + d - m] = root;
        // Deflation: P ← P / (z − root)
        for k in 1..m {
            let prev = p_all[k - 1];
            p_all[k] += root * prev;
        }
        m -= 1;
    }
    roots[lo + d - 1] = -(p_all[1] / p_all[0]);
    // Nachiteration mit dem Newton-Verfahren am ursprünglichen Polynom (gegen Fehler der Deflation)
    for r in roots[lo..hi].iter_mut() {
        for _ in 0..2 {
            let (mut pz, mut dp) = (coef(hi), C::ZERO);
            for k in (lo..hi).rev() {
                dp = dp * *r + pz;
                pz = pz * *r + coef(k);
            }
            if is_zero(dp) || !pz.is_finite() {
                break;
            }
            let delta = pz / dp;
            if !delta.is_finite() || delta.abs() > T::from_f64(0.5) * r.abs().max(T::MIN_POSITIVE) {
                break; // mehrfache Nullstelle oder schlechte Näherung: nicht nachpolieren
            }
            *r -= delta;
            result.last_correction = result
                .last_correction
                .max(delta.abs() / r.abs().max(T::MIN_POSITIVE));
        }
    }
    Ok(result)
}

/// Alle Nullstellen eines Polynoms mit **reellen** Koeffizienten nach **Jenkins und Traub** (CPOLY, dreistufig).
///
/// Das Verfahren erzeugt aus `P′` eine Folge von Polynomen `H`, die sich mit einem festen und dann einem variablen Shift
/// zu einem Linearfaktor des Polynoms verdichten; die Nullstellen kommen etwa nach wachsendem Betrag, was die Deflation
/// stabil hält. Zum Schluss poliert je zwei Newton-Schritte am ursprünglichen Polynom. Es braucht keine Startwerte und
/// ist die klassische Wahl für Polynome höheren Grades.
///
/// - `coeffs`: aufsteigend (`c[0] + c[1]·x + …`); führende Nullkoeffizienten werden ignoriert, Nullstellen bei 0 exakt als `0`.
/// - `roots`: Ausgabe (mindestens Grad), `work`: mindestens [`jenkins_traub_work_len`]`(Grad)` Elemente.
/// - `opts.tol` steuert die Annahme einer Nullstelle (Auswertungsfehler, relative Korrektur); `max_iter` begrenzt die
///   Schritte der letzten Stufe je Versuch.
///
/// Dies ist eine vereinfachte Fassung des Originalalgorithmus (kein Zertifikat für die Konvergenz der Stufe 2, Versuche
/// mit gedrehtem Shift bis zu 6 Mal, danach ein Rückfall auf das Laguerre-Verfahren); bei Misserfolg [`RootError::NoConvergence`].
///
/// Fehler: [`RootError::InvalidInput`], [`RootError::NotFinite`], [`RootError::NoConvergence`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::complex::Complex;
/// use azmath::roots::{jenkins_traub, jenkins_traub_work_len, RootOptions};
///
/// // x² + 1 = 0
/// let mut z = [Complex::new(0.0f64, 0.0); 2];
/// let mut work = [Complex::new(0.0, 0.0); jenkins_traub_work_len(2)];
/// jenkins_traub(&[1.0, 0.0, 1.0], &mut z, &mut work, RootOptions::default()).unwrap();
/// assert!(z.iter().all(|r| r.re.abs() < 1e-12 && (r.im.abs() - 1.0).abs() < 1e-12));
/// # }
/// ```
pub fn jenkins_traub<T: Real>(
    coeffs: &[T],
    roots: &mut [C<T>],
    work: &mut [C<T>],
    opts: RootOptions<T>,
) -> Result<PolyRootsResult<T>, RootError> {
    solve(|k| C::from_real(coeffs[k]), coeffs.len(), roots, work, opts)
}

/// Wie [`jenkins_traub`] für **komplexe** Koeffizienten.
pub fn jenkins_traub_complex<T: Real>(
    coeffs: &[C<T>],
    roots: &mut [C<T>],
    work: &mut [C<T>],
    opts: RootOptions<T>,
) -> Result<PolyRootsResult<T>, RootError> {
    solve(|k| coeffs[k], coeffs.len(), roots, work, opts)
}
