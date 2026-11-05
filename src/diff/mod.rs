//! Numerisches Differenzieren: Differenzenquotienten, Richardson-Extrapolation (Ridders), Gradient, Hessematrix, Jacobi-Matrix.
//!
//! Differenzenquotienten verlieren Genauigkeit durch Auslöschung: Der Fehler eines zentralen Quotienten ist etwa
//! `h²·f‴/6 + ε·|f|/h`, am kleinsten bei `h ≈ ε^(1/3)` (bei `f64` etwa `6·10⁻⁶`), mit einem Fehler von ungefähr `ε^(2/3) ≈ 4·10⁻¹¹`.
//! [`derivative_ridders`] extrapoliert mehrere Schrittweiten und liefert zusätzlich eine Fehlerschätzung, meist mit deutlich
//! mehr gültigen Stellen. Wo es möglich ist, sind **exakte Ableitungen** durch [`autodiff`](crate::autodiff) vorzuziehen.
//!
//! Ungültige bzw. in der Gleitkommadarstellung zusammenfallende Stützpunkte ergeben `NaN`,
//! ohne dort die Benutzerfunktion auszuwerten. Nichtendliche Funktionswerte ergeben ebenfalls
//! `NaN`. Ridders meldet in solchen Fällen `(NaN, ∞)`; fehlen nur weitere darstellbare
//! Verfeinerungen, bleibt die beste bisherige Schätzung erhalten (ohne belastbare Fehlerschätzung ggf. `∞`).
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::diff::{central_difference, derivative_ridders};
//!
//! let f = |x: f64| x.sin();
//! let d1 = central_difference(f, 1.0, 1e-5);
//! let (d2, err) = derivative_ridders(f, 1.0, 0.1);
//! assert!((d1 - 1.0f64.cos()).abs() < 1e-9);
//! assert!((d2 - 1.0f64.cos()).abs() < 1e-12 && err < 1e-10);
//! # }
//! ```

use crate::Real;

/// Endliche, getrennte Stützpunkte sind Voraussetzung für einen Differenzenquotienten.
fn nodes<T: Real>(x: T, h: T) -> Option<(T, T)> {
    if !x.is_finite() || !h.is_finite() || h == T::ZERO {
        return None;
    }
    let (plus, minus) = (x + h, x - h);
    if !plus.is_finite() || !minus.is_finite() || plus == x || minus == x || plus == minus {
        None
    } else {
        Some((plus, minus))
    }
}

fn quotient<T: Real>(a: T, b: T, h: T, factor: T) -> T {
    if !a.is_finite() || !b.is_finite() {
        return T::NAN;
    }
    let difference = a - b;
    let denominator = factor * h;
    if difference.is_finite() && denominator.is_finite() {
        difference / denominator
    } else if difference.is_finite() {
        (difference / factor) / h
    } else if factor > T::ONE {
        (a / factor - b / factor) / h
    } else {
        a / h - b / h
    }
}

/// Quotient mit dem tatsächlich darstellbaren Abstand der Stützpunkte.
fn span_quotient<T: Real>(a: T, b: T, plus: T, minus: T) -> T {
    let width = plus - minus;
    if width.is_finite() {
        quotient(a, b, width, T::ONE)
    } else {
        quotient(
            a,
            b,
            plus * T::from_f64(0.5) - minus * T::from_f64(0.5),
            T::from_f64(2.0),
        )
    }
}

fn second_from_samples<T: Real>(fp: T, f0: T, fm: T, plus: T, x: T, minus: T) -> T {
    let forward = span_quotient(fp, f0, plus, x);
    let backward = span_quotient(f0, fm, x, minus);
    let width = plus - minus;
    let half = T::from_f64(0.5);
    let half_width = if width.is_finite() {
        width * half
    } else {
        plus * half - minus * half
    };
    quotient(forward, backward, half_width, T::ONE)
}

/// Vorwärtsdifferenz `(f(x + h) − f(x))/h` (Fehlerordnung `h`).
pub fn forward_difference<T: Real>(mut f: impl FnMut(T) -> T, x: T, h: T) -> T {
    if !x.is_finite() || !h.is_finite() || h == T::ZERO || !(x + h).is_finite() || x + h == x {
        return T::NAN;
    }
    span_quotient(f(x + h), f(x), x + h, x)
}

/// Rückwärtsdifferenz `(f(x) − f(x − h))/h` (Fehlerordnung `h`).
pub fn backward_difference<T: Real>(mut f: impl FnMut(T) -> T, x: T, h: T) -> T {
    if !x.is_finite() || !h.is_finite() || h == T::ZERO || !(x - h).is_finite() || x - h == x {
        return T::NAN;
    }
    span_quotient(f(x), f(x - h), x, x - h)
}

/// Zentrale Differenz `(f(x + h) − f(x − h))/(2h)` (Fehlerordnung `h²`).
pub fn central_difference<T: Real>(mut f: impl FnMut(T) -> T, x: T, h: T) -> T {
    let Some((plus, minus)) = nodes(x, h) else {
        return T::NAN;
    };
    span_quotient(f(plus), f(minus), plus, minus)
}

/// Zentrale Differenz vierter Ordnung `(−f(x+2h) + 8f(x+h) − 8f(x−h) + f(x−2h))/(12h)`.
pub fn central_difference4<T: Real>(mut f: impl FnMut(T) -> T, x: T, h: T) -> T {
    let two = T::from_f64(2.0);
    let Some((plus, minus)) = nodes(x, h) else {
        return T::NAN;
    };
    let Some((plus2, minus2)) = nodes(x, two * h) else {
        return T::NAN;
    };
    if plus == plus2 || minus == minus2 {
        return T::NAN;
    }
    let near = span_quotient(f(plus), f(minus), plus, minus);
    let far = span_quotient(f(plus2), f(minus2), plus2, minus2);
    near + (near - far) / T::from_f64(3.0)
}

/// Zweite Ableitung `(f(x + h) − 2f(x) + f(x − h))/h²` (Fehlerordnung `h²`; optimal `h ≈ ε^(1/4)`).
pub fn second_derivative<T: Real>(mut f: impl FnMut(T) -> T, x: T, h: T) -> T {
    let Some((plus, minus)) = nodes(x, h) else {
        return T::NAN;
    };
    let (fp, f0, fm) = (f(plus), f(x), f(minus));
    second_from_samples(fp, f0, fm, plus, x, minus)
}

/// Ableitung per **Ridders-Extrapolation**: zentrale Differenzen mit geometrisch fallender Schrittweite (Faktor 1,4,
/// Start `h`) werden im Neville-Schema extrapoliert. Rückgabe `(Ableitung, Fehlerschätzung)`. `h` sollte nicht zu klein sein
/// (eher ein Zehntel der Längenskala der Funktion als `10⁻⁸`).
pub fn derivative_ridders<T: Real>(mut f: impl FnMut(T) -> T, x: T, h: T) -> (T, T) {
    const NTAB: usize = 10;
    let con = T::from_f64(1.4);
    let con2 = con * con;
    let safe = T::from_f64(2.0);
    if nodes(x, h).is_none() {
        return (T::NAN, T::INFINITY);
    }
    let mut a = [[T::ZERO; NTAB]; NTAB];
    let mut hh = h;
    let mut roundoff = [T::ZERO; NTAB];
    let mut measure = |step: T| {
        let (plus, minus) = nodes(x, step).expect("Stützpunkte vorher geprüft");
        let (fp, fm) = (f(plus), f(minus));
        let derivative = span_quotient(fp, fm, plus, minus);
        let eps = T::from_f64(8.0) * T::EPSILON;
        let noise = span_quotient(fp.abs() * eps, -fm.abs() * eps, plus, minus).abs();
        (derivative, noise)
    };
    (a[0][0], roundoff[0]) = measure(hh);
    if !a[0][0].is_finite() {
        return (T::NAN, T::INFINITY);
    }
    let mut err = T::INFINITY;
    let mut ans = a[0][0];
    for i in 1..NTAB {
        hh /= con;
        if nodes(x, hh).is_none() {
            break;
        }
        (a[0][i], roundoff[i]) = measure(hh);
        if !a[0][i].is_finite() {
            return (T::NAN, T::INFINITY);
        }
        let mut fac = con2;
        for j in 1..=i {
            a[j][i] = a[j - 1][i] + (a[j - 1][i] - a[j - 1][i - 1]) / (fac - T::ONE);
            if !a[j][i].is_finite() {
                return (T::NAN, T::INFINITY);
            }
            fac *= con2;
            let errt = (a[j][i] - a[j - 1][i])
                .abs()
                .max((a[j][i] - a[j - 1][i - 1]).abs())
                .max(T::from_f64(4.0) * roundoff[i].max(roundoff[i - 1]));
            if errt <= err {
                err = errt;
                ans = a[j][i];
            }
        }
        if (a[i][i] - a[i - 1][i - 1]).abs() >= safe * err {
            break;
        }
    }
    (ans, err)
}

/// Zwischenspeicher für [`gradient`] und [`hessian`] bei `n` Variablen.
pub const fn gradient_work_len(n: usize) -> usize {
    n
}

/// Gradient von `f: ℝⁿ → ℝ` per zentralen Differenzen mit Schrittweite `ε^(1/3)·max(1, |xᵢ|)` (`2n` Auswertungen).
/// `work`: mindestens [`gradient_work_len`]`(n)`. Panik bei zu kurzen Slices.
pub fn gradient<T: Real>(mut f: impl FnMut(&[T]) -> T, x: &[T], grad: &mut [T], work: &mut [T]) {
    let n = x.len();
    assert!(
        grad.len() >= n && work.len() >= gradient_work_len(n),
        "gradient: Slices zu kurz"
    );
    let xp = &mut work[..n];
    xp.copy_from_slice(x);
    let cbrt_eps = T::EPSILON.cbrt();
    if x.iter()
        .any(|v| nodes(*v, cbrt_eps * T::ONE.max(v.abs())).is_none())
    {
        grad[..n].fill(T::NAN);
        return;
    }
    for i in 0..n {
        let h = cbrt_eps * T::ONE.max(x[i].abs());
        xp[i] = x[i] + h;
        let fp = f(xp);
        xp[i] = x[i] - h;
        let fm = f(xp);
        xp[i] = x[i];
        grad[i] = span_quotient(fp, fm, x[i] + h, x[i] - h);
    }
}

/// Hessematrix (`n × n`, zeilenweise) von `f: ℝⁿ → ℝ` per zentralen Differenzen mit Schrittweite
/// `ε^(1/4)·max(1, |xᵢ|)`; symmetrisch. `work`: mindestens [`gradient_work_len`]`(n)`. Aufwand `O(n²)` Auswertungen.
/// Genauigkeit nur etwa `√ε`; mit [`hessian_ad`](crate::autodiff::hessian_ad) geht es exakt.
pub fn hessian<T: Real>(mut f: impl FnMut(&[T]) -> T, x: &[T], h_out: &mut [T], work: &mut [T]) {
    let n = x.len();
    assert!(
        h_out.len() >= n * n && work.len() >= gradient_work_len(n),
        "hessian: Slices zu kurz"
    );
    let xp = &mut work[..n];
    xp.copy_from_slice(x);
    let root = T::EPSILON.sqrt().sqrt();
    if x.iter()
        .any(|v| nodes(*v, root * T::ONE.max(v.abs())).is_none())
    {
        h_out[..n * n].fill(T::NAN);
        return;
    }
    let f0 = f(xp);
    if !f0.is_finite() {
        h_out[..n * n].fill(T::NAN);
        return;
    }
    for i in 0..n {
        let hi = root * T::ONE.max(x[i].abs());
        xp[i] = x[i] + hi;
        let fp = f(xp);
        xp[i] = x[i] - hi;
        let fm = f(xp);
        xp[i] = x[i];
        h_out[i * n + i] = second_from_samples(fp, f0, fm, x[i] + hi, x[i], x[i] - hi);
        for j in (i + 1)..n {
            let hj = root * T::ONE.max(x[j].abs());
            xp[i] = x[i] + hi;
            xp[j] = x[j] + hj;
            let fpp = f(xp);
            xp[j] = x[j] - hj;
            let fpm = f(xp);
            xp[i] = x[i] - hi;
            let fmm = f(xp);
            xp[j] = x[j] + hj;
            let fmp = f(xp);
            xp[i] = x[i];
            xp[j] = x[j];
            let v = span_quotient(
                span_quotient(fpp, fpm, x[j] + hj, x[j] - hj),
                span_quotient(fmp, fmm, x[j] + hj, x[j] - hj),
                x[i] + hi,
                x[i] - hi,
            );
            h_out[i * n + j] = v;
            h_out[j * n + i] = v;
        }
    }
}

/// Zwischenspeicher für [`jacobian`] bei `m` Funktionswerten und `n` Variablen.
pub const fn jacobian_work_len(m: usize, n: usize) -> usize {
    n + 2 * m
}

/// Jacobi-Matrix (`m × n`, zeilenweise) von `f: ℝⁿ → ℝᵐ` per zentralen Differenzen (`2n` Auswertungen). `f(x, out)` schreibt
/// die `m` Werte. `work`: mindestens [`jacobian_work_len`]`(m, n)`. Genauer als [`numerical_jacobian`](crate::optimize::numerical_jacobian)
/// (dort Vorwärtsdifferenzen), aber doppelt so teuer.
pub fn jacobian<T: Real>(
    mut f: impl FnMut(&[T], &mut [T]),
    x: &[T],
    m: usize,
    jac: &mut [T],
    work: &mut [T],
) {
    let n = x.len();
    assert!(
        jac.len() >= m * n && work.len() >= jacobian_work_len(m, n),
        "jacobian: Slices zu kurz"
    );
    let (xp, rest) = work.split_at_mut(n);
    let (fp, rest) = rest.split_at_mut(m);
    let fm = &mut rest[..m];
    xp.copy_from_slice(x);
    let cbrt_eps = T::EPSILON.cbrt();
    if x.iter()
        .any(|v| nodes(*v, cbrt_eps * T::ONE.max(v.abs())).is_none())
    {
        jac[..m * n].fill(T::NAN);
        return;
    }
    for k in 0..n {
        let h = cbrt_eps * T::ONE.max(x[k].abs());
        xp[k] = x[k] + h;
        f(xp, fp);
        xp[k] = x[k] - h;
        f(xp, fm);
        xp[k] = x[k];
        for i in 0..m {
            jac[i * n + k] = span_quotient(fp[i], fm[i], x[k] + h, x[k] - h);
        }
    }
}
