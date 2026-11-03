//! Wahrscheinlichkeitsverteilungen: Dichte (`*_pdf`), Verteilungsfunktion (`*_cdf`) und Quantilfunktion (`*_quantile`).
//!
//! Ungültige Parameter oder Argumente liefern `NaN`. Die Quantile werden durch Intervallhalbierung auf der Verteilungsfunktion
//! bestimmt (robust, etwa 100 Auswertungen). Grundlage sind [`gamma_p`], [`beta_inc`] und [`erfc`](crate::special::erfc).

use crate::Real;
use crate::special::{beta_inc, gamma_p, ln_beta, ln_gamma, normal_cdf, normal_quantile};

fn nan<T: Real>() -> T {
    T::NAN
}

/// Löst `cdf(x) = p` für eine monotone Verteilungsfunktion; `lo` ist die untere Stützstelle des Suchbereichs, `symmetric`
/// erlaubt negative Werte (Intervall `[−hi, hi]`).
fn invert<T: Real>(cdf: impl Fn(T) -> T, p: T, lo: T, symmetric: bool) -> T {
    if p.is_nan() || p < T::ZERO || p > T::ONE {
        return nan();
    }
    if p == T::ZERO {
        return if symmetric { -T::INFINITY } else { lo };
    }
    if p == T::ONE {
        return T::INFINITY;
    }
    let two = T::from_f64(2.0);
    let mut hi = lo + T::ONE;
    let mut a = lo;
    let mut n = 0;
    if symmetric {
        a = -hi;
        while cdf(a) > p && n < 2000 {
            a = a * two;
            n += 1;
        }
    }
    n = 0;
    while cdf(hi) < p && n < 2000 {
        hi = lo + (hi - lo) * two;
        n += 1;
    }
    let mut b = hi;
    for _ in 0..200 {
        let m = (a + b) * T::from_f64(0.5);
        if m == a || m == b {
            break;
        }
        if cdf(m) < p {
            a = m;
        } else {
            b = m;
        }
    }
    (a + b) * T::from_f64(0.5)
}

// ---- Normalverteilung ----

/// Dichte der Normalverteilung `N(mean, sd²)` (`sd > 0`).
pub fn normal_pdf<T: Real>(x: T, mean: T, sd: T) -> T {
    if !(sd > T::ZERO) {
        return nan();
    }
    let z = (x - mean) / sd;
    (-T::from_f64(0.5) * z * z).exp() / (sd * (T::from_f64(2.0) * T::PI).sqrt())
}

/// Verteilungsfunktion der Normalverteilung `N(mean, sd²)`.
pub fn normal_cdf_with<T: Real>(x: T, mean: T, sd: T) -> T {
    if !(sd > T::ZERO) {
        return nan();
    }
    normal_cdf((x - mean) / sd)
}

/// Quantilfunktion der Normalverteilung `N(mean, sd²)`.
pub fn normal_quantile_with<T: Real>(p: T, mean: T, sd: T) -> T {
    if !(sd > T::ZERO) {
        return nan();
    }
    mean + sd * normal_quantile(p)
}

// ---- Exponential, Gleichverteilung ----

/// Dichte der Exponentialverteilung mit Rate `lambda > 0`.
pub fn exponential_pdf<T: Real>(x: T, lambda: T) -> T {
    if !(lambda > T::ZERO) {
        return nan();
    }
    if x < T::ZERO {
        T::ZERO
    } else {
        lambda * (-lambda * x).exp()
    }
}

/// Verteilungsfunktion der Exponentialverteilung.
pub fn exponential_cdf<T: Real>(x: T, lambda: T) -> T {
    if !(lambda > T::ZERO) {
        return nan();
    }
    if x < T::ZERO {
        T::ZERO
    } else {
        -(-lambda * x).exp_m1()
    }
}

/// Quantilfunktion der Exponentialverteilung.
pub fn exponential_quantile<T: Real>(p: T, lambda: T) -> T {
    if !(lambda > T::ZERO) || p.is_nan() || p < T::ZERO || p > T::ONE {
        return nan();
    }
    -(-p).ln_1p() / lambda
}

/// Dichte der Gleichverteilung auf `[a, b]` (`a < b`).
pub fn uniform_pdf<T: Real>(x: T, a: T, b: T) -> T {
    if !(a < b) {
        return nan();
    }
    if x < a || x > b {
        T::ZERO
    } else {
        T::ONE / (b - a)
    }
}

/// Verteilungsfunktion der Gleichverteilung auf `[a, b]`.
pub fn uniform_cdf<T: Real>(x: T, a: T, b: T) -> T {
    if !(a < b) {
        return nan();
    }
    ((x - a) / (b - a)).max(T::ZERO).min(T::ONE)
}

/// Quantilfunktion der Gleichverteilung auf `[a, b]`.
pub fn uniform_quantile<T: Real>(p: T, a: T, b: T) -> T {
    if !(a < b) || p.is_nan() || p < T::ZERO || p > T::ONE {
        return nan();
    }
    a + p * (b - a)
}

// ---- Gamma, χ² ----

/// Dichte der Gammaverteilung mit Form `k > 0` und Skala `theta > 0`.
pub fn gamma_pdf<T: Real>(x: T, k: T, theta: T) -> T {
    if !(k > T::ZERO) || !(theta > T::ZERO) {
        return nan();
    }
    if x < T::ZERO {
        return T::ZERO;
    }
    if x == T::ZERO {
        return if k < T::ONE {
            T::INFINITY
        } else if k == T::ONE {
            T::ONE / theta
        } else {
            T::ZERO
        };
    }
    ((k - T::ONE) * x.ln() - x / theta - ln_gamma(k) - k * theta.ln()).exp()
}

/// Verteilungsfunktion der Gammaverteilung.
pub fn gamma_cdf<T: Real>(x: T, k: T, theta: T) -> T {
    if !(k > T::ZERO) || !(theta > T::ZERO) {
        return nan();
    }
    if x <= T::ZERO {
        T::ZERO
    } else {
        gamma_p(k, x / theta)
    }
}

/// Quantilfunktion der Gammaverteilung.
pub fn gamma_quantile<T: Real>(p: T, k: T, theta: T) -> T {
    if !(k > T::ZERO) || !(theta > T::ZERO) {
        return nan();
    }
    invert(|x| gamma_cdf(x, k, theta), p, T::ZERO, false)
}

/// Dichte der χ²-Verteilung mit `df > 0` Freiheitsgraden.
pub fn chi2_pdf<T: Real>(x: T, df: T) -> T {
    gamma_pdf(x, df * T::from_f64(0.5), T::from_f64(2.0))
}

/// Verteilungsfunktion der χ²-Verteilung.
pub fn chi2_cdf<T: Real>(x: T, df: T) -> T {
    gamma_cdf(x, df * T::from_f64(0.5), T::from_f64(2.0))
}

/// Quantilfunktion der χ²-Verteilung.
pub fn chi2_quantile<T: Real>(p: T, df: T) -> T {
    gamma_quantile(p, df * T::from_f64(0.5), T::from_f64(2.0))
}

// ---- Beta ----

/// Dichte der Betaverteilung `Beta(a, b)` auf `[0, 1]` (`a, b > 0`).
pub fn beta_pdf<T: Real>(x: T, a: T, b: T) -> T {
    if !(a > T::ZERO) || !(b > T::ZERO) {
        return nan();
    }
    if x < T::ZERO || x > T::ONE {
        return T::ZERO;
    }
    if x == T::ZERO || x == T::ONE {
        let e = if x == T::ZERO { a } else { b };
        return if e < T::ONE {
            T::INFINITY
        } else if e == T::ONE {
            (T::ONE / ln_beta(a, b).exp()).max(T::ZERO)
        } else {
            T::ZERO
        };
    }
    ((a - T::ONE) * x.ln() + (b - T::ONE) * (-x).ln_1p() - ln_beta(a, b)).exp()
}

/// Verteilungsfunktion der Betaverteilung.
pub fn beta_cdf<T: Real>(x: T, a: T, b: T) -> T {
    if !(a > T::ZERO) || !(b > T::ZERO) {
        return nan();
    }
    beta_inc(a, b, x.max(T::ZERO).min(T::ONE))
}

/// Quantilfunktion der Betaverteilung (Ergebnis in `[0, 1]`).
pub fn beta_quantile<T: Real>(p: T, a: T, b: T) -> T {
    if !(a > T::ZERO) || !(b > T::ZERO) || p.is_nan() || p < T::ZERO || p > T::ONE {
        return nan();
    }
    if p == T::ZERO {
        return T::ZERO;
    }
    if p == T::ONE {
        return T::ONE;
    }
    let (mut lo, mut hi) = (T::ZERO, T::ONE);
    for _ in 0..200 {
        let m = (lo + hi) * T::from_f64(0.5);
        if m == lo || m == hi {
            break;
        }
        if beta_cdf(m, a, b) < p {
            lo = m;
        } else {
            hi = m;
        }
    }
    (lo + hi) * T::from_f64(0.5)
}

// ---- Student-t ----

/// Dichte der Student-t-Verteilung mit `df > 0` Freiheitsgraden.
pub fn student_t_pdf<T: Real>(x: T, df: T) -> T {
    if !(df > T::ZERO) {
        return nan();
    }
    let half = T::from_f64(0.5);
    let l = ln_gamma((df + T::ONE) * half)
        - ln_gamma(df * half)
        - half * (df * T::PI).ln()
        - (df + T::ONE) * half * (x * x / df).ln_1p();
    l.exp()
}

/// Verteilungsfunktion der Student-t-Verteilung.
pub fn student_t_cdf<T: Real>(x: T, df: T) -> T {
    if !(df > T::ZERO) || x.is_nan() {
        return nan();
    }
    let half = T::from_f64(0.5);
    let z = df / (df + x * x);
    let tail = half * beta_inc(df * half, half, z);
    if x > T::ZERO { T::ONE - tail } else { tail }
}

/// Quantilfunktion der Student-t-Verteilung.
pub fn student_t_quantile<T: Real>(p: T, df: T) -> T {
    if !(df > T::ZERO) {
        return nan();
    }
    if p == T::from_f64(0.5) {
        return T::ZERO;
    }
    invert(|x| student_t_cdf(x, df), p, T::ZERO, true)
}

// ---- F ----

/// Dichte der F-Verteilung mit `d1, d2 > 0` Freiheitsgraden.
pub fn f_pdf<T: Real>(x: T, d1: T, d2: T) -> T {
    if !(d1 > T::ZERO) || !(d2 > T::ZERO) {
        return nan();
    }
    if x <= T::ZERO {
        return T::ZERO;
    }
    let half = T::from_f64(0.5);
    let l = half * d1 * (d1 * x).ln() + half * d2 * d2.ln()
        - half * (d1 + d2) * (d1 * x + d2).ln()
        - x.ln()
        - ln_beta(d1 * half, d2 * half);
    l.exp()
}

/// Verteilungsfunktion der F-Verteilung.
pub fn f_cdf<T: Real>(x: T, d1: T, d2: T) -> T {
    if !(d1 > T::ZERO) || !(d2 > T::ZERO) {
        return nan();
    }
    if x <= T::ZERO {
        return T::ZERO;
    }
    let half = T::from_f64(0.5);
    beta_inc(d1 * half, d2 * half, d1 * x / (d1 * x + d2))
}

/// Quantilfunktion der F-Verteilung.
pub fn f_quantile<T: Real>(p: T, d1: T, d2: T) -> T {
    if !(d1 > T::ZERO) || !(d2 > T::ZERO) {
        return nan();
    }
    invert(|x| f_cdf(x, d1, d2), p, T::ZERO, false)
}
