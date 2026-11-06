//! Wahrscheinlichkeitsverteilungen: Dichte (`*_pdf`), CDF (`*_cdf`),
//! Überlebensfunktion (`*_sf`), Quantil (`*_quantile`) und inverser oberer Tail (`*_isf`).
//!
//! Parameter müssen endlich und gültig sein; ungültige Eingaben liefern `NaN`.
//! Für kleine obere Wahrscheinlichkeiten direkt `sf`/`isf` verwenden: `1-cdf`
//! bzw. `quantile(1-q)` verlieren sie durch Rundung. Numerische Quantile
//! invertieren den kleineren Tail mit darstellungsabhängigem Abbruch.
use crate::Real;
use crate::special::{
    beta_inc, beta_inc_complement, gamma_p, gamma_q, ln_beta, ln_gamma, normal_cdf, normal_isf,
    normal_quantile, normal_sf,
};
fn positive<T: Real>(x: T) -> bool {
    x.is_finite() && x > T::ZERO
}
fn probability<T: Real>(p: T) -> bool {
    p.is_finite() && p >= T::ZERO && p <= T::ONE
}
fn interval<T: Real>(a: T, b: T) -> bool {
    a.is_finite() && b.is_finite() && a < b
}

// Invert a positive-support tail. Finite endpoint for beta, expanding bracket
// otherwise. 1200 halvings cover the f64 exponent range down to subnormals.
fn invert<T: Real>(tail: impl Fn(T) -> T, p: T, upper: bool, bounded: bool) -> T {
    let half = T::from_f64(0.5);
    let two = T::from_f64(2.0);
    let (mut lo, mut hi) = (T::ZERO, T::ONE);
    let below = |v: T| if upper { v > p } else { v < p };
    if !bounded {
        let largest = (T::ONE - T::EPSILON * half) / T::MIN_POSITIVE * T::from_f64(4.0);
        for _ in 0..2048 {
            let value = tail(hi);
            if value.is_nan() {
                return T::NAN;
            }
            if !below(value) {
                break;
            }
            lo = hi;
            if hi == largest {
                return T::INFINITY;
            }
            hi = if hi > largest * half {
                largest
            } else {
                hi * two
            };
        }
    }
    for _ in 0..1200 {
        let mid = lo + (hi - lo) * half;
        if mid == lo || mid == hi {
            return mid;
        }
        let value = tail(mid);
        if value.is_nan() {
            return T::NAN;
        }
        if below(value) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo + (hi - lo) * half
}
fn zscore<T: Real>(x: T, mean: T, sd: T) -> T {
    let delta = x - mean;
    if delta.is_infinite() && x.is_finite() {
        x / sd - mean / sd
    } else {
        delta / sd
    }
}
/// Dichte von `N(mean,sd²)` mit endlichem `mean` und `sd > 0`.
pub fn normal_pdf<T: Real>(x: T, mean: T, sd: T) -> T {
    if !mean.is_finite() || !positive(sd) {
        return T::NAN;
    }
    let z = zscore(x, mean, sd);
    (-T::from_f64(0.5) * z * z).exp() / T::from_f64(core::f64::consts::TAU).sqrt() / sd
}
/// CDF von `N(mean,sd²)`.
pub fn normal_cdf_with<T: Real>(x: T, mean: T, sd: T) -> T {
    if !mean.is_finite() || !positive(sd) {
        return T::NAN;
    }
    normal_cdf(zscore(x, mean, sd))
}
/// Direkte obere Schwanzwahrscheinlichkeit von `N(mean,sd²)`.
pub fn normal_sf_with<T: Real>(x: T, mean: T, sd: T) -> T {
    if !mean.is_finite() || !positive(sd) {
        return T::NAN;
    }
    normal_sf(zscore(x, mean, sd))
}
/// Quantil von `N(mean,sd²)`.
pub fn normal_quantile_with<T: Real>(p: T, mean: T, sd: T) -> T {
    if !mean.is_finite() || !positive(sd) {
        return T::NAN;
    }
    sd.mul_add(normal_quantile(p), mean)
}
/// Inverser oberer Tail von `N(mean,sd²)`; `sf(x)=q`.
pub fn normal_isf_with<T: Real>(q: T, mean: T, sd: T) -> T {
    if !mean.is_finite() || !positive(sd) {
        return T::NAN;
    }
    sd.mul_add(normal_isf(q), mean)
}
/// Exponentialdichte mit Rate `lambda > 0`.
pub fn exponential_pdf<T: Real>(x: T, lambda: T) -> T {
    if !positive(lambda) {
        return T::NAN;
    }
    if x < T::ZERO {
        T::ZERO
    } else {
        lambda * (-lambda * x).exp()
    }
}
/// CDF der Exponentialverteilung.
pub fn exponential_cdf<T: Real>(x: T, lambda: T) -> T {
    if !positive(lambda) {
        return T::NAN;
    }
    if x < T::ZERO {
        T::ZERO
    } else {
        -(-lambda * x).exp_m1()
    }
}
/// Direkter oberer Tail der Exponentialverteilung.
pub fn exponential_sf<T: Real>(x: T, lambda: T) -> T {
    if !positive(lambda) {
        return T::NAN;
    }
    if x < T::ZERO {
        T::ONE
    } else {
        (-lambda * x).exp()
    }
}
/// Quantil der Exponentialverteilung.
pub fn exponential_quantile<T: Real>(p: T, lambda: T) -> T {
    if !positive(lambda) || !probability(p) {
        return T::NAN;
    }
    -(-p).ln_1p() / lambda
}
/// Inverser oberer Tail der Exponentialverteilung.
pub fn exponential_isf<T: Real>(q: T, lambda: T) -> T {
    if !positive(lambda) || !probability(q) {
        return T::NAN;
    }
    -q.ln() / lambda
}
/// Gleichverteilungsdichte auf einem endlichen Intervall `[a,b]`, `a<b`.
pub fn uniform_pdf<T: Real>(x: T, a: T, b: T) -> T {
    if !interval(a, b) || x.is_nan() {
        return T::NAN;
    }
    if x < a || x > b {
        return T::ZERO;
    }
    let width = b - a;
    if width.is_finite() {
        T::ONE / width
    } else {
        T::from_f64(0.5) / (b * T::from_f64(0.5) - a * T::from_f64(0.5))
    }
}
/// CDF der Gleichverteilung.
pub fn uniform_cdf<T: Real>(x: T, a: T, b: T) -> T {
    if !interval(a, b) || x.is_nan() {
        return T::NAN;
    }
    if x <= a {
        return T::ZERO;
    }
    if x >= b {
        return T::ONE;
    }
    let width = b - a;
    if width.is_finite() {
        (x - a) / width
    } else {
        (x * T::from_f64(0.5) - a * T::from_f64(0.5))
            / (b * T::from_f64(0.5) - a * T::from_f64(0.5))
    }
}
/// Direkter oberer Tail der Gleichverteilung.
pub fn uniform_sf<T: Real>(x: T, a: T, b: T) -> T {
    if !interval(a, b) || x.is_nan() {
        return T::NAN;
    }
    if x <= a {
        return T::ONE;
    }
    if x >= b {
        return T::ZERO;
    }
    let width = b - a;
    if width.is_finite() {
        (b - x) / width
    } else {
        (b * T::from_f64(0.5) - x * T::from_f64(0.5))
            / (b * T::from_f64(0.5) - a * T::from_f64(0.5))
    }
}
/// Quantil der Gleichverteilung.
pub fn uniform_quantile<T: Real>(p: T, a: T, b: T) -> T {
    if !interval(a, b) || !probability(p) {
        return T::NAN;
    }
    (T::ONE - p) * a + p * b
}
/// Inverser oberer Tail der Gleichverteilung.
pub fn uniform_isf<T: Real>(q: T, a: T, b: T) -> T {
    if !interval(a, b) || !probability(q) {
        return T::NAN;
    }
    q * a + (T::ONE - q) * b
}
/// Gammadichte mit Form `k>0` und Skala `theta>0`.
pub fn gamma_pdf<T: Real>(x: T, k: T, theta: T) -> T {
    if !positive(k) || !positive(theta) || x.is_nan() {
        return T::NAN;
    }
    if x < T::ZERO || x == T::INFINITY {
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
    if (x / theta).is_infinite() {
        return T::ZERO;
    }
    ((k - T::ONE) * (x.ln() - theta.ln()) - x / theta - ln_gamma(k) - theta.ln()).exp()
}
/// CDF der Gammaverteilung.
pub fn gamma_cdf<T: Real>(x: T, k: T, theta: T) -> T {
    if !positive(k) || !positive(theta) {
        return T::NAN;
    }
    if x <= T::ZERO {
        T::ZERO
    } else {
        gamma_p(k, x / theta)
    }
}
/// Direkter oberer Tail der Gammaverteilung.
pub fn gamma_sf<T: Real>(x: T, k: T, theta: T) -> T {
    if !positive(k) || !positive(theta) {
        return T::NAN;
    }
    if x <= T::ZERO {
        T::ONE
    } else {
        gamma_q(k, x / theta)
    }
}
/// Quantil der Gammaverteilung.
pub fn gamma_quantile<T: Real>(p: T, k: T, theta: T) -> T {
    if !positive(k) || !positive(theta) || !probability(p) {
        return T::NAN;
    }
    if p == T::ZERO {
        return T::ZERO;
    }
    if p == T::ONE {
        return T::INFINITY;
    }
    let x = if p <= T::from_f64(0.5) {
        invert(|x| gamma_p(k, x), p, false, false)
    } else {
        invert(|x| gamma_q(k, x), T::ONE - p, true, false)
    };
    x * theta
}
/// Inverser oberer Tail der Gammaverteilung.
pub fn gamma_isf<T: Real>(q: T, k: T, theta: T) -> T {
    if !positive(k) || !positive(theta) || !probability(q) {
        return T::NAN;
    }
    if q == T::ZERO {
        return T::INFINITY;
    }
    if q == T::ONE {
        return T::ZERO;
    }
    let x = if q <= T::from_f64(0.5) {
        invert(|x| gamma_q(k, x), q, true, false)
    } else {
        invert(|x| gamma_p(k, x), T::ONE - q, false, false)
    };
    x * theta
}
/// χ²-Dichte mit `df>0` Freiheitsgraden.
pub fn chi2_pdf<T: Real>(x: T, df: T) -> T {
    gamma_pdf(x, df * T::from_f64(0.5), T::from_f64(2.0))
}
/// CDF der χ²-Verteilung.
pub fn chi2_cdf<T: Real>(x: T, df: T) -> T {
    gamma_cdf(x, df * T::from_f64(0.5), T::from_f64(2.0))
}
/// Direkter oberer Tail der χ²-Verteilung.
pub fn chi2_sf<T: Real>(x: T, df: T) -> T {
    gamma_sf(x, df * T::from_f64(0.5), T::from_f64(2.0))
}
/// Quantil der χ²-Verteilung.
pub fn chi2_quantile<T: Real>(p: T, df: T) -> T {
    gamma_quantile(p, df * T::from_f64(0.5), T::from_f64(2.0))
}
/// Inverser oberer Tail der χ²-Verteilung.
pub fn chi2_isf<T: Real>(q: T, df: T) -> T {
    gamma_isf(q, df * T::from_f64(0.5), T::from_f64(2.0))
}
/// Betadichte auf `[0,1]`, `a,b>0`.
pub fn beta_pdf<T: Real>(x: T, a: T, b: T) -> T {
    if !positive(a) || !positive(b) || x.is_nan() {
        return T::NAN;
    }
    if x < T::ZERO || x > T::ONE {
        return T::ZERO;
    }
    if x == T::ZERO || x == T::ONE {
        let e = if x == T::ZERO { a } else { b };
        return if e < T::ONE {
            T::INFINITY
        } else if e == T::ONE {
            (-ln_beta(a, b)).exp()
        } else {
            T::ZERO
        };
    }
    ((a - T::ONE) * x.ln() + (b - T::ONE) * (-x).ln_1p() - ln_beta(a, b)).exp()
}
/// CDF der Betaverteilung.
pub fn beta_cdf<T: Real>(x: T, a: T, b: T) -> T {
    if !positive(a) || !positive(b) || x.is_nan() {
        return T::NAN;
    }
    beta_inc(a, b, x.max(T::ZERO).min(T::ONE))
}
/// Direkter oberer Tail der Betaverteilung.
pub fn beta_sf<T: Real>(x: T, a: T, b: T) -> T {
    if !positive(a) || !positive(b) || x.is_nan() {
        return T::NAN;
    }
    beta_inc_complement(a, b, x.max(T::ZERO).min(T::ONE))
}
/// Quantil der Betaverteilung auf `[0,1]`.
pub fn beta_quantile<T: Real>(p: T, a: T, b: T) -> T {
    if !positive(a) || !positive(b) || !probability(p) {
        return T::NAN;
    }
    if p == T::ZERO || p == T::ONE {
        return p;
    }
    if p <= T::from_f64(0.5) {
        invert(|x| beta_cdf(x, a, b), p, false, true)
    } else {
        T::ONE - invert(|x| beta_cdf(x, b, a), T::ONE - p, false, true)
    }
}
/// Inverser oberer Tail der Betaverteilung.
pub fn beta_isf<T: Real>(q: T, a: T, b: T) -> T {
    if !positive(a) || !positive(b) || !probability(q) {
        return T::NAN;
    }
    if q == T::ZERO || q == T::ONE {
        return T::ONE - q;
    }
    if q <= T::from_f64(0.5) {
        T::ONE - invert(|x| beta_cdf(x, b, a), q, false, true)
    } else {
        invert(|x| beta_cdf(x, a, b), T::ONE - q, false, true)
    }
}
/// Student-t-Dichte mit `df>0`.
pub fn student_t_pdf<T: Real>(x: T, df: T) -> T {
    if !positive(df) {
        return T::NAN;
    }
    let half = T::from_f64(0.5);
    let r = x.abs() / df.sqrt();
    let log = if r <= T::ONE {
        (r * r).ln_1p()
    } else {
        T::from_f64(2.0) * r.ln() + ((T::ONE / r) * (T::ONE / r)).ln_1p()
    };
    (ln_gamma((df + T::ONE) * half)
        - ln_gamma(df * half)
        - half * (df.ln() + T::PI.ln())
        - (df + T::ONE) * half * log)
        .exp()
}
fn t_tail<T: Real>(x: T, df: T) -> T {
    let half = T::from_f64(0.5);
    let a = df * half;
    let r = x.abs() / df.sqrt();
    if r <= T::ONE {
        let v = r * r;
        return half - half * beta_inc(half, a, v / (T::ONE + v));
    }
    let inv = T::ONE / r;
    let v = inv * inv;
    if v >= T::MIN_POSITIVE {
        return half * beta_inc(a, half, v / (T::ONE + v));
    }
    // x*x may overflow or its reciprocal underflow although the t tail fits.
    let log_z = df.ln() - T::from_f64(2.0) * x.abs().ln();
    (a * log_z - a.ln() - ln_beta(a, half) - T::from_f64(core::f64::consts::LN_2)).exp()
}
/// CDF der Student-t-Verteilung.
pub fn student_t_cdf<T: Real>(x: T, df: T) -> T {
    if !positive(df) || x.is_nan() {
        return T::NAN;
    }
    let tail = t_tail(x, df);
    if x > T::ZERO { T::ONE - tail } else { tail }
}
/// Direkter oberer Tail der Student-t-Verteilung.
pub fn student_t_sf<T: Real>(x: T, df: T) -> T {
    if !positive(df) || x.is_nan() {
        return T::NAN;
    }
    let tail = t_tail(x, df);
    if x < T::ZERO { T::ONE - tail } else { tail }
}
/// Quantil der Student-t-Verteilung.
pub fn student_t_quantile<T: Real>(p: T, df: T) -> T {
    if !positive(df) || !probability(p) {
        return T::NAN;
    }
    if p == T::ZERO {
        return T::NEG_INFINITY;
    }
    if p == T::ONE {
        return T::INFINITY;
    }
    if p == T::from_f64(0.5) {
        return T::ZERO;
    }
    if p < T::from_f64(0.5) {
        -invert(|x| t_tail(x, df), p, true, false)
    } else {
        invert(|x| t_tail(x, df), T::ONE - p, true, false)
    }
}
/// Inverser oberer Tail der Student-t-Verteilung.
pub fn student_t_isf<T: Real>(q: T, df: T) -> T {
    -student_t_quantile(q, df)
}
fn f_tails<T: Real>(x: T, d1: T, d2: T) -> (T, T) {
    let a = d1 * T::from_f64(0.5);
    let b = d2 * T::from_f64(0.5);
    let log = x.ln() + d1.ln() - d2.ln();
    if log <= T::ZERO {
        let v = log.exp();
        let c = beta_inc(a, b, v / (T::ONE + v));
        (c, T::ONE - c)
    } else {
        let v = (-log).exp();
        let s = beta_inc(b, a, v / (T::ONE + v));
        (T::ONE - s, s)
    }
}
/// F-Dichte mit `d1,d2>0` Freiheitsgraden.
pub fn f_pdf<T: Real>(x: T, d1: T, d2: T) -> T {
    if !positive(d1) || !positive(d2) || x.is_nan() {
        return T::NAN;
    }
    if x < T::ZERO || x == T::INFINITY {
        return T::ZERO;
    }
    if x == T::ZERO {
        return if d1 < T::from_f64(2.0) {
            T::INFINITY
        } else if d1 == T::from_f64(2.0) {
            T::ONE
        } else {
            T::ZERO
        };
    }
    let a = d1 * T::from_f64(0.5);
    let b = d2 * T::from_f64(0.5);
    let log = x.ln() + d1.ln() - d2.ln();
    let (lz, lq) = if log <= T::ZERO {
        let l = log.exp().ln_1p();
        (log - l, -l)
    } else {
        let l = (-log).exp().ln_1p();
        (-l, -log - l)
    };
    (a * lz + b * lq - x.ln() - ln_beta(a, b)).exp()
}
/// CDF der F-Verteilung.
pub fn f_cdf<T: Real>(x: T, d1: T, d2: T) -> T {
    if !positive(d1) || !positive(d2) || x.is_nan() {
        return T::NAN;
    }
    if x <= T::ZERO {
        return T::ZERO;
    }
    if x == T::INFINITY {
        return T::ONE;
    }
    f_tails(x, d1, d2).0
}
/// Direkter oberer Tail der F-Verteilung.
pub fn f_sf<T: Real>(x: T, d1: T, d2: T) -> T {
    if !positive(d1) || !positive(d2) || x.is_nan() {
        return T::NAN;
    }
    if x <= T::ZERO {
        return T::ONE;
    }
    if x == T::INFINITY {
        return T::ZERO;
    }
    f_tails(x, d1, d2).1
}
/// Quantil der F-Verteilung.
pub fn f_quantile<T: Real>(p: T, d1: T, d2: T) -> T {
    if !positive(d1) || !positive(d2) || !probability(p) {
        return T::NAN;
    }
    if p == T::ZERO {
        return T::ZERO;
    }
    if p == T::ONE {
        return T::INFINITY;
    }
    if p <= T::from_f64(0.5) {
        invert(|x| f_cdf(x, d1, d2), p, false, false)
    } else {
        invert(|x| f_sf(x, d1, d2), T::ONE - p, true, false)
    }
}
/// Inverser oberer Tail der F-Verteilung.
pub fn f_isf<T: Real>(q: T, d1: T, d2: T) -> T {
    if !positive(d1) || !positive(d2) || !probability(q) {
        return T::NAN;
    }
    if q == T::ZERO {
        return T::INFINITY;
    }
    if q == T::ONE {
        return T::ZERO;
    }
    if q <= T::from_f64(0.5) {
        invert(|x| f_sf(x, d1, d2), q, true, false)
    } else {
        invert(|x| f_cdf(x, d1, d2), T::ONE - q, false, false)
    }
}
