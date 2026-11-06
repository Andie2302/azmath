//! Hypothesentests und lineare Regression.

use super::dist::{chi2_sf, student_t_quantile, student_t_sf};
use super::{mean, variance};
use crate::Real;

/// Ergebnis eines t-Tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TTestResult<T> {
    /// Teststatistik `t`.
    pub statistic: T,
    /// Freiheitsgrade (beim Welch-Test gebrochen).
    pub df: T,
    /// Zweiseitiger p-Wert.
    pub p_value: T,
}

fn two_sided_p<T: Real>(t: T, df: T) -> T {
    T::from_f64(2.0) * student_t_sf(t.abs(), df)
}

/// Einstichproben-t-Test: `H₀: Erwartungswert = mu0` (zweiseitig). `NaN`-Ergebnisse bei weniger als 2 Werten.
pub fn t_test_one_sample<T: Real>(x: &[T], mu0: T) -> TTestResult<T> {
    let n = T::from_usize(x.len());
    let t = (mean(x) - mu0) / (variance(x) / n).sqrt();
    let df = n - T::ONE;
    TTestResult {
        statistic: t,
        df,
        p_value: two_sided_p(t, df),
    }
}

/// Zweistichproben-t-Test nach Welch (ungleiche Varianzen erlaubt), zweiseitig.
pub fn t_test_welch<T: Real>(x: &[T], y: &[T]) -> TTestResult<T> {
    let (nx, ny) = (T::from_usize(x.len()), T::from_usize(y.len()));
    let (vx, vy) = (variance(x) / nx, variance(y) / ny);
    let t = (mean(x) - mean(y)) / (vx + vy).sqrt();
    let df = (vx + vy) * (vx + vy) / (vx * vx / (nx - T::ONE) + vy * vy / (ny - T::ONE));
    TTestResult {
        statistic: t,
        df,
        p_value: two_sided_p(t, df),
    }
}

/// Ergebnis des χ²-Anpassungstests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChiSquareResult<T> {
    /// Teststatistik `Σ (Oᵢ − Eᵢ)²/Eᵢ`.
    pub statistic: T,
    /// Freiheitsgrade `k − 1 − fitted_params`.
    pub df: T,
    /// p-Wert (obere Schwanzwahrscheinlichkeit).
    pub p_value: T,
}

/// χ²-Anpassungstest: beobachtete gegen erwartete Häufigkeiten (`expected > 0`); `fitted_params` geschätzte Parameter.
/// `NaN` bei ungleicher Länge oder nicht positiven Freiheitsgraden.
pub fn chi_square_gof<T: Real>(
    observed: &[T],
    expected: &[T],
    fitted_params: usize,
) -> ChiSquareResult<T> {
    let k = observed.len();
    if k != expected.len() || k < 2 + fitted_params || expected.iter().any(|&e| !(e > T::ZERO)) {
        return ChiSquareResult {
            statistic: T::NAN,
            df: T::NAN,
            p_value: T::NAN,
        };
    }
    let mut s = T::ZERO;
    for i in 0..k {
        let d = observed[i] - expected[i];
        s += d * d / expected[i];
    }
    let df = T::from_usize(k - 1 - fitted_params);
    ChiSquareResult {
        statistic: s,
        df,
        p_value: chi2_sf(s, df),
    }
}

/// Ergebnis des Kolmogorow-Smirnow-Tests.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KsResult<T> {
    /// Größter Abstand `D` zwischen empirischer und theoretischer Verteilungsfunktion.
    pub statistic: T,
    /// Asymptotischer p-Wert.
    pub p_value: T,
}

/// Kolmogorow-Smirnow-Test einer Stichprobe gegen die Verteilungsfunktion `cdf` (asymptotischer p-Wert mit Stephens-Korrektur).
#[cfg(feature = "alloc")]
pub fn ks_test_one_sample<T: Real>(sample: &[T], cdf: impl Fn(T) -> T) -> KsResult<T> {
    let n = sample.len();
    if n == 0 || sample.iter().any(|v| v.is_nan()) {
        return KsResult {
            statistic: T::NAN,
            p_value: T::NAN,
        };
    }
    let mut s = alloc::vec::Vec::from(sample);
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    let nf = T::from_usize(n);
    let mut d = T::ZERO;
    for (i, &x) in s.iter().enumerate() {
        let f = cdf(x);
        let up = T::from_usize(i + 1) / nf - f;
        let down = f - T::from_usize(i) / nf;
        d = d.max(up).max(down);
    }
    let sq = nf.sqrt();
    let lambda = (sq + T::from_f64(0.12) + T::from_f64(0.11) / sq) * d;
    let mut p = T::ZERO;
    let mut sign = T::ONE;
    for k in 1..=100usize {
        let kf = T::from_usize(k);
        let term = (-T::from_f64(2.0) * kf * kf * lambda * lambda).exp();
        p += sign * term;
        sign = -sign;
        if term < T::EPSILON * T::from_f64(1.0e-3) {
            break;
        }
    }
    KsResult {
        statistic: d,
        p_value: (T::from_f64(2.0) * p).max(T::ZERO).min(T::ONE),
    }
}

/// Ergebnis der einfachen linearen Regression `y = intercept + slope·x`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegressionResult<T> {
    /// Steigung.
    pub slope: T,
    /// Achsenabschnitt.
    pub intercept: T,
    /// Bestimmtheitsmaß `R²`.
    pub r_squared: T,
    /// Standardfehler der Steigung.
    pub slope_se: T,
    /// Standardfehler des Achsenabschnitts.
    pub intercept_se: T,
    /// Standardabweichung der Residuen (`√(RSS/(n−2))`).
    pub residual_std: T,
    /// Freiheitsgrade `n − 2`.
    pub df: T,
}

impl<T: Real> RegressionResult<T> {
    /// Zweiseitiges Konfidenzintervall der Steigung zum Niveau `level` (zum Beispiel `0.95`).
    pub fn slope_ci(&self, level: T) -> (T, T) {
        let t = student_t_quantile((T::ONE + level) * T::from_f64(0.5), self.df);
        (
            self.slope - t * self.slope_se,
            self.slope + t * self.slope_se,
        )
    }

    /// Zweiseitiges Konfidenzintervall des Achsenabschnitts zum Niveau `level`.
    pub fn intercept_ci(&self, level: T) -> (T, T) {
        let t = student_t_quantile((T::ONE + level) * T::from_f64(0.5), self.df);
        (
            self.intercept - t * self.intercept_se,
            self.intercept + t * self.intercept_se,
        )
    }
}

/// Einfache lineare Regression durch kleinste Quadrate. `None` bei ungleicher Länge, weniger als 3 Punkten oder konstantem `x`.
pub fn linear_regression<T: Real>(x: &[T], y: &[T]) -> Option<RegressionResult<T>> {
    let n = x.len();
    if y.len() != n || n < 3 {
        return None;
    }
    let (mx, my) = (mean(x), mean(y));
    let (mut sxx, mut sxy, mut syy) = (T::ZERO, T::ZERO, T::ZERO);
    for i in 0..n {
        let (dx, dy) = (x[i] - mx, y[i] - my);
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if !(sxx > T::ZERO) {
        return None;
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let rss = (syy - slope * sxy).max(T::ZERO);
    let df = T::from_usize(n - 2);
    let s2 = rss / df;
    let nf = T::from_usize(n);
    Some(RegressionResult {
        slope,
        intercept,
        r_squared: if syy > T::ZERO {
            T::ONE - rss / syy
        } else {
            T::ONE
        },
        slope_se: (s2 / sxx).sqrt(),
        intercept_se: (s2 * (T::ONE / nf + mx * mx / sxx)).sqrt(),
        residual_std: s2.sqrt(),
        df,
    })
}
