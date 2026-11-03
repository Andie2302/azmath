//! Tanh-Sinh-Quadratur für Endpunktsingularitäten sowie halb- und ganz unendliche Intervalle.

use super::{IntegralResult, IntegrateError, finite};
use crate::Real;

/// Integral von `f` über `[a, b]` mit der doppelt-exponentiellen Tanh-Sinh-Regel auf relative Genauigkeit `tol`.
///
/// Wertet `f` nie an den Endpunkten aus; algebraische und logarithmische Singularitäten dort sind erlaubt
/// (`∫₀¹ 1/√x dx`, `∫₀¹ ln x dx`). Nicht endliche Funktionswerte werden übergangen. Das Intervall wird bis zu 10-mal
/// verfeinert.
///
/// Fehler: [`IntegrateError::NotFinite`] (Grenzen), [`IntegrateError::InvalidInput`] (`a ≥ b`),
/// [`IntegrateError::NoConvergence`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::integrate::tanh_sinh;
/// let r = tanh_sinh(|x: f64| 1.0 / x.sqrt(), 0.0, 1.0, 1e-10).unwrap();
/// assert!((r.value - 2.0).abs() < 1e-8);
/// # }
/// ```
pub fn tanh_sinh<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    tol: T,
) -> Result<IntegralResult<T>, IntegrateError> {
    if !finite(a) || !finite(b) {
        return Err(IntegrateError::NotFinite);
    }
    if !(a < b) {
        return Err(IntegrateError::InvalidInput);
    }
    let half = T::from_f64(0.5);
    let c = (a + b) * half;
    let hw = (b - a) * half;
    let pi2 = T::PI * half;
    let t_max = T::from_f64(6.0);
    let mut evals = 0usize;

    // Beitrag der Parameter ±t (t > 0: zwei Punkte symmetrisch zu den Endpunkten; t = 0: Mittelpunkt)
    let mut term = |t: T, f: &mut dyn FnMut(T) -> T| -> T {
        let u = pi2 * t.sinh();
        let cu = u.cosh();
        let w = pi2 * t.cosh() / (cu * cu);
        let mut s = T::ZERO;
        if t == T::ZERO {
            evals += 1;
            let v = f(c);
            if finite(v) {
                s += w * v * hw;
            }
            return s;
        }
        let dist = hw * (-u).exp() / cu; // Abstand zum Endpunkt
        for x in [a + dist, b - dist] {
            if x <= a || x >= b {
                continue;
            }
            evals += 1;
            let v = f(x);
            if finite(v) {
                s += w * v * hw;
            }
        }
        s
    };

    let mut h = T::ONE;
    let mut sum = term(T::ZERO, &mut f);
    let mut k = T::ONE;
    while k * h <= t_max {
        sum += term(k * h, &mut f);
        k += T::ONE;
    }
    let mut prev = sum * h;
    let mut est = prev;
    for _ in 0..10 {
        h = h * half;
        let mut k = T::ONE;
        let mut add = T::ZERO;
        while k * h <= t_max {
            add += term(k * h, &mut f);
            k += T::from_f64(2.0);
        }
        sum += add;
        est = sum * h;
        let err = (est - prev).abs();
        if err <= tol * est.abs().max(T::EPSILON) || err <= T::EPSILON * T::from_f64(4.0) {
            return Ok(IntegralResult {
                value: est,
                error_estimate: err,
                evaluations: evals,
            });
        }
        prev = est;
    }
    let _ = est;
    Err(IntegrateError::NoConvergence)
}

/// Integral von `f` über `[a, ∞)` (Substitution `x = a + t/(1−t)` und [`tanh_sinh`]).
pub fn integrate_semi_infinite<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    tol: T,
) -> Result<IntegralResult<T>, IntegrateError> {
    tanh_sinh(
        |t| {
            let om = T::ONE - t;
            f(a + t / om) / (om * om)
        },
        T::ZERO,
        T::ONE,
        tol,
    )
}

/// Integral von `f` über `(−∞, ∞)` (Substitution `x = t/(1−t²)` und [`tanh_sinh`]).
pub fn integrate_infinite<T: Real>(
    mut f: impl FnMut(T) -> T,
    tol: T,
) -> Result<IntegralResult<T>, IntegrateError> {
    tanh_sinh(
        |t| {
            let d = T::ONE - t * t;
            f(t / d) * (T::ONE + t * t) / (d * d)
        },
        -T::ONE,
        T::ONE,
        tol,
    )
}
