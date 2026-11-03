//! Tests für die Spezialfunktionen. Die Vergleichswerte stammen aus Standardtabellen (Abramowitz/Stegun, NIST DLMF).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::special::{
    bessel_j, beta, beta_inc, digamma, erf, erf_inv, erfc, gamma, gamma_p, gamma_q, ln_beta,
    ln_gamma, normal_cdf, normal_quantile,
};
use core::f64::consts::PI;

fn rel(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs().max(1e-300),
        "{a:e} gegen {b:e}"
    );
}

const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

#[test]
fn gamma_function() {
    rel(gamma(5.0), 24.0, 1e-14);
    rel(gamma(1.0), 1.0, 1e-14);
    rel(gamma(0.5), PI.sqrt(), 1e-14);
    rel(gamma(1.5), PI.sqrt() / 2.0, 1e-14);
    rel(gamma(-0.5), -2.0 * PI.sqrt(), 1e-13);
    rel(gamma(10.5), 1_133_278.388_948_785_5, 1e-13);
    rel(gamma(20.0), 121_645_100_408_832_000.0, 1e-14);
    assert!(gamma(0.0f64).is_nan() && gamma(-3.0f64).is_nan());
    assert!(gamma(172.0f64).is_infinite());
    // Funktionalgleichung Γ(x+1) = x·Γ(x)
    for x in [0.3, 1.7, 4.2, 25.5, 60.1] {
        rel(gamma(x + 1.0), x * gamma(x), 1e-13);
    }
}

#[test]
fn log_gamma_and_beta() {
    rel(ln_gamma(100.0), 359.134_205_369_575_4, 1e-14);
    rel(ln_gamma(0.5), (PI.sqrt()).ln(), 1e-13);
    rel(ln_gamma(1.0) + 1.0, 1.0, 1e-13);
    rel(ln_gamma(1000.0), 5_905.220_423_209_181, 1e-14);
    assert!(ln_gamma(0.0f64).is_infinite() && ln_gamma(-2.0f64).is_infinite());
    rel(ln_gamma(-0.5), (2.0 * PI.sqrt()).ln(), 1e-13);
    rel(beta(2.0, 3.0), 1.0 / 12.0, 1e-13);
    rel(ln_beta(0.5, 0.5), PI.ln(), 1e-13);
    rel(beta(1.0, 1.0), 1.0, 1e-13);
}

#[test]
fn digamma_function() {
    rel(digamma(1.0), -EULER_GAMMA, 1e-13);
    rel(digamma(0.5), -EULER_GAMMA - 2.0 * 2f64.ln(), 1e-13);
    rel(digamma(10.0), 2.251_752_589_066_721, 1e-13);
    // Rekursion ψ(x+1) = ψ(x) + 1/x und Reflexion
    for x in [0.2, 1.3, 7.5, -0.7] {
        rel(digamma(x + 1.0), digamma(x) + 1.0 / x, 1e-12);
    }
    assert!(digamma(0.0f64).is_nan() && digamma(-1.0f64).is_nan());
}

#[test]
fn error_function() {
    rel(erf(0.5), 0.520_499_877_813_046_5, 1e-14);
    rel(erf(1.0), 0.842_700_792_949_714_9, 1e-14);
    rel(erf(3.0), 0.999_977_909_503_001_4, 1e-14);
    assert_eq!(erf(0.0), 0.0);
    rel(erf(-1.0), -0.842_700_792_949_714_9, 1e-14);
    rel(erfc(2.0), 0.004_677_734_981_047_266, 1e-13);
    rel(erfc(5.0), 1.537_459_794_428_035e-12, 1e-12);
    rel(erfc(-1.0), 1.842_700_792_949_715, 1e-14);
    // erf + erfc = 1
    for x in [-2.0, -0.3, 0.1, 1.5, 2.5] {
        rel(erf(x) + erfc(x), 1.0, 1e-14);
    }
    assert!(erf(f64::NAN).is_nan());
}

#[test]
fn inverse_error_function_and_the_normal_distribution() {
    for y in [-0.999, -0.5, -1e-6, 1e-9, 0.3, 0.9, 0.999_999] {
        rel(erf(erf_inv(y)), y, 1e-13);
    }
    assert!(erf_inv(1.0f64).is_infinite() && erf_inv(-1.0f64).is_infinite());
    assert!(erf_inv(1.5f64).is_nan());
    rel(normal_cdf(1.96), 0.975_002_104_851_779_5, 1e-13);
    rel(normal_cdf(0.0), 0.5, 1e-15);
    rel(normal_cdf(-3.0), 0.001_349_898_031_630_094_6, 1e-12);
    rel(normal_quantile(0.975), 1.959_963_984_540_054, 1e-12);
    assert_eq!(normal_quantile(0.5), 0.0);
    for p in [0.001, 0.2, 0.5, 0.9, 0.999] {
        rel(normal_cdf(normal_quantile(p)), p, 1e-12);
    }
    assert!(normal_quantile(1.2f64).is_nan());
}

#[test]
fn incomplete_gamma_and_beta() {
    // P(2, 1) = 1 − 2/e,  P(1, x) = 1 − e^{−x}
    rel(gamma_p(2.0, 1.0), 1.0 - 2.0 / 1f64.exp(), 1e-13);
    rel(gamma_p(1.0, 3.0), 1.0 - (-3.0f64).exp(), 1e-13);
    rel(gamma_q(0.5, 0.5), 0.317_310_507_862_914_04, 1e-12);
    for (a, x) in [
        (0.7, 0.3),
        (3.0, 2.0),
        (3.0, 9.0),
        (10.0, 4.0),
        (10.0, 25.0),
    ] {
        rel(gamma_p(a, x) + gamma_q(a, x), 1.0, 1e-13);
    }
    assert_eq!(gamma_p(2.0, 0.0), 0.0);
    assert_eq!(gamma_q(2.0, 0.0), 1.0);
    assert!(gamma_p(-1.0f64, 1.0).is_nan() && gamma_p(1.0f64, -1.0).is_nan());
    // I_x(2, 3) bei x = 0,4 ist 0,5248 (Binomialsumme); Symmetrie I_x(a, b) = 1 − I_{1−x}(b, a)
    rel(beta_inc(2.0, 3.0, 0.4), 0.5248, 1e-13);
    rel(beta_inc(1.0, 1.0, 0.37), 0.37, 1e-13);
    for (a, b, x) in [(0.5, 0.5, 0.3), (2.5, 4.0, 0.8), (7.0, 1.5, 0.1)] {
        rel(beta_inc(a, b, x) + beta_inc(b, a, 1.0 - x), 1.0, 1e-12);
    }
    assert_eq!(beta_inc(2.0, 2.0, 0.0), 0.0);
    assert_eq!(beta_inc(2.0, 2.0, 1.0), 1.0);
    assert!(beta_inc(2.0f64, 2.0, 1.5).is_nan());
}

#[test]
fn bessel_functions_of_the_first_kind() {
    rel(bessel_j(0, 1.0), 0.765_197_686_557_966_6, 1e-13);
    rel(bessel_j(1, 1.0), 0.440_050_585_744_933_5, 1e-13);
    rel(bessel_j(2, 2.0), 0.352_834_028_615_637_7, 1e-13);
    rel(bessel_j(0, 10.0), -0.245_935_764_451_348_3, 1e-12);
    rel(bessel_j(5, 10.0), -0.234_061_528_186_793_6, 1e-12);
    assert_eq!(bessel_j(0, 0.0), 1.0);
    assert_eq!(bessel_j(3, 0.0), 0.0);
    // Vorzeichen für negative Argumente: Jₙ(−x) = (−1)ⁿ·Jₙ(x)
    rel(bessel_j(1, -1.0), -bessel_j(1, 1.0), 1e-14);
    rel(bessel_j(2, -2.0), bessel_j(2, 2.0), 1e-14);
    // Rekursion J_{n−1} + J_{n+1} = (2n/x)·Jₙ
    for (n, x) in [(1u32, 3.7), (4, 12.0), (9, 2.2), (6, 40.0)] {
        let lhs = bessel_j(n - 1, x) + bessel_j(n + 1, x);
        let rhs = 2.0 * n as f64 / x * bessel_j(n, x);
        assert!((lhs - rhs).abs() < 1e-12, "n = {n}, x = {x}");
    }
    // hohe Ordnung: J₁₀(1) ≈ 2,63·10⁻¹⁰
    rel(bessel_j(10, 1.0), 2.630_615_123_687_453e-10, 1e-10);
}

#[test]
fn works_for_f32() {
    assert!((gamma(5.0f32) - 24.0).abs() < 1e-3);
    assert!((erf(1.0f32) - 0.842_700_8).abs() < 1e-5);
    assert!((ln_gamma(10.0f32) - 12.801_827).abs() < 1e-3);
}
