//! Tests für Verteilungen, Hypothesentests, Regression und die neuen Spezialfunktionen.
#![cfg(any(feature = "std", feature = "libm"))]
#![allow(
    clippy::needless_range_loop,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::excessive_precision
)]

use azmath::random::{Pcg32, Rng};
use azmath::special::{bessel_i, elliptic_e, elliptic_k};
use azmath::stats::dist::*;
use azmath::stats::*;
use core::f64::consts::PI;

#[test]
fn table_values() {
    assert!((normal_cdf_with(1.96f64, 0.0, 1.0) - 0.9750021).abs() < 1e-6);
    assert!((normal_quantile_with(0.975f64, 0.0, 1.0) - 1.959964).abs() < 1e-5);
    assert!((student_t_quantile(0.975f64, 10.0) - 2.228139).abs() < 1e-5);
    assert!((student_t_cdf(2.228139f64, 10.0) - 0.975).abs() < 1e-6);
    assert!((chi2_cdf(3.841459f64, 1.0) - 0.95).abs() < 1e-6);
    assert!((chi2_quantile(0.95f64, 5.0) - 11.0705).abs() < 1e-3);
    assert!((f_quantile(0.95f64, 3.0, 10.0) - 3.7083).abs() < 1e-3);
    assert!((exponential_cdf(1.0f64, 2.0) - (1.0 - (-2.0f64).exp())).abs() < 1e-14);
    assert!((gamma_cdf(2.0f64, 1.0, 1.0) - (1.0 - (-2.0f64).exp())).abs() < 1e-12);
    assert!((beta_cdf(0.5f64, 2.0, 2.0) - 0.5).abs() < 1e-12);
    assert_eq!(uniform_cdf(0.25f64, 0.0, 1.0), 0.25);
}

#[test]
fn quantile_inverts_cdf() {
    for &p in &[0.01f64, 0.2, 0.5, 0.8, 0.99] {
        assert!((gamma_cdf(gamma_quantile(p, 2.5, 1.5), 2.5, 1.5) - p).abs() < 1e-10);
        assert!((beta_cdf(beta_quantile(p, 2.0, 5.0), 2.0, 5.0) - p).abs() < 1e-10);
        assert!((student_t_cdf(student_t_quantile(p, 4.0), 4.0) - p).abs() < 1e-10);
        assert!((f_cdf(f_quantile(p, 4.0, 7.0), 4.0, 7.0) - p).abs() < 1e-10);
        assert!((exponential_cdf(exponential_quantile(p, 3.0), 3.0) - p).abs() < 1e-12);
        assert!((normal_cdf_with(normal_quantile_with(p, 1.0, 2.0), 1.0, 2.0) - p).abs() < 1e-10);
    }
    assert!(student_t_quantile(0.0f64, 3.0).is_infinite());
    assert!(chi2_quantile(1.5f64, 3.0).is_nan());
    assert!(normal_pdf(0.0f64, 0.0, -1.0).is_nan());
}

#[test]
fn densities_integrate_to_one() {
    let n = 20000;
    let h = 20.0 / n as f64;
    let integral =
        |f: &dyn Fn(f64) -> f64| (0..n).map(|i| f(-10.0 + (i as f64 + 0.5) * h)).sum::<f64>() * h;
    assert!((integral(&|x| normal_pdf(x, 0.0, 1.0)) - 1.0).abs() < 1e-6);
    assert!((integral(&|x| student_t_pdf(x, 5.0)) - 1.0).abs() < 1e-4);
    let h2 = 40.0 / n as f64;
    let int_pos =
        |f: &dyn Fn(f64) -> f64| (0..n).map(|i| f((i as f64 + 0.5) * h2)).sum::<f64>() * h2;
    assert!((int_pos(&|x| chi2_pdf(x, 4.0)) - 1.0).abs() < 1e-4);
    assert!((int_pos(&|x| f_pdf(x, 5.0, 8.0)) - 1.0).abs() < 2e-3);
}

#[test]
fn t_tests() {
    let x = [5.1f64, 4.9, 5.6, 5.8, 6.0, 5.3, 5.5, 4.7];
    let r = t_test_one_sample(&x, 5.0);
    assert!((r.df - 7.0).abs() < 1e-12);
    assert!(r.p_value > 0.0 && r.p_value < 0.2);
    let r0 = t_test_one_sample(&x, mean(&x));
    assert!(r0.statistic.abs() < 1e-12 && (r0.p_value - 1.0).abs() < 1e-9);
    let mut rng = Pcg32::new(11, 5);
    let a: Vec<f64> = (0..200).map(|_| rng.normal_with(0.0, 1.0)).collect();
    let b: Vec<f64> = (0..200).map(|_| rng.normal_with(1.0, 1.5)).collect();
    let c: Vec<f64> = (0..200).map(|_| rng.normal_with(0.0, 1.0)).collect();
    assert!(t_test_welch(&a, &b).p_value < 1e-6);
    assert!(t_test_welch(&a, &c).p_value > 1e-3);
}

#[test]
fn chi_square_and_ks() {
    let r = chi_square_gof(&[16.0f64, 18.0, 16.0, 14.0, 12.0, 12.0], &[14.667; 6], 0);
    assert!((r.df - 5.0).abs() < 1e-12 && r.p_value > 0.5);
    let r = chi_square_gof(&[50.0f64, 10.0, 10.0, 10.0], &[20.0; 4], 0);
    assert!(r.p_value < 1e-6);
    assert!(chi_square_gof(&[1.0f64], &[1.0], 0).statistic.is_nan());
    #[cfg(feature = "alloc")]
    {
        let mut rng = Pcg32::new(3, 9);
        let u: Vec<f64> = (0..500).map(|_| rng.uniform()).collect();
        let ok = ks_test_one_sample(&u, |x| uniform_cdf(x, 0.0, 1.0));
        assert!(ok.p_value > 0.01 && ok.statistic < 0.1);
        let bad = ks_test_one_sample(&u, |x| uniform_cdf(x, 0.0, 0.5));
        assert!(bad.p_value < 1e-6);
    }
}

#[test]
fn regression() {
    let x: Vec<f64> = (0..20).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| 3.0 + 2.0 * v).collect();
    let r = linear_regression(&x, &y).unwrap();
    assert!((r.slope - 2.0).abs() < 1e-12 && (r.intercept - 3.0).abs() < 1e-11);
    assert!((r.r_squared - 1.0).abs() < 1e-12 && r.residual_std < 1e-6);
    let mut rng = Pcg32::new(21, 1);
    let yn: Vec<f64> = x
        .iter()
        .map(|&v| 3.0 + 2.0 * v + rng.normal_with(0.0, 1.0))
        .collect();
    let r = linear_regression(&x, &yn).unwrap();
    let (lo, hi) = r.slope_ci(0.99);
    assert!(lo < 2.0 && 2.0 < hi, "{lo} {hi}");
    let (lo, hi) = r.intercept_ci(0.99);
    assert!(lo < 3.0 && 3.0 < hi, "{lo} {hi}");
    assert!(linear_regression(&[1.0f64, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_none());
    assert!(linear_regression(&[1.0f64, 2.0], &[1.0, 2.0]).is_none());
}

#[test]
fn elliptic_and_bessel_i() {
    assert!((elliptic_k(0.0f64) - PI / 2.0).abs() < 1e-14);
    assert!((elliptic_e(0.0f64) - PI / 2.0).abs() < 1e-14);
    assert!((elliptic_k(0.5f64) - 1.854074677301372).abs() < 1e-13);
    assert!((elliptic_e(0.5f64) - 1.350643881047675).abs() < 1e-13);
    assert!(elliptic_k(1.0f64).is_infinite() && (elliptic_e(1.0f64) - 1.0).abs() < 1e-15);
    assert!(elliptic_k(1.5f64).is_nan());
    assert!((bessel_i(0, 1.0f64) - 1.2660658777520084).abs() < 1e-13);
    assert!((bessel_i(1, 1.0f64) - 0.5651591039924851).abs() < 1e-13);
    assert!((bessel_i(2, 2.0f64) - 0.6889484476987382).abs() < 1e-13);
    assert_eq!(bessel_i(1, -1.0f64), -bessel_i(1, 1.0f64));
}

#[test]
fn bessel_y_and_k() {
    use azmath::special::{bessel_k, bessel_y};
    assert!((bessel_k(0, 1.0f64) - 0.42102443824070834).abs() < 1e-12);
    assert!((bessel_k(1, 1.0f64) - 0.6019072301972346).abs() < 1e-12);
    assert!((bessel_k(2, 2.0f64) - 0.25375975456605586).abs() < 1e-12);
    assert!((bessel_k(0, 0.1f64) - 2.4270690247020164).abs() < 1e-11);
    assert!(bessel_k(0, 0.0f64).is_infinite() && bessel_k(0, -1.0f64).is_nan());
    assert!((bessel_y(0, 1.0f64) - 0.08825696421567697).abs() < 1e-9);
    assert!((bessel_y(1, 1.0f64) + 0.7812128213002887).abs() < 1e-9);
    assert!((bessel_y(2, 3.0f64) - (-0.16040039348492374)).abs() < 1e-9);
    assert!((bessel_y(0, 10.0f64) - 0.05567116728359939).abs() < 1e-9);
    assert!(bessel_y(0, 0.0f64) == f64::NEG_INFINITY && bessel_y(1, -2.0f64).is_nan());
}
