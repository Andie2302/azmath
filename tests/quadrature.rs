//! Tests für Gauß-Regeln, Tanh-Sinh und Kubatur.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::integrate::*;
use core::f64::consts::PI;

fn rule(family: Family<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut x, mut w, mut work) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    gauss_rule(family, n, &mut x, &mut w, &mut work).unwrap();
    (x, w)
}

fn moment(x: &[f64], w: &[f64], k: i32) -> f64 {
    x.iter().zip(w).map(|(x, w)| w * x.powi(k)).sum()
}

#[test]
fn legendre_exactness() {
    for n in [1usize, 2, 5, 10, 20] {
        let (x, w) = rule(Family::Legendre, n);
        for k in 0..(2 * n as i32) {
            let exact = if k % 2 == 1 {
                0.0
            } else {
                2.0 / (k as f64 + 1.0)
            };
            assert!(
                (moment(&x, &w, k) - exact).abs() < 1e-11,
                "n = {n}, k = {k}"
            );
        }
        assert!(x.windows(2).all(|p| p[0] < p[1]));
    }
    // bekannte 2-Punkte-Regel
    let (x, w) = rule(Family::Legendre, 2);
    assert!((x[1] - 1.0 / 3f64.sqrt()).abs() < 1e-14 && (w[0] - 1.0).abs() < 1e-14);
}

#[test]
fn laguerre_hermite_moments() {
    let (x, w) = rule(Family::Laguerre(0.0), 6);
    for k in 0..12 {
        let fact: f64 = (1..=k).map(|i| i as f64).product();
        assert!((moment(&x, &w, k) / fact - 1.0).abs() < 1e-9, "k = {k}");
    }
    let (x, w) = rule(Family::Hermite, 8);
    assert!((moment(&x, &w, 0) - PI.sqrt()).abs() < 1e-12);
    assert!((moment(&x, &w, 2) - PI.sqrt() / 2.0).abs() < 1e-12);
    assert!((moment(&x, &w, 4) - 3.0 * PI.sqrt() / 4.0).abs() < 1e-11);
    assert!(moment(&x, &w, 3).abs() < 1e-11);
}

#[test]
fn chebyshev_and_jacobi() {
    let (x, w) = rule(Family::ChebyshevT, 7);
    assert!((moment(&x, &w, 0) - PI).abs() < 1e-12);
    assert!((moment(&x, &w, 2) - PI / 2.0).abs() < 1e-12);
    let (x, w) = rule(Family::ChebyshevU, 7);
    assert!((moment(&x, &w, 0) - PI / 2.0).abs() < 1e-12);
    // Jacobi(0, 0) = Legendre
    let (xj, wj) = rule(Family::Jacobi(0.0, 0.0), 6);
    let (xl, wl) = rule(Family::Legendre, 6);
    for i in 0..6 {
        assert!((xj[i] - xl[i]).abs() < 1e-12 && (wj[i] - wl[i]).abs() < 1e-12);
    }
    // Jacobi(1, 0): ∫(1−x) dx = 2, ∫(1−x)·x dx = −2/3
    let (x, w) = rule(Family::Jacobi(1.0, 0.0), 5);
    assert!((moment(&x, &w, 0) - 2.0).abs() < 1e-12);
    assert!((moment(&x, &w, 1) + 2.0 / 3.0).abs() < 1e-12);
}

#[test]
fn gauss_rule_errors() {
    let (mut x, mut w, mut work) = ([0.0; 3], [0.0; 3], [0.0; 3]);
    assert!(gauss_rule(Family::Legendre, 0, &mut x, &mut w, &mut work).is_err());
    assert!(gauss_rule(Family::Legendre, 5, &mut x, &mut w, &mut work).is_err());
    assert!(gauss_rule(Family::Laguerre(-1.5), 3, &mut x, &mut w, &mut work).is_err());
}

#[test]
fn tanh_sinh_singularities() {
    let r = tanh_sinh(|x: f64| 1.0 / x.sqrt(), 0.0, 1.0, 1e-10).unwrap();
    assert!((r.value - 2.0).abs() < 1e-8);
    let r = tanh_sinh(|x: f64| x.ln(), 0.0, 1.0, 1e-10).unwrap();
    assert!((r.value + 1.0).abs() < 1e-8);
    let r = tanh_sinh(|x: f64| x.sin(), 0.0, PI, 1e-12).unwrap();
    assert!((r.value - 2.0).abs() < 1e-10);
    assert!(tanh_sinh(|x: f64| x, 1.0, 0.0, 1e-8).is_err());
}

#[test]
fn infinite_intervals() {
    let r = integrate_semi_infinite(|x: f64| (-x).exp(), 0.0, 1e-10).unwrap();
    assert!((r.value - 1.0).abs() < 1e-7);
    let r = integrate_infinite(|x: f64| (-x * x).exp(), 1e-10).unwrap();
    assert!((r.value - PI.sqrt()).abs() < 1e-7);
    let r = integrate_semi_infinite(|x: f64| 1.0 / (1.0 + x * x), 0.0, 1e-9).unwrap();
    assert!((r.value - PI / 2.0).abs() < 1e-6);
}

#[test]
fn tensor_product() {
    let mut work = [0.0; tensor_gauss_work_len(3, 4)];
    // ∫∫∫ x·y²·z³ über [0,1]×[0,2]×[1,2]
    let v = tensor_gauss(
        |p: &[f64]| p[0] * p[1] * p[1] * p[2].powi(3),
        &[0.0, 0.0, 1.0],
        &[1.0, 2.0, 2.0],
        4,
        &mut work,
    )
    .unwrap();
    let exact = 0.5 * (8.0 / 3.0) * (15.0 / 4.0);
    assert!((v - exact).abs() < 1e-12);
    let mut work = [0.0; tensor_gauss_work_len(2, 12)];
    let v = tensor_gauss(
        |p: &[f64]| (p[0] + p[1]).exp(),
        &[0.0, 0.0],
        &[1.0, 1.0],
        12,
        &mut work,
    )
    .unwrap();
    let e1 = core::f64::consts::E - 1.0;
    assert!((v - e1 * e1).abs() < 1e-12);
    assert!(tensor_gauss(|_: &[f64]| 1.0, &[], &[], 3, &mut work).is_err());
}
