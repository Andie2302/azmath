//! Tests für `random` und `stats`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::random::{Pcg32, Rng, Xoshiro256pp};
use azmath::stats::*;

#[test]
fn pcg32_reference_vector() {
    // Referenzwerte aus pcg32-demo (seed 42, stream 54)
    let mut r = Pcg32::new(42, 54);
    let got: Vec<u32> = (0..6).map(|_| r.next32()).collect();
    assert_eq!(
        got,
        [
            0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e
        ]
    );
}

#[test]
fn deterministic_and_different_seeds() {
    let mut a = Xoshiro256pp::new(1);
    let mut b = Xoshiro256pp::new(1);
    let mut c = Xoshiro256pp::new(2);
    let (x, y, z) = (a.next_u64(), b.next_u64(), c.next_u64());
    assert_eq!(x, y);
    assert_ne!(x, z);
}

#[test]
fn uniform_moments() {
    let mut r = Xoshiro256pp::new(2024);
    let xs: Vec<f64> = (0..20000).map(|_| r.uniform()).collect();
    assert!(xs.iter().all(|&u| (0.0..1.0).contains(&u)));
    assert!((mean(&xs) - 0.5).abs() < 0.01);
    assert!((variance(&xs) - 1.0 / 12.0).abs() < 0.005);
}

#[test]
fn normal_and_exponential_moments() {
    let mut r = Pcg32::new(7, 1);
    let xs: Vec<f64> = (0..40000).map(|_| r.normal_with(3.0, 2.0)).collect();
    assert!((mean(&xs) - 3.0).abs() < 0.05);
    assert!((std_dev(&xs) - 2.0).abs() < 0.05);
    let es: Vec<f64> = (0..40000).map(|_| r.exponential(4.0)).collect();
    assert!((mean(&es) - 0.25).abs() < 0.01);
}

#[test]
fn below_and_shuffle() {
    let mut r = Pcg32::new(3, 3);
    let mut counts = [0usize; 6];
    for _ in 0..6000 {
        counts[r.below(6) as usize] += 1;
    }
    assert!(counts.iter().all(|&c| (800..1200).contains(&c)));
    let mut v: Vec<u32> = (0..50).collect();
    r.shuffle(&mut v);
    let mut s = v.clone();
    s.sort();
    assert_eq!(s, (0..50).collect::<Vec<u32>>());
    assert_ne!(v, s);
}

#[test]
fn descriptive_statistics() {
    let x = [2.0f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    assert_eq!(mean(&x), 5.0);
    assert!((variance(&x) - 32.0 / 7.0).abs() < 1e-14);
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    assert!((correlation(&x, &y) - 1.0).abs() < 1e-14);
    assert!((covariance(&x, &y) - 2.0 * variance(&x)).abs() < 1e-13);
    assert!(mean::<f64>(&[]).is_nan());
    assert!(variance(&[1.0f64]).is_nan());
    assert!(covariance(&[1.0f64, 2.0], &[1.0]).is_nan());
}

#[test]
fn quantiles() {
    let s = [1.0f64, 2.0, 3.0, 4.0];
    assert_eq!(quantile_sorted(&s, 0.0), 1.0);
    assert_eq!(quantile_sorted(&s, 1.0), 4.0);
    assert!((quantile_sorted(&s, 0.5) - 2.5).abs() < 1e-14);
    #[cfg(feature = "alloc")]
    assert_eq!(median(&[5.0, 1.0, 3.0]), 3.0);
    assert!(quantile_sorted(&s, 1.5).is_nan());
}

#[test]
fn monte_carlo_integration() {
    let mut r = Xoshiro256pp::new(99);
    let mut w = [0.0; 2];
    // ∫∫ x·y über [0,2]² = 4
    let (est, err) = mc_integrate(
        |p: &[f64]| p[0] * p[1],
        &[0.0, 0.0],
        &[2.0, 2.0],
        100_000,
        &mut r,
        &mut w,
    );
    assert!((est - 4.0).abs() < 5.0 * err.max(1e-3), "{est} ± {err}");
    assert!(err > 0.0 && err < 0.1);
}
