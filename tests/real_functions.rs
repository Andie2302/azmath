//! Tests für die zusätzlichen `Real`-Methoden (`copysign`, `exp_m1`, `ln_1p`, `asinh`, `rem_euclid`, `to_f64`, …).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::Real;

fn close(a: f64, b: f64, rel: f64) {
    assert!(
        (a - b).abs() <= rel * b.abs().max(1e-300),
        "{a:e} gegen {b:e}"
    );
}

#[test]
fn finite_copysign_and_conversions() {
    assert!(Real::is_finite(1.5f64));
    assert!(!Real::is_finite(f64::NAN) && !Real::is_finite(f64::INFINITY));
    assert_eq!(Real::copysign(3.0f64, -0.5), -3.0);
    assert_eq!(Real::copysign(-3.0f64, 2.0), 3.0);
    assert!(Real::copysign(1.0f64, -0.0).is_sign_negative());
    assert_eq!(<f64 as Real>::from_usize(7), 7.0);
    assert_eq!(Real::to_f64(2.5f32), 2.5);
    assert_eq!(Real::to_f64(0.1f64), 0.1);
    let (s, c) = Real::sin_cos(0.3f64);
    assert_eq!((s, c), (0.3f64.sin(), 0.3f64.cos()));
}

#[test]
fn exp_m1_and_ln_1p_are_accurate_for_small_arguments() {
    for x in [1e-20, 1e-10, 1e-5, 0.1, -1e-12, -0.5, 3.0] {
        close(Real::exp_m1(x), x.exp_m1(), 1e-13);
        close(Real::ln_1p(x), x.ln_1p(), 1e-13);
    }
    assert_eq!(Real::exp_m1(0.0f64), 0.0);
    assert_eq!(Real::ln_1p(0.0f64), 0.0);
    assert_eq!(Real::exp_m1(-800.0f64), -1.0);
    assert!(Real::ln_1p(-1.0f64).is_infinite());
    assert!(Real::ln_1p(-2.0f64).is_nan());
}

#[test]
fn inverse_hyperbolic_functions() {
    for x in [-50.0, -1.0, -1e-9, 0.0, 1e-9, 0.5, 2.0, 1e3, 1e12] {
        close(Real::asinh(x), x.asinh(), 1e-12);
        close(Real::sinh(Real::asinh(x)), x, 1e-9);
    }
    for x in [1.0, 1.0 + 1e-12, 2.0, 100.0, 1e12] {
        close(Real::acosh(x), x.acosh(), 1e-12);
    }
    assert!(Real::acosh(0.5f64).is_nan());
    for x in [-0.99, -0.5, 1e-9, 0.3, 0.999] {
        close(Real::atanh(x), x.atanh(), 1e-12);
    }
    assert!(Real::atanh(2.0f64).is_nan());
}

#[test]
fn powers_logs_and_euclidean_remainder() {
    close(Real::exp2(10.0f64), 1024.0, 1e-14);
    close(Real::exp2(0.5f64), 2f64.sqrt(), 1e-14);
    close(Real::log(8.0f64, 2.0), 3.0, 1e-14);
    assert_eq!(Real::rem_euclid(7.0f64, 3.0), 1.0);
    assert_eq!(Real::rem_euclid(-7.0f64, 3.0), 2.0);
    assert_eq!(Real::rem_euclid(7.0f64, -3.0), 1.0);
    assert_eq!(Real::rem_euclid(-7.5f64, 2.0), 0.5);
    for x in [-9.3, -0.1, 0.0, 2.2, 11.9] {
        let r = Real::rem_euclid(x, 4.0f64);
        assert!((0.0..4.0).contains(&r), "{x}: {r}");
    }
}

#[test]
fn new_methods_work_for_f32() {
    assert!((Real::exp_m1(1e-8f32) - 1e-8).abs() < 1e-12);
    assert!((Real::asinh(1.0f32) - 0.881_373_6).abs() < 1e-6);
    assert!((Real::ln_1p(1e-8f32) - 1e-8).abs() < 1e-12);
}

#[test]
fn jets_get_correct_derivatives_from_the_default_methods() {
    use azmath::autodiff::Jet;
    let x = Jet::<f64, 1>::variable(0.7, 0);
    // d/dx asinh = 1/√(1+x²), d/dx atanh = 1/(1−x²), d/dx exp_m1 = eˣ, d/dx ln_1p = 1/(1+x)
    close(Real::asinh(x).d[0], 1.0 / (1.0 + 0.49f64).sqrt(), 1e-10);
    close(
        Real::atanh(Jet::<f64, 1>::variable(0.4, 0)).d[0],
        1.0 / (1.0 - 0.16),
        1e-10,
    );
    close(Real::exp_m1(x).d[0], 0.7f64.exp(), 1e-10);
    close(Real::ln_1p(x).d[0], 1.0 / 1.7, 1e-10);
    assert_eq!(Real::to_f64(x), 0.7);
}
