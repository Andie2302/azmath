//! Tests für round, trunc, fract, signum, log10/log2, hypot, mul_add,
//! Hyperbelfunktionen, cbrt, clamp, recip und Winkelumrechnung.
#![cfg(any(feature = "std", feature = "libm"))]

mod common;

use azmath_core::Real;
use common::assert_close_f32 as close;

#[test]
fn round_trunc_fract_f32() {
    assert_eq!(Real::round(2.5f32), 3.0);
    assert_eq!(Real::round(-2.5f32), -3.0);
    assert_eq!(Real::round(2.4f32), 2.0);
    assert_eq!(Real::trunc(2.9f32), 2.0);
    assert_eq!(Real::trunc(-2.9f32), -2.0);
    close(Real::fract(2.75f32), 0.75);
    close(Real::fract(-2.75f32), -0.75);
}

#[test]
fn signum_f32() {
    assert_eq!(Real::signum(5.0f32), 1.0);
    assert_eq!(Real::signum(-5.0f32), -1.0);
    assert_eq!(Real::signum(0.0f32), 1.0);
    assert_eq!(Real::signum(-0.0f32), -1.0);
    assert_eq!(Real::signum(f32::INFINITY), 1.0);
    assert_eq!(Real::signum(f32::NEG_INFINITY), -1.0);
    assert!(Real::signum(f32::NAN).is_nan());
}

#[test]
fn logarithms_f32() {
    close(Real::log10(1000.0f32), 3.0);
    close(Real::log10(1.0f32), 0.0);
    close(Real::log2(8.0f32), 3.0);
    close(Real::log2(1.0f32), 0.0);
}

#[test]
fn hypot_and_mul_add_f32() {
    close(Real::hypot(3.0f32, 4.0), 5.0);
    close(Real::hypot(5.0f32, 12.0), 13.0);
    close(Real::mul_add(2.0f32, 3.0, 4.0), 10.0);
    close(Real::mul_add(-2.0f32, 3.0, 1.0), -5.0);
}

#[test]
fn hyperbolic_f32() {
    close(Real::sinh(0.0f32), 0.0);
    close(Real::cosh(0.0f32), 1.0);
    close(Real::tanh(0.0f32), 0.0);
    close(Real::sinh(1.0f32), 1.175_201_2);
    close(Real::cosh(1.0f32), 1.543_080_6);
    close(Real::tanh(1.0f32), 0.761_594_2);
    // cosh² - sinh² = 1
    let x = 0.8f32;
    let (s, c) = (Real::sinh(x), Real::cosh(x));
    close(c * c - s * s, 1.0);
}

#[test]
fn cbrt_f32() {
    close(Real::cbrt(27.0f32), 3.0);
    close(Real::cbrt(-8.0f32), -2.0);
    close(Real::cbrt(1.0f32), 1.0);
    assert_eq!(Real::cbrt(0.0f32), 0.0);
    assert!(Real::cbrt(f32::NAN).is_nan());
}

#[test]
fn clamp_recip_angles_f32() {
    assert_eq!(Real::clamp(5.0f32, 0.0, 1.0), 1.0);
    assert_eq!(Real::clamp(-5.0f32, 0.0, 1.0), 0.0);
    assert_eq!(Real::clamp(0.5f32, 0.0, 1.0), 0.5);
    assert_eq!(Real::recip(4.0f32), 0.25);
    close(Real::to_radians(180.0f32), core::f32::consts::PI);
    close(Real::to_degrees(core::f32::consts::PI), 180.0);
    close(Real::to_degrees(Real::to_radians(42.0f32)), 42.0);
}

mod f64_tests {
    use super::common::assert_close_f64 as close;
    use azmath_core::Real;

    #[test]
    fn round_trunc_fract_f64() {
        assert_eq!(Real::round(2.5f64), 3.0);
        assert_eq!(Real::round(-2.5f64), -3.0);
        assert_eq!(Real::trunc(-2.9f64), -2.0);
        close(Real::fract(2.75f64), 0.75);
    }

    #[test]
    fn signum_f64() {
        assert_eq!(Real::signum(5.0f64), 1.0);
        assert_eq!(Real::signum(-5.0f64), -1.0);
        assert_eq!(Real::signum(-0.0f64), -1.0);
        assert!(Real::signum(f64::NAN).is_nan());
    }

    #[test]
    fn logarithms_f64() {
        close(Real::log10(1000.0f64), 3.0);
        close(Real::log2(1024.0f64), 10.0);
    }

    #[test]
    fn hypot_and_mul_add_f64() {
        close(Real::hypot(3.0f64, 4.0), 5.0);
        close(Real::mul_add(2.0f64, 3.0, 4.0), 10.0);
    }

    #[test]
    fn hyperbolic_f64() {
        close(Real::sinh(1.0f64), 1.1752011936438014);
        close(Real::cosh(1.0f64), 1.5430806348152437);
        close(Real::tanh(1.0f64), 0.7615941559557649);
    }

    #[test]
    fn cbrt_f64() {
        close(Real::cbrt(27.0f64), 3.0);
        close(Real::cbrt(-64.0f64), -4.0);
    }

    #[test]
    fn angles_f64() {
        close(Real::to_radians(90.0f64), core::f64::consts::FRAC_PI_2);
        close(Real::to_degrees(core::f64::consts::PI), 180.0);
    }
}
