//! Tests für die Backend-abhängigen Funktionen (sqrt, trig, exp, ln, pow, floor, ceil).
#![cfg(any(feature = "std", feature = "libm", feature = "micromath"))]

mod common;

use azmath::Real;
use common::assert_close_f32 as close;
use core::f32::consts::{E, FRAC_PI_2, FRAC_PI_4, PI};

#[test]
fn sqrt_f32() {
    close(Real::sqrt(4.0f32), 2.0);
    close(Real::sqrt(9.0f32), 3.0);
    close(Real::sqrt(0.0f32), 0.0);
    assert!(Real::sqrt(-1.0f32).is_nan());
}

#[test]
fn trigonometry_f32() {
    close(Real::sin(0.0f32), 0.0);
    close(Real::sin(FRAC_PI_2), 1.0);
    close(Real::cos(0.0f32), 1.0);
    close(Real::cos(PI), -1.0);
    close(Real::tan(FRAC_PI_4), 1.0);
}

#[test]
fn inverse_trigonometry_f32() {
    close(Real::asin(0.0f32), 0.0);
    close(Real::asin(1.0f32), FRAC_PI_2);
    close(Real::acos(1.0f32), 0.0);
    close(Real::acos(0.0f32), FRAC_PI_2);
    close(Real::atan(1.0f32), FRAC_PI_4);
    close(Real::atan2(1.0f32, 1.0), FRAC_PI_4);
    close(Real::atan2(0.0f32, -1.0), PI);
}

#[test]
fn exp_and_ln_f32() {
    close(Real::exp(0.0f32), 1.0);
    close(Real::exp(1.0f32), E);
    close(Real::ln(1.0f32), 0.0);
    close(Real::ln(E), 1.0);
    // micromath 2.1.0 läuft bei ln(negativ) in Debug-Builds in einen
    // Integer-Überlauf (Upstream-Bug), daher nur mit genauen Backends testen.
    #[cfg(any(feature = "std", feature = "libm"))]
    assert!(Real::ln(-1.0f32).is_nan());
    // exp und ln sind Umkehrfunktionen
    close(Real::ln(Real::exp(2.0f32)), 2.0);
}

#[test]
fn powers_f32() {
    close(Real::powf(2.0f32, 3.0), 8.0);
    close(Real::powf(4.0f32, 0.5), 2.0);
    close(Real::powi(2.0f32, 10), 1024.0);
    close(Real::powi(2.0f32, 0), 1.0);
    close(Real::powi(2.0f32, -2), 0.25);
    close(Real::powi(-2.0f32, 3), -8.0);
    close(Real::powi(10.0f32, 1), 10.0);
}

#[test]
fn rounding_f32() {
    close(Real::floor(2.7f32), 2.0);
    close(Real::floor(-2.1f32), -3.0);
    close(Real::floor(3.0f32), 3.0);
    close(Real::ceil(2.1f32), 3.0);
    close(Real::ceil(-2.7f32), -2.0);
    close(Real::ceil(3.0f32), 3.0);
}

#[cfg(any(feature = "std", feature = "libm"))]
mod f64_tests {
    use super::common::assert_close_f64 as close;
    use azmath::Real;
    use core::f64::consts::{E, FRAC_PI_2, FRAC_PI_4, PI};

    #[test]
    fn sqrt_f64() {
        close(Real::sqrt(2.0f64), core::f64::consts::SQRT_2);
        close(Real::sqrt(144.0f64), 12.0);
        assert!(Real::sqrt(-1.0f64).is_nan());
    }

    #[test]
    fn trigonometry_f64() {
        close(Real::sin(FRAC_PI_2), 1.0);
        close(Real::cos(PI), -1.0);
        close(Real::tan(FRAC_PI_4), 1.0);
        close(Real::asin(1.0f64), FRAC_PI_2);
        close(Real::acos(-1.0f64), PI);
        close(Real::atan(1.0f64), FRAC_PI_4);
        close(Real::atan2(1.0f64, -1.0), 3.0 * FRAC_PI_4);
    }

    #[test]
    fn exp_ln_f64() {
        close(Real::exp(1.0f64), E);
        close(Real::ln(E), 1.0);
        close(Real::ln(Real::exp(3.5f64)), 3.5);
        assert!(Real::ln(-1.0f64).is_nan());
    }

    #[test]
    fn powers_f64() {
        close(Real::powf(2.0f64, 0.5), core::f64::consts::SQRT_2);
        close(Real::powi(3.0f64, 5), 243.0);
        close(Real::powi(2.0f64, -3), 0.125);
        close(Real::powi(5.0f64, 0), 1.0);
    }

    #[test]
    fn rounding_f64() {
        close(Real::floor(-0.5f64), -1.0);
        close(Real::ceil(-0.5f64), 0.0);
        close(Real::floor(1e10 + 0.5), 1e10);
    }
}
