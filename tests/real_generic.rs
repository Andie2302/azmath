//! Tests für generischen Code über `T: Real` (Zusammenspiel mehrerer Methoden).
#![cfg(any(feature = "std", feature = "libm"))]

mod common;

use azmath::Real;
use common::assert_close_f32 as close32;

fn circle_area<T: Real>(r: T) -> T {
    T::PI * r * r
}

fn hypot<T: Real>(a: T, b: T) -> T {
    (a * a + b * b).sqrt()
}

fn clamp<T: Real>(x: T, lo: T, hi: T) -> T {
    x.max(lo).min(hi)
}

fn sum_of_squares<T: Real>(values: &[T]) -> T {
    let mut acc = T::ZERO;
    for &v in values {
        acc += v * v;
    }
    acc
}

fn celsius_to_fahrenheit<T: Real>(c: T) -> T {
    c * T::from_f64(1.8) + T::from_f64(32.0)
}

#[test]
fn circle_area_f32() {
    close32(circle_area(1.0f32), core::f32::consts::PI);
    close32(circle_area(2.0f32), 4.0 * core::f32::consts::PI);
}

#[test]
fn hypot_f32() {
    close32(hypot(3.0f32, 4.0), 5.0);
    close32(hypot(0.0f32, 0.0), 0.0);
}

#[test]
fn clamp_f32() {
    assert_eq!(clamp(5.0f32, 0.0, 1.0), 1.0);
    assert_eq!(clamp(-5.0f32, 0.0, 1.0), 0.0);
    assert_eq!(clamp(0.5f32, 0.0, 1.0), 0.5);
}

#[test]
fn sum_of_squares_f32() {
    assert_eq!(sum_of_squares(&[1.0f32, 2.0, 3.0]), 14.0);
    assert_eq!(sum_of_squares::<f32>(&[]), 0.0);
}

#[test]
fn from_f64_in_formula_f32() {
    close32(celsius_to_fahrenheit(100.0f32), 212.0);
    close32(celsius_to_fahrenheit(0.0f32), 32.0);
}

#[test]
fn pythagorean_identity_f32() {
    for i in 0..8 {
        let x = i as f32 * 0.4;
        let s = x.sin();
        let c = x.cos();
        // sin² + cos² = 1
        close32(s * s + c * c, 1.0);
    }
}

mod f64_tests {
    use super::*;
    use common::assert_close_f64 as close64;

    #[test]
    fn same_formula_both_types() {
        close64(circle_area(1.0f64), core::f64::consts::PI);
        close32(circle_area(1.0f32), core::f32::consts::PI);
        close64(hypot(5.0f64, 12.0), 13.0);
        close64(celsius_to_fahrenheit(37.0f64), 98.6);
    }

    #[test]
    fn f64_is_more_precise_than_f32() {
        let a = <f32 as Real>::from_f64(0.1) as f64;
        let b = <f64 as Real>::from_f64(0.1);
        assert_ne!(a, b);
        assert_eq!(b, 0.1);
    }
}
