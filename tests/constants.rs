//! Tests für die Konstanten des `Real`-Traits.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::Real;

#[test]
fn f32_constants() {
    assert_eq!(<f32 as Real>::ZERO, 0.0);
    assert_eq!(<f32 as Real>::ONE, 1.0);
    assert_eq!(<f32 as Real>::PI, core::f32::consts::PI);
    assert_eq!(<f32 as Real>::E, core::f32::consts::E);
    assert_eq!(<f32 as Real>::EPSILON, f32::EPSILON);
}

#[test]
fn f64_constants() {
    assert_eq!(<f64 as Real>::ZERO, 0.0);
    assert_eq!(<f64 as Real>::ONE, 1.0);
    assert_eq!(<f64 as Real>::PI, core::f64::consts::PI);
    assert_eq!(<f64 as Real>::E, core::f64::consts::E);
    assert_eq!(<f64 as Real>::EPSILON, f64::EPSILON);
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn epsilon_is_positive_and_small() {
    assert!(<f32 as Real>::EPSILON > 0.0);
    assert!(<f32 as Real>::EPSILON < 1e-5);
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn f32_special_values() {
    assert!(<f32 as Real>::NAN.is_nan());
    assert_eq!(<f32 as Real>::INFINITY, f32::INFINITY);
    assert_eq!(<f32 as Real>::NEG_INFINITY, f32::NEG_INFINITY);
    assert_eq!(<f32 as Real>::MIN_POSITIVE, f32::MIN_POSITIVE);
    assert!(<f32 as Real>::INFINITY.is_infinite());
    assert!(<f32 as Real>::MIN_POSITIVE > <f32 as Real>::ZERO);
}

#[test]
fn f64_special_values() {
    assert!(<f64 as Real>::NAN.is_nan());
    assert_eq!(<f64 as Real>::INFINITY, f64::INFINITY);
    assert_eq!(<f64 as Real>::NEG_INFINITY, f64::NEG_INFINITY);
    assert_eq!(<f64 as Real>::MIN_POSITIVE, f64::MIN_POSITIVE);
}
