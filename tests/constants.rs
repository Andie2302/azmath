//! Tests für die Konstanten des `Real`-Traits.
#![cfg(any(feature = "std", feature = "libm", feature = "micromath"))]

use azmath::Real;

#[test]
fn f32_constants() {
    assert_eq!(<f32 as Real>::ZERO, 0.0);
    assert_eq!(<f32 as Real>::ONE, 1.0);
    assert_eq!(<f32 as Real>::PI, core::f32::consts::PI);
    assert_eq!(<f32 as Real>::E, core::f32::consts::E);
    assert_eq!(<f32 as Real>::EPSILON, f32::EPSILON);
}

#[cfg(any(feature = "std", feature = "libm"))]
#[test]
fn f64_constants() {
    assert_eq!(<f64 as Real>::ZERO, 0.0);
    assert_eq!(<f64 as Real>::ONE, 1.0);
    assert_eq!(<f64 as Real>::PI, core::f64::consts::PI);
    assert_eq!(<f64 as Real>::E, core::f64::consts::E);
    assert_eq!(<f64 as Real>::EPSILON, f64::EPSILON);
}

#[test]
fn epsilon_is_positive_and_small() {
    assert!(<f32 as Real>::EPSILON > 0.0);
    assert!(<f32 as Real>::EPSILON < 1e-5);
}
