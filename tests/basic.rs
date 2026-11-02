//! Tests für die Basis-Methoden: abs, min, max, is_nan, is_infinite, from_f64.
#![cfg(any(feature = "std", feature = "libm", feature = "micromath"))]

use azmath::Real;

#[test]
fn abs_f32() {
    assert_eq!(Real::abs(-3.5f32), 3.5);
    assert_eq!(Real::abs(3.5f32), 3.5);
    assert_eq!(Real::abs(0.0f32), 0.0);
    // -0.0 wird zu +0.0 (Vorzeichenbit gelöscht)
    assert!(Real::abs(-0.0f32).is_sign_positive());
    assert_eq!(Real::abs(f32::NEG_INFINITY), f32::INFINITY);
    assert!(Real::abs(f32::NAN).is_nan());
}

#[test]
fn min_max_f32() {
    assert_eq!(Real::min(1.0f32, 2.0), 1.0);
    assert_eq!(Real::max(1.0f32, 2.0), 2.0);
    assert_eq!(Real::min(-1.0f32, -2.0), -2.0);
    assert_eq!(Real::max(-1.0f32, -2.0), -1.0);
    // NaN wird ignoriert, sofern der andere Wert eine Zahl ist
    assert_eq!(Real::min(f32::NAN, 1.0), 1.0);
    assert_eq!(Real::max(1.0f32, f32::NAN), 1.0);
}

#[test]
fn nan_and_infinity_f32() {
    assert!(Real::is_nan(f32::NAN));
    assert!(!Real::is_nan(1.0f32));
    assert!(!Real::is_nan(f32::INFINITY));

    assert!(Real::is_infinite(f32::INFINITY));
    assert!(Real::is_infinite(f32::NEG_INFINITY));
    assert!(!Real::is_infinite(1.0f32));
    assert!(!Real::is_infinite(f32::NAN));
}

#[test]
fn from_f64_f32() {
    assert_eq!(<f32 as Real>::from_f64(0.5), 0.5f32);
    assert_eq!(<f32 as Real>::from_f64(0.1), 0.1f32);
    assert_eq!(<f32 as Real>::from_f64(-2.0), -2.0f32);
    assert!(<f32 as Real>::from_f64(f64::NAN).is_nan());
}

#[cfg(any(feature = "std", feature = "libm"))]
mod f64_tests {
    use azmath::Real;

    #[test]
    fn abs_f64() {
        assert_eq!(Real::abs(-3.5f64), 3.5);
        assert!(Real::abs(-0.0f64).is_sign_positive());
        assert_eq!(Real::abs(f64::NEG_INFINITY), f64::INFINITY);
        assert!(Real::abs(f64::NAN).is_nan());
    }

    #[test]
    fn min_max_f64() {
        assert_eq!(Real::min(1.0f64, 2.0), 1.0);
        assert_eq!(Real::max(1.0f64, 2.0), 2.0);
        assert_eq!(Real::min(f64::NAN, 1.0), 1.0);
    }

    #[test]
    fn nan_and_infinity_f64() {
        assert!(Real::is_nan(f64::NAN));
        assert!(!Real::is_nan(0.0f64));
        assert!(Real::is_infinite(f64::INFINITY));
        assert!(!Real::is_infinite(1e308f64));
    }

    #[test]
    fn from_f64_f64_is_identity() {
        assert_eq!(<f64 as Real>::from_f64(0.1), 0.1f64);
        assert_eq!(<f64 as Real>::from_f64(1e300), 1e300f64);
    }
}
