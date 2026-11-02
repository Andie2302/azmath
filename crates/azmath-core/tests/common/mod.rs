//! Gemeinsame Hilfsfunktionen für die Integrationstests.
#![allow(dead_code)]

/// Erlaubte relative Abweichung für f32-Vergleiche.
pub const TOL_F32: f32 = 1e-5;

/// Erlaubte Abweichung für f64-Vergleiche .
pub const TOL_F64: f64 = 1e-12;

pub fn assert_close_f32(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= TOL_F32 * expected.abs().max(1.0),
        "f32: erwartet {expected}, erhalten {actual} (Toleranz {TOL_F32})"
    );
}

pub fn assert_close_f64(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= TOL_F64 * expected.abs().max(1.0),
        "f64: erwartet {expected}, erhalten {actual} (Toleranz {TOL_F64})"
    );
}
