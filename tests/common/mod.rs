//! Gemeinsame Hilfsfunktionen für die Integrationstests.
#![allow(dead_code)]

/// Erlaubte Abweichung für f32-Vergleiche.
///
/// `std` und `libm` sind genau; `micromath` ist nur eine schnelle Näherung
/// (Fehler im Bereich von 10 %), daher dort eine großzügige Toleranz.
pub const TOL_F32: f32 = if cfg!(any(feature = "std", feature = "libm")) {
    1e-5
} else {
    0.2
};

/// Erlaubte Abweichung für f64-Vergleiche (nur mit `std`/`libm` verfügbar).
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
