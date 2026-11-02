use core::fmt;

/// Fehler der linearen Algebra.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinalgError {
    /// Die Matrix ist singulär (oder numerisch nicht von einer singulären zu unterscheiden).
    Singular,
    /// Die Matrix enthält NaN oder unendliche Werte.
    NotFinite,
    /// Die Dimensionen der Operanden passen nicht zusammen.
    DimensionMismatch,
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LinalgError::Singular => "Matrix ist singulär",
            LinalgError::NotFinite => "Matrix enthält NaN oder unendliche Werte",
            LinalgError::DimensionMismatch => "Dimensionen passen nicht zusammen",
        })
    }
}

impl core::error::Error for LinalgError {}
