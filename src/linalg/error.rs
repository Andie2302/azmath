use core::fmt;

/// Fehler der linearen Algebra.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LinalgError {
    /// Die Matrix ist singulär bzw. bei QR rangdefizient (oder numerisch nicht davon zu unterscheiden).
    Singular,
    /// Die Matrix enthält NaN oder unendliche Werte.
    NotFinite,
    /// Die Dimensionen der Operanden passen nicht zusammen.
    DimensionMismatch,
    /// Die Matrix ist nicht symmetrisch (Voraussetzung für Cholesky).
    NotSymmetric,
    /// Die Matrix ist nicht positiv definit (Cholesky bzw. CG nicht möglich).
    NotPositiveDefinite,
    /// Das Iterationsverfahren hat die geforderte Genauigkeit in `max_iter` Schritten nicht erreicht.
    NoConvergence,
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            LinalgError::Singular => "Matrix ist singulär",
            LinalgError::NotFinite => "Matrix enthält NaN oder unendliche Werte",
            LinalgError::DimensionMismatch => "Dimensionen passen nicht zusammen",
            LinalgError::NotSymmetric => "Matrix ist nicht symmetrisch",
            LinalgError::NotPositiveDefinite => "Matrix ist nicht positiv definit",
            LinalgError::NoConvergence => "Iteration ist nicht konvergiert",
        })
    }
}

impl core::error::Error for LinalgError {}
