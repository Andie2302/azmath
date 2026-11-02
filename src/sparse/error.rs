use core::fmt;

/// Fehler der Sparse-Formate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SparseError {
    /// Zeilen- oder Spaltenindex außerhalb der Matrix.
    IndexOutOfBounds,
    /// Die CSR-Struktur ist ungültig (Längen, Monotonie von `row_ptr`, Spaltenindizes nicht streng aufsteigend).
    InvalidStructure,
    /// Die Größen passen nicht zusammen (Summe, Skalierungsvektor, Vektorlänge).
    DimensionMismatch,
    /// NaN oder Unendlich in einem Wert.
    NotFinite,
}

impl fmt::Display for SparseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            SparseError::IndexOutOfBounds => "Index außerhalb der Matrix",
            SparseError::InvalidStructure => "ungültige CSR-Struktur",
            SparseError::DimensionMismatch => "Größen passen nicht zusammen",
            SparseError::NotFinite => "NaN oder unendlicher Wert",
        })
    }
}

impl core::error::Error for SparseError {}
