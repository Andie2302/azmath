//! Optimierung mit Nebenbedingungen: lineare Programme (LP) und später QP, NLP und gemischt-ganzzahlige Probleme.
//!
//! Dieses Modul braucht das Feature `alloc` (die Löser arbeiten auf Tableaus mit zur Laufzeit bestimmter Größe).
//!
//! | Funktion | Problem |
//! |---|---|
//! | [`simplex`] | `min cᵀx` unter `A·x ≤ / = / ≥ b`, `x ≥ 0` (Zwei-Phasen-Simplex, dichtes Tableau) |
//!
//! Die Minimierung ohne Nebenbedingungen (BFGS, L-BFGS, Newton, Levenberg-Marquardt) steht in
//! [`optimize`](crate::optimize).

mod simplex;

pub use simplex::{LpSolution, LpStatus, Relation, SimplexOptions, simplex};

use core::fmt;

/// Fehler der Optimierungslöser dieses Moduls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OptError {
    /// Ungültige Größen (leere Aufgabe, Längen passen nicht zusammen).
    InvalidInput,
    /// NaN oder Unendlich in den Eingabedaten.
    NotFinite,
}

impl fmt::Display for OptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            OptError::InvalidInput => "ungültige Größen",
            OptError::NotFinite => "NaN oder unendlicher Wert in den Eingabedaten",
        })
    }
}

impl core::error::Error for OptError {}
