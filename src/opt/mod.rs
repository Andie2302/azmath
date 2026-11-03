//! Optimierung mit Nebenbedingungen: lineare Programme (LP) und später QP, NLP und gemischt-ganzzahlige Probleme.
//!
//! Dieses Modul braucht das Feature `alloc` (die Löser arbeiten auf Tableaus mit zur Laufzeit bestimmter Größe).
//!
//! | Funktion | Problem |
//! |---|---|
//! | [`simplex`] | `min cᵀx` unter `A·x ≤ / = / ≥ b`, `x ≥ 0` (Zwei-Phasen-Simplex, dichtes Tableau) |
//! | [`simplex_bounded`] | wie `simplex`, aber mit Schranken `lᵢ ≤ xᵢ ≤ uᵢ` je Variable (auch unendlich, freie Variablen) |
//! | [`branch_and_bound`] | gemischt-ganzzahliges LP (`xᵢ ∈ ℤ` für markierte Variablen), Branch-and-Bound über Simplex-Relaxationen |
//! | [`interior_point_nlp`] | nichtlineares Programm per primal-dualem Innere-Punkte-Verfahren (exakte Hessematrix, Barriere, ℓ₁-Merit-Funktion) |
//! | [`interior_point_qp`] | konvexes QP `min ½xᵀQx + cᵀx` unter `A·x = b`, `G·x ≤ h` (primal-duales Innere-Punkte-Verfahren) |
//! | [`sqp`] | nichtlineares Programm `min f(x)`, `c(x) = 0`, `c(x) ≤ 0` (SQP mit BFGS und ℓ₁-Merit-Funktion) |
//! | [`interior_point_lp`] | LP `min cᵀx`, `A·x = b`, `x ≥ 0` per Innere-Punkte-Verfahren |
//!
//! Die Minimierung ohne Nebenbedingungen (BFGS, L-BFGS, Newton, Levenberg-Marquardt) steht in
//! [`optimize`](crate::optimize).

mod milp;
mod nlp_ip;
mod qp;
mod simplex;
mod sqp;

pub use milp::{MilpOptions, MilpSolution, MilpStatus, NodeSelection, branch_and_bound};
pub use nlp_ip::{NlpOptions, NlpResult, NlpStatus, interior_point_nlp};
pub use qp::{QpOptions, QpSolution, QpStatus, interior_point_lp, interior_point_qp};
pub use simplex::{LpSolution, LpStatus, Relation, SimplexOptions, simplex, simplex_bounded};
pub use sqp::{SqpOptions, SqpResult, SqpStatus, sqp};

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
