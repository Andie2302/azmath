//! Iterative Löser für große lineare Gleichungssysteme.
//!
//! - [`cg`] / [`pcg`]: konjugierte Gradienten für symmetrisch positiv definite Systeme.
//! - [`bicgstab`] / [`pbicgstab`]: für beliebige (auch nichtsymmetrische) reguläre Systeme.
//!
//! Die Verfahren arbeiten **matrixfrei** gegen ein [`LinearOperator`](super::LinearOperator) und
//! brauchen keinen Allokator: Der Arbeitsspeicher wird als Slice übergeben. Bequeme Methoden
//! (`Matrix::solve_cg`, mit `alloc` auch `DynMatrix::solve_cg`) kümmern sich selbst darum.
//!
//! Das Abbruchkriterium ist die **relative Residuumsnorm** `‖b − A·x‖ ≤ tol · ‖b‖`.
//!
//! Hinweis zu `micromath`: Mit der bewusst ungenauen Näherung erreichen iterative Verfahren eine
//! strenge Toleranz unter Umständen nie (`NoConvergence`). Für verlässliche Ergebnisse `std` oder
//! `libm` verwenden.

mod bicgstab;
mod cg;
mod precond;

pub use bicgstab::{bicgstab, pbicgstab};
pub use cg::{cg, pcg};
pub use precond::{Identity, Jacobi, Preconditioner};

#[cfg(feature = "alloc")]
pub use precond::DynJacobi;

use crate::Real;

/// Einstellungen für iterative Löser.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterOptions<T> {
    /// Relative Toleranz für `‖b − A·x‖ / ‖b‖`.
    pub tol: T,
    /// Maximale Anzahl an Iterationen.
    pub max_iter: usize,
}

impl<T: Real> IterOptions<T> {
    /// Erzeugt Einstellungen mit der gegebenen Toleranz und Iterationsgrenze.
    pub fn new(tol: T, max_iter: usize) -> Self {
        Self { tol, max_iter }
    }
}

impl<T: Real> Default for IterOptions<T> {
    /// Toleranz `√ε` (etwa `1.5e-8` bei `f64`) und höchstens 1000 Iterationen.
    fn default() -> Self {
        Self {
            tol: T::EPSILON.sqrt(),
            max_iter: 1000,
        }
    }
}

/// Ergebnis eines erfolgreichen iterativen Lösers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterResult<T> {
    /// Anzahl der ausgeführten Iterationen (0, wenn die Startnäherung schon genügt).
    pub iterations: usize,
    /// Euklidische Norm des Residuums `‖b − A·x‖` der Lösung.
    pub residual_norm: T,
}

/// Skalarprodukt zweier gleich langer Slices.
pub(crate) fn dot<T: Real>(a: &[T], b: &[T]) -> T {
    let mut s = T::ZERO;
    for (x, y) in a.iter().zip(b) {
        s += *x * *y;
    }
    s
}

/// `y += alpha · x`
pub(crate) fn axpy<T: Real>(alpha: T, x: &[T], y: &mut [T]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += alpha * *xi;
    }
}
