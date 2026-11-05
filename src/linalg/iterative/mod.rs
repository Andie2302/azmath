//! Iterative Löser für große lineare Gleichungssysteme.
//!
//! - [`cg`] / [`pcg`]: konjugierte Gradienten für symmetrisch positiv definite Systeme.
//! - [`bicgstab`] / [`pbicgstab`]: für beliebige (auch nichtsymmetrische) reguläre Systeme.
//! - [`gmres`] / [`pgmres`]: GMRES(m) für beliebige reguläre Systeme; in exakter Arithmetik monoton und ohne Zusammenbruch,
//!   auch bei symmetrisch indefiniten und schiefsymmetrischen Matrizen. Braucht `m + 1` Basisvektoren.
//!
//! Die Verfahren arbeiten **matrixfrei** gegen ein [`LinearOperator`](super::LinearOperator) und
//! brauchen keinen Allokator: Der Arbeitsspeicher wird als Slice übergeben. Bequeme Methoden
//! (`Matrix::solve_cg`, mit `alloc` auch `DynMatrix::solve_cg`) kümmern sich selbst darum.
//!
//! Das Abbruchkriterium ist die **relative Residuumsnorm** `‖b − A·x‖ ≤ tol · ‖b‖`.
//! Erfolgreiche Rückgaben werden durch ein neu berechnetes Residuum bestätigt. Bei
//! [`pgmres_left`] gilt das dokumentierte Kriterium des links vorkonditionierten Systems.
//! Negative oder nichtendliche Toleranzen ergeben [`LinalgError::InvalidInput`](super::LinalgError::InvalidInput).
//! `max_iter = 0` erlaubt nur eine bereits ausreichende Startnäherung; `tol = 0` verlangt ein Nullresiduum.
//!
//! Hinweis zu `micromath`: Mit der bewusst ungenauen Näherung erreichen iterative Verfahren eine
//! strenge Toleranz unter Umständen nie (`NoConvergence`). Für verlässliche Ergebnisse `std` oder
//! `libm` verwenden.

mod bicgstab;
mod cg;
mod gmres;
mod precond;

pub use bicgstab::{bicgstab, pbicgstab};
pub use cg::{cg, pcg};
pub use gmres::{gmres, gmres_work_len, pgmres, pgmres_left, pgmres_left_work_len};
pub use precond::{Identity, Jacobi, Preconditioner};

#[cfg(feature = "alloc")]
pub use precond::DynJacobi;

use crate::Real;

/// Einstellungen für iterative Löser.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterOptions<T> {
    /// Relative Toleranz für `‖b − A·x‖ / ‖b‖`; endlich und nichtnegativ (0 verlangt ein exaktes Residuum).
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

/// Stabile euklidische Norm ohne direktes Quadrieren der Eingabewerte.
fn norm<T: Real>(values: &[T]) -> T {
    super::norm::frobenius(values)
}

fn finite<T: Real>(values: &[T]) -> Result<(), super::LinalgError> {
    if values.iter().any(|v| !v.is_finite()) {
        Err(super::LinalgError::NotFinite)
    } else {
        Ok(())
    }
}

fn validate<T: Real>(b: &[T], x: &[T], opts: IterOptions<T>) -> Result<(), super::LinalgError> {
    if !opts.tol.is_finite() || opts.tol < T::ZERO {
        return Err(super::LinalgError::InvalidInput);
    }
    finite(b)?;
    finite(x)
}

fn converged<T: Real>(residual: T, bnorm: T, tol: T) -> bool {
    if bnorm == T::ZERO || tol == T::ZERO {
        residual == T::ZERO
    } else {
        residual / bnorm <= tol
    }
}

/// Wahres Residuum neu berechnen; niemals allein einer rekursiven Schätzung vertrauen.
fn residual<T: Real, A: super::LinearOperator<T> + ?Sized>(
    a: &A,
    b: &[T],
    x: &[T],
    r: &mut [T],
) -> Result<T, super::LinalgError> {
    finite(x)?;
    a.apply(x, r);
    finite(r)?;
    for (ri, bi) in r.iter_mut().zip(b) {
        *ri = *bi - *ri;
    }
    let rn = norm(r);
    if !rn.is_finite() {
        return Err(super::LinalgError::NotFinite);
    }
    Ok(rn)
}

/// Rückskalierung der normierten Suchrichtung beim Aktualisieren von x.
fn scaled_axpy<T: Real>(alpha: T, scale: T, p: &[T], x: &mut [T]) {
    let factor = alpha * scale;
    for (xi, pi) in x.iter_mut().zip(p) {
        *xi += if factor.is_finite() && factor != T::ZERO {
            factor * *pi
        } else {
            alpha * (scale * *pi)
        };
    }
}
