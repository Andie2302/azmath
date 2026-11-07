//! Zellzentrierter 2D-Poisson-Operator `A ≈ −Δ` mit Dirichlet-, Neumann- und periodischen Rändern (AZM-040).
//!
//! **Konventionen**
//! - `nx × ny` Zellen, Zellmitten, Index `k = j·nx + i`; Abstände `hx`, `hy`.
//! - `A` approximiert **−Δ** (positiv semidefinit): `(A·u)ₖ = Σ_Achsen (2uₖ − u₋ − u₊)/h²`.
//! - Dirichlet (homogen, `u = 0` auf der Randfläche): Geisterwert `−uₖ`, Diagonale `+1/h²`.
//! - Neumann (homogen, `∂u/∂n = 0`): Geisterwert `uₖ`, der Randnachbar entfällt.
//! - Periodisch: Nachbar umlaufend, mindestens 3 Zellen in dieser Richtung.
//! - **Flussbilanz:** Bei Neumann/periodisch in beiden Richtungen sind alle Spaltensummen 0, also
//!   `Σₖ (A·u)ₖ·hx·hy = 0` und `A·1 = 0` (Nullraum = Konstanten).
//!
//! Bei Nullraum ist `A·u = b` nur lösbar, wenn `Σ b = 0` gilt; [`check_compatibility`] weist andere
//! rechte Seiten ab, [`project_mean_zero`] projiziert sie ausdrücklich. [`solve_poisson_2d`] löst auf dem
//! Unterraum mit Mittelwert 0 (CG), die Lösung hat Mittelwert 0.

use crate::Real;
use crate::linalg::LinalgError;
use crate::linalg::iterative::{IterOptions, IterResult, cg};
use crate::sparse::{CooMatrix, CsrMatrix, SparseError};
use alloc::vec;
use alloc::vec::Vec;

/// Randbedingung entlang einer Achse (beide Seiten gleich).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Boundary {
    /// Homogenes Dirichlet.
    Dirichlet,
    /// Homogenes Neumann.
    Neumann,
    /// Periodisch.
    Periodic,
}

/// Fehler der Poisson-Routinen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoissonError {
    /// Zu wenige Zellen (periodisch: < 3, sonst < 2), Längenfehler oder Abstand nicht endlich/positiv.
    InvalidGrid,
    /// Rechte Seite verletzt `Σ b = 0` bei vorhandenem Nullraum.
    IncompatibleRhs,
    /// Nichtendlicher Wert in der rechten Seite.
    NotFinite,
    /// Fehler des iterativen Lösers.
    Solver(LinalgError),
    /// Fehler beim Aufbau der Matrix.
    Sparse(SparseError),
}

impl core::fmt::Display for PoissonError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidGrid => f.write_str("ungültiges Gitter"),
            Self::IncompatibleRhs => f.write_str("rechte Seite nicht mit dem Nullraum verträglich"),
            Self::NotFinite => f.write_str("NaN oder unendlicher Wert"),
            Self::Solver(e) => write!(f, "Löser: {e:?}"),
            Self::Sparse(e) => write!(f, "Matrix: {e}"),
        }
    }
}

impl core::error::Error for PoissonError {}

/// `true`, wenn der Operator einen Nullraum (Konstanten) hat: keine der beiden Achsen ist Dirichlet.
pub fn has_nullspace(bcx: Boundary, bcy: Boundary) -> bool {
    bcx != Boundary::Dirichlet && bcy != Boundary::Dirichlet
}

fn axis_ok<T: Real>(n: usize, h: T, bc: Boundary) -> bool {
    let min = if bc == Boundary::Periodic { 3 } else { 2 };
    n >= min && h.is_finite() && h > T::ZERO
}

/// Baut `A ≈ −Δ` als CSR-Matrix der Größe `nx·ny`. Siehe Moduldoku für Konventionen.
pub fn poisson_2d<T: Real>(
    nx: usize,
    ny: usize,
    hx: T,
    hy: T,
    bcx: Boundary,
    bcy: Boundary,
) -> Result<CsrMatrix<T>, PoissonError> {
    if !axis_ok(nx, hx, bcx) || !axis_ok(ny, hy, bcy) {
        return Err(PoissonError::InvalidGrid);
    }
    let n = nx * ny;
    let (ix2, iy2) = (T::ONE / (hx * hx), T::ONE / (hy * hy));
    let mut coo = CooMatrix::with_capacity(n, n, 5 * n);
    let push = |c: &mut CooMatrix<T>, i, j, v| c.push(i, j, v).map_err(PoissonError::Sparse);
    for j in 0..ny {
        for i in 0..nx {
            let k = j * nx + i;
            // pro Achse: (Position, Zellenzahl, Randtyp, Gewicht 1/h², Index-Schritt)
            for (pos, len, bc, w, stride) in [(i, nx, bcx, ix2, 1usize), (j, ny, bcy, iy2, nx)] {
                let lo = if pos > 0 {
                    Some(k - stride)
                } else if bc == Boundary::Periodic {
                    Some(k + (len - 1) * stride)
                } else {
                    None
                };
                let hi = if pos + 1 < len {
                    Some(k + stride)
                } else if bc == Boundary::Periodic {
                    Some(k - (len - 1) * stride)
                } else {
                    None
                };
                for nb in [lo, hi] {
                    match nb {
                        Some(m) => {
                            push(&mut coo, k, k, w)?;
                            push(&mut coo, k, m, -w)?;
                        }
                        None => {
                            // Randseite: Dirichlet Geisterwert −u → 2w, Neumann → 0
                            if bc == Boundary::Dirichlet {
                                push(&mut coo, k, k, w + w)?;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(CsrMatrix::from_coo(&coo))
}

fn sum_abs<T: Real>(b: &[T]) -> (T, T) {
    b.iter()
        .fold((T::ZERO, T::ZERO), |(s, a), &v| (s + v, a + v.abs()))
}

/// Prüft `|Σ b| ≤ tol·Σ|b|` (relativ). Gibt [`PoissonError::IncompatibleRhs`] zurück, sonst `Ok`.
pub fn check_compatibility<T: Real>(b: &[T], tol: T) -> Result<(), PoissonError> {
    if b.iter().any(|v| !v.is_finite()) || !tol.is_finite() || tol < T::ZERO {
        return Err(PoissonError::NotFinite);
    }
    let (s, a) = sum_abs(b);
    if s.abs() <= tol * a {
        Ok(())
    } else {
        Err(PoissonError::IncompatibleRhs)
    }
}

/// Zieht den Mittelwert ab (orthogonale Projektion auf `1⊥`). Gibt den entfernten Mittelwert zurück.
pub fn project_mean_zero<T: Real>(b: &mut [T]) -> T {
    if b.is_empty() {
        return T::ZERO;
    }
    let mean = b.iter().fold(T::ZERO, |s, &v| s + v) / T::from_f64(b.len() as f64);
    b.iter_mut().for_each(|v| *v -= mean);
    mean
}

/// Löst `A·u = b`. Bei Nullraum muss `b` verträglich sein (relative Toleranz `compat_tol`, sonst
/// [`PoissonError::IncompatibleRhs`]); die Lösung hat dann Mittelwert 0. Mit Dirichlet-Achse wird
/// direkt mit CG gelöst.
pub fn solve_poisson_2d<T: Real>(
    a: &CsrMatrix<T>,
    b: &[T],
    nullspace: bool,
    compat_tol: T,
    opts: IterOptions<T>,
) -> Result<(Vec<T>, IterResult<T>), PoissonError> {
    let n = a.rows();
    if b.len() != n || a.cols() != n {
        return Err(PoissonError::InvalidGrid);
    }
    let mut rhs = b.to_vec();
    if nullspace {
        check_compatibility(b, compat_tol)?;
        project_mean_zero(&mut rhs);
    } else if b.iter().any(|v| !v.is_finite()) {
        return Err(PoissonError::NotFinite);
    }
    let mut x = vec![T::ZERO; n];
    let mut work = vec![T::ZERO; 4 * n];
    let r = cg(a, &rhs, &mut x, opts, &mut work).map_err(PoissonError::Solver)?;
    if nullspace {
        project_mean_zero(&mut x);
    }
    Ok((x, r))
}
