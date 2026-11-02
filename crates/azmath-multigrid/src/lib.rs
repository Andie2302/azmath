//! Geometrisches Multigrid für die Poisson-Gleichung `−Δu = f` mit Dirichlet-Rand `u = 0`.
//!
//! Multigrid löst das diskretisierte System in `O(N)` Operationen (`N` = Zahl der Unbekannten) und mit einer
//! von der Gittergröße **unabhängigen** Konvergenzrate: Ein glättendes Verfahren (Jacobi, Gauß-Seidel)
//! dämpft die hochfrequenten Fehleranteile auf dem feinen Gitter, die verbleibenden glatten Anteile
//! werden auf einem gröberen Gitter korrigiert, rekursiv bis zu einem einzigen Unbekannten. Zum Vergleich:
//! CG braucht auf dem 2D-Gitter mit `127 × 127` Punkten rund 200 Iterationen, ein Multigrid-Zyklus reicht
//! für einen Faktor 10 im Fehler.
//!
//! | Funktion | Problem |
//! |---|---|
//! | [`solve_1d`], [`solve_2d`] | löst `−u'' = f` (1D) bzw. `−Δu = f` (2D, 5-Punkte-Stern) |
//! | [`cycle_1d`], [`cycle_2d`] | ein einzelner V- oder W-Zyklus auf einer Näherung |
//! | [`MultigridPreconditioner`] | ein symmetrischer Zyklus als Vorkonditionierer für [`pcg`](azmath_linalg::iterative::pcg) |
//! | [`Poisson1d`], [`Poisson2d`] | der Operator selbst als [`LinearOperator`](azmath_linalg::LinearOperator) |
//!
//! # Gitter und Skalierung
//!
//! Das Gebiet ist das Einheitsintervall bzw. Einheitsquadrat, `n` ist die Zahl der **inneren** Punkte je
//! Richtung und muss die Form `2ᵏ − 1` haben (`1, 3, 7, 15, 31, 63, 127, …`), damit sich das Gitter bis auf
//! einen Punkt halbieren lässt. Die Maschenweite ist `h = 1/(n+1)`, das System lautet
//! `(2uᵢ − uᵢ₋₁ − uᵢ₊₁)/h² = fᵢ` (1D) bzw. `(4uᵢⱼ − Nachbarn)/h² = fᵢⱼ` (2D, Zeilen `i` zeilenweise
//! gespeichert, `u[i·n + j]`). Die rechte Seite `f` ist also der Wert der Funktion an den Gitterpunkten,
//! nicht mit `h²` multipliziert.
//!
//! # Speicher
//!
//! Alle Funktionen brauchen keinen Allokator: Der Arbeitsspeicher wird als Slice übergeben
//! ([`work_len_1d`], [`work_len_2d`] als `const fn`). Mit `alloc` kümmern sich [`solve_1d_vec`] und
//! [`solve_2d_vec`] selbst darum.
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath_multigrid::{solve_1d, work_len_1d, MultigridOptions, SolveOptions};
//!
//! // −u'' = π²·sin(πx) auf (0, 1) hat die Lösung sin(πx)
//! const N: usize = 63;
//! let h = 1.0 / (N as f64 + 1.0);
//! let pi = core::f64::consts::PI;
//! let f: [f64; N] = core::array::from_fn(|i| pi * pi * (pi * (i + 1) as f64 * h).sin());
//! let mut u = [0.0; N];
//! let mut work = [0.0; work_len_1d(N)];
//! let r = solve_1d(&f, &mut u, N, MultigridOptions::default(), SolveOptions::default(), &mut work)
//!     .unwrap();
//! assert!(r.cycles <= 15);
//! // Diskretisierungsfehler ∝ h²
//! assert!((u[N / 2] - 1.0).abs() < 1e-3);
//! # }
//! ```
//!
//! Hinweis zu `micromath`: Das Verfahren selbst nutzt nur `+ - * /` (die Norm benötigt `sqrt`); die Tests
//! laufen dennoch nur mit `std` und `libm`.

#![no_std]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

mod oned;
mod precond;
mod twod;

pub use oned::{cycle_1d, solve_1d, work_len_1d};
pub use precond::{MultigridPreconditioner, Poisson1d, Poisson2d};
pub use twod::{cycle_2d, solve_2d, work_len_2d};

#[cfg(feature = "alloc")]
mod owned;
#[cfg(feature = "alloc")]
pub use owned::{solve_1d_vec, solve_2d_vec};

use core::fmt;

use azmath_core::Real;

/// Zyklusform: wie oft das Grobgitterproblem pro Ebene gelöst wird.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleType {
    /// V-Zyklus: ein Grobgitterschritt je Ebene. Am billigsten, meist ausreichend.
    V,
    /// W-Zyklus: zwei Grobgitterschritte je Ebene. Etwas teurer, robuster bei schwierigen Problemen.
    W,
}

/// Glätter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Smoother<T> {
    /// Gewichtetes Jacobi-Verfahren mit Dämpfungsfaktor `ω` (1D: `2/3`, 2D mit 5-Punkte-Stern: `4/5`).
    /// Unabhängig von der Reihenfolge, daher selbstadjungiert.
    WeightedJacobi(T),
    /// Gauß-Seidel in lexikografischer Reihenfolge: vorwärts vor, rückwärts nach dem Grobgitterschritt
    /// (dadurch bleibt der Zyklus symmetrisch). Meist schneller als Jacobi.
    GaussSeidel,
}

/// Einstellungen für einen Multigrid-Zyklus.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultigridOptions<T> {
    /// Glättungsschritte vor dem Grobgitterschritt.
    pub pre_smooth: usize,
    /// Glättungsschritte nach dem Grobgitterschritt. Für einen symmetrischen Vorkonditionierer muss
    /// `post_smooth == pre_smooth` gelten.
    pub post_smooth: usize,
    /// V- oder W-Zyklus.
    pub cycle: CycleType,
    /// Der Glätter.
    pub smoother: Smoother<T>,
}

impl<T: Real> Default for MultigridOptions<T> {
    /// V(2,2)-Zyklus mit Gauß-Seidel-Glättung.
    fn default() -> Self {
        Self {
            pre_smooth: 2,
            post_smooth: 2,
            cycle: CycleType::V,
            smoother: Smoother::GaussSeidel,
        }
    }
}

/// Abbruchkriterien für die Löser [`solve_1d`] und [`solve_2d`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolveOptions<T> {
    /// Relative Toleranz: Abbruch bei `‖f − A·u‖ ≤ tol · ‖f‖`.
    pub tol: T,
    /// Höchstzahl der Zyklen.
    pub max_cycles: usize,
}

impl<T: Real> Default for SolveOptions<T> {
    /// `tol = ε^0,75` (etwa `1.8e-12` bei `f64`), höchstens 100 Zyklen.
    fn default() -> Self {
        Self {
            tol: T::EPSILON.powf(T::from_f64(0.75)),
            max_cycles: 100,
        }
    }
}

/// Ergebnis eines erfolgreichen Multigrid-Lösers (die Lösung steht im übergebenen `u`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultigridResult<T> {
    /// Anzahl der ausgeführten Zyklen (0, wenn die Startnäherung schon genügt).
    pub cycles: usize,
    /// Euklidische Norm des Residuums `‖f − A·u‖` der Lösung (mit den Skalierungen aus dem
    /// [Modul](self)).
    pub residual_norm: T,
}

/// Fehler der Multigrid-Funktionen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MultigridError {
    /// `n` hat nicht die Form `2ᵏ − 1` (mit `k ≥ 1`).
    InvalidGridSize,
    /// Slices haben die falsche Länge, der Arbeitsspeicher ist zu kurz, ein Jacobi-Gewicht ist nicht
    /// positiv oder `pre_smooth ≠ post_smooth` bei einem Vorkonditionierer.
    InvalidInput,
    /// NaN oder Unendlich in der rechten Seite, der Startnäherung oder im Verlauf.
    NotFinite,
    /// Die Toleranz wurde in `max_cycles` Zyklen nicht erreicht (`u` enthält die letzte Näherung).
    NoConvergence,
}

impl fmt::Display for MultigridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            MultigridError::InvalidGridSize => "Gittergröße ist nicht von der Form 2^k − 1",
            MultigridError::InvalidInput => "ungültige Eingabe oder zu wenig Arbeitsspeicher",
            MultigridError::NotFinite => "NaN oder unendlicher Wert aufgetreten",
            MultigridError::NoConvergence => "Toleranz in max_cycles Zyklen nicht erreicht",
        })
    }
}

impl core::error::Error for MultigridError {}

/// Anzahl `k` der Ebenen, wenn `n = 2ᵏ − 1` (mit `k ≥ 1`), sonst `None`.
///
/// ```
/// use azmath_multigrid::grid_levels;
///
/// assert_eq!(grid_levels(1), Some(1));
/// assert_eq!(grid_levels(63), Some(6));
/// assert_eq!(grid_levels(64), None);
/// ```
pub const fn grid_levels(n: usize) -> Option<usize> {
    if n == 0 || (n + 1) & n != 0 {
        return None;
    }
    let mut k = 0;
    let mut m = n + 1;
    while m > 1 {
        m >>= 1;
        k += 1;
    }
    Some(k)
}

pub(crate) fn check_options<T: Real>(opts: &MultigridOptions<T>) -> Result<(), MultigridError> {
    if let Smoother::WeightedJacobi(omega) = opts.smoother {
        if omega.is_nan() || omega <= T::ZERO || omega.is_infinite() {
            return Err(MultigridError::InvalidInput);
        }
    }
    Ok(())
}

/// Wie oft das Grobgitterproblem pro Ebene gelöst wird.
pub(crate) fn gamma(c: CycleType) -> usize {
    match c {
        CycleType::V => 1,
        CycleType::W => 2,
    }
}

pub(crate) fn finite_slice<T: Real>(v: &[T]) -> bool {
    v.iter().all(|x| !(x.is_nan() || x.is_infinite()))
}
