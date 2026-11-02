//! Lineare Algebra: Vektoren, Matrizen und der LU-Löser.
//!
//! - [`Vector`] und [`Matrix`] haben eine zur Compile-Zeit feste Größe (Const Generics),
//!   liegen auf dem Stack und brauchen keinen Allokator.
//! - Mit dem Feature `alloc` kommen [`DynVector`] und [`DynMatrix`] mit zur Laufzeit
//!   festgelegter Größe hinzu.
//! - [`LuDecomposition`] zerlegt quadratische Matrizen (mit Teilpivotisierung) und löst
//!   damit Gleichungssysteme, berechnet Determinanten und Inverse.
//! - Dreieckssysteme löst man direkt mit [`Matrix::solve_lower`] und [`Matrix::solve_upper`]
//!   (sowie den `_unit`- und `_transposed`-Varianten); sie sind die Bausteine der Zerlegungen.
//! - [`iterative`] enthält matrixfreie iterative Löser (CG) für große Systeme, die gegen ein
//!   [`LinearOperator`] arbeiten.
//! - [`solve_tridiagonal`] und [`solve_cyclic_tridiagonal`] lösen tridiagonale Systeme (Thomas-Algorithmus)
//!   in `O(n)`, etwa für Splines und Differenzenverfahren.
//! - [`eigen`] berechnet Eigenwerte und -vektoren (Jacobi, Potenzmethode) und die Singulärwertzerlegung.
//! - Matrixnormen: `norm_frobenius`, `norm_1`, `norm_inf` und `norm_2` (größter Singulärwert), siehe `Matrix::norm_2`.
//! - Äquilibrierung (Ruiz, Sinkhorn): `Matrix::equilibrate_ruiz`, `Matrix::equilibrate_sinkhorn`, Ergebnis [`Equilibrated`].
//! - Orthonormalisierung und Polarzerlegung: `Matrix::orthonormalize_mgs`, `QrDecomposition::thin_q`, `Matrix::polar`,
//!   `Matrix::nearest_rotation`.
//! - Gauß-Elimination als benannte Funktionen: `Matrix::gauss_solve`, `Matrix::rank`, `Matrix::row_echelon`.
//! - [`Cholesky`] löst symmetrisch positiv definite Systeme schneller als LU.
//! - [`QrDecomposition`] (Householder) löst überbestimmte Systeme im Sinne der kleinsten
//!   Quadrate, etwa für Ausgleichsgeraden.
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::linalg::{Matrix, Vector};
//!
//! // 2x + y = 5,  x + 3y = 10  ->  x = 1, y = 3
//! let a = Matrix::new([[2.0f64, 1.0], [1.0, 3.0]]);
//! let b = Vector::new([5.0, 10.0]);
//! let x = a.solve(&b).unwrap();
//! assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 3.0).abs() < 1e-12);
//! # }
//! ```

mod cholesky;
pub mod eigen;
mod equilibrate;
mod error;
mod gauss;
pub mod iterative;
/// Interne Rechenkerne auf Slices (auch von `optimize` genutzt). Keine öffentliche API.
pub(crate) mod kernel;
mod lu;
mod matrix;
mod norm;
mod operator;
mod orthogonal;
mod qr;
mod tridiagonal;
mod vector;

#[cfg(feature = "alloc")]
mod dynamic;

pub use cholesky::Cholesky;
pub use equilibrate::{EquilibrateOptions, Equilibrated};
pub use error::LinalgError;
pub use gauss::RowEchelon;
pub use lu::LuDecomposition;
pub use matrix::Matrix;
pub use operator::{FnOperator, LinearOperator};
pub use orthogonal::{Orthonormalized, Polar};
pub use qr::QrDecomposition;
pub use tridiagonal::{
    cyclic_tridiagonal_work_len, solve_cyclic_tridiagonal, solve_tridiagonal, tridiagonal_work_len,
};
pub use vector::Vector;

#[cfg(feature = "alloc")]
pub use dynamic::{DynCholesky, DynLu, DynMatrix, DynQr, DynVector};
#[cfg(feature = "alloc")]
pub use equilibrate::DynEquilibrated;
#[cfg(feature = "alloc")]
pub use gauss::DynRowEchelon;
#[cfg(feature = "alloc")]
pub use orthogonal::{DynOrthonormalized, DynPolar};
