//! Lineare Algebra: Vektoren, Matrizen und der LU-Löser.
//!
//! - [`Vector`] und [`Matrix`] haben eine zur Compile-Zeit feste Größe (Const Generics),
//!   liegen auf dem Stack und brauchen keinen Allokator.
//! - Mit dem Feature `alloc` kommen [`DynVector`] und [`DynMatrix`] mit zur Laufzeit
//!   festgelegter Größe hinzu.
//! - [`LuDecomposition`] zerlegt quadratische Matrizen (mit Teilpivotisierung) und löst
//!   damit Gleichungssysteme, berechnet Determinanten und Inverse.
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

mod error;
mod kernel;
mod lu;
mod matrix;
mod vector;

#[cfg(feature = "alloc")]
mod dynamic;

pub use error::LinalgError;
pub use lu::LuDecomposition;
pub use matrix::Matrix;
pub use vector::Vector;

#[cfg(feature = "alloc")]
pub use dynamic::{DynLu, DynMatrix, DynVector};
