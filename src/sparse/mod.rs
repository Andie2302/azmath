//! Dünn besetzte Matrizen: Formate, die nur die Einträge ungleich 0 speichern.
//!
//! | Typ | Zweck | Allokator |
//! |---|---|---|
//! | [`CsrView`] | geliehene Sicht auf CSR-Arrays (Zeilenzeiger, Spaltenindizes, Werte) | nein |
//! | `CsrMatrix` | CSR-Matrix mit eigenem Speicher: Matrix-Vektor-Produkt, Transponierte, Summe, Umwandlungen | `alloc` |
//! | `CooMatrix` | Tripletliste zum Aufbauen (Duplikate werden summiert) | `alloc` |
//! | `BandMatrix` | Bandmatrix mit direkten Lösern `BandLu` (Pivotisierung) und `BandCholesky` (SPD) | `alloc` |
//!
//! CSR-Matrizen implementieren [`LinearOperator`](crate::linalg::LinearOperator) und laufen damit unverändert
//! mit den iterativen Lösern [`cg`](crate::linalg::iterative::cg), [`gmres`](crate::linalg::iterative::gmres) und
//! [`bicgstab`](crate::linalg::iterative::bicgstab).
//!
//! ```
//! use azmath::sparse::CsrView;
//! use azmath::linalg::LinearOperator;
//!
//! // [[2, 0, 1],
//! //  [0, 3, 0]]   ohne Allokator, direkt über Slices
//! let a = CsrView::new(2, 3, &[0, 2, 3], &[0, 2, 1], &[2.0f64, 1.0, 3.0]).unwrap();
//! let mut y = [0.0; 2];
//! a.apply(&[1.0, 1.0, 1.0], &mut y);
//! assert_eq!(y, [3.0, 3.0]);
//! ```

#[cfg(feature = "alloc")]
mod band;
#[cfg(feature = "alloc")]
mod coo;
#[cfg(feature = "alloc")]
mod csr;
mod error;
mod view;

#[cfg(feature = "alloc")]
pub use band::{BandCholesky, BandLu, BandMatrix};
#[cfg(feature = "alloc")]
pub use coo::CooMatrix;
#[cfg(feature = "alloc")]
pub use csr::CsrMatrix;
pub use error::SparseError;
pub use view::CsrView;
