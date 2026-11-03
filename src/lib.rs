//! Ein generisches [`Real`]-Trait für `f32` und `f64` mit austauschbarem Backend
//! (`std`, `libm` oder `micromath`) und numerische Verfahren für `no_std`-Umgebungen.
//!
//! Alles liegt in diesem einen Crate, gegliedert in Module:
//!
//! | Pfad | Inhalt |
//! |---|---|
//! | [`Real`] | Zahlentrait mit Backends |
//! | [`linalg`] | Zerlegungen, iterative Löser, Eigenwerte, SVD |
//! | [`roots`] | Nullstellen in einer Veränderlichen |
//! | [`integrate`] | numerische Integration |
//! | [`interp`] | Interpolation, Splines |
//! | [`optimize`] | Newton, Gauß-Newton, Levenberg-Marquardt |
//! | [`ode`] | gewöhnliche Differentialgleichungen |
//! | [`multigrid`] | Multigrid für Poisson |
//! | [`autodiff`] | Automatisches Differenzieren (`Jet`) |
//! | [`complex`] | komplexe Zahlen `Complex<T>` |
//! | [`diff`] | numerisches Differenzieren (Differenzen, Ridders, Gradient, Hessematrix, Jacobi-Matrix) |
//! | [`fft`] | schnelle Fourier-Transformation (beliebige Länge) |
//! | [`geometry`] | Quaternionen, Rotationen, Euler-Winkel |
//! | `opt` | lineare Programme (Simplex), später QP/NLP/MILP (Feature `alloc`) |
//! | [`sparse`] | dünn besetzte Matrizen (CSR, COO) |

#![no_std]
#![warn(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod real;
pub use real::*;

pub mod autodiff;
pub mod complex;
pub mod diff;
pub mod fft;
pub mod geometry;
pub mod integrate;
pub mod interp;
pub mod linalg;
pub mod multigrid;
pub mod ode;
#[cfg(feature = "alloc")]
pub mod opt;
pub mod optimize;
pub mod roots;
pub mod sparse;
pub use complex::*;
