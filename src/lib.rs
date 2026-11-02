//! Ein generisches [`Real`]-Trait für `f32` und `f64` mit austauschbarem Backend
//! (`std`, `libm` oder `micromath`) für `no_std`-Umgebungen.

#![no_std]
#![warn(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "alloc")]
extern crate alloc;

/// Kern des Crates: das [`Real`]-Trait und seine Implementierungen.
pub mod real;
pub use real::*;

pub mod linalg;

pub mod roots;

pub mod integrate;

pub mod optimize;

pub mod ode;
