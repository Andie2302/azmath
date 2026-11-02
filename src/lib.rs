//! Ein generisches [`Real`]-Trait für `f32` und `f64` mit austauschbarem Backend
//! (`std`, `libm` oder `micromath`) für `no_std`-Umgebungen.

#![no_std]
#![warn(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

/// Kern des Crates: das [`Real`]-Trait und seine Implementierungen.
pub mod real;
pub use real::*;
