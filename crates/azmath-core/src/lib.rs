//! Grundlage von azmath: das [`Real`]-Trait für `f32` und `f64` mit austauschbarem Backend
//! (`std`, `libm` oder `micromath`) für `no_std`-Umgebungen.
//!
//! Dieses Crate bildet die Basis aller anderen `azmath-*`-Crates. Es wird meist über die Fassade
//! [`azmath`](https://docs.rs/azmath) verwendet.

#![no_std]
#![warn(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

/// Das `Real`-Trait und die Backend-Implementierungen.
pub mod real;
pub use real::*;
