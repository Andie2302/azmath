//! Ein generisches [`Real`]-Trait für `f32` und `f64` mit austauschbarem Backend
//! (`std`, `libm` oder `micromath`) und numerische Verfahren für `no_std`-Umgebungen.
//!
//! Diese Fassade fasst die Teil-Crates zusammen und behält die gewohnten Pfade bei
//! (`azmath::Real`, `azmath::linalg`, `azmath::roots`, …). Wer nur einen Teil braucht, kann die
//! Crates (`azmath-core`, `azmath-linalg`, …) auch einzeln verwenden.
//!
//! | Pfad | Crate | Inhalt |
//! |---|---|---|
//! | [`Real`] | `azmath-core` | Zahlentrait mit Backends |
//! | [`linalg`] | `azmath-linalg` | Zerlegungen, iterative Löser, Eigenwerte, SVD |
//! | [`roots`] | `azmath-roots` | Nullstellen in einer Veränderlichen |
//! | [`integrate`] | `azmath-integrate` | numerische Integration |
//! | [`interp`] | `azmath-interp` | Interpolation, Splines |
//! | [`optimize`] | `azmath-optimize` | Newton, Gauß-Newton, Levenberg-Marquardt |
//! | [`ode`] | `azmath-ode` | gewöhnliche Differentialgleichungen |
//! | [`multigrid`] | `azmath-multigrid` | Multigrid für Poisson |

#![no_std]
#![warn(missing_docs)]

pub use azmath_core::*;

pub use azmath_integrate as integrate;
pub use azmath_interp as interp;
pub use azmath_linalg as linalg;
pub use azmath_multigrid as multigrid;
pub use azmath_ode as ode;
pub use azmath_optimize as optimize;
pub use azmath_roots as roots;
