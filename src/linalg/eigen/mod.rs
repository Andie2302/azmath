//! Eigenwerte, Eigenvektoren und Singulärwertzerlegung.
//!
//! | Funktion / Typ | Problem |
//! |---|---|
//! | [`Matrix::symmetric_eigen`](crate::linalg::Matrix::symmetric_eigen) | **alle** Eigenwerte und -vektoren einer symmetrischen Matrix (Jacobi) |
//! | [`lanczos`], [`lanczos_ritz`] | extreme Eigenwerte großer symmetrischer Operatoren (Krylov-Raum, Ritz-Werte) |
//! | [`arnoldi`] | Arnoldi-Zerlegung `A·V = V·H` für beliebige Operatoren (Ritz-Werte folgen mit dem nichtsymmetrischen Eigenlöser) |
//! | [`Matrix::svd`](crate::linalg::Matrix::svd) | Singulärwertzerlegung `A = U·Σ·Vᵀ`; Rang, Konditionszahl, Pseudoinverse, Lösung bei Rangmangel |
//! | [`power_iteration`] | betragsgrößter Eigenwert, **matrixfrei** (nur `A·x`) |
//! | [`inverse_iteration`] | Eigenwert nahe einem Wert (`shift`) |
//!
//! Die Jacobi-Verfahren liefern sehr genaue Ergebnisse (auch kleine Eigen- und Singulärwerte),
//! kosten aber `O(n³)` pro Sweep und lohnen daher für kleine bis mittlere Matrizen. Die Potenzmethode
//! eignet sich für große, dünn besetzte Operatoren, wenn nur der größte Eigenwert gebraucht wird.
//! Mit dem Feature `alloc` gibt es die dynamischen Gegenstücke [`DynSymmetricEigen`] und [`DynSvd`]
//! (Methoden direkt auf `DynMatrix`).
//!
//! Nicht enthalten sind bisher das vollständige Eigenwertproblem für **nichtsymmetrische** Matrizen
//! (QR-Algorithmus, komplexe Eigenwerte) und Teilspektren großer Matrizen (Lanczos, Arnoldi).
//!
//! Hinweis zu `micromath`: Die Verfahren nutzen `sqrt`; mit der bewusst ungenauen Näherung werden
//! die Ergebnisse entsprechend ungenau. Für verlässliche Werte `std` oder `libm` verwenden.

mod jacobi;
mod kernel;
mod lanczos;
mod power;
mod svd;

#[cfg(feature = "alloc")]
mod dynamic;

pub use jacobi::SymmetricEigen;
pub use lanczos::{
    arnoldi, arnoldi_work_len, lanczos, lanczos_ritz, lanczos_ritz_work_len, lanczos_work_len,
};
pub use power::{EigenResult, inverse_iteration, power_iteration, power_iteration_work_len};
pub use svd::Svd;

#[cfg(feature = "alloc")]
pub use dynamic::{DynSvd, DynSymmetricEigen};

/// Höchstzahl der Jacobi-Sweeps (üblich sind 5 bis 10).
pub(crate) const MAX_SWEEPS: usize = 60;

/// Füllt `x` mit einem **deterministischen Pseudozufallsvektor** mit Einträgen in `[0,5; 1,5)`.
///
/// Als Startvektor für die Potenzmethode ist er robuster als ein glatter Vektor: er hat Anteile in allen
/// Frequenzen, also auch im oft hochfrequenten (alternierenden) dominanten Eigenvektor, und ist dennoch
/// reproduzierbar (feste Saat).
pub(crate) fn fill_start_vector<T: crate::Real>(x: &mut [T]) {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for v in x.iter_mut() {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let r = (state >> 11) as f64 / (1u64 << 53) as f64;
        *v = T::from_f64(0.5 + r);
    }
}
