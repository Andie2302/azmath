//! Finite Volumen für Erhaltungsgleichungen `uₜ + f(u)ₓ = 0` auf gleichmäßigem 1D-Gitter, ohne Allokator.
//!
//! | Funktion | Inhalt |
//! |---|---|
//! | [`rusanov`], [`lax_friedrichs`] | numerische Flüsse `F(uₗ, uᵣ)` |
//! | [`fvm_rhs_1d`] | Linienmethode erster Ordnung: `duᵢ/dt = −(Fᵢ₊½ − Fᵢ₋½)/Δx` |
//! | [`muscl_rhs_1d`] | zweite Ordnung durch lineare Rekonstruktion mit Minmod-Limiter |
//! | [`Boundary`] | periodisch, Nullgradient (Ausfluss) oder feste Randwerte (Geisterzellen) |
//!
//! Die rechten Seiten passen als `f` zu `rk4`, `rk45` und den impliziten Lösern. Für Erhaltungsschemata gilt: Bei periodischem Rand
//! bleibt `Σ uᵢ` bis auf Rundung konstant. Die Zeitschrittweite muss die CFL-Bedingung `Δt·max|f′| ≤ Δx` (für `muscl` mit einem
//! Strong-Stability-Verfahren etwa `½` davon) erfüllen.
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::fvm::{fvm_rhs_1d, rusanov, Boundary};
//!
//! // lineare Advektion u_t + u_x = 0: ein Euler-Schritt, Masse bleibt erhalten
//! let mut u = [0.0f64, 1.0, 0.0, 0.0];
//! let mut du = [0.0; 4];
//! let flux = |a: f64, b: f64| rusanov(|u| u, |_| 1.0, a, b);
//! fvm_rhs_1d(&u, 0.1, Boundary::Periodic, flux, &mut du);
//! assert!(du.iter().sum::<f64>().abs() < 1e-12);
//! # }
//! ```

use crate::Real;

/// Randbedingung über Geisterzellen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Boundary<T> {
    /// Periodisch.
    Periodic,
    /// Nullgradient: Geisterzelle = Randzelle (freier Ausfluss).
    ZeroGradient,
    /// Feste Werte `(links, rechts)` in den Geisterzellen.
    Fixed(T, T),
}

/// Rusanov-Fluss (lokales Lax-Friedrichs): `½(f(uₗ) + f(uᵣ)) − ½·a·(uᵣ − uₗ)`, `a = max(|f′(uₗ)|, |f′(uᵣ)|)`.
pub fn rusanov<T: Real>(f: impl Fn(T) -> T, speed: impl Fn(T) -> T, ul: T, ur: T) -> T {
    let half = T::from_f64(0.5);
    let a = speed(ul).abs().max(speed(ur).abs());
    half * (f(ul) + f(ur)) - half * a * (ur - ul)
}

/// Lax-Friedrichs-Fluss mit fester Dissipation `alpha ≥ max|f′|`: `½(f(uₗ) + f(uᵣ)) − ½·alpha·(uᵣ − uₗ)`.
pub fn lax_friedrichs<T: Real>(f: impl Fn(T) -> T, alpha: T, ul: T, ur: T) -> T {
    let half = T::from_f64(0.5);
    half * (f(ul) + f(ur)) - half * alpha * (ur - ul)
}

fn ghost<T: Real>(u: &[T], i: isize, bc: Boundary<T>) -> T {
    let n = u.len() as isize;
    if i >= 0 && i < n {
        return u[i as usize];
    }
    match bc {
        Boundary::Periodic => u[i.rem_euclid(n) as usize],
        Boundary::ZeroGradient => {
            if i < 0 {
                u[0]
            } else {
                u[(n - 1) as usize]
            }
        }
        Boundary::Fixed(l, r) => {
            if i < 0 {
                l
            } else {
                r
            }
        }
    }
}

/// Rechte Seite der Semidiskretisierung erster Ordnung: `du[i] = −(F(uᵢ, uᵢ₊₁) − F(uᵢ₋₁, uᵢ))/dx`. `u.len() ≥ 2`, `du.len() ≥ u.len()`,
/// `dx > 0` (sonst Panik).
pub fn fvm_rhs_1d<T: Real>(
    u: &[T],
    dx: T,
    bc: Boundary<T>,
    numerical_flux: impl Fn(T, T) -> T,
    du: &mut [T],
) {
    let n = u.len();
    assert!(
        n >= 2 && du.len() >= n && dx > T::ZERO,
        "fvm_rhs_1d: ungültige Eingabe"
    );
    let mut f_left = numerical_flux(ghost(u, -1, bc), u[0]);
    for i in 0..n {
        let f_right = numerical_flux(u[i], ghost(u, i as isize + 1, bc));
        du[i] = -(f_right - f_left) / dx;
        f_left = f_right;
    }
}

fn minmod<T: Real>(a: T, b: T) -> T {
    if a * b <= T::ZERO {
        T::ZERO
    } else if a.abs() < b.abs() {
        a
    } else {
        b
    }
}

/// Wie [`fvm_rhs_1d`], aber zweiter Ordnung: lineare Rekonstruktion mit Minmod-Steigungen (MUSCL). Für die Zeitintegration ein
/// Verfahren mindestens zweiter Ordnung wählen (zum Beispiel `rk4`).
pub fn muscl_rhs_1d<T: Real>(
    u: &[T],
    dx: T,
    bc: Boundary<T>,
    numerical_flux: impl Fn(T, T) -> T,
    du: &mut [T],
) {
    let n = u.len();
    assert!(
        n >= 2 && du.len() >= n && dx > T::ZERO,
        "muscl_rhs_1d: ungültige Eingabe"
    );
    let half = T::from_f64(0.5);
    let slope = |i: isize| {
        let (a, b, c) = (ghost(u, i - 1, bc), ghost(u, i, bc), ghost(u, i + 1, bc));
        minmod(b - a, c - b)
    };
    // Fluss an der Grenze zwischen Zelle i und i+1
    let interface = |i: isize| {
        let ul = ghost(u, i, bc) + half * slope(i);
        let ur = ghost(u, i + 1, bc) - half * slope(i + 1);
        numerical_flux(ul, ur)
    };
    let mut f_left = interface(-1);
    for i in 0..n {
        let f_right = interface(i as isize);
        du[i] = -(f_right - f_left) / dx;
        f_left = f_right;
    }
}
