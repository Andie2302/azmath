//! Finite Differenzen: Stencil-Gewichte für beliebige Gitter (Fornberg) und Differenzenoperatoren auf gleichmäßigen
//! Gittern, ohne Allokator.
//!
//! | Funktion | Inhalt |
//! |---|---|
//! | [`fornberg_weights`] | Gewichte der `m`-ten Ableitung für beliebige Knoten |
//! | [`first_derivative_1d`], [`second_derivative_1d`] | Ableitungen auf gleichmäßigem 1D-Gitter (2. Ordnung, auch am Rand) |
//! | [`laplacian_2d`] | 5-Punkte-Laplace-Operator auf einem Rechteckgitter |
//! | `poisson` | Zellzentrierter 2D-Poisson-Operator (CSR) mit Dirichlet/Neumann/periodisch, Nullraum (Feature `alloc`) |
//! | [`heat_rhs_1d`] | Rechte Seite der Linienmethode für `u_t = α·u_xx` mit Dirichlet-Rand, passend zu den ODE-Lösern |
//!
//! Slices zu kurz oder Gitterabstand/Dimensionen ungültig: **Panik** (wie bei [`diff`](crate::diff)).
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::fdm::{fornberg_weights, fornberg_work_len};
//!
//! // zentrale zweite Ableitung auf h = 0,5: Gewichte [1, −2, 1]/h²
//! let mut w = [0.0f64; 3];
//! let mut work = [0.0; fornberg_work_len(3, 2)];
//! fornberg_weights(2, 0.0, &[-0.5, 0.0, 0.5], &mut w, &mut work);
//! assert!((w[0] - 4.0).abs() < 1e-12 && (w[1] + 8.0).abs() < 1e-12 && (w[2] - 4.0).abs() < 1e-12);
//! # }
//! ```

use crate::Real;

#[cfg(feature = "alloc")]
pub mod poisson;

/// Arbeitsspeicher für [`fornberg_weights`] bei `n_nodes` Knoten und Ableitungsordnung `order`.
pub const fn fornberg_work_len(n_nodes: usize, order: usize) -> usize {
    n_nodes * (order + 1)
}

/// Gewichte `w[j]` mit `f^(order)(x0) ≈ Σ w[j]·f(nodes[j])` (Fornberg 1988), exakt für Polynome vom Grad `< n_nodes`.
///
/// `weights`: mindestens `nodes.len()` Elemente; `work`: mindestens [`fornberg_work_len`]. Die Knoten müssen verschieden
/// sein und `order < nodes.len()` gelten (sonst Panik).
pub fn fornberg_weights<T: Real>(
    order: usize,
    x0: T,
    nodes: &[T],
    weights: &mut [T],
    work: &mut [T],
) {
    let n = nodes.len();
    let m = order;
    assert!(n > m, "fornberg_weights: zu wenige Knoten");
    assert!(
        weights.len() >= n && work.len() >= fornberg_work_len(n, m),
        "fornberg_weights: Slices zu kurz"
    );
    let w = m + 1;
    let c = &mut work[..n * w];
    c.iter_mut().for_each(|v| *v = T::ZERO);
    let mut c1 = T::ONE;
    let mut c4 = nodes[0] - x0;
    c[0] = T::ONE;
    for i in 1..n {
        let mn = i.min(m);
        let mut c2 = T::ONE;
        let c5 = c4;
        c4 = nodes[i] - x0;
        for j in 0..i {
            let c3 = nodes[i] - nodes[j];
            assert!(
                c3 != T::ZERO,
                "fornberg_weights: Knoten müssen verschieden sein"
            );
            c2 *= c3;
            if j + 1 == i {
                for k in (1..=mn).rev() {
                    c[i * w + k] = c1
                        * (T::from_usize(k) * c[(i - 1) * w + k - 1] - c5 * c[(i - 1) * w + k])
                        / c2;
                }
                c[i * w] = -c1 * c5 * c[(i - 1) * w] / c2;
            }
            for k in (1..=mn).rev() {
                c[j * w + k] = (c4 * c[j * w + k] - T::from_usize(k) * c[j * w + k - 1]) / c3;
            }
            c[j * w] = c4 * c[j * w] / c3;
        }
        c1 = c2;
    }
    for j in 0..n {
        weights[j] = c[j * w + m];
    }
}

/// Erste Ableitung auf einem gleichmäßigen Gitter mit Abstand `h` (`u.len() ≥ 3`): zentral im Inneren, einseitig am Rand,
/// jeweils 2. Ordnung. `out.len() ≥ u.len()`.
pub fn first_derivative_1d<T: Real>(u: &[T], h: T, out: &mut [T]) {
    let n = u.len();
    assert!(
        n >= 3 && out.len() >= n && h != T::ZERO,
        "first_derivative_1d: ungültige Eingabe"
    );
    let (two, three, four) = (T::from_f64(2.0), T::from_f64(3.0), T::from_f64(4.0));
    for i in 1..n - 1 {
        out[i] = (u[i + 1] - u[i - 1]) / (two * h);
    }
    out[0] = (-three * u[0] + four * u[1] - u[2]) / (two * h);
    out[n - 1] = (three * u[n - 1] - four * u[n - 2] + u[n - 3]) / (two * h);
}

/// Zweite Ableitung auf einem gleichmäßigen Gitter mit Abstand `h` (`u.len() ≥ 4`): zentral im Inneren (2. Ordnung),
/// einseitig am Rand (2. Ordnung). `out.len() ≥ u.len()`.
pub fn second_derivative_1d<T: Real>(u: &[T], h: T, out: &mut [T]) {
    let n = u.len();
    assert!(
        n >= 4 && out.len() >= n && h != T::ZERO,
        "second_derivative_1d: ungültige Eingabe"
    );
    let (two, four, five) = (T::from_f64(2.0), T::from_f64(4.0), T::from_f64(5.0));
    let h2 = h * h;
    for i in 1..n - 1 {
        out[i] = (u[i - 1] - two * u[i] + u[i + 1]) / h2;
    }
    out[0] = (two * u[0] - five * u[1] + four * u[2] - u[3]) / h2;
    out[n - 1] = (two * u[n - 1] - five * u[n - 2] + four * u[n - 3] - u[n - 4]) / h2;
}

/// 5-Punkte-Laplace-Operator `u_xx + u_yy` auf einem `nx × ny`-Gitter (zeilenweise, Index `j·nx + i`) mit Abständen
/// `hx`, `hy`. Nur innere Punkte werden berechnet, Randwerte von `out` werden auf 0 gesetzt.
pub fn laplacian_2d<T: Real>(u: &[T], nx: usize, ny: usize, hx: T, hy: T, out: &mut [T]) {
    assert!(
        nx >= 3
            && ny >= 3
            && u.len() >= nx * ny
            && out.len() >= nx * ny
            && hx != T::ZERO
            && hy != T::ZERO,
        "laplacian_2d: ungültige Eingabe"
    );
    let (ix2, iy2) = (T::ONE / (hx * hx), T::ONE / (hy * hy));
    let two = T::from_f64(2.0);
    out[..nx * ny].iter_mut().for_each(|v| *v = T::ZERO);
    for j in 1..ny - 1 {
        for i in 1..nx - 1 {
            let k = j * nx + i;
            let c = u[k];
            out[k] =
                (u[k - 1] - two * c + u[k + 1]) * ix2 + (u[k - nx] - two * c + u[k + nx]) * iy2;
        }
    }
}

/// Rechte Seite der Linienmethode für `u_t = α·u_xx` mit festen Randwerten (Dirichlet): `du[i] = α·(u[i−1] − 2u[i] + u[i+1])/h²`
/// im Inneren, `du = 0` an den Rändern. Als `f` für `rk45`, `radau5` usw. verwendbar.
pub fn heat_rhs_1d<T: Real>(alpha: T, u: &[T], h: T, du: &mut [T]) {
    let n = u.len();
    assert!(
        n >= 3 && du.len() >= n && h != T::ZERO,
        "heat_rhs_1d: ungültige Eingabe"
    );
    let k = alpha / (h * h);
    let two = T::from_f64(2.0);
    du[0] = T::ZERO;
    du[n - 1] = T::ZERO;
    for i in 1..n - 1 {
        du[i] = k * (u[i - 1] - two * u[i] + u[i + 1]);
    }
}
