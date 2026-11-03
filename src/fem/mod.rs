//! Finite Elemente (Feature `alloc`): lineare Elemente (P1) in 1D und auf Dreiecksnetzen in 2D, Assemblierung nach
//! [`CsrMatrix`] und Dirichlet-Randbedingungen.
//!
//! | Funktion | Inhalt |
//! |---|---|
//! | [`assemble_1d`] | Steifigkeits- und Massenmatrix, Lastvektor auf beliebigem 1D-Gitter |
//! | [`assemble_2d`] | Steifigkeits- und Massenmatrix auf Dreiecksnetzen |
//! | [`load_vector_2d`] | Lastvektor `∫ f·φᵢ` (Quadratur in den Kantenmitten, exakt bis Grad 2) |
//! | [`apply_dirichlet`] | symmetrisches Einbringen fester Randwerte |
//! | [`rectangle_mesh`], [`boundary_nodes`] | Hilfsfunktionen für Rechtecknetze |
//!
//! Die Matrizen sind symmetrisch; mit `SparseCholesky` aus [`sparse`](crate::sparse) lässt sich das System direkt lösen.
//! Dreiecke müssen gegen den Uhrzeigersinn orientiert sein oder dürfen beliebig orientiert sein (es zählt der Betrag der Fläche).
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::fem::{apply_dirichlet, assemble_1d};
//! use azmath::sparse::{Ordering, SparseCholesky};
//!
//! // −u'' = 1 auf (0, 1), u(0) = u(1) = 0  ⇒  u = x(1 − x)/2
//! let x: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
//! let (k, _m, f) = assemble_1d(&x, |_| 1.0);
//! let (k, f) = apply_dirichlet(&k, &f, &[(0, 0.0), (10, 0.0)]);
//! let u = SparseCholesky::new(&k, Ordering::Natural).unwrap().solve(&f).unwrap();
//! assert!((u[5] - 0.125).abs() < 1e-12);
//! # }
//! ```

use crate::Real;
use crate::sparse::{CooMatrix, CsrMatrix};
use alloc::vec;
use alloc::vec::Vec;

/// P1-Elemente auf einem 1D-Gitter `x[0] < x[1] < …`: liefert `(K, M, f)` mit `K` Steifigkeitsmatrix (`∫ φᵢ′φⱼ′`), `M`
/// Massenmatrix (`∫ φᵢφⱼ`) und Lastvektor `f` (`∫ source·φᵢ`, Trapezregel). Panik bei weniger als 2 Knoten oder nicht aufsteigendem Gitter.
pub fn assemble_1d<T: Real>(
    x: &[T],
    source: impl Fn(T) -> T,
) -> (CsrMatrix<T>, CsrMatrix<T>, Vec<T>) {
    let n = x.len();
    assert!(n >= 2, "assemble_1d: mindestens 2 Knoten");
    let mut k = CooMatrix::new(n, n);
    let mut m = CooMatrix::new(n, n);
    let mut f = vec![T::ZERO; n];
    let (half, sixth, third) = (
        T::from_f64(0.5),
        T::from_f64(1.0 / 6.0),
        T::from_f64(1.0 / 3.0),
    );
    for e in 0..n - 1 {
        let h = x[e + 1] - x[e];
        assert!(h > T::ZERO, "assemble_1d: Gitter muss aufsteigend sein");
        let (a, b) = (e, e + 1);
        for &(i, j, v) in &[
            (a, a, T::ONE / h),
            (b, b, T::ONE / h),
            (a, b, -T::ONE / h),
            (b, a, -T::ONE / h),
        ] {
            let _ = k.push(i, j, v);
        }
        for &(i, j, v) in &[
            (a, a, third * h),
            (b, b, third * h),
            (a, b, sixth * h),
            (b, a, sixth * h),
        ] {
            let _ = m.push(i, j, v);
        }
        f[a] += half * h * source(x[a]);
        f[b] += half * h * source(x[b]);
    }
    (CsrMatrix::from_coo(&k), CsrMatrix::from_coo(&m), f)
}

/// P1-Elemente auf einem Dreiecksnetz: `(K, M)` mit `K` der Steifigkeitsmatrix (`∫ ∇φᵢ·∇φⱼ`) und `M` der Massenmatrix.
/// Panik bei Knotenindizes außerhalb oder entarteten Dreiecken (Fläche 0).
pub fn assemble_2d<T: Real>(
    points: &[[T; 2]],
    triangles: &[[usize; 3]],
) -> (CsrMatrix<T>, CsrMatrix<T>) {
    let n = points.len();
    let mut k = CooMatrix::new(n, n);
    let mut m = CooMatrix::new(n, n);
    let (half, twelfth) = (T::from_f64(0.5), T::from_f64(1.0 / 12.0));
    for tri in triangles {
        let p = [points[tri[0]], points[tri[1]], points[tri[2]]];
        let det =
            (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
        let area = half * det.abs();
        assert!(area > T::ZERO, "assemble_2d: entartetes Dreieck");
        // Gradienten der Hutfunktionen: ∇φᵢ = (yⱼ − yₖ, xₖ − xⱼ)/det
        let g = [
            [(p[1][1] - p[2][1]) / det, (p[2][0] - p[1][0]) / det],
            [(p[2][1] - p[0][1]) / det, (p[0][0] - p[2][0]) / det],
            [(p[0][1] - p[1][1]) / det, (p[1][0] - p[0][0]) / det],
        ];
        for i in 0..3 {
            for j in 0..3 {
                let kij = area * (g[i][0] * g[j][0] + g[i][1] * g[j][1]);
                let mij = if i == j { T::from_f64(2.0) } else { T::ONE } * twelfth * area;
                let _ = k.push(tri[i], tri[j], kij);
                let _ = m.push(tri[i], tri[j], mij);
            }
        }
    }
    (CsrMatrix::from_coo(&k), CsrMatrix::from_coo(&m))
}

/// Lastvektor `fᵢ = ∫ source·φᵢ` auf einem Dreiecksnetz (Quadratur in den drei Kantenmitten, exakt für Polynome bis Grad 2).
pub fn load_vector_2d<T: Real>(
    points: &[[T; 2]],
    triangles: &[[usize; 3]],
    source: impl Fn(T, T) -> T,
) -> Vec<T> {
    let mut f = vec![T::ZERO; points.len()];
    let half = T::from_f64(0.5);
    for tri in triangles {
        let p = [points[tri[0]], points[tri[1]], points[tri[2]]];
        let det =
            (p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[2][0] - p[0][0]) * (p[1][1] - p[0][1]);
        let area = half * det.abs();
        // Kantenmitten (a, b): Hutfunktion φᵢ = ½ an den beiden Eckpunkten der Kante, 0 am dritten
        for (a, b, c) in [(0usize, 1usize, 2usize), (1, 2, 0), (2, 0, 1)] {
            let mx = (p[a][0] + p[b][0]) * half;
            let my = (p[a][1] + p[b][1]) * half;
            let v = source(mx, my) * area / T::from_f64(3.0);
            f[tri[a]] += v * half;
            f[tri[b]] += v * half;
            let _ = c;
        }
    }
    f
}

/// Bringt Dirichlet-Werte `(Knoten, Wert)` symmetrisch ein: Die rechte Seite wird um `K[:, i]·g` korrigiert, Zeile und Spalte `i`
/// werden durch die Einheitszeile ersetzt. Gibt `(K', f')` zurück.
pub fn apply_dirichlet<T: Real>(
    k: &CsrMatrix<T>,
    f: &[T],
    fixed: &[(usize, T)],
) -> (CsrMatrix<T>, Vec<T>) {
    let n = k.rows();
    let mut value = vec![None; n];
    for &(i, g) in fixed {
        value[i] = Some(g);
    }
    let mut rhs = f.to_vec();
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        for (j, v) in k.row(i) {
            match (value[i], value[j]) {
                (None, None) => {
                    let _ = coo.push(i, j, v);
                }
                (None, Some(g)) => rhs[i] -= v * g,
                _ => {}
            }
        }
    }
    for i in 0..n {
        if let Some(g) = value[i] {
            let _ = coo.push(i, i, T::ONE);
            rhs[i] = g;
        }
    }
    (CsrMatrix::from_coo(&coo), rhs)
}

/// Regelmäßiges Dreiecksnetz auf `[0, lx] × [0, ly]` mit `nx × ny` Zellen (je zwei Dreiecke); Knoten zeilenweise
/// (`index = j·(nx+1) + i`). Gibt `(Punkte, Dreiecke)` zurück.
pub fn rectangle_mesh<T: Real>(
    nx: usize,
    ny: usize,
    lx: T,
    ly: T,
) -> (Vec<[T; 2]>, Vec<[usize; 3]>) {
    assert!(nx >= 1 && ny >= 1, "rectangle_mesh: nx, ny ≥ 1");
    let mut pts = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        for i in 0..=nx {
            pts.push([
                lx * T::from_usize(i) / T::from_usize(nx),
                ly * T::from_usize(j) / T::from_usize(ny),
            ]);
        }
    }
    let idx = |i: usize, j: usize| j * (nx + 1) + i;
    let mut tris = Vec::with_capacity(2 * nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            tris.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
            tris.push([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
        }
    }
    (pts, tris)
}

/// Indizes der Randknoten eines mit [`rectangle_mesh`] erzeugten Netzes.
pub fn boundary_nodes(nx: usize, ny: usize) -> Vec<usize> {
    let mut v = Vec::new();
    for j in 0..=ny {
        for i in 0..=nx {
            if i == 0 || j == 0 || i == nx || j == ny {
                v.push(j * (nx + 1) + i);
            }
        }
    }
    v
}
