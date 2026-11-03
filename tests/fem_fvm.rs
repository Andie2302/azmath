//! Tests für `fem` und `fvm`.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::fem::*;
use azmath::fvm::*;
use azmath::ode::{Rk45Options, rk4, rk4_work_len};
use azmath::sparse::{Ordering, SparseCholesky};
use core::f64::consts::PI;

#[test]
fn fem_1d_poisson_nodally_exact() {
    // nichtäquidistantes Gitter: P1 ist in 1D knotenexakt
    let x = [0.0f64, 0.1, 0.35, 0.5, 0.8, 0.9, 1.0];
    let n = x.len();
    let (k, m, f) = assemble_1d(&x, |_| 1.0);
    let (kd, fd) = apply_dirichlet(&k, &f, &[(0, 0.0), (n - 1, 0.0)]);
    let u = SparseCholesky::new(&kd, Ordering::Natural)
        .unwrap()
        .solve(&fd)
        .unwrap();
    for i in 0..n {
        assert!((u[i] - x[i] * (1.0 - x[i]) / 2.0).abs() < 1e-12, "i = {i}");
    }
    // Massenmatrix: Summe aller Einträge = Länge
    let total: f64 = (0..n).map(|i| m.row(i).map(|(_, v)| v).sum::<f64>()).sum();
    assert!((total - 1.0).abs() < 1e-14);
    // Steifigkeitsmatrix: Zeilensummen 0
    for i in 0..n {
        assert!(k.row(i).map(|(_, v)| v).sum::<f64>().abs() < 1e-12);
    }
}

#[test]
fn fem_2d_matrices() {
    let (pts, tris) = rectangle_mesh(4, 3, 2.0f64, 1.5);
    let (k, m) = assemble_2d(&pts, &tris);
    let n = pts.len();
    for i in 0..n {
        assert!(k.row(i).map(|(_, v)| v).sum::<f64>().abs() < 1e-12);
    }
    let total: f64 = (0..n).map(|i| m.row(i).map(|(_, v)| v).sum::<f64>()).sum();
    assert!((total - 3.0).abs() < 1e-12); // Fläche 2·1,5
    // Energie von u = x: ∫|∇u|² = Fläche
    let u: Vec<f64> = pts.iter().map(|p| p[0]).collect();
    let mut ku = vec![0.0; n];
    k.matvec(&u, &mut ku).unwrap();
    let energy: f64 = u.iter().zip(&ku).map(|(a, b)| a * b).sum();
    assert!((energy - 3.0).abs() < 1e-12);
    assert_eq!(boundary_nodes(4, 3).len(), 2 * (4 + 3));
}

#[test]
fn fem_2d_poisson_converges() {
    let solve = |nx: usize| {
        let (pts, tris) = rectangle_mesh(nx, nx, 1.0f64, 1.0);
        let (k, _) = assemble_2d(&pts, &tris);
        let f = load_vector_2d(&pts, &tris, |x, y| {
            2.0 * PI * PI * (PI * x).sin() * (PI * y).sin()
        });
        let fixed: Vec<(usize, f64)> = boundary_nodes(nx, nx)
            .into_iter()
            .map(|i| (i, 0.0))
            .collect();
        let (kd, fd) = apply_dirichlet(&k, &f, &fixed);
        let u = SparseCholesky::new(&kd, Ordering::MinimumDegree)
            .unwrap()
            .solve(&fd)
            .unwrap();
        pts.iter()
            .zip(&u)
            .map(|(p, &v)| (v - (PI * p[0]).sin() * (PI * p[1]).sin()).abs())
            .fold(0.0, f64::max)
    };
    let (e8, e16) = (solve(8), solve(16));
    assert!(e16 < 2e-2, "{e16}");
    assert!(e8 / e16 > 3.0, "{e8} {e16}");
}

fn advect_error(muscl: bool, n: usize) -> f64 {
    let dx = 1.0 / n as f64;
    let mut u: Vec<f64> = (0..n)
        .map(|i| (-200.0 * ((i as f64 + 0.5) * dx - 0.5).powi(2)).exp())
        .collect();
    let u0 = u.clone();
    let mut work = vec![0.0; rk4_work_len(n)];
    let flux = |a: f64, b: f64| rusanov(|u| u, |_| 1.0, a, b);
    let f = |_t: f64, u: &[f64], du: &mut [f64]| {
        if muscl {
            muscl_rhs_1d(u, dx, Boundary::Periodic, flux, du)
        } else {
            fvm_rhs_1d(u, dx, Boundary::Periodic, flux, du)
        }
    };
    rk4(
        f,
        0.0,
        1.0,
        &mut u,
        (2.0 / dx) as usize,
        &mut work,
        |_, _| {},
    )
    .unwrap();
    u.iter()
        .zip(&u0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

#[test]
fn advection_muscl_beats_first_order() {
    let e1 = advect_error(false, 100);
    let e2 = advect_error(true, 100);
    assert!(e2 < e1 * 0.5, "{e1} {e2}");
    assert!(e2 < 0.1);
}

#[test]
fn burgers_conserves_mass_and_stays_bounded() {
    let n = 100;
    let dx = 1.0 / n as f64;
    let mut u: Vec<f64> = (0..n)
        .map(|i| 0.5 + 0.5 * (2.0 * PI * (i as f64 + 0.5) * dx).sin())
        .collect();
    let mass0: f64 = u.iter().sum();
    let flux = |a: f64, b: f64| rusanov(|u| 0.5 * u * u, |u| u, a, b);
    let dt = 0.4 * dx;
    let mut du = vec![0.0; n];
    for _ in 0..(0.5 / dt) as usize {
        fvm_rhs_1d(&u, dx, Boundary::Periodic, flux, &mut du);
        for i in 0..n {
            u[i] += dt * du[i];
        }
    }
    assert!((u.iter().sum::<f64>() - mass0).abs() < 1e-10);
    assert!(u.iter().all(|&v| v > -1e-12 && v < 1.0 + 1e-12));
    // Randbedingungen: Ausfluss und feste Werte laufen ohne Panik
    let mut du = vec![0.0; n];
    fvm_rhs_1d(&u, dx, Boundary::ZeroGradient, flux, &mut du);
    fvm_rhs_1d(&u, dx, Boundary::Fixed(0.0, 1.0), flux, &mut du);
    let lf = lax_friedrichs(|u: f64| u, 1.0, 1.0, 0.0);
    assert!((lf - 1.0).abs() < 1e-15);
    let _ = Rk45Options::<f64>::default();
}
