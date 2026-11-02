//! Tests für das geometrische Multigrid (1D und 2D) und den Multigrid-Vorkonditionierer.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath_core::Real;
use azmath_linalg::iterative::{IterOptions, Preconditioner, cg, pcg};
use azmath_linalg::{LinearOperator, solve_tridiagonal, tridiagonal_work_len};
use azmath_multigrid::{
    CycleType, MultigridError, MultigridOptions, MultigridPreconditioner, Poisson1d, Poisson2d,
    Smoother, SolveOptions, cycle_1d, cycle_2d, grid_levels, solve_1d, solve_2d, work_len_1d,
    work_len_2d,
};
use core::f64::consts::PI;

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "erwartet {b}, erhalten {a} (Δ = {:e})",
        (a - b).abs()
    );
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn opts() -> MultigridOptions<f64> {
    MultigridOptions::default()
}

fn solve_opts() -> SolveOptions<f64> {
    SolveOptions::default()
}

/// Eigenwert des diskreten 1D-Operators zur p-ten Sinusschwingung.
fn lambda_1d(p: usize, n: usize) -> f64 {
    let h = 1.0 / (n as f64 + 1.0);
    4.0 / (h * h) * Real::sin(p as f64 * PI * h / 2.0).powi(2)
}

fn sine_1d(p: usize, n: usize) -> Vec<f64> {
    let h = 1.0 / (n as f64 + 1.0);
    (0..n)
        .map(|i| Real::sin(p as f64 * PI * (i + 1) as f64 * h))
        .collect()
}

fn sine_2d(p: usize, q: usize, n: usize) -> Vec<f64> {
    let h = 1.0 / (n as f64 + 1.0);
    (0..n * n)
        .map(|k| {
            let (i, j) = (k / n, k % n);
            Real::sin(p as f64 * PI * (i + 1) as f64 * h)
                * Real::sin(q as f64 * PI * (j + 1) as f64 * h)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Gitter und Hilfsfunktionen
// ---------------------------------------------------------------------------

#[test]
fn grid_sizes() {
    for (n, k) in [(1, 1), (3, 2), (7, 3), (15, 4), (63, 6), (1023, 10)] {
        assert_eq!(grid_levels(n), Some(k));
    }
    for n in [0, 2, 4, 5, 6, 8, 62, 64, 100] {
        assert_eq!(grid_levels(n), None, "n = {n}");
    }
}

#[test]
fn work_lengths() {
    assert_eq!(work_len_1d(1), 3);
    assert_eq!(work_len_1d(3), 3 * (3 + 1));
    assert_eq!(work_len_1d(7), 3 * (7 + 3 + 1));
    assert_eq!(work_len_2d(1), 3);
    assert_eq!(work_len_2d(3), 3 * (9 + 1));
    assert_eq!(work_len_2d(7), 3 * (49 + 9 + 1));
    // Der Bedarf wächst nur linear in der Zahl der Unbekannten (≈ 6N in 1D, 4N in 2D)
    assert!(work_len_1d(1023) < 6 * 1023 + 100);
    assert!(work_len_2d(127) < 4 * 127 * 127 + 100);
}

#[test]
fn poisson_operators_have_the_analytic_eigenpairs() {
    let n = 15;
    let a1 = Poisson1d::new(n);
    assert_eq!(
        (
            LinearOperator::<f64>::rows(&a1),
            LinearOperator::<f64>::cols(&a1)
        ),
        (n, n)
    );
    for p in [1, 4, 15] {
        let phi = sine_1d(p, n);
        let mut y = vec![0.0; n];
        a1.apply(&phi, &mut y);
        for i in 0..n {
            close(y[i], lambda_1d(p, n) * phi[i], 1e-9);
        }
    }
    let a2 = Poisson2d::new(n);
    assert_eq!(LinearOperator::<f64>::rows(&a2), n * n);
    let phi = sine_2d(2, 3, n);
    let mut y = vec![0.0; n * n];
    a2.apply(&phi, &mut y);
    let lambda = lambda_1d(2, n) + lambda_1d(3, n);
    for k in 0..n * n {
        close(y[k], lambda * phi[k], 1e-8);
    }
}

// ---------------------------------------------------------------------------
// Löser 1D
// ---------------------------------------------------------------------------

#[test]
fn one_d_solution_matches_the_direct_tridiagonal_solver() {
    let n = 63;
    let h = 1.0 / (n as f64 + 1.0);
    let f: Vec<f64> = (0..n)
        .map(|i| 1.0 + Real::cos(3.0 * (i + 1) as f64 * h))
        .collect();
    let mut u = vec![0.0; n];
    let mut work = vec![0.0; work_len_1d(n)];
    solve_1d(&f, &mut u, n, opts(), solve_opts(), &mut work).unwrap();

    let (a, b, c) = (
        vec![-1.0 / (h * h); n],
        vec![2.0 / (h * h); n],
        vec![-1.0 / (h * h); n],
    );
    let mut direct = vec![0.0; n];
    let mut scratch = vec![0.0; tridiagonal_work_len(n)];
    solve_tridiagonal(&a, &b, &c, &f, &mut direct, &mut scratch).unwrap();
    assert!(
        max_diff(&u, &direct) < 1e-9,
        "Abweichung {:e}",
        max_diff(&u, &direct)
    );
}

#[test]
fn manufactured_eigenvector_solutions() {
    // f = λ·φ  ->  u = φ (algebraisch exakt, ohne Diskretisierungsfehler)
    let n = 63;
    for p in [1usize, 3, 10] {
        let phi = sine_1d(p, n);
        let f: Vec<f64> = phi.iter().map(|v| lambda_1d(p, n) * v).collect();
        let mut u = vec![0.0; n];
        let mut work = vec![0.0; work_len_1d(n)];
        solve_1d(&f, &mut u, n, opts(), solve_opts(), &mut work).unwrap();
        assert!(
            max_diff(&u, &phi) < 1e-9,
            "p = {p}: {:e}",
            max_diff(&u, &phi)
        );
    }
}

#[test]
fn one_d_convergence_does_not_depend_on_the_grid_size() {
    let mut counts = Vec::new();
    for k in 4..=9 {
        let n = (1usize << k) - 1;
        let f = vec![1.0; n];
        let mut u = vec![0.0; n];
        let mut work = vec![0.0; work_len_1d(n)];
        let r = solve_1d(&f, &mut u, n, opts(), solve_opts(), &mut work).unwrap();
        counts.push(r.cycles);
    }
    let (lo, hi) = (*counts.iter().min().unwrap(), *counts.iter().max().unwrap());
    assert!(hi <= 14 && hi - lo <= 2, "Zyklenzahlen {counts:?}");
}

#[test]
fn discretization_error_is_second_order() {
    // −u'' = π²·sin(πx), u = sin(πx)
    let err = |n: usize| {
        let h = 1.0 / (n as f64 + 1.0);
        let f: Vec<f64> = (0..n)
            .map(|i| PI * PI * Real::sin(PI * (i + 1) as f64 * h))
            .collect();
        let mut u = vec![0.0; n];
        let mut work = vec![0.0; work_len_1d(n)];
        solve_1d(&f, &mut u, n, opts(), solve_opts(), &mut work).unwrap();
        (0..n)
            .map(|i| (u[i] - Real::sin(PI * (i + 1) as f64 * h)).abs())
            .fold(0.0f64, f64::max)
    };
    let ratio = err(31) / err(63);
    assert!((ratio - 4.0).abs() < 0.1, "Verhältnis {ratio}");
}

#[test]
fn smallest_grid_is_solved_directly() {
    let mut u = [0.0];
    let mut work = [0.0; work_len_1d(1)];
    let r = solve_1d(&[2.0], &mut u, 1, opts(), solve_opts(), &mut work).unwrap();
    // h = ½: 2u/h² = 8u = 2  ->  u = 1/4
    close(u[0], 0.25, 1e-15);
    assert!(r.cycles <= 1);
    let mut u2 = [0.0];
    let mut work2 = [0.0; work_len_2d(1)];
    solve_2d(&[2.0], &mut u2, 1, opts(), solve_opts(), &mut work2).unwrap();
    close(u2[0], 2.0 / 16.0, 1e-15);
}

// ---------------------------------------------------------------------------
// Löser 2D
// ---------------------------------------------------------------------------

#[test]
fn two_d_manufactured_eigenvector_solution() {
    let n = 31;
    let phi = sine_2d(2, 3, n);
    let lambda = lambda_1d(2, n) + lambda_1d(3, n);
    let f: Vec<f64> = phi.iter().map(|v| lambda * v).collect();
    let mut u = vec![0.0; n * n];
    let mut work = vec![0.0; work_len_2d(n)];
    solve_2d(&f, &mut u, n, opts(), solve_opts(), &mut work).unwrap();
    assert!(
        max_diff(&u, &phi) < 1e-9,
        "Abweichung {:e}",
        max_diff(&u, &phi)
    );
}

#[test]
fn two_d_agrees_with_cg() {
    let n = 15;
    let f: Vec<f64> = (0..n * n).map(|k| 1.0 + ((k * 7) % 5) as f64).collect();
    let mut u = vec![0.0; n * n];
    let mut work = vec![0.0; work_len_2d(n)];
    solve_2d(&f, &mut u, n, opts(), solve_opts(), &mut work).unwrap();
    let a = Poisson2d::new(n);
    let mut x = vec![0.0; n * n];
    let mut cw = vec![0.0; 4 * n * n];
    cg(&a, &f, &mut x, IterOptions::new(1e-13, 1000), &mut cw).unwrap();
    assert!(
        max_diff(&u, &x) < 1e-8 * norm(&x).max(1.0),
        "Abweichung {:e}",
        max_diff(&u, &x)
    );
}

#[test]
fn two_d_convergence_does_not_depend_on_the_grid_size() {
    let mut counts = Vec::new();
    for k in 3..=7 {
        let n = (1usize << k) - 1;
        let f = vec![1.0; n * n];
        let mut u = vec![0.0; n * n];
        let mut work = vec![0.0; work_len_2d(n)];
        let r = solve_2d(&f, &mut u, n, opts(), solve_opts(), &mut work).unwrap();
        counts.push(r.cycles);
    }
    let (lo, hi) = (*counts.iter().min().unwrap(), *counts.iter().max().unwrap());
    assert!(hi <= 14 && hi - lo <= 2, "Zyklenzahlen {counts:?}");
}

#[test]
fn two_d_discretization_error_is_second_order() {
    // −Δu = 2π²·sin(πx)·sin(πy), u = sin(πx)·sin(πy)
    let err = |n: usize| {
        let h = 1.0 / (n as f64 + 1.0);
        let exact: Vec<f64> = (0..n * n)
            .map(|k| {
                Real::sin(PI * ((k / n) + 1) as f64 * h) * Real::sin(PI * ((k % n) + 1) as f64 * h)
            })
            .collect();
        let f: Vec<f64> = exact.iter().map(|v| 2.0 * PI * PI * v).collect();
        let mut u = vec![0.0; n * n];
        let mut work = vec![0.0; work_len_2d(n)];
        solve_2d(&f, &mut u, n, opts(), solve_opts(), &mut work).unwrap();
        max_diff(&u, &exact)
    };
    let ratio = err(15) / err(31);
    assert!((ratio - 4.0).abs() < 0.2, "Verhältnis {ratio}");
}

#[test]
fn a_single_cycle_contracts_the_error_strongly() {
    let n = 63;
    let phi = sine_2d(3, 5, n);
    let lambda = lambda_1d(3, n) + lambda_1d(5, n);
    let f: Vec<f64> = phi.iter().map(|v| lambda * v).collect();
    let mut work = vec![0.0; work_len_2d(n)];
    // Startnäherung: gemischte Fehleranteile (hoch- und niederfrequent)
    let mut u: Vec<f64> = (0..n * n)
        .map(|k| ((k * 31) % 17) as f64 / 17.0 - 0.5)
        .collect();
    let mut previous = max_diff(&u, &phi);
    for step in 0..6 {
        cycle_2d(&f, &mut u, n, opts(), &mut work).unwrap();
        let now = max_diff(&u, &phi);
        if step > 0 {
            assert!(
                now < 0.25 * previous,
                "Schritt {step}: {previous:e} -> {now:e}"
            );
        }
        previous = now;
    }
}

// ---------------------------------------------------------------------------
// Optionen: Zyklus, Glätter
// ---------------------------------------------------------------------------

fn count_cycles_2d(o: MultigridOptions<f64>, n: usize) -> usize {
    let f = vec![1.0; n * n];
    let mut u = vec![0.0; n * n];
    let mut work = vec![0.0; work_len_2d(n)];
    solve_2d(&f, &mut u, n, o, solve_opts(), &mut work)
        .unwrap()
        .cycles
}

#[test]
fn w_cycle_needs_fewer_cycles_than_v_cycle() {
    let v = count_cycles_2d(opts(), 31);
    let w = count_cycles_2d(
        MultigridOptions {
            cycle: CycleType::W,
            ..opts()
        },
        31,
    );
    assert!(w <= v, "W {w} vs. V {v}");
    assert!(w >= 5);
}

#[test]
fn smoothers_and_sweep_counts() {
    let gs22 = count_cycles_2d(opts(), 31);
    let gs11 = count_cycles_2d(
        MultigridOptions {
            pre_smooth: 1,
            post_smooth: 1,
            ..opts()
        },
        31,
    );
    let jac = count_cycles_2d(
        MultigridOptions {
            smoother: Smoother::WeightedJacobi(0.8),
            ..opts()
        },
        31,
    );
    assert!(gs22 < gs11, "V(2,2) {gs22} vs. V(1,1) {gs11}");
    assert!(gs22 < jac, "Gauß-Seidel {gs22} vs. Jacobi {jac}");
    assert!(jac < 40, "Jacobi braucht {jac} Zyklen");
}

#[test]
fn jacobi_in_one_dimension() {
    let n = 63;
    let phi = sine_1d(5, n);
    let f: Vec<f64> = phi.iter().map(|v| lambda_1d(5, n) * v).collect();
    let mut u = vec![0.0; n];
    let mut work = vec![0.0; work_len_1d(n)];
    let o = MultigridOptions {
        smoother: Smoother::WeightedJacobi(2.0 / 3.0),
        ..opts()
    };
    let r = solve_1d(&f, &mut u, n, o, solve_opts(), &mut work).unwrap();
    assert!(max_diff(&u, &phi) < 1e-9);
    assert!(r.cycles < 25, "{} Zyklen", r.cycles);
}

#[test]
fn start_guess_is_used_and_zero_rhs_gives_zero() {
    let n = 31;
    let phi = sine_1d(2, n);
    let f: Vec<f64> = phi.iter().map(|v| lambda_1d(2, n) * v).collect();
    let mut work = vec![0.0; work_len_1d(n)];
    // exakte Startnäherung: 0 Zyklen
    let mut u = phi.clone();
    let r = solve_1d(
        &f,
        &mut u,
        n,
        opts(),
        SolveOptions {
            tol: 1e-6,
            ..solve_opts()
        },
        &mut work,
    )
    .unwrap();
    assert_eq!(r.cycles, 0);
    // rechte Seite 0
    let mut z = vec![5.0; n];
    let r = solve_1d(&vec![0.0; n], &mut z, n, opts(), solve_opts(), &mut work).unwrap();
    assert!(z.iter().all(|&v| v == 0.0) && r.cycles == 0);
}

#[test]
fn repeated_single_cycles_reach_the_solver_result() {
    let n = 31;
    let f: Vec<f64> = (0..n).map(|i| 1.0 + (i % 4) as f64).collect();
    let mut work = vec![0.0; work_len_1d(n)];
    let mut by_solver = vec![0.0; n];
    let r = solve_1d(&f, &mut by_solver, n, opts(), solve_opts(), &mut work).unwrap();
    let mut by_cycles = vec![0.0; n];
    for _ in 0..r.cycles {
        cycle_1d(&f, &mut by_cycles, n, opts(), &mut work).unwrap();
    }
    assert!(max_diff(&by_solver, &by_cycles) < 1e-14);
}

// ---------------------------------------------------------------------------
// Vorkonditionierer
// ---------------------------------------------------------------------------

#[test]
fn preconditioner_is_symmetric_positive_definite() {
    let n = 15;
    let mut work = vec![0.0; work_len_2d(n)];
    let b = MultigridPreconditioner::new_2d(n, opts(), &mut work).unwrap();
    let m = n * n;
    let x: Vec<f64> = (0..m).map(|k| Real::sin(0.7 * k as f64) + 0.3).collect();
    let y: Vec<f64> = (0..m).map(|k| Real::cos(1.3 * k as f64) - 0.2).collect();
    let (mut bx, mut by) = (vec![0.0; m], vec![0.0; m]);
    b.apply(&x, &mut bx);
    b.apply(&y, &mut by);
    // Symmetrie: ⟨Bx, y⟩ = ⟨x, By⟩
    close(
        dot(&bx, &y),
        dot(&x, &by),
        1e-11 * dot(&bx, &y).abs().max(1.0),
    );
    // positiv definit
    assert!(dot(&bx, &x) > 0.0 && dot(&by, &y) > 0.0);
    // linear: B(2x) = 2·Bx
    let x2: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
    let mut bx2 = vec![0.0; m];
    b.apply(&x2, &mut bx2);
    for k in 0..m {
        close(bx2[k], 2.0 * bx[k], 1e-12);
    }
}

#[test]
fn multigrid_pcg_has_grid_independent_iteration_counts_unlike_plain_cg() {
    let mut cg_counts = Vec::new();
    let mut mg_counts = Vec::new();
    for k in 3..=6 {
        let n = (1usize << k) - 1;
        let a = Poisson2d::new(n);
        let b = vec![1.0; n * n];
        let mut cw = vec![0.0; 4 * n * n];
        let mut x = vec![0.0; n * n];
        cg_counts.push(
            cg(&a, &b, &mut x, IterOptions::new(1e-10, 10_000), &mut cw)
                .unwrap()
                .iterations,
        );

        let mut mw = vec![0.0; work_len_2d(n)];
        let mg = MultigridPreconditioner::new_2d(n, opts(), &mut mw).unwrap();
        let mut y = vec![0.0; n * n];
        mg_counts.push(
            pcg(&a, &mg, &b, &mut y, IterOptions::new(1e-10, 1000), &mut cw)
                .unwrap()
                .iterations,
        );
        assert!(max_diff(&x, &y) < 1e-7, "n = {n}");
    }
    assert!(
        cg_counts.windows(2).all(|w| w[1] > w[0]),
        "CG: {cg_counts:?}"
    );
    assert!(*cg_counts.last().unwrap() > 100, "CG: {cg_counts:?}");
    let (lo, hi) = (
        *mg_counts.iter().min().unwrap(),
        *mg_counts.iter().max().unwrap(),
    );
    assert!(hi <= 9 && hi - lo <= 1, "MG-PCG: {mg_counts:?}");
}

#[test]
fn preconditioner_in_one_dimension_and_with_other_options() {
    let n = 127;
    let a = Poisson1d::new(n);
    let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
    let mut cw = vec![0.0; 4 * n];
    for o in [
        opts(),
        MultigridOptions {
            cycle: CycleType::W,
            ..opts()
        },
        MultigridOptions {
            smoother: Smoother::WeightedJacobi(2.0 / 3.0),
            ..opts()
        },
    ] {
        let mut mw = vec![0.0; work_len_1d(n)];
        let mg = MultigridPreconditioner::new_1d(n, o, &mut mw).unwrap();
        let mut x = vec![0.0; n];
        let r = pcg(&a, &mg, &b, &mut x, IterOptions::new(1e-10, 200), &mut cw).unwrap();
        assert!(r.iterations <= 12, "{} Iterationen mit {o:?}", r.iterations);
        let mut ax = vec![0.0; n];
        a.apply(&x, &mut ax);
        assert!(max_diff(&ax, &b) < 1e-6 * norm(&b));
    }
}

#[test]
fn preconditioner_rejects_asymmetric_or_invalid_settings() {
    let n = 7;
    let mut work = vec![0.0; work_len_1d(n)];
    let uneven = MultigridOptions {
        pre_smooth: 1,
        post_smooth: 2,
        ..opts()
    };
    assert!(matches!(
        MultigridPreconditioner::new_1d(n, uneven, &mut work),
        Err(MultigridError::InvalidInput)
    ));
    assert!(matches!(
        MultigridPreconditioner::new_1d(6, opts(), &mut work),
        Err(MultigridError::InvalidGridSize)
    ));
    let mut short = vec![0.0; work_len_1d(n) - 1];
    assert!(matches!(
        MultigridPreconditioner::new_1d(n, opts(), &mut short),
        Err(MultigridError::InvalidInput)
    ));
    let bad_omega = MultigridOptions {
        smoother: Smoother::WeightedJacobi(0.0),
        ..opts()
    };
    assert!(matches!(
        MultigridPreconditioner::new_1d(n, bad_omega, &mut work),
        Err(MultigridError::InvalidInput)
    ));
}

// ---------------------------------------------------------------------------
// Fehlerfälle und Sonstiges
// ---------------------------------------------------------------------------

#[test]
fn error_cases() {
    let mut work = vec![0.0; 10_000];
    // Gittergröße
    for n in [0usize, 2, 6, 64] {
        let f = vec![1.0; n];
        let mut u = vec![0.0; n];
        assert_eq!(
            solve_1d(&f, &mut u, n, opts(), solve_opts(), &mut work),
            Err(MultigridError::InvalidGridSize),
            "n = {n}"
        );
    }
    // Längen
    let mut u = vec![0.0; 7];
    assert_eq!(
        solve_1d(&[1.0; 6], &mut u, 7, opts(), solve_opts(), &mut work),
        Err(MultigridError::InvalidInput)
    );
    let mut u2 = vec![0.0; 49];
    assert_eq!(
        solve_2d(&[1.0; 7], &mut u2, 7, opts(), solve_opts(), &mut work),
        Err(MultigridError::InvalidInput)
    );
    let mut short = vec![0.0; 5];
    assert_eq!(
        solve_1d(&[1.0; 7], &mut u, 7, opts(), solve_opts(), &mut short),
        Err(MultigridError::InvalidInput)
    );
    assert_eq!(
        cycle_2d(&[1.0; 49], &mut u2, 7, opts(), &mut short),
        Err(MultigridError::InvalidInput)
    );
    // NaN
    let mut f = vec![1.0; 7];
    f[3] = f64::NAN;
    assert_eq!(
        solve_1d(&f, &mut u, 7, opts(), solve_opts(), &mut work),
        Err(MultigridError::NotFinite)
    );
    let mut bad_u = vec![0.0; 7];
    bad_u[0] = f64::INFINITY;
    assert_eq!(
        cycle_1d(&[1.0; 7], &mut bad_u, 7, opts(), &mut work),
        Err(MultigridError::NotFinite)
    );
    // ungültiges Jacobi-Gewicht
    let neg = MultigridOptions {
        smoother: Smoother::WeightedJacobi(-1.0),
        ..opts()
    };
    assert_eq!(
        solve_1d(&[1.0; 7], &mut u, 7, neg, solve_opts(), &mut work),
        Err(MultigridError::InvalidInput)
    );
}

#[test]
fn no_convergence_with_too_few_cycles() {
    let n = 63;
    let f = vec![1.0; n];
    let mut u = vec![0.0; n];
    let mut work = vec![0.0; work_len_1d(n)];
    let few = SolveOptions {
        tol: 1e-14,
        max_cycles: 2,
    };
    assert_eq!(
        solve_1d(&f, &mut u, n, opts(), few, &mut work),
        Err(MultigridError::NoConvergence)
    );
    assert!(u.iter().any(|&v| v != 0.0), "u enthält die letzte Näherung");
}

#[test]
fn f32_works() {
    let n = 31;
    let f = vec![1.0f32; n * n];
    let mut u = vec![0.0f32; n * n];
    let mut work = vec![0.0f32; work_len_2d(n)];
    let r = solve_2d(
        &f,
        &mut u,
        n,
        MultigridOptions::default(),
        SolveOptions {
            tol: 1e-5f32,
            max_cycles: 50,
        },
        &mut work,
    )
    .unwrap();
    assert!(r.cycles <= 12);
    // −Δu = 1: u(½, ½) ≈ 0.0737
    assert!((u[(n / 2) * n + n / 2] - 0.0737).abs() < 2e-3);
}

#[test]
fn defaults_and_display() {
    let o = MultigridOptions::<f64>::default();
    assert_eq!(
        (o.pre_smooth, o.post_smooth, o.cycle, o.smoother),
        (2, 2, CycleType::V, Smoother::GaussSeidel)
    );
    let s = SolveOptions::<f64>::default();
    assert!(s.tol < 1e-10 && s.max_cycles == 100);
    extern crate std;
    use std::string::ToString;
    for e in [
        MultigridError::InvalidGridSize,
        MultigridError::InvalidInput,
        MultigridError::NotFinite,
        MultigridError::NoConvergence,
    ] {
        assert!(!e.to_string().is_empty());
    }
}

#[cfg(feature = "alloc")]
mod owned {
    use super::*;
    use azmath_multigrid::{solve_1d_vec, solve_2d_vec};

    #[test]
    fn vec_variants_match_the_slice_variants() {
        let n = 31;
        let f: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();
        let (u, r) = solve_1d_vec(&f, opts(), solve_opts()).unwrap();
        let mut u2 = vec![0.0; n];
        let mut work = vec![0.0; work_len_1d(n)];
        let r2 = solve_1d(&f, &mut u2, n, opts(), solve_opts(), &mut work).unwrap();
        assert_eq!((u, r), (u2, r2));

        let m = 15;
        let f2 = vec![1.0; m * m];
        let (v, rv) = solve_2d_vec(&f2, m, opts(), solve_opts()).unwrap();
        let mut v2 = vec![0.0; m * m];
        let mut w2 = vec![0.0; work_len_2d(m)];
        let rv2 = solve_2d(&f2, &mut v2, m, opts(), solve_opts(), &mut w2).unwrap();
        assert_eq!((v, rv), (v2, rv2));
    }

    #[test]
    fn vec_variants_report_errors() {
        assert_eq!(
            solve_1d_vec(&[1.0f64; 6], opts(), solve_opts()).unwrap_err(),
            MultigridError::InvalidGridSize
        );
        assert_eq!(
            solve_2d_vec(&[1.0f64; 9], 5, opts(), solve_opts()).unwrap_err(),
            MultigridError::InvalidGridSize
        );
        assert_eq!(
            solve_2d_vec(&[1.0f64; 8], 3, opts(), solve_opts()).unwrap_err(),
            MultigridError::InvalidInput
        );
    }
}
