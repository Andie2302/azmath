//! Tests für Schrittweitensuche, BFGS und L-BFGS.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]
#![allow(clippy::needless_range_loop)]

use azmath::optimize::{
    LineSearchError, OptimizeError, OptimizeOptions, Termination, WolfeOptions, bfgs,
    bfgs_work_len, lbfgs, lbfgs_work_len, line_search_wolfe,
};

fn rosen(x: &[f64]) -> f64 {
    (0..x.len() - 1)
        .map(|i| (1.0 - x[i]).powi(2) + 100.0 * (x[i + 1] - x[i] * x[i]).powi(2))
        .sum()
}

fn rosen_grad(x: &[f64], g: &mut [f64]) {
    g.fill(0.0);
    for i in 0..x.len() - 1 {
        let t = x[i + 1] - x[i] * x[i];
        g[i] += -2.0 * (1.0 - x[i]) - 400.0 * x[i] * t;
        g[i + 1] += 200.0 * t;
    }
}

fn opts() -> OptimizeOptions<f64> {
    OptimizeOptions::default()
}

#[test]
fn wolfe_search_satisfies_both_conditions() {
    let f = |x: &[f64]| x[0] * x[0] + 10.0 * x[1] * x[1];
    let grad = |x: &[f64], g: &mut [f64]| {
        g[0] = 2.0 * x[0];
        g[1] = 20.0 * x[1];
    };
    let x = [1.0, 1.0];
    let (f0, mut g0) = (f(&x), [0.0; 2]);
    grad(&x, &mut g0);
    let d = [-g0[0], -g0[1]];
    let o = WolfeOptions::default();
    for alpha0 in [1.0, 1e-3, 50.0] {
        let (mut xn, mut gn) = ([0.0; 2], [0.0; 2]);
        let r = line_search_wolfe(f, grad, &x, f0, &g0, &d, alpha0, &o, &mut xn, &mut gn).unwrap();
        let dphi0 = g0[0] * d[0] + g0[1] * d[1];
        let dphi = gn[0] * d[0] + gn[1] * d[1];
        assert!(
            r.f_value <= f0 + o.c1 * r.alpha * dphi0,
            "Armijo bei alpha0 = {alpha0}"
        );
        assert!(
            dphi.abs() <= -o.c2 * dphi0,
            "Krümmung bei alpha0 = {alpha0}"
        );
        assert!((f(&xn) - r.f_value).abs() < 1e-15);
    }
}

#[test]
fn wolfe_search_errors() {
    let f = |x: &[f64]| x[0] * x[0];
    let grad = |x: &[f64], g: &mut [f64]| g[0] = 2.0 * x[0];
    let (mut xn, mut gn) = ([0.0], [0.0]);
    let o = WolfeOptions::default();
    // Aufstiegsrichtung
    let e = line_search_wolfe(
        f,
        grad,
        &[1.0],
        1.0,
        &[2.0],
        &[1.0],
        1.0,
        &o,
        &mut xn,
        &mut gn,
    );
    assert_eq!(e, Err(LineSearchError::NotDescent));
    // NaN in f
    let nan = |_: &[f64]| f64::NAN;
    let e = line_search_wolfe(
        nan,
        grad,
        &[1.0],
        1.0,
        &[2.0],
        &[-1.0],
        1.0,
        &o,
        &mut xn,
        &mut gn,
    );
    assert_eq!(e, Err(LineSearchError::NotFinite));
    // unbeschränkt fallend: alpha_max erreicht
    let lin = |x: &[f64]| -x[0];
    let lg = |_: &[f64], g: &mut [f64]| g[0] = -1.0;
    let e = line_search_wolfe(
        lin,
        lg,
        &[0.0],
        0.0,
        &[-1.0],
        &[1.0],
        1.0,
        &o,
        &mut xn,
        &mut gn,
    );
    assert_eq!(e, Err(LineSearchError::Failed));
}

#[test]
fn bfgs_minimizes_rosenbrock() {
    let mut x = [-1.2, 1.0];
    let mut work = [0.0; bfgs_work_len(2)];
    let r = bfgs(rosen, rosen_grad, &mut x, opts(), &mut work).unwrap();
    assert!(
        (x[0] - 1.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6,
        "{x:?}"
    );
    assert!(r.f_value < 1e-12);
    assert!(r.iterations < 60, "{} Iterationen", r.iterations);
    assert!(r.evaluations >= r.iterations);
}

#[test]
fn bfgs_solves_a_quadratic_in_about_n_steps() {
    // f = ½ xᵀAx − bᵀx mit A = diag(1..10)
    let n = 10;
    let a = |i: usize| (i + 1) as f64;
    let f = |x: &[f64]| (0..n).map(|i| 0.5 * a(i) * x[i] * x[i] - x[i]).sum::<f64>();
    let grad = |x: &[f64], g: &mut [f64]| {
        for i in 0..n {
            g[i] = a(i) * x[i] - 1.0;
        }
    };
    let mut x = [0.0; 10];
    let mut work = [0.0; bfgs_work_len(10)];
    let r = bfgs(f, grad, &mut x, opts(), &mut work).unwrap();
    for i in 0..n {
        assert!((x[i] - 1.0 / a(i)).abs() < 1e-7, "x[{i}] = {}", x[i]);
    }
    assert!(r.iterations <= 30, "{} Iterationen", r.iterations);
}

#[test]
fn lbfgs_minimizes_rosenbrock_in_higher_dimension() {
    let mut x = [-1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0, -1.2, 1.0];
    let mut work = [0.0; lbfgs_work_len(10, 8)];
    let o = OptimizeOptions {
        max_iter: 500,
        ..opts()
    };
    let r = lbfgs(rosen, rosen_grad, &mut x, 8, o, &mut work).unwrap();
    for v in x {
        assert!((v - 1.0).abs() < 1e-5, "{x:?}");
    }
    assert!(r.f_value < 1e-10);
}

#[test]
fn lbfgs_memory_size_one_still_converges_on_a_quadratic() {
    let f = |x: &[f64]| x[0] * x[0] + 5.0 * x[1] * x[1];
    let grad = |x: &[f64], g: &mut [f64]| {
        g[0] = 2.0 * x[0];
        g[1] = 10.0 * x[1];
    };
    let mut x = [3.0, -2.0];
    let mut work = [0.0; lbfgs_work_len(2, 1)];
    let r = lbfgs(f, grad, &mut x, 1, opts(), &mut work).unwrap();
    assert!(x[0].abs() < 1e-6 && x[1].abs() < 1e-6);
    assert!(matches!(
        r.termination,
        Termination::Gradient | Termination::Step | Termination::Cost
    ));
}

#[test]
fn famous_test_functions() {
    // Himmelblau: vier Minima, jedes ist ein gültiges Ergebnis
    let f = |x: &[f64]| (x[0] * x[0] + x[1] - 11.0).powi(2) + (x[0] + x[1] * x[1] - 7.0).powi(2);
    let grad = |x: &[f64], g: &mut [f64]| {
        let (a, b) = (x[0] * x[0] + x[1] - 11.0, x[0] + x[1] * x[1] - 7.0);
        g[0] = 4.0 * x[0] * a + 2.0 * b;
        g[1] = 2.0 * a + 4.0 * x[1] * b;
    };
    let mut x = [0.0, 0.0];
    let mut work = [0.0; bfgs_work_len(2)];
    let r = bfgs(f, grad, &mut x, opts(), &mut work).unwrap();
    assert!(r.f_value < 1e-14, "{x:?}");
    // Beale-Funktion, Minimum bei (3, 0.5)
    let beale = |x: &[f64]| {
        (1.5 - x[0] + x[0] * x[1]).powi(2)
            + (2.25 - x[0] + x[0] * x[1] * x[1]).powi(2)
            + (2.625 - x[0] + x[0] * x[1].powi(3)).powi(2)
    };
    let bg = |x: &[f64], g: &mut [f64]| {
        let (a, b, c) = (
            1.5 - x[0] + x[0] * x[1],
            2.25 - x[0] + x[0] * x[1] * x[1],
            2.625 - x[0] + x[0] * x[1].powi(3),
        );
        g[0] =
            2.0 * a * (x[1] - 1.0) + 2.0 * b * (x[1] * x[1] - 1.0) + 2.0 * c * (x[1].powi(3) - 1.0);
        g[1] = 2.0 * a * x[0] + 4.0 * b * x[0] * x[1] + 6.0 * c * x[0] * x[1] * x[1];
    };
    let mut x = [1.0, 1.0];
    bfgs(beale, bg, &mut x, opts(), &mut work).unwrap();
    assert!(
        (x[0] - 3.0).abs() < 1e-5 && (x[1] - 0.5).abs() < 1e-5,
        "{x:?}"
    );
}

#[test]
fn start_at_the_minimum_returns_immediately() {
    let mut x = [1.0, 1.0];
    let mut work = [0.0; bfgs_work_len(2)];
    let r = bfgs(rosen, rosen_grad, &mut x, opts(), &mut work).unwrap();
    assert_eq!((r.iterations, r.termination), (0, Termination::Gradient));
    let mut work = [0.0; lbfgs_work_len(2, 3)];
    let r = lbfgs(rosen, rosen_grad, &mut x, 3, opts(), &mut work).unwrap();
    assert_eq!((r.iterations, r.termination), (0, Termination::Gradient));
}

#[test]
fn input_errors() {
    let mut work = [0.0; 64];
    let mut empty: [f64; 0] = [];
    assert_eq!(
        bfgs(rosen, rosen_grad, &mut empty, opts(), &mut work),
        Err(OptimizeError::InvalidInput)
    );
    let mut x = [0.0, 0.0];
    let mut short = [0.0; 5];
    assert_eq!(
        bfgs(rosen, rosen_grad, &mut x, opts(), &mut short),
        Err(OptimizeError::InvalidInput)
    );
    assert_eq!(
        lbfgs(rosen, rosen_grad, &mut x, 0, opts(), &mut work),
        Err(OptimizeError::InvalidInput)
    );
    assert_eq!(
        lbfgs(rosen, rosen_grad, &mut x, 50, opts(), &mut work),
        Err(OptimizeError::InvalidInput)
    );
    let mut bad = [f64::NAN, 0.0];
    assert_eq!(
        bfgs(rosen, rosen_grad, &mut bad, opts(), &mut work),
        Err(OptimizeError::NotFinite)
    );
    let nan_f = |_: &[f64]| f64::NAN;
    assert_eq!(
        bfgs(nan_f, rosen_grad, &mut x, opts(), &mut work),
        Err(OptimizeError::NotFinite)
    );
}

#[test]
fn iteration_limit_gives_no_convergence() {
    let mut x = [-1.2, 1.0];
    let mut work = [0.0; bfgs_work_len(2)];
    let o = OptimizeOptions {
        max_iter: 3,
        ..opts()
    };
    assert_eq!(
        bfgs(rosen, rosen_grad, &mut x, o, &mut work),
        Err(OptimizeError::NoConvergence)
    );
}

#[test]
fn works_with_autodiff_gradients() {
    use azmath::Real;
    use azmath::autodiff::{Jet, VectorFn, ad_scratch_len, jacobian_ad};
    struct Rosen;
    impl VectorFn for Rosen {
        fn eval<S: Real>(&self, x: &[S], out: &mut [S]) {
            let c = |v: f64| S::from_f64(v);
            let (a, b) = (c(1.0) - x[0], x[1] - x[0] * x[0]);
            out[0] = a * a + c(100.0) * b * b;
        }
    }
    let mut scratch = [Jet::<f64, 2>::constant(0.0); ad_scratch_len(1, 2)];
    let grad = |x: &[f64], g: &mut [f64]| jacobian_ad::<f64, 2, _>(&Rosen, x, 1, g, &mut scratch);
    let mut x = [-1.2, 1.0];
    let mut work = [0.0; bfgs_work_len(2)];
    bfgs(rosen, grad, &mut x, opts(), &mut work).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6);
}

#[test]
fn works_for_f32() {
    let f = |x: &[f32]| (x[0] - 1.0).powi(2) + 2.0 * (x[1] + 0.5).powi(2);
    let grad = |x: &[f32], g: &mut [f32]| {
        g[0] = 2.0 * (x[0] - 1.0);
        g[1] = 4.0 * (x[1] + 0.5);
    };
    let mut x = [4.0f32, 3.0];
    let mut work = [0.0f32; bfgs_work_len(2)];
    bfgs(f, grad, &mut x, OptimizeOptions::default(), &mut work).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-3 && (x[1] + 0.5).abs() < 1e-3);
}

// ---------------------------------------------------------------------------
// Broyden
// ---------------------------------------------------------------------------

mod broyden_tests {
    use azmath::optimize::{
        OptimizeError, OptimizeOptions, Termination, broyden, broyden_fd, broyden_work_len,
        newton_system, newton_work_len,
    };

    fn circle_line(x: &[f64], out: &mut [f64]) {
        out[0] = x[0] * x[0] + x[1] * x[1] - 4.0;
        out[1] = x[1] - x[0];
    }
    fn circle_line_jac(x: &[f64], j: &mut [f64]) {
        j.copy_from_slice(&[2.0 * x[0], 2.0 * x[1], -1.0, 1.0]);
    }

    #[test]
    fn solves_circle_and_line() {
        let mut x = [1.0, 0.5];
        let mut work = [0.0; broyden_work_len(2)];
        let r = broyden(
            circle_line,
            circle_line_jac,
            &mut x,
            OptimizeOptions::default(),
            &mut work,
        )
        .unwrap();
        let s = 2f64.sqrt();
        assert!((x[0] - s).abs() < 1e-7 && (x[1] - s).abs() < 1e-7, "{x:?}");
        assert!(r.residual_norm < 1e-7);
        let mut y = [1.0, 0.5];
        broyden_fd(circle_line, &mut y, OptimizeOptions::default(), &mut work).unwrap();
        assert!((y[0] - s).abs() < 1e-6);
    }

    #[test]
    fn evaluates_the_jacobian_only_at_the_start() {
        let mut calls = 0;
        let mut x = [1.0, 0.5];
        let mut work = [0.0; broyden_work_len(2)];
        let jac = |x: &[f64], j: &mut [f64]| {
            calls += 1;
            circle_line_jac(x, j);
        };
        let r = broyden(
            circle_line,
            jac,
            &mut x,
            OptimizeOptions::default(),
            &mut work,
        )
        .unwrap();
        assert_eq!(
            calls, 1,
            "Jacobi-Matrix nur am Start (Iterationen: {})",
            r.iterations
        );
    }

    #[test]
    fn solves_a_ten_dimensional_system_like_newton() {
        // F_i(x) = x_i + 0,1·Σ_j sin(x_j) − 1,   Startwert 0
        let n = 10;
        let f = |x: &[f64], out: &mut [f64]| {
            let s: f64 = x.iter().map(|v| v.sin()).sum();
            for i in 0..n {
                out[i] = x[i] + 0.1 * s - 1.0;
            }
        };
        let jac = |x: &[f64], j: &mut [f64]| {
            for i in 0..n {
                for k in 0..n {
                    j[i * n + k] = 0.1 * x[k].cos() + if i == k { 1.0 } else { 0.0 };
                }
            }
        };
        let mut xb = [0.0; 10];
        let mut wb = [0.0; broyden_work_len(10)];
        let rb = broyden(f, jac, &mut xb, OptimizeOptions::default(), &mut wb).unwrap();
        let mut xn = [0.0; 10];
        let mut wn = [0.0; newton_work_len(10)];
        newton_system(f, jac, &mut xn, OptimizeOptions::default(), &mut wn).unwrap();
        for i in 0..n {
            assert!(
                (xb[i] - xn[i]).abs() < 1e-8,
                "x[{i}]: {} gegen {}",
                xb[i],
                xn[i]
            );
        }
        assert!(rb.iterations <= 25, "{} Iterationen", rb.iterations);
    }

    #[test]
    fn rosenbrock_as_a_system() {
        let f = |x: &[f64], out: &mut [f64]| {
            out[0] = 10.0 * (x[1] - x[0] * x[0]);
            out[1] = 1.0 - x[0];
        };
        let mut x = [-1.2, 1.0];
        let mut work = [0.0; broyden_work_len(2)];
        broyden_fd(f, &mut x, OptimizeOptions::default(), &mut work).unwrap();
        assert!(
            (x[0] - 1.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6,
            "{x:?}"
        );
    }

    #[test]
    fn already_solved_and_errors() {
        let s = 2f64.sqrt();
        let mut x = [s, s];
        let mut work = [0.0; broyden_work_len(2)];
        let o = OptimizeOptions {
            tol_f: 1e-12,
            ..OptimizeOptions::default()
        };
        let r = broyden_fd(circle_line, &mut x, o, &mut work).unwrap();
        assert_eq!((r.iterations, r.termination), (0, Termination::Residual));

        let mut empty: [f64; 0] = [];
        assert_eq!(
            broyden_fd(
                circle_line,
                &mut empty,
                OptimizeOptions::default(),
                &mut work
            ),
            Err(OptimizeError::InvalidInput)
        );
        let mut short = [0.0; 10];
        let mut x = [1.0, 0.5];
        assert_eq!(
            broyden_fd(circle_line, &mut x, OptimizeOptions::default(), &mut short),
            Err(OptimizeError::InvalidInput)
        );
        let mut bad = [f64::NAN, 0.0];
        assert_eq!(
            broyden_fd(circle_line, &mut bad, OptimizeOptions::default(), &mut work),
            Err(OptimizeError::NotFinite)
        );
        // singuläre Anfangs-Jacobi-Matrix: x² − 1 = 0 bei x = 0
        let mut z = [0.0];
        let mut w1 = [0.0; broyden_work_len(1)];
        let sq = |x: &[f64], out: &mut [f64]| out[0] = x[0] * x[0] - 1.0;
        let dsq = |x: &[f64], j: &mut [f64]| j[0] = 2.0 * x[0];
        assert_eq!(
            broyden(sq, dsq, &mut z, OptimizeOptions::default(), &mut w1),
            Err(OptimizeError::SingularJacobian)
        );
        // keine Lösung: x² + 1 = 0
        let mut z = [1.0];
        let none = |x: &[f64], out: &mut [f64]| out[0] = x[0] * x[0] + 1.0;
        assert!(broyden_fd(none, &mut z, OptimizeOptions::default(), &mut w1).is_err());
    }
}
