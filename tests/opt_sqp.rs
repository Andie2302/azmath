//! Tests für das SQP-Verfahren.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm` und brauchen `alloc`.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::Real;
use azmath::autodiff::{Jet, VectorFn, ad_scratch_len, jacobian_ad};
use azmath::opt::{OptError, SqpOptions, SqpStatus, sqp};

fn opts() -> SqpOptions<f64> {
    SqpOptions::default()
}

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} gegen {b}");
}

#[test]
fn linear_equality_constraint() {
    // min x² + y² unter x + y = 1 -> (½, ½), Multiplikator −1
    let mut x = [3.0, -2.0];
    let r = sqp(
        |x: &[f64]| x[0] * x[0] + x[1] * x[1],
        |x: &[f64], g: &mut [f64]| {
            g[0] = 2.0 * x[0];
            g[1] = 2.0 * x[1];
        },
        |x: &[f64], c: &mut [f64]| c[0] = x[0] + x[1] - 1.0,
        |_x: &[f64], j: &mut [f64]| j.copy_from_slice(&[1.0, 1.0]),
        1,
        0,
        &mut x,
        opts(),
    )
    .unwrap();
    assert_eq!(r.status, SqpStatus::Optimal);
    close(x[0], 0.5, 1e-6);
    close(x[1], 0.5, 1e-6);
    close(r.lambda_eq[0], -1.0, 1e-5);
    close(r.f_value, 0.5, 1e-6);
    assert!(r.violation < 1e-7);
}

#[test]
fn nonlinear_equality_constraint() {
    // min x + y unter x² + y² = 2 -> (−1, −1)
    let mut x = [-0.5, -1.5];
    let r = sqp(
        |x: &[f64]| x[0] + x[1],
        |_x: &[f64], g: &mut [f64]| g.copy_from_slice(&[1.0, 1.0]),
        |x: &[f64], c: &mut [f64]| c[0] = x[0] * x[0] + x[1] * x[1] - 2.0,
        |x: &[f64], j: &mut [f64]| j.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]),
        1,
        0,
        &mut x,
        opts(),
    )
    .unwrap();
    assert_eq!(r.status, SqpStatus::Optimal);
    close(x[0], -1.0, 1e-5);
    close(x[1], -1.0, 1e-5);
}

#[test]
fn inequality_active_and_inactive() {
    // min (x − 2)² + (y − 1)² unter x + y ≤ 2: aktiv -> (1,5; 0,5); unter x + y ≤ 5: inaktiv -> (2, 1)
    let run = |rhs: f64| {
        let mut x = [0.0, 0.0];
        let r = sqp(
            |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2),
            |x: &[f64], g: &mut [f64]| g.copy_from_slice(&[2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
            move |x: &[f64], c: &mut [f64]| c[0] = x[0] + x[1] - rhs,
            |_x: &[f64], j: &mut [f64]| j.copy_from_slice(&[1.0, 1.0]),
            0,
            1,
            &mut x,
            opts(),
        )
        .unwrap();
        (r, x)
    };
    let (r, x) = run(2.0);
    assert_eq!(r.status, SqpStatus::Optimal);
    close(x[0], 1.5, 1e-5);
    close(x[1], 0.5, 1e-5);
    assert!(r.lambda_ineq[0] > 0.9);
    let (r, x) = run(5.0);
    assert_eq!(r.status, SqpStatus::Optimal);
    close(x[0], 2.0, 1e-5);
    close(x[1], 1.0, 1e-5);
    assert!(r.lambda_ineq[0].abs() < 1e-6);
}

#[test]
fn rosenbrock_inside_a_disc() {
    // min Rosenbrock unter x² + y² ≤ 2: das freie Minimum (1, 1) liegt genau auf dem Rand
    let mut x = [-1.2, 1.0];
    let r = sqp(
        |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
        |x: &[f64], g: &mut [f64]| {
            g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            g[1] = 200.0 * (x[1] - x[0] * x[0]);
        },
        |x: &[f64], c: &mut [f64]| c[0] = x[0] * x[0] + x[1] * x[1] - 2.0,
        |x: &[f64], j: &mut [f64]| j.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]),
        0,
        1,
        &mut x,
        SqpOptions {
            max_iter: 500,
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(r.status, SqpStatus::Optimal);
    close(x[0], 1.0, 1e-4);
    close(x[1], 1.0, 1e-4);
}

#[test]
fn hock_schittkowski_71() {
    // min x₁x₄(x₁ + x₂ + x₃) + x₃  unter  x₁x₂x₃x₄ ≥ 25,  Σxᵢ² = 40,  1 ≤ xᵢ ≤ 5
    let f = |x: &[f64]| x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
    let grad = |x: &[f64], g: &mut [f64]| {
        g[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + 1.0;
        g[3] = x[0] * (x[0] + x[1] + x[2]);
    };
    // Gleichung zuerst, dann: 25 − Πx ≤ 0, 1 − xᵢ ≤ 0, xᵢ − 5 ≤ 0
    let cons = |x: &[f64], c: &mut [f64]| {
        c[0] = x.iter().map(|v| v * v).sum::<f64>() - 40.0;
        c[1] = 25.0 - x[0] * x[1] * x[2] * x[3];
        for i in 0..4 {
            c[2 + i] = 1.0 - x[i];
            c[6 + i] = x[i] - 5.0;
        }
    };
    let jac = |x: &[f64], j: &mut [f64]| {
        j.fill(0.0);
        for i in 0..4 {
            j[i] = 2.0 * x[i];
        }
        j[4] = -x[1] * x[2] * x[3];
        j[5] = -x[0] * x[2] * x[3];
        j[6] = -x[0] * x[1] * x[3];
        j[7] = -x[0] * x[1] * x[2];
        for i in 0..4 {
            j[(2 + i) * 4 + i] = -1.0;
            j[(6 + i) * 4 + i] = 1.0;
        }
    };
    let mut x = [1.0, 5.0, 5.0, 1.0];
    let r = sqp(
        f,
        grad,
        cons,
        jac,
        1,
        9,
        &mut x,
        SqpOptions {
            max_iter: 300,
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(r.status, SqpStatus::Optimal, "{x:?}");
    close(r.f_value, 17.014_017, 1e-3);
    close(x[0], 1.0, 1e-3);
    close(x[1], 4.743, 2e-3);
    close(x[2], 3.821, 2e-3);
    close(x[3], 1.379, 2e-3);
}

#[test]
fn derivatives_from_automatic_differentiation() {
    // min (x − 1)² + (y − 2)² unter x² + y² ≤ 1; Ziel und Nebenbedingung als VectorFn, Ableitungen per Autodiff
    struct Obj;
    impl VectorFn for Obj {
        fn eval<S: Real>(&self, x: &[S], out: &mut [S]) {
            let c = |v: f64| S::from_f64(v);
            out[0] = (x[0] - c(1.0)) * (x[0] - c(1.0)) + (x[1] - c(2.0)) * (x[1] - c(2.0));
        }
    }
    struct Con;
    impl VectorFn for Con {
        fn eval<S: Real>(&self, x: &[S], out: &mut [S]) {
            out[0] = x[0] * x[0] + x[1] * x[1] - S::ONE;
        }
    }
    let mut s1 = [Jet::<f64, 2>::constant(0.0); ad_scratch_len(1, 2)];
    let mut s2 = [Jet::<f64, 2>::constant(0.0); ad_scratch_len(1, 2)];
    let mut x = [0.0, 0.0];
    let r = sqp(
        |x: &[f64]| {
            let mut o = [0.0];
            Obj.eval::<f64>(x, &mut o);
            o[0]
        },
        |x: &[f64], g: &mut [f64]| jacobian_ad::<f64, 2, _>(&Obj, x, 1, g, &mut s1),
        |x: &[f64], c: &mut [f64]| Con.eval::<f64>(x, c),
        |x: &[f64], j: &mut [f64]| jacobian_ad::<f64, 2, _>(&Con, x, 1, j, &mut s2),
        0,
        1,
        &mut x,
        opts(),
    )
    .unwrap();
    assert_eq!(r.status, SqpStatus::Optimal);
    // Projektion von (1, 2) auf den Einheitskreis
    let n = 5f64.sqrt();
    close(x[0], 1.0 / n, 1e-5);
    close(x[1], 2.0 / n, 1e-5);
}

#[test]
fn unconstrained_problems_reduce_to_a_quasi_newton_method() {
    let mut x = [-1.2, 1.0];
    let r = sqp(
        |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
        |x: &[f64], g: &mut [f64]| {
            g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            g[1] = 200.0 * (x[1] - x[0] * x[0]);
        },
        |_x: &[f64], _c: &mut [f64]| {},
        |_x: &[f64], _j: &mut [f64]| {},
        0,
        0,
        &mut x,
        SqpOptions {
            max_iter: 500,
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(r.status, SqpStatus::Optimal);
    close(x[0], 1.0, 1e-4);
    close(x[1], 1.0, 1e-4);
}

#[test]
fn failure_statuses_and_errors() {
    // x = 1 und x = 2 gleichzeitig: das linearisierte Teilproblem ist unlösbar
    let mut x = [0.0];
    let r = sqp(
        |x: &[f64]| x[0] * x[0],
        |x: &[f64], g: &mut [f64]| g[0] = 2.0 * x[0],
        |x: &[f64], c: &mut [f64]| {
            c[0] = x[0] - 1.0;
            c[1] = x[0] - 2.0;
        },
        |_x: &[f64], j: &mut [f64]| j.copy_from_slice(&[1.0, 1.0]),
        2,
        0,
        &mut x,
        opts(),
    )
    .unwrap();
    assert_ne!(r.status, SqpStatus::Optimal);
    // Iterationslimit
    let mut y = [3.0, -2.0];
    let r = sqp(
        |x: &[f64]| x[0] * x[0] + x[1] * x[1],
        |x: &[f64], g: &mut [f64]| g.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]),
        |x: &[f64], c: &mut [f64]| c[0] = x[0] * x[0] + x[1] * x[1] - 2.0,
        |x: &[f64], j: &mut [f64]| j.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]),
        1,
        0,
        &mut y,
        SqpOptions {
            max_iter: 1,
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(r.status, SqpStatus::MaxIterations);
    // Eingabefehler
    let mut empty: [f64; 0] = [];
    let noop_g = |_x: &[f64], _g: &mut [f64]| {};
    assert_eq!(
        sqp(
            |_x: &[f64]| 0.0,
            noop_g,
            |_x: &[f64], _c: &mut [f64]| {},
            noop_g,
            0,
            0,
            &mut empty,
            opts()
        ),
        Err(OptError::InvalidInput)
    );
    let mut bad = [f64::NAN];
    assert_eq!(
        sqp(
            |x: &[f64]| x[0],
            noop_g,
            |_x: &[f64], _c: &mut [f64]| {},
            noop_g,
            0,
            0,
            &mut bad,
            opts()
        ),
        Err(OptError::NotFinite)
    );
}

#[test]
fn works_for_f32() {
    let mut x = [3.0f32, -2.0];
    let r = sqp(
        |x: &[f32]| x[0] * x[0] + x[1] * x[1],
        |x: &[f32], g: &mut [f32]| g.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]),
        |x: &[f32], c: &mut [f32]| c[0] = x[0] + x[1] - 1.0,
        |_x: &[f32], j: &mut [f32]| j.copy_from_slice(&[1.0, 1.0]),
        1,
        0,
        &mut x,
        SqpOptions {
            tol: 1e-3,
            max_iter: 100,
        },
    )
    .unwrap();
    assert!(matches!(
        r.status,
        SqpStatus::Optimal | SqpStatus::MaxIterations | SqpStatus::QpFailed
    ));
    assert!(
        (x[0] - 0.5).abs() < 1e-2 && (x[1] - 0.5).abs() < 1e-2,
        "{x:?}"
    );
}
