//! Tests für das Innere-Punkte-Verfahren für nichtlineare Programme.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm` und brauchen `alloc`.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::Real;
use azmath::autodiff::{VectorFn, hessian_ad};
use azmath::opt::{
    NlpOptions, NlpStatus, OptError, SqpOptions, SqpStatus, interior_point_nlp, sqp,
};

fn opts() -> NlpOptions<f64> {
    NlpOptions::default()
}

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} gegen {b}");
}

#[test]
fn linear_equality_constraint() {
    // min x² + y² unter x + y = 1 -> (½, ½), Multiplikator −1
    let mut x = [3.0, -2.0];
    let r = interior_point_nlp(
        |x: &[f64]| x[0] * x[0] + x[1] * x[1],
        |x: &[f64], g: &mut [f64]| g.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]),
        |x: &[f64], c: &mut [f64]| c[0] = x[0] + x[1] - 1.0,
        |_x: &[f64], j: &mut [f64]| j.copy_from_slice(&[1.0, 1.0]),
        |_x: &[f64], _l: &[f64], h: &mut [f64]| h.copy_from_slice(&[2.0, 0.0, 0.0, 2.0]),
        1,
        0,
        &mut x,
        opts(),
    )
    .unwrap();
    assert_eq!(r.status, NlpStatus::Optimal);
    close(x[0], 0.5, 1e-6);
    close(x[1], 0.5, 1e-6);
    close(r.lambda_eq[0], -1.0, 1e-5);
    assert!(r.violation < 1e-7);
}

#[test]
fn nonlinear_equality_constraint() {
    // min x + y unter x² + y² = 2 -> (−1, −1); W = 2·λ₀·I
    let mut x = [-0.5, -1.5];
    let r = interior_point_nlp(
        |x: &[f64]| x[0] + x[1],
        |_x: &[f64], g: &mut [f64]| g.copy_from_slice(&[1.0, 1.0]),
        |x: &[f64], c: &mut [f64]| c[0] = x[0] * x[0] + x[1] * x[1] - 2.0,
        |x: &[f64], j: &mut [f64]| j.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]),
        |_x: &[f64], l: &[f64], h: &mut [f64]| {
            h.copy_from_slice(&[2.0 * l[0], 0.0, 0.0, 2.0 * l[0]])
        },
        1,
        0,
        &mut x,
        opts(),
    )
    .unwrap();
    assert_eq!(r.status, NlpStatus::Optimal);
    close(x[0], -1.0, 1e-5);
    close(x[1], -1.0, 1e-5);
}

fn halfspace(rhs: f64) -> (NlpStatusAndX, f64) {
    let mut x = [0.0, 0.0];
    let r = interior_point_nlp(
        |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2),
        |x: &[f64], g: &mut [f64]| g.copy_from_slice(&[2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        move |x: &[f64], c: &mut [f64]| c[0] = x[0] + x[1] - rhs,
        |_x: &[f64], j: &mut [f64]| j.copy_from_slice(&[1.0, 1.0]),
        |_x: &[f64], _l: &[f64], h: &mut [f64]| h.copy_from_slice(&[2.0, 0.0, 0.0, 2.0]),
        0,
        1,
        &mut x,
        opts(),
    )
    .unwrap();
    ((r.status, x), r.lambda_ineq[0])
}

type NlpStatusAndX = (NlpStatus, [f64; 2]);

#[test]
fn inequality_active_and_inactive() {
    let ((status, x), z) = halfspace(2.0);
    assert_eq!(status, NlpStatus::Optimal);
    close(x[0], 1.5, 1e-5);
    close(x[1], 0.5, 1e-5);
    assert!(z > 0.9);
    let ((status, x), z) = halfspace(5.0);
    assert_eq!(status, NlpStatus::Optimal);
    close(x[0], 2.0, 1e-5);
    close(x[1], 1.0, 1e-5);
    assert!(z.abs() < 1e-5);
}

#[test]
fn rosenbrock_inside_a_disc() {
    let mut x = [-1.2, 1.0];
    let r = interior_point_nlp(
        |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
        |x: &[f64], g: &mut [f64]| {
            g[0] = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]);
            g[1] = 200.0 * (x[1] - x[0] * x[0]);
        },
        |x: &[f64], c: &mut [f64]| c[0] = x[0] * x[0] + x[1] * x[1] - 2.0,
        |x: &[f64], j: &mut [f64]| j.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]),
        |x: &[f64], l: &[f64], h: &mut [f64]| {
            h[0] = 2.0 - 400.0 * x[1] + 1200.0 * x[0] * x[0] + 2.0 * l[0];
            h[1] = -400.0 * x[0];
            h[2] = -400.0 * x[0];
            h[3] = 200.0 + 2.0 * l[0];
        },
        0,
        1,
        &mut x,
        NlpOptions {
            max_iter: 500,
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(r.status, NlpStatus::Optimal);
    close(x[0], 1.0, 1e-4);
    close(x[1], 1.0, 1e-4);
}

#[test]
fn hock_schittkowski_71() {
    let f = |x: &[f64]| x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2];
    let grad = |x: &[f64], g: &mut [f64]| {
        g[0] = x[3] * (2.0 * x[0] + x[1] + x[2]);
        g[1] = x[0] * x[3];
        g[2] = x[0] * x[3] + 1.0;
        g[3] = x[0] * (x[0] + x[1] + x[2]);
    };
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
    let hess = |x: &[f64], l: &[f64], h: &mut [f64]| {
        let mut m = [[0.0; 4]; 4];
        m[0][0] = 2.0 * x[3];
        m[0][1] = x[3];
        m[0][2] = x[3];
        m[0][3] = 2.0 * x[0] + x[1] + x[2];
        m[2][3] = x[0];
        m[1][3] = x[0];
        for i in 0..4 {
            for j in (i + 1)..4 {
                m[j][i] = m[i][j];
            }
        }
        for i in 0..4 {
            m[i][i] += 2.0 * l[0]; // Gleichung Σxᵢ² = 40
            for j in 0..4 {
                if i != j {
                    // −∂²(x₁x₂x₃x₄)/∂xᵢ∂xⱼ = −Produkt der beiden anderen Variablen
                    let others: f64 = (0..4)
                        .filter(|k| *k != i && *k != j)
                        .map(|k| x[k])
                        .product();
                    m[i][j] -= l[1] * others;
                }
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                h[i * 4 + j] = m[i][j];
            }
        }
    };
    let mut x = [1.0, 5.0, 5.0, 1.0];
    let r = interior_point_nlp(
        f,
        grad,
        cons,
        jac,
        hess,
        1,
        9,
        &mut x,
        NlpOptions {
            max_iter: 400,
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(r.status, NlpStatus::Optimal, "{x:?}");
    close(r.f_value, 17.014_017, 1e-3);
    close(x[0], 1.0, 1e-3);
    close(x[1], 4.743, 2e-3);
    close(x[2], 3.821, 2e-3);
    close(x[3], 1.379, 2e-3);
}

#[test]
fn many_bound_constraints() {
    // min Σ (xᵢ − aᵢ)² unter xᵢ ≥ 1 (n = 20), aᵢ = 0,3 + 0,15·i  ->  xᵢ = max(aᵢ, 1)
    let n = 20;
    let a: Vec<f64> = (0..n).map(|i| 0.3 + 0.15 * i as f64).collect();
    let a2 = a.clone();
    let mut x = vec![0.0; n];
    let r = interior_point_nlp(
        move |x: &[f64]| x.iter().zip(&a2).map(|(v, a)| (v - a) * (v - a)).sum(),
        {
            let a = a.clone();
            move |x: &[f64], g: &mut [f64]| {
                for i in 0..x.len() {
                    g[i] = 2.0 * (x[i] - a[i]);
                }
            }
        },
        |x: &[f64], c: &mut [f64]| {
            for i in 0..x.len() {
                c[i] = 1.0 - x[i];
            }
        },
        |x: &[f64], j: &mut [f64]| {
            j.fill(0.0);
            for i in 0..x.len() {
                j[i * x.len() + i] = -1.0;
            }
        },
        |x: &[f64], _l: &[f64], h: &mut [f64]| {
            h.fill(0.0);
            for i in 0..x.len() {
                h[i * x.len() + i] = 2.0;
            }
        },
        0,
        n,
        &mut x,
        opts(),
    )
    .unwrap();
    assert_eq!(r.status, NlpStatus::Optimal);
    for i in 0..n {
        close(x[i], a[i].max(1.0), 1e-5);
    }
}

#[test]
fn hessian_from_automatic_differentiation() {
    // min (x − 1)² + (y − 2)² unter x² + y² ≤ 1: Lagrange-Funktion als VectorFn mit festem Multiplikator
    struct Lagrangian {
        lambda: f64,
    }
    impl VectorFn for Lagrangian {
        fn eval<S: Real>(&self, x: &[S], out: &mut [S]) {
            let c = |v: f64| S::from_f64(v);
            out[0] = (x[0] - c(1.0)) * (x[0] - c(1.0))
                + (x[1] - c(2.0)) * (x[1] - c(2.0))
                + c(self.lambda) * (x[0] * x[0] + x[1] * x[1] - c(1.0));
        }
    }
    let mut x = [0.0, 0.0];
    let r = interior_point_nlp(
        |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2),
        |x: &[f64], g: &mut [f64]| g.copy_from_slice(&[2.0 * (x[0] - 1.0), 2.0 * (x[1] - 2.0)]),
        |x: &[f64], c: &mut [f64]| c[0] = x[0] * x[0] + x[1] * x[1] - 1.0,
        |x: &[f64], j: &mut [f64]| j.copy_from_slice(&[2.0 * x[0], 2.0 * x[1]]),
        |x: &[f64], l: &[f64], h: &mut [f64]| {
            let hm = hessian_ad::<f64, 2, _>(&Lagrangian { lambda: l[0] }, &[x[0], x[1]]);
            for i in 0..2 {
                for j in 0..2 {
                    h[i * 2 + j] = hm[i][j];
                }
            }
        },
        0,
        1,
        &mut x,
        opts(),
    )
    .unwrap();
    assert_eq!(r.status, NlpStatus::Optimal);
    let n = 5f64.sqrt();
    close(x[0], 1.0 / n, 1e-5);
    close(x[1], 2.0 / n, 1e-5);
}

#[test]
fn agrees_with_sqp_on_a_shared_problem() {
    let f = |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2);
    let g = |x: &[f64], g: &mut [f64]| g.copy_from_slice(&[2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]);
    let c = |x: &[f64], c: &mut [f64]| c[0] = x[0] + x[1] - 2.0;
    let j = |_x: &[f64], j: &mut [f64]| j.copy_from_slice(&[1.0, 1.0]);
    let mut xs = [0.0, 0.0];
    let rs = sqp(f, g, c, j, 0, 1, &mut xs, SqpOptions::default()).unwrap();
    let mut xi = [0.0, 0.0];
    let ri = interior_point_nlp(
        f,
        g,
        c,
        j,
        |_x: &[f64], _l: &[f64], h: &mut [f64]| h.copy_from_slice(&[2.0, 0.0, 0.0, 2.0]),
        0,
        1,
        &mut xi,
        opts(),
    )
    .unwrap();
    assert_eq!(rs.status, SqpStatus::Optimal);
    assert_eq!(ri.status, NlpStatus::Optimal);
    close(xs[0], xi[0], 1e-5);
    close(xs[1], xi[1], 1e-5);
    close(rs.f_value, ri.f_value, 1e-6);
}

#[test]
fn indefinite_hessian_triggers_the_inertia_correction() {
    // f = −x² + 0,1·x⁴ auf |x| ≤ 2: im Start x = 0,1 ist f'' < 0; das Minimum liegt am Rand x = 2 (f = −2,4)
    let mut x = [0.1];
    let r = interior_point_nlp(
        |x: &[f64]| -x[0] * x[0] + 0.1 * x[0].powi(4),
        |x: &[f64], g: &mut [f64]| g[0] = -2.0 * x[0] + 0.4 * x[0].powi(3),
        |x: &[f64], c: &mut [f64]| {
            c[0] = x[0] - 2.0;
            c[1] = -x[0] - 2.0;
        },
        |_x: &[f64], j: &mut [f64]| j.copy_from_slice(&[1.0, -1.0]),
        |x: &[f64], _l: &[f64], h: &mut [f64]| h[0] = -2.0 + 1.2 * x[0] * x[0],
        0,
        2,
        &mut x,
        opts(),
    )
    .unwrap();
    assert_eq!(r.status, NlpStatus::Optimal);
    close(x[0].abs(), 2.0, 1e-5);
    close(r.f_value, -2.4, 1e-5);
}

#[test]
fn status_values_and_errors() {
    let f = |x: &[f64]| x[0] * x[0];
    let g = |x: &[f64], g: &mut [f64]| g[0] = 2.0 * x[0];
    let h = |_x: &[f64], _l: &[f64], h: &mut [f64]| h[0] = 2.0;
    // Iterationslimit
    let mut x = [5.0];
    let r = interior_point_nlp(
        f,
        g,
        |x: &[f64], c: &mut [f64]| c[0] = 1.0 - x[0],
        |_x: &[f64], j: &mut [f64]| j[0] = -1.0,
        h,
        0,
        1,
        &mut x,
        NlpOptions {
            max_iter: 2,
            ..opts()
        },
    )
    .unwrap();
    assert_eq!(r.status, NlpStatus::MaxIterations);
    // unzulässig: x ≥ 1 und x ≤ 0
    let mut y = [0.5];
    let r = interior_point_nlp(
        f,
        g,
        |x: &[f64], c: &mut [f64]| {
            c[0] = 1.0 - x[0];
            c[1] = x[0];
        },
        |_x: &[f64], j: &mut [f64]| j.copy_from_slice(&[-1.0, 1.0]),
        h,
        0,
        2,
        &mut y,
        opts(),
    )
    .unwrap();
    assert_ne!(r.status, NlpStatus::Optimal);
    // Eingabefehler
    let mut empty: [f64; 0] = [];
    let noop = |_x: &[f64], _g: &mut [f64]| {};
    let noop_h = |_x: &[f64], _l: &[f64], _h: &mut [f64]| {};
    assert_eq!(
        interior_point_nlp(
            |_x: &[f64]| 0.0,
            noop,
            noop,
            noop,
            noop_h,
            0,
            0,
            &mut empty,
            opts()
        ),
        Err(OptError::InvalidInput)
    );
    let mut bad = [f64::NAN];
    assert_eq!(
        interior_point_nlp(
            |x: &[f64]| x[0],
            noop,
            noop,
            noop,
            noop_h,
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
    let mut x = [0.0f32, 0.0];
    let r = interior_point_nlp(
        |x: &[f32]| (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2),
        |x: &[f32], g: &mut [f32]| g.copy_from_slice(&[2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        |x: &[f32], c: &mut [f32]| c[0] = x[0] + x[1] - 2.0,
        |_x: &[f32], j: &mut [f32]| j.copy_from_slice(&[1.0, 1.0]),
        |_x: &[f32], _l: &[f32], h: &mut [f32]| h.copy_from_slice(&[2.0, 0.0, 0.0, 2.0]),
        0,
        1,
        &mut x,
        NlpOptions {
            tol: 1e-3,
            ..NlpOptions::default()
        },
    )
    .unwrap();
    assert!(matches!(
        r.status,
        NlpStatus::Optimal | NlpStatus::MaxIterations | NlpStatus::LineSearchFailed
    ));
    assert!(
        (x[0] - 1.5).abs() < 1e-2 && (x[1] - 0.5).abs() < 1e-2,
        "{x:?}"
    );
}
