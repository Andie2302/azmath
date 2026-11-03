//! Tests für das Innere-Punkte-Verfahren (QP und LP).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm` und brauchen `alloc`.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::opt::{
    LpStatus, OptError, QpOptions, QpStatus, Relation, SimplexOptions, interior_point_lp,
    interior_point_qp, simplex,
};

fn opts() -> QpOptions<f64> {
    QpOptions::default()
}

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} gegen {b}");
}

/// Deterministische Pseudozufallszahlen in [0, 1).
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

#[test]
fn unconstrained_qp_solves_the_normal_equations() {
    // min ½xᵀQx + cᵀx mit Q = [[4, 1], [1, 3]], c = (1, 2)  ->  x = −Q⁻¹c = (−1/11, −7/11)
    let sol = interior_point_qp(
        &[4.0, 1.0, 1.0, 3.0],
        &[1.0, 2.0],
        &[],
        &[],
        &[],
        &[],
        opts(),
    )
    .unwrap();
    assert_eq!(sol.status, QpStatus::Optimal);
    close(sol.x[0], -1.0 / 11.0, 1e-8);
    close(sol.x[1], -7.0 / 11.0, 1e-8);
    close(
        sol.objective,
        -0.5 * (1.0 * 1.0 / 11.0 + 2.0 * 7.0 / 11.0),
        1e-8,
    );
}

#[test]
fn equality_constrained_qp() {
    // min ½(x² + y²)  unter  x + y = 2  ->  (1, 1), Multiplikator y = −1
    let sol = interior_point_qp(
        &[1.0, 0.0, 0.0, 1.0],
        &[0.0, 0.0],
        &[1.0, 1.0],
        &[2.0],
        &[],
        &[],
        opts(),
    )
    .unwrap();
    assert_eq!(sol.status, QpStatus::Optimal);
    close(sol.x[0], 1.0, 1e-7);
    close(sol.x[1], 1.0, 1e-7);
    close(sol.y[0], -1.0, 1e-6);
    close(sol.objective, 1.0, 1e-7);
}

#[test]
fn projection_onto_a_half_space() {
    // Projektion von (2, 1) auf x + y ≤ 2: (1,5; 0,5), Multiplikator 0,5
    let sol = interior_point_qp(
        &[1.0, 0.0, 0.0, 1.0],
        &[-2.0, -1.0],
        &[],
        &[],
        &[1.0, 1.0],
        &[2.0],
        opts(),
    )
    .unwrap();
    assert_eq!(sol.status, QpStatus::Optimal);
    close(sol.x[0], 1.5, 1e-6);
    close(sol.x[1], 0.5, 1e-6);
    close(sol.z[0], 0.5, 1e-5);
    // Punkt im Inneren: Restriktion inaktiv, Multiplikator ≈ 0
    let inner = interior_point_qp(
        &[1.0, 0.0, 0.0, 1.0],
        &[-0.5, -0.5],
        &[],
        &[],
        &[1.0, 1.0],
        &[2.0],
        opts(),
    )
    .unwrap();
    close(inner.x[0], 0.5, 1e-6);
    close(inner.x[1], 0.5, 1e-6);
    assert!(inner.z[0].abs() < 1e-6);
}

#[test]
fn bound_constraints_active_and_inactive() {
    // min ½(x − 3)² + ½(y + 1)² mit x ≥ 0, y ≥ 0, also −x ≤ 0, −y ≤ 0  ->  (3, 0)
    let sol = interior_point_qp(
        &[1.0, 0.0, 0.0, 1.0],
        &[-3.0, 1.0],
        &[],
        &[],
        &[-1.0, 0.0, 0.0, -1.0],
        &[0.0, 0.0],
        opts(),
    )
    .unwrap();
    assert_eq!(sol.status, QpStatus::Optimal);
    close(sol.x[0], 3.0, 1e-6);
    close(sol.x[1], 0.0, 1e-6);
    close(sol.z[0], 0.0, 1e-5);
    close(sol.z[1], 1.0, 1e-5);
}

#[test]
fn random_convex_qps_satisfy_the_kkt_conditions() {
    let mut s = 777u64;
    for (n, m) in [(3, 2), (5, 4), (6, 3)] {
        // Q = MᵀM + I (SPD), Ungleichungen G·x ≤ h mit h > 0 (der Ursprung ist zulässig)
        let mm: Vec<f64> = (0..n * n).map(|_| lcg(&mut s) - 0.5).collect();
        let mut q = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                q[i * n + j] = (0..n).map(|k| mm[k * n + i] * mm[k * n + j]).sum::<f64>()
                    + if i == j { 1.0 } else { 0.0 };
            }
        }
        let c: Vec<f64> = (0..n).map(|_| 4.0 * (lcg(&mut s) - 0.5)).collect();
        let g: Vec<f64> = (0..m * n).map(|_| 2.0 * (lcg(&mut s) - 0.5)).collect();
        let h: Vec<f64> = (0..m).map(|_| 0.2 + lcg(&mut s)).collect();
        let sol = interior_point_qp(&q, &c, &[], &[], &g, &h, opts()).unwrap();
        assert_eq!(sol.status, QpStatus::Optimal, "n = {n}, m = {m}");
        // Stationarität: Qx + c + Gᵀz = 0
        for j in 0..n {
            let qx: f64 = (0..n).map(|k| q[j * n + k] * sol.x[k]).sum();
            let gz: f64 = (0..m).map(|r| g[r * n + j] * sol.z[r]).sum();
            assert!((qx + c[j] + gz).abs() < 1e-5, "Stationarität in {j}");
        }
        // Zulässigkeit, Vorzeichen und Komplementarität
        for r in 0..m {
            let gx: f64 = (0..n).map(|k| g[r * n + k] * sol.x[k]).sum();
            assert!(gx <= h[r] + 1e-6, "Zeile {r} verletzt");
            assert!(sol.z[r] >= -1e-8);
            assert!((h[r] - gx) * sol.z[r] < 1e-5, "Komplementarität in {r}");
        }
    }
}

#[test]
fn lp_diet_problem_agrees_with_the_simplex() {
    // min 2x + 3y  unter  x + y − s₁ = 4,  x + 3y − s₂ = 6,  alle Variablen ≥ 0   (Optimum 9)
    let c = [2.0, 3.0, 0.0, 0.0];
    let a = [1.0, 1.0, -1.0, 0.0, 1.0, 3.0, 0.0, -1.0];
    let b = [4.0, 6.0];
    let ip = interior_point_lp(&c, &a, &b, opts()).unwrap();
    assert_eq!(ip.status, QpStatus::Optimal);
    close(ip.objective, 9.0, 1e-5);
    let sx = simplex(&c, &a, &b, &[Relation::Eq; 2], SimplexOptions::default()).unwrap();
    assert_eq!(sx.status, LpStatus::Optimal);
    close(ip.objective, sx.objective, 1e-5);
}

#[test]
fn lp_transportation_problem() {
    let c = [8.0, 6.0, 10.0, 9.0, 12.0, 13.0];
    let a = [
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, //
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
    ];
    let b = [20.0, 30.0, 10.0, 25.0, 15.0]; // eine Gleichung ist redundant: die Regularisierung fängt das ab
    let sol = interior_point_lp(&c, &a, &b, opts()).unwrap();
    assert_eq!(sol.status, QpStatus::Optimal);
    close(sol.objective, 465.0, 1e-3);
    assert!(sol.x.iter().all(|&v| v >= -1e-6));
}

#[test]
fn infeasible_and_unbounded_problems_do_not_report_optimal() {
    // x ≥ 1 und x ≤ 0 (unzulässig)
    let inf =
        interior_point_qp(&[0.0], &[1.0], &[], &[], &[-1.0, 1.0], &[-1.0, 0.0], opts()).unwrap();
    assert_ne!(inf.status, QpStatus::Optimal);
    // min −x unter x ≥ 0 (unbeschränkt)
    let unb = interior_point_lp(&[-1.0], &[], &[], opts()).unwrap();
    assert_ne!(unb.status, QpStatus::Optimal);
}

#[test]
fn errors_and_iteration_limit() {
    assert_eq!(
        interior_point_qp::<f64>(&[], &[], &[], &[], &[], &[], opts()),
        Err(OptError::InvalidInput)
    );
    assert_eq!(
        interior_point_qp(&[1.0], &[1.0, 2.0], &[], &[], &[], &[], opts()),
        Err(OptError::InvalidInput)
    );
    assert_eq!(
        interior_point_qp(&[1.0], &[1.0], &[1.0, 1.0], &[1.0], &[], &[], opts()),
        Err(OptError::InvalidInput)
    );
    assert_eq!(
        interior_point_qp(&[1.0], &[1.0], &[], &[], &[1.0, 1.0], &[1.0], opts()),
        Err(OptError::InvalidInput)
    );
    assert_eq!(
        interior_point_qp(&[f64::NAN], &[1.0], &[], &[], &[], &[], opts()),
        Err(OptError::NotFinite)
    );
    let one = QpOptions {
        max_iter: 1,
        ..opts()
    };
    let r = interior_point_qp(
        &[1.0, 0.0, 0.0, 1.0],
        &[-2.0, -1.0],
        &[],
        &[],
        &[1.0, 1.0],
        &[2.0],
        one,
    )
    .unwrap();
    assert_eq!(r.status, QpStatus::MaxIterations);
}

#[test]
fn works_for_f32() {
    let sol = interior_point_qp(
        &[1.0f32, 0.0, 0.0, 1.0],
        &[-2.0, -1.0],
        &[],
        &[],
        &[1.0, 1.0],
        &[2.0],
        QpOptions {
            tol: 1e-4,
            ..QpOptions::default()
        },
    )
    .unwrap();
    assert!(matches!(
        sol.status,
        QpStatus::Optimal | QpStatus::MaxIterations
    ));
    assert!((sol.x[0] - 1.5).abs() < 1e-2 && (sol.x[1] - 0.5).abs() < 1e-2);
}
