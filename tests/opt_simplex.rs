//! Tests für das Simplex-Verfahren.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm` und brauchen `alloc`.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::opt::{LpStatus, OptError, Relation, SimplexOptions, simplex};

use Relation::{Eq, Ge, Le};

fn opts() -> SimplexOptions<f64> {
    SimplexOptions::default()
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-8, "{a} gegen {b}");
}

/// Deterministische Pseudozufallszahlen in [0, 1).
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

#[test]
fn textbook_maximization() {
    // maximiere 3x + 2y  unter  x + y ≤ 4,  x + 3y ≤ 6,  x ≤ 3
    let sol = simplex(
        &[-3.0, -2.0],
        &[1.0, 1.0, 1.0, 3.0, 1.0, 0.0],
        &[4.0, 6.0, 3.0],
        &[Le, Le, Le],
        opts(),
    )
    .unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    close(sol.x[0], 3.0);
    close(sol.x[1], 1.0);
    close(sol.objective, -11.0);
    assert!(sol.iterations >= 1);
}

#[test]
fn minimization_with_greater_or_equal_constraints() {
    // Diät: minimiere 2x + 3y unter x + y ≥ 4, x + 3y ≥ 6
    let sol = simplex(
        &[2.0, 3.0],
        &[1.0, 1.0, 1.0, 3.0],
        &[4.0, 6.0],
        &[Ge, Ge],
        opts(),
    )
    .unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    close(sol.x[0], 3.0);
    close(sol.x[1], 1.0);
    close(sol.objective, 9.0);
}

#[test]
fn equality_constraint() {
    // minimiere x + y unter x + 2y = 4
    let sol = simplex(&[1.0, 1.0], &[1.0, 2.0], &[4.0], &[Eq], opts()).unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    close(sol.x[0], 0.0);
    close(sol.x[1], 2.0);
    close(sol.objective, 2.0);
}

#[test]
fn negative_right_hand_side_is_normalized() {
    // minimiere x + y unter x − y ≤ −1  (also y ≥ x + 1)
    let sol = simplex(&[1.0, 1.0], &[1.0, -1.0], &[-1.0], &[Le], opts()).unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    close(sol.x[0], 0.0);
    close(sol.x[1], 1.0);
    close(sol.objective, 1.0);
}

#[test]
fn infeasible_and_unbounded_problems() {
    // x ≤ 1 und x ≥ 2
    let inf = simplex(&[1.0], &[1.0, 1.0], &[1.0, 2.0], &[Le, Ge], opts()).unwrap();
    assert_eq!(inf.status, LpStatus::Infeasible);
    // minimiere −x unter x − y ≤ 1: x und y wachsen gemeinsam ohne Grenze
    let unb = simplex(&[-1.0, 0.0], &[1.0, -1.0], &[1.0], &[Le], opts()).unwrap();
    assert_eq!(unb.status, LpStatus::Unbounded);
    // widersprüchliche Gleichungen
    let inf2 = simplex(
        &[1.0, 1.0],
        &[1.0, 1.0, 1.0, 1.0],
        &[1.0, 2.0],
        &[Eq, Eq],
        opts(),
    )
    .unwrap();
    assert_eq!(inf2.status, LpStatus::Infeasible);
}

#[test]
fn redundant_equalities_are_accepted() {
    // x + y = 2 und 2x + 2y = 4 (doppelt), minimiere x
    let sol = simplex(
        &[1.0, 0.0],
        &[1.0, 1.0, 2.0, 2.0],
        &[2.0, 4.0],
        &[Eq, Eq],
        opts(),
    )
    .unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    close(sol.x[0], 0.0);
    close(sol.x[1], 2.0);
}

#[test]
fn beales_cycling_example_terminates_at_the_optimum() {
    // Beale (1955): bei naiver Pivotwahl zyklisch; Optimum −1/20 bei (1/25, 0, 1, 0)
    let c = [-0.75, 150.0, -0.02, 6.0];
    let a = [
        0.25, -60.0, -0.04, 9.0, //
        0.5, -90.0, -0.02, 3.0, //
        0.0, 0.0, 1.0, 0.0,
    ];
    let sol = simplex(&c, &a, &[0.0, 0.0, 1.0], &[Le, Le, Le], opts()).unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    close(sol.objective, -0.05);
}

#[test]
fn transportation_problem_with_a_redundant_row() {
    // 2 Lieferanten (20, 30), 3 Abnehmer (10, 25, 15); Kosten [[8, 6, 10], [9, 12, 13]]
    let c = [8.0, 6.0, 10.0, 9.0, 12.0, 13.0];
    let a = [
        1.0, 1.0, 1.0, 0.0, 0.0, 0.0, // Lieferant 1
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, // Lieferant 2
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, // Abnehmer 1
        0.0, 1.0, 0.0, 0.0, 1.0, 0.0, // Abnehmer 2
        0.0, 0.0, 1.0, 0.0, 0.0, 1.0, // Abnehmer 3
    ];
    let b = [20.0, 30.0, 10.0, 25.0, 15.0];
    let sol = simplex(&c, &a, &b, &[Eq; 5], opts()).unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    close(sol.objective, 465.0);
    for (row, rhs) in b.iter().enumerate() {
        let lhs: f64 = (0..6).map(|j| a[row * 6 + j] * sol.x[j]).sum();
        close(lhs, *rhs);
    }
    assert!(sol.x.iter().all(|&v| v >= -1e-9));
}

#[test]
fn strong_duality_on_random_problems() {
    // Primal: min cᵀx, A·x ≥ b, x ≥ 0    Dual: max bᵀy, Aᵀ·y ≤ c, y ≥ 0   (A, b, c > 0: beide zulässig)
    let mut s = 12345u64;
    for (m, n) in [(2, 3), (3, 3), (4, 6), (5, 4)] {
        let a: Vec<f64> = (0..m * n).map(|_| 0.1 + lcg(&mut s)).collect();
        let b: Vec<f64> = (0..m).map(|_| 0.5 + 1.5 * lcg(&mut s)).collect();
        let c: Vec<f64> = (0..n).map(|_| 0.5 + 1.5 * lcg(&mut s)).collect();
        let primal = simplex(&c, &a, &b, &vec![Ge; m], opts()).unwrap();
        // Dual als Minimierung von −bᵀy mit Aᵀ (n × m) ≤ c
        let at: Vec<f64> = (0..n)
            .flat_map(|j| (0..m).map(|i| (i, j)).collect::<Vec<_>>())
            .map(|(i, j)| a[i * n + j])
            .collect();
        let neg_b: Vec<f64> = b.iter().map(|v| -v).collect();
        let dual = simplex(&neg_b, &at, &c, &vec![Le; n], opts()).unwrap();
        assert_eq!(primal.status, LpStatus::Optimal);
        assert_eq!(dual.status, LpStatus::Optimal);
        assert!(
            (primal.objective + dual.objective).abs() < 1e-7,
            "{m}×{n}: {} gegen {}",
            primal.objective,
            -dual.objective
        );
    }
}

#[test]
fn errors_and_iteration_limit() {
    assert_eq!(
        simplex::<f64>(&[], &[], &[], &[], opts()),
        Err(OptError::InvalidInput)
    );
    assert_eq!(
        simplex(&[1.0], &[1.0, 2.0], &[1.0], &[Le], opts()),
        Err(OptError::InvalidInput)
    );
    assert_eq!(
        simplex(&[1.0], &[1.0], &[1.0], &[Le, Le], opts()),
        Err(OptError::InvalidInput)
    );
    assert_eq!(
        simplex(&[f64::NAN], &[1.0], &[1.0], &[Le], opts()),
        Err(OptError::NotFinite)
    );
    assert_eq!(
        simplex(&[1.0], &[f64::INFINITY], &[1.0], &[Le], opts()),
        Err(OptError::NotFinite)
    );
    assert!(OptError::InvalidInput.to_string().contains("Größen"));
    let none = SimplexOptions {
        max_iter: 0,
        ..opts()
    };
    let r = simplex(&[-3.0, -2.0], &[1.0, 1.0], &[4.0], &[Le], none).unwrap();
    assert_eq!(r.status, LpStatus::IterationLimit);
}

#[test]
fn zero_constraints_and_trivial_cases() {
    // keine Restriktionen, c ≥ 0: Optimum im Ursprung
    let sol = simplex(&[1.0, 2.0], &[], &[], &[], opts()).unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    assert_eq!(sol.x, vec![0.0, 0.0]);
    // keine Restriktionen, c < 0: unbeschränkt
    let unb = simplex(&[-1.0], &[], &[], &[], opts()).unwrap();
    assert_eq!(unb.status, LpStatus::Unbounded);
}

#[test]
fn works_for_f32() {
    let sol = simplex(
        &[-3.0f32, -2.0],
        &[1.0, 1.0, 1.0, 3.0, 1.0, 0.0],
        &[4.0, 6.0, 3.0],
        &[Le, Le, Le],
        SimplexOptions::default(),
    )
    .unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    assert!((sol.objective + 11.0).abs() < 1e-3);
}
