//! Tests für das Simplex-Verfahren.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm` und brauchen `alloc`.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::opt::{LpStatus, OptError, Relation, SimplexOptions, simplex, simplex_bounded};

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

// ---------------------------------------------------------------------------
// Variablenschranken
// ---------------------------------------------------------------------------

const INF: f64 = f64::INFINITY;

#[test]
fn box_bounds_with_negative_lower_bound() {
    // maximiere x + y mit −2 ≤ x ≤ 3, 0 ≤ y ≤ 4, x + y ≤ 5
    let sol = simplex_bounded(
        &[-1.0, -1.0],
        &[1.0, 1.0],
        &[5.0],
        &[Le],
        &[(-2.0, 3.0), (0.0, 4.0)],
        opts(),
    )
    .unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    close(sol.objective, -5.0);
    assert!(sol.x[0] >= -2.0 - 1e-9 && sol.x[0] <= 3.0 + 1e-9);
    assert!(sol.x[1] >= -1e-9 && sol.x[1] <= 4.0 + 1e-9);
    close(sol.x[0] + sol.x[1], 5.0);
}

#[test]
fn bounds_alone_determine_the_optimum() {
    // keine Restriktionen: minimiere x − y mit x ∈ [−2, 3], y ∈ [1, 7] -> x = −2, y = 7
    let sol = simplex_bounded(
        &[1.0, -1.0],
        &[],
        &[],
        &[],
        &[(-2.0, 3.0), (1.0, 7.0)],
        opts(),
    )
    .unwrap();
    assert_eq!(sol.status, LpStatus::Optimal);
    close(sol.x[0], -2.0);
    close(sol.x[1], 7.0);
    close(sol.objective, -9.0);
}

#[test]
fn upper_bound_only_and_free_variables() {
    // maximiere x mit x ≤ 4 (−∞ < x ≤ 4)
    let up = simplex_bounded(&[-1.0], &[], &[], &[], &[(-INF, 4.0)], opts()).unwrap();
    assert_eq!(up.status, LpStatus::Optimal);
    close(up.x[0], 4.0);
    // freie Variable: minimiere x unter x ≥ −5 (als Restriktion)
    let free = simplex_bounded(&[1.0], &[1.0], &[-5.0], &[Ge], &[(-INF, INF)], opts()).unwrap();
    assert_eq!(free.status, LpStatus::Optimal);
    close(free.x[0], -5.0);
    // freie Variable ohne untere Begrenzung: unbeschränkt
    let unb = simplex_bounded(&[1.0], &[], &[], &[], &[(-INF, INF)], opts()).unwrap();
    assert_eq!(unb.status, LpStatus::Unbounded);
    // zwei freie Variablen mit Gleichung: minimiere x + 2y unter x − y = 3, x ≥ 0 -> y = x − 3, Ziel 3x − 6, x = 0 -> (0, −3)
    let two = simplex_bounded(
        &[1.0, 2.0],
        &[1.0, -1.0],
        &[3.0],
        &[Eq],
        &[(0.0, INF), (-INF, INF)],
        opts(),
    )
    .unwrap();
    assert_eq!(two.status, LpStatus::Optimal);
    close(two.x[0], 0.0);
    close(two.x[1], -3.0);
    close(two.objective, -6.0);
}

#[test]
fn fixed_and_contradictory_bounds() {
    // x fest auf 2: minimiere y unter x + y ≥ 5 -> y = 3
    let fixed = simplex_bounded(
        &[0.0, 1.0],
        &[1.0, 1.0],
        &[5.0],
        &[Ge],
        &[(2.0, 2.0), (0.0, INF)],
        opts(),
    )
    .unwrap();
    assert_eq!(fixed.status, LpStatus::Optimal);
    close(fixed.x[0], 2.0);
    close(fixed.x[1], 3.0);
    // l > u
    let bad = simplex_bounded(&[1.0], &[], &[], &[], &[(3.0, 1.0)], opts()).unwrap();
    assert_eq!(bad.status, LpStatus::Infeasible);
    // Schranken im Widerspruch zur Restriktion: x ≤ 1, aber x ≥ 2
    let inf = simplex_bounded(&[1.0], &[1.0], &[2.0], &[Ge], &[(0.0, 1.0)], opts()).unwrap();
    assert_eq!(inf.status, LpStatus::Infeasible);
}

#[test]
fn bounds_agree_with_explicit_constraints_on_random_problems() {
    let mut s = 4242u64;
    for (m, n) in [(2, 3), (3, 4), (4, 4)] {
        let a: Vec<f64> = (0..m * n).map(|_| 2.0 * lcg(&mut s) - 0.5).collect();
        let b: Vec<f64> = (0..m).map(|_| 1.0 + 3.0 * lcg(&mut s)).collect();
        let c: Vec<f64> = (0..n).map(|_| lcg(&mut s) - 0.5).collect();
        let lo: Vec<f64> = (0..n).map(|_| -1.0 - lcg(&mut s)).collect();
        let hi: Vec<f64> = (0..n).map(|k| lo[k] + 1.0 + 2.0 * lcg(&mut s)).collect();
        let bounds: Vec<(f64, f64)> = lo.iter().copied().zip(hi.iter().copied()).collect();
        let bounded = simplex_bounded(&c, &a, &b, &vec![Le; m], &bounds, opts()).unwrap();

        // gleiches Problem mit Verschiebung x = lo + y und expliziten Zeilen y ≤ hi − lo
        let mut a2 = vec![0.0; (m + n) * n];
        let mut b2 = vec![0.0; m + n];
        for i in 0..m {
            b2[i] = b[i];
            for j in 0..n {
                a2[i * n + j] = a[i * n + j];
                b2[i] -= a[i * n + j] * lo[j];
            }
        }
        for j in 0..n {
            a2[(m + j) * n + j] = 1.0;
            b2[m + j] = hi[j] - lo[j];
        }
        let explicit = simplex(&c, &a2, &b2, &vec![Le; m + n], opts()).unwrap();
        assert_eq!(bounded.status, explicit.status, "{m}×{n}");
        if bounded.status == LpStatus::Optimal {
            let shift: f64 = c.iter().zip(&lo).map(|(ci, li)| ci * li).sum();
            close(bounded.objective, explicit.objective + shift);
        }
    }
}

#[test]
fn bounded_errors() {
    assert_eq!(
        simplex_bounded(&[1.0], &[], &[], &[], &[], opts()),
        Err(OptError::InvalidInput)
    );
    assert_eq!(
        simplex_bounded(&[1.0], &[], &[], &[], &[(0.0, 1.0), (0.0, 1.0)], opts()),
        Err(OptError::InvalidInput)
    );
    assert_eq!(
        simplex_bounded(&[1.0], &[], &[], &[], &[(INF, INF)], opts()),
        Err(OptError::InvalidInput)
    );
    assert_eq!(
        simplex_bounded(&[1.0], &[], &[], &[], &[(-INF, -INF)], opts()),
        Err(OptError::InvalidInput)
    );
    assert_eq!(
        simplex_bounded(&[1.0], &[], &[], &[], &[(f64::NAN, 1.0)], opts()),
        Err(OptError::NotFinite)
    );
}
