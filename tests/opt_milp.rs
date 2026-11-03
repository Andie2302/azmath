//! Tests für Branch-and-Bound (gemischt-ganzzahlige LPs).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm` und brauchen `alloc`.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::opt::{
    LpStatus, MilpOptions, MilpStatus, NodeSelection, OptError, Relation, SimplexOptions,
    branch_and_bound, simplex_bounded,
};

use Relation::{Eq, Ge, Le};

const INF: f64 = f64::INFINITY;

fn opts() -> MilpOptions<f64> {
    MilpOptions::default()
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-7, "{a} gegen {b}");
}

/// Deterministische Pseudozufallszahlen in [0, 1).
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

#[test]
fn textbook_integer_program() {
    // maximiere 5x + 4y, 6x + 4y ≤ 24, x + 2y ≤ 6, ganzzahlig -> 20 bei (4, 0); LP-Relaxation wäre 21
    let (c, a, b) = ([-5.0, -4.0], [6.0, 4.0, 1.0, 2.0], [24.0, 6.0]);
    let bounds = [(0.0, INF); 2];
    let lp = simplex_bounded(&c, &a, &b, &[Le; 2], &bounds, SimplexOptions::default()).unwrap();
    close(lp.objective, -21.0);
    let sol = branch_and_bound(&c, &a, &b, &[Le; 2], &bounds, &[true, true], opts()).unwrap();
    assert_eq!(sol.status, MilpStatus::Optimal);
    close(sol.objective, -20.0);
    close(sol.best_bound, -20.0);
    assert!(sol.x.iter().all(|v| (v - v.round()).abs() < 1e-9));
    assert!(sol.nodes > 1);
}

#[test]
fn zero_one_knapsack() {
    // Gewichte [12, 2, 1, 1, 4], Werte [4, 2, 1, 2, 10], Kapazität 15 -> beste Wahl: Gegenstände 2..5 (Wert 15)
    let weights = [12.0, 2.0, 1.0, 1.0, 4.0];
    let values = [4.0, 2.0, 1.0, 2.0, 10.0];
    let c: Vec<f64> = values.iter().map(|v| -v).collect();
    let sol = branch_and_bound(
        &c,
        &weights,
        &[15.0],
        &[Le],
        &[(0.0, 1.0); 5],
        &[true; 5],
        opts(),
    )
    .unwrap();
    assert_eq!(sol.status, MilpStatus::Optimal);
    close(sol.objective, -15.0);
    let w: f64 = sol.x.iter().zip(&weights).map(|(x, w)| x * w).sum();
    assert!(w <= 15.0 + 1e-9);
}

#[test]
fn assignment_problem_matches_enumeration_of_all_permutations() {
    let cost = [[9.0, 2.0, 7.0], [3.0, 6.0, 1.0], [5.0, 8.0, 4.0]];
    let c: Vec<f64> = cost.iter().flatten().copied().collect();
    // x_ij binär; jede Zeile und jede Spalte genau einmal
    let mut a = vec![0.0; 6 * 9];
    for i in 0..3 {
        for j in 0..3 {
            a[i * 9 + i * 3 + j] = 1.0; // Zeile i
            a[(3 + j) * 9 + i * 3 + j] = 1.0; // Spalte j
        }
    }
    let sol = branch_and_bound(
        &c,
        &a,
        &[1.0; 6],
        &[Eq; 6],
        &[(0.0, 1.0); 9],
        &[true; 9],
        opts(),
    )
    .unwrap();
    assert_eq!(sol.status, MilpStatus::Optimal);
    let perms = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let best = perms
        .iter()
        .map(|p| (0..3).map(|i| cost[i][p[i]]).sum::<f64>())
        .fold(INF, f64::min);
    close(sol.objective, best);
}

#[test]
fn mixed_problem_and_pure_continuous_case() {
    // minimiere −x − y mit x + y ≤ 3,5, x ganzzahlig, y stetig -> Optimum −3,5 (x beliebig ganz, y füllt auf)
    let mixed = branch_and_bound(
        &[-1.0, -1.0],
        &[1.0, 1.0],
        &[3.5],
        &[Le],
        &[(0.0, INF); 2],
        &[true, false],
        opts(),
    )
    .unwrap();
    assert_eq!(mixed.status, MilpStatus::Optimal);
    close(mixed.objective, -3.5);
    assert!((mixed.x[0] - mixed.x[0].round()).abs() < 1e-9);
    // keine ganzzahligen Variablen: identisch zum LP, ein einziger Knoten
    let none = branch_and_bound(
        &[-1.0, -1.0],
        &[1.0, 1.0],
        &[3.5],
        &[Le],
        &[(0.0, INF); 2],
        &[false, false],
        opts(),
    )
    .unwrap();
    close(none.objective, -3.5);
    assert_eq!(none.nodes, 1);
}

#[test]
fn random_binary_programs_match_complete_enumeration() {
    let mut s = 2024u64;
    for case in 0..6 {
        let (n, m) = (8, 3);
        let c: Vec<f64> = (0..n).map(|_| 6.0 * lcg(&mut s) - 3.0).collect();
        let a: Vec<f64> = (0..m * n).map(|_| 5.0 * lcg(&mut s)).collect();
        let b: Vec<f64> = (0..m).map(|_| 6.0 + 4.0 * lcg(&mut s)).collect();
        // vollständige Enumeration
        let mut best = INF;
        for mask in 0u32..(1 << n) {
            let x: Vec<f64> = (0..n).map(|k| ((mask >> k) & 1) as f64).collect();
            let feasible =
                (0..m).all(|i| (0..n).map(|k| a[i * n + k] * x[k]).sum::<f64>() <= b[i] + 1e-9);
            if feasible {
                best = best.min(c.iter().zip(&x).map(|(ci, xi)| ci * xi).sum());
            }
        }
        for sel in [NodeSelection::BestFirst, NodeSelection::DepthFirst] {
            let o = MilpOptions {
                node_selection: sel,
                ..opts()
            };
            let sol = branch_and_bound(
                &c,
                &a,
                &b,
                &vec![Le; m],
                &vec![(0.0, 1.0); n],
                &vec![true; n],
                o,
            )
            .unwrap();
            assert_eq!(sol.status, MilpStatus::Optimal, "Fall {case}");
            close(sol.objective, best);
        }
    }
}

#[test]
fn infeasible_and_unbounded_cases() {
    // 2x = 1 mit x ganzzahlig: Relaxation zulässig (x = ½), ganzzahlig unzulässig
    let sol = branch_and_bound(
        &[1.0],
        &[2.0],
        &[1.0],
        &[Eq],
        &[(0.0, 10.0)],
        &[true],
        opts(),
    )
    .unwrap();
    assert_eq!(sol.status, MilpStatus::Infeasible);
    assert!(sol.objective.is_infinite());
    // Relaxation unzulässig: x ≥ 2 und x ≤ 1
    let inf = branch_and_bound(
        &[1.0],
        &[1.0, 1.0],
        &[2.0, 1.0],
        &[Ge, Le],
        &[(0.0, INF)],
        &[true],
        opts(),
    )
    .unwrap();
    assert_eq!(inf.status, MilpStatus::Infeasible);
    // unbeschränkt
    let unb = branch_and_bound(&[-1.0], &[], &[], &[], &[(0.0, INF)], &[true], opts()).unwrap();
    assert_eq!(unb.status, MilpStatus::Unbounded);
}

#[test]
fn node_limit_reports_a_valid_bound() {
    let weights = [12.0, 2.0, 1.0, 1.0, 4.0];
    let c: Vec<f64> = [4.0, 2.0, 1.0, 2.0, 10.0].iter().map(|v| -v).collect();
    let o = MilpOptions {
        max_nodes: 1,
        ..opts()
    };
    let sol = branch_and_bound(
        &c,
        &weights,
        &[15.0],
        &[Le],
        &[(0.0, 1.0); 5],
        &[true; 5],
        o,
    )
    .unwrap();
    assert_eq!(sol.status, MilpStatus::NodeLimit);
    assert_eq!(sol.nodes, 1);
    assert!(
        sol.best_bound <= -15.0 + 1e-9,
        "Schranke {}",
        sol.best_bound
    );
}

#[test]
fn integer_bounds_are_tightened() {
    // x ganzzahlig in [0,5; 2,5] -> {1, 2}; minimiere −x -> x = 2
    let sol = branch_and_bound(&[-1.0], &[], &[], &[], &[(0.5, 2.5)], &[true], opts()).unwrap();
    assert_eq!(sol.status, MilpStatus::Optimal);
    close(sol.x[0], 2.0);
}

#[test]
fn errors() {
    assert_eq!(
        branch_and_bound(
            &[1.0, 1.0],
            &[],
            &[],
            &[],
            &[(0.0, 1.0); 2],
            &[true],
            opts()
        ),
        Err(OptError::InvalidInput)
    );
    assert_eq!(
        branch_and_bound(
            &[1.0, 1.0],
            &[],
            &[],
            &[],
            &[(0.0, 1.0)],
            &[true, true],
            opts()
        ),
        Err(OptError::InvalidInput)
    );
    assert_eq!(
        branch_and_bound(&[f64::NAN], &[], &[], &[], &[(0.0, 1.0)], &[true], opts()),
        Err(OptError::NotFinite)
    );
    // auch die Teilprobleme melden ungültige Eingaben weiter (Zeilen passen nicht zu b)
    assert_eq!(
        branch_and_bound(
            &[1.0],
            &[1.0, 2.0],
            &[1.0],
            &[Le],
            &[(0.0, 1.0)],
            &[true],
            opts()
        ),
        Err(OptError::InvalidInput)
    );
}

#[test]
fn works_for_f32() {
    let sol = branch_and_bound(
        &[-5.0f32, -4.0],
        &[6.0, 4.0, 1.0, 2.0],
        &[24.0, 6.0],
        &[Le; 2],
        &[(0.0, f32::INFINITY); 2],
        &[true, true],
        MilpOptions::default(),
    )
    .unwrap();
    assert_eq!(sol.status, MilpStatus::Optimal);
    assert!((sol.objective + 20.0).abs() < 1e-3);
    let _ = LpStatus::Optimal;
}
