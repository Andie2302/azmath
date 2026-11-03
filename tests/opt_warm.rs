//! Tests für `LpWarmStart` (Dual-Simplex) und `branch_and_cut`.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]
#![allow(
    clippy::needless_range_loop,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::excessive_precision
)]

use azmath::opt::{
    CutOptions, LpStatus, LpWarmStart, MilpOptions, MilpStatus, NodeSelection, Relation,
    SimplexOptions, branch_and_bound, branch_and_cut, simplex,
};

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

#[test]
fn warm_start_matches_fresh_solve() {
    let c = [-1.0f64, -1.0];
    let a = [1.0, 1.0, 1.0, 0.0];
    let b = [4.0, 3.0];
    let mut lp =
        LpWarmStart::new(&c, &a, &b, &[Relation::Le; 2], SimplexOptions::default()).unwrap();
    assert!((lp.solution().objective + 4.0).abs() < 1e-9);
    assert_eq!(
        lp.add_constraint(&[0.0, 1.0], Relation::Le, 0.5).unwrap(),
        LpStatus::Optimal
    );
    let fresh = simplex(
        &c,
        &[1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        &[4.0, 3.0, 0.5],
        &[Relation::Le; 3],
        SimplexOptions::default(),
    )
    .unwrap();
    assert!((lp.solution().objective - fresh.objective).abs() < 1e-9);
    // Gleichung hinzufügen: x − y = 1
    assert_eq!(
        lp.add_constraint(&[1.0, -1.0], Relation::Eq, 1.0).unwrap(),
        LpStatus::Optimal
    );
    let s = lp.solution();
    assert!((s.x[0] - s.x[1] - 1.0).abs() < 1e-9);
    assert!(s.x[1] <= 0.5 + 1e-9 && s.x[0] <= 3.0 + 1e-9);
    // unzulässig
    assert_eq!(
        lp.add_constraint(&[0.0, 1.0], Relation::Ge, 2.0).unwrap(),
        LpStatus::Infeasible
    );
    assert_eq!(lp.status(), LpStatus::Infeasible);
    assert!(lp.add_constraint(&[1.0], Relation::Le, 1.0).is_err());
    assert!(
        lp.add_constraint(&[f64::NAN, 1.0], Relation::Le, 1.0)
            .is_err()
    );
}

#[test]
fn warm_start_random_lps_agree_with_simplex() {
    let mut rng = Lcg(5);
    for _ in 0..20 {
        let n = 4;
        let m = 3;
        let c: Vec<f64> = (0..n).map(|_| -(rng.next() + 0.1)).collect();
        let mut a: Vec<f64> = (0..m * n).map(|_| rng.next()).collect();
        let b: Vec<f64> = (0..m).map(|_| 1.0 + 4.0 * rng.next()).collect();
        // Obergrenzen als Zeilen
        for j in 0..n {
            let mut row = vec![0.0; n];
            row[j] = 1.0;
            a.extend(row);
        }
        let mut bb = b.clone();
        bb.extend(vec![3.0; n]);
        let rel = vec![Relation::Le; m + n];
        let mut lp = LpWarmStart::new(&c, &a, &bb, &rel, SimplexOptions::default()).unwrap();
        // eine zusätzliche Zeile
        let extra: Vec<f64> = (0..n).map(|_| rng.next()).collect();
        let rhs = 0.5 + rng.next();
        let st = lp.add_constraint(&extra, Relation::Le, rhs).unwrap();
        let mut a2 = a.clone();
        a2.extend(&extra);
        let mut b2 = bb.clone();
        b2.push(rhs);
        let mut rel2 = rel.clone();
        rel2.push(Relation::Le);
        let fresh = simplex(&c, &a2, &b2, &rel2, SimplexOptions::default()).unwrap();
        assert_eq!(st, fresh.status);
        if st == LpStatus::Optimal {
            assert!((lp.solution().objective - fresh.objective).abs() < 1e-7);
        }
    }
}

#[test]
fn branch_and_cut_small_problems() {
    let sol = branch_and_cut(
        &[-5.0f64, -4.0],
        &[6.0, 4.0, 1.0, 2.0],
        &[24.0, 6.0],
        &[Relation::Le; 2],
        &[true, true],
        CutOptions::default(),
    )
    .unwrap();
    assert_eq!(sol.status, MilpStatus::Optimal);
    assert!((sol.objective + 20.0).abs() < 1e-9);
    assert!(sol.x.iter().all(|v| (v - v.round()).abs() < 1e-9));
    // unzulässig: 2x = 1 mit x ganzzahlig
    let inf = branch_and_cut(
        &[1.0f64],
        &[2.0],
        &[1.0],
        &[Relation::Eq],
        &[true],
        CutOptions::default(),
    )
    .unwrap();
    assert_eq!(inf.status, MilpStatus::Infeasible);
    // unbeschränkt
    let unb = branch_and_cut(
        &[-1.0f64],
        &[1.0],
        &[0.0],
        &[Relation::Ge],
        &[true],
        CutOptions::default(),
    )
    .unwrap();
    assert_eq!(unb.status, MilpStatus::Unbounded);
    assert!(
        branch_and_cut(
            &[1.0f64, 1.0],
            &[1.0, 1.0],
            &[1.0],
            &[Relation::Le],
            &[true],
            CutOptions::default()
        )
        .is_err()
    );
}

#[test]
fn branch_and_cut_agrees_with_branch_and_bound_on_knapsacks() {
    let mut rng = Lcg(2024);
    for case in 0..15 {
        let n = 6 + case % 3;
        let weights: Vec<f64> = (0..n).map(|_| (2.0 + 8.0 * rng.next()).floor()).collect();
        let values: Vec<f64> = (0..n).map(|_| (3.0 + 10.0 * rng.next()).floor()).collect();
        let cap = (weights.iter().sum::<f64>() * 0.5).floor();
        let c: Vec<f64> = values.iter().map(|v| -v).collect();
        let mut a = weights.clone();
        let mut b = vec![cap];
        let mut rel = vec![Relation::Le];
        for j in 0..n {
            let mut row = vec![0.0; n];
            row[j] = 1.0;
            a.extend(row);
            b.push(1.0);
            rel.push(Relation::Le);
        }
        let integer = vec![true; n];
        let bounds = vec![(0.0, f64::INFINITY); n];
        let reference =
            branch_and_bound(&c, &a, &b, &rel, &bounds, &integer, MilpOptions::default()).unwrap();
        for sel in [NodeSelection::BestFirst, NodeSelection::DepthFirst] {
            let mut opts = CutOptions::default();
            opts.milp.node_selection = sel;
            let cut = branch_and_cut(&c, &a, &b, &rel, &integer, opts).unwrap();
            assert_eq!(cut.status, MilpStatus::Optimal, "case {case}");
            assert!(
                (cut.objective - reference.objective).abs() < 1e-6,
                "case {case}: {} vs {}",
                cut.objective,
                reference.objective
            );
        }
        // ohne Schnitte (reines Warmstart-B&B)
        let mut opts = CutOptions::default();
        opts.cut_rounds = 0;
        let plain = branch_and_cut(&c, &a, &b, &rel, &integer, opts).unwrap();
        assert!((plain.objective - reference.objective).abs() < 1e-6);
    }
}
