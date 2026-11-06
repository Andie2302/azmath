//! Eigenschaftstests mit festem Seed für `sparse`: Formatumwandlung, Umordnungen,
//! direkte Löser und Refaktorisierung bei gleichem Muster.
#![cfg(all(feature = "alloc", any(feature = "std", feature = "libm")))]

use azmath::linalg::LinalgError;
use azmath::sparse::{
    CooMatrix, CsrMatrix, Ordering, SparseCholesky, SparseLdl, SparseLu, minimum_degree_ordering,
    permute_symmetric, rcm_ordering,
};

const ORDERINGS: [Ordering; 3] = [
    Ordering::Natural,
    Ordering::ReverseCuthillMcKee,
    Ordering::MinimumDegree,
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (((self.next() + 1.0) * 0.5 * n as f64) as usize).min(n - 1)
    }
}

/// Zufällige Tripletliste (mit Duplikaten) samt dichter Referenzsumme.
fn random_coo(rng: &mut Rng, m: usize, n: usize, count: usize) -> (CooMatrix<f64>, Vec<f64>) {
    let mut coo = CooMatrix::new(m, n);
    let mut dense = vec![0.0; m * n];
    for _ in 0..count {
        let (i, j, v) = (rng.below(m), rng.below(n), rng.next());
        coo.push(i, j, v).unwrap();
        dense[i * n + j] += v;
    }
    (coo, dense)
}

/// Strukturell symmetrische, streng diagonaldominante Matrix; `spd` macht sie symmetrisch positiv definit.
fn dominant(rng: &mut Rng, n: usize, spd: bool) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    let mut rowsum = vec![0.0f64; n];
    for _ in 0..3 * n {
        let (i, j) = (rng.below(n), rng.below(n));
        if i == j {
            continue;
        }
        let v = rng.next();
        let w = if spd { v } else { rng.next() };
        coo.push(i, j, v).unwrap();
        coo.push(j, i, w).unwrap();
        rowsum[i] += v.abs();
        rowsum[j] += w.abs();
    }
    for (i, s) in rowsum.iter().enumerate() {
        coo.push(i, i, s + 1.0 + rng.next().abs()).unwrap();
    }
    CsrMatrix::from_coo(&coo)
}

fn residual_inf(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    (0..a.rows())
        .map(|i| (a.row(i).map(|(j, v)| v * x[j]).sum::<f64>() - b[i]).abs())
        .fold(0.0, f64::max)
}

fn rhs(rng: &mut Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.next()).collect()
}

#[test]
fn coo_to_csr_sums_duplicates_and_matches_dense() {
    let mut rng = Rng(11);
    for case in 0..200 {
        let (m, n) = (1 + rng.below(12), 1 + rng.below(12));
        let count = rng.below(60);
        let (coo, dense) = random_coo(&mut rng, m, n, count);
        let a = CsrMatrix::from_coo(&coo);
        a.validate().unwrap();
        let d = a.to_dense();
        for i in 0..m {
            for j in 0..n {
                assert!((d[(i, j)] - dense[i * n + j]).abs() < 1e-14, "case {case}");
            }
        }
        // Spalten je Zeile streng aufsteigend (keine Duplikate)
        for i in 0..m {
            let cols: Vec<_> = a.row(i).map(|(j, _)| j).collect();
            assert!(cols.windows(2).all(|w| w[0] < w[1]), "case {case}");
        }
        // matvec / matvec_transposed gegen die dichte Matrix
        let x = rhs(&mut rng, n);
        let mut y = vec![0.0; m];
        a.matvec(&x, &mut y).unwrap();
        for i in 0..m {
            let e: f64 = (0..n).map(|j| dense[i * n + j] * x[j]).sum();
            assert!((y[i] - e).abs() < 1e-12, "case {case} matvec");
        }
        let xt = rhs(&mut rng, m);
        let mut yt = vec![0.0; n];
        a.matvec_transposed(&xt, &mut yt).unwrap();
        for j in 0..n {
            let e: f64 = (0..m).map(|i| dense[i * n + j] * xt[i]).sum();
            assert!((yt[j] - e).abs() < 1e-12, "case {case} matvec_t");
        }
        // zweifache Transposition ist die Identität, Summe mit sich selbst verdoppelt
        let tt = a.transpose().transpose();
        assert_eq!(
            (tt.row_ptr(), tt.col_idx(), tt.values()),
            (a.row_ptr(), a.col_idx(), a.values()),
            "case {case} transpose"
        );
        let s = a.add(&a).unwrap().to_dense();
        for i in 0..m {
            for j in 0..n {
                assert!((s[(i, j)] - 2.0 * d[(i, j)]).abs() < 1e-14);
            }
        }
    }
}

fn is_permutation(p: &[usize], n: usize) -> bool {
    let mut seen = vec![false; n];
    p.len() == n
        && p.iter()
            .all(|&i| i < n && !std::mem::replace(&mut seen[i], true))
}

#[test]
fn orderings_are_permutations_and_symmetric_permutation_preserves_entries() {
    let mut rng = Rng(12);
    for case in 0..100 {
        let n = 1 + rng.below(25);
        let a = dominant(&mut rng, n, true);
        for perm in [rcm_ordering(&a), minimum_degree_ordering(&a)] {
            assert!(is_permutation(&perm, n), "case {case}: keine Permutation");
            let b = permute_symmetric(&a, &perm);
            b.validate().unwrap();
            assert_eq!(b.nnz(), a.nnz());
            // perm[neu] = alt  =>  b[neu_i, neu_j] = a[perm[i], perm[j]]
            for i in 0..n {
                for j in 0..n {
                    assert_eq!(b.get(i, j), a.get(perm[i], perm[j]), "case {case}");
                }
            }
        }
    }
}

#[test]
fn direct_solvers_have_small_residual_for_every_ordering() {
    let mut rng = Rng(13);
    for case in 0..120 {
        let n = 1 + rng.below(30);
        let spd = dominant(&mut rng, n, true);
        let general = dominant(&mut rng, n, false);
        let b = rhs(&mut rng, n);
        for ordering in ORDERINGS {
            let x = SparseCholesky::new(&spd, ordering)
                .unwrap()
                .solve(&b)
                .unwrap();
            assert!(
                residual_inf(&spd, &x, &b) < 1e-10,
                "case {case} chol {ordering:?}"
            );
            let x = SparseLdl::new(&spd, ordering).unwrap().solve(&b).unwrap();
            assert!(
                residual_inf(&spd, &x, &b) < 1e-10,
                "case {case} ldl {ordering:?}"
            );
            let lu = SparseLu::new(&general, ordering).unwrap();
            let x = lu.solve(&b).unwrap();
            assert!(
                residual_inf(&general, &x, &b) < 1e-10,
                "case {case} lu {ordering:?}"
            );
            // Aᵀ·x = b
            let at = general.transpose();
            let xt = lu.solve_transposed(&b).unwrap();
            assert!(
                residual_inf(&at, &xt, &b) < 1e-10,
                "case {case} lu^T {ordering:?}"
            );
        }
    }
}

#[test]
fn ldl_inertia_matches_sign_pattern_of_diagonal_matrices() {
    let mut rng = Rng(14);
    for case in 0..100 {
        let n = 1 + rng.below(20);
        let mut coo = CooMatrix::new(n, n);
        let (mut pos, mut neg) = (0, 0);
        for i in 0..n {
            let v = rng.next();
            let v = if v.abs() < 0.1 { 1.0 } else { v };
            if v > 0.0 {
                pos += 1
            } else {
                neg += 1
            }
            coo.push(i, i, v).unwrap();
        }
        let a = CsrMatrix::from_coo(&coo);
        for ordering in ORDERINGS {
            let f = SparseLdl::new(&a, ordering).unwrap();
            assert_eq!(f.inertia(), (pos, neg), "case {case} {ordering:?}");
        }
    }
}

#[test]
fn refactor_with_same_pattern_equals_fresh_factorization() {
    let mut rng = Rng(15);
    for case in 0..80 {
        let n = 2 + rng.below(25);
        let mut spd = dominant(&mut rng, n, true);
        let mut general = dominant(&mut rng, n, false);
        for ordering in ORDERINGS {
            let mut ldl = SparseLdl::new(&spd, ordering).unwrap();
            let mut chol = SparseCholesky::new(&spd, ordering).unwrap();
            let mut lu = SparseLu::new(&general, ordering).unwrap();
            for round in 0..3 {
                // gleiche Struktur, neue Werte (Diagonale bleibt dominant)
                for v in spd.values_mut() {
                    *v *= 1.0 + 0.25 * rng.next().abs();
                }
                let sym = spd.transpose();
                // Symmetrie wiederherstellen: (A + Aᵀ)/2 hätte gleiches Muster
                let mut half = spd.add(&sym).unwrap();
                for v in half.values_mut() {
                    *v *= 0.5;
                }
                spd = half;
                for v in general.values_mut() {
                    *v *= 1.0 + 0.25 * rng.next().abs();
                }
                ldl.refactor(&spd).unwrap();
                chol.refactor(&spd).unwrap();
                lu.refactor(&general).unwrap();
                let b = rhs(&mut rng, n);
                let tag = format!("case {case} round {round} {ordering:?}");
                let fresh = SparseLdl::new(&spd, ordering).unwrap().solve(&b).unwrap();
                let x = ldl.solve(&b).unwrap();
                assert!(
                    x.iter().zip(&fresh).all(|(p, q)| (p - q).abs() < 1e-9),
                    "{tag} ldl"
                );
                assert!(
                    residual_inf(&spd, &chol.solve(&b).unwrap(), &b) < 1e-9,
                    "{tag} chol"
                );
                assert!(
                    residual_inf(&general, &lu.solve(&b).unwrap(), &b) < 1e-9,
                    "{tag} lu"
                );
            }
        }
    }
}

#[test]
fn failed_numeric_refactor_invalidates_and_next_good_refactor_recovers() {
    let mut rng = Rng(16);
    for case in 0..60 {
        let n = 2 + rng.below(15);
        let good = dominant(&mut rng, n, true);
        let mut bad = good.clone();
        // alle Werte null: gleiches Muster, aber singulär
        for v in bad.values_mut() {
            *v = 0.0;
        }
        for ordering in ORDERINGS {
            let mut ldl = SparseLdl::new(&good, ordering).unwrap();
            let b = rhs(&mut rng, n);
            let before = ldl.solve(&b).unwrap();
            assert!(ldl.refactor(&bad).is_err(), "case {case}");
            assert!(!ldl.is_valid());
            assert_eq!(ldl.solve(&b), Err(LinalgError::InvalidInput), "case {case}");
            ldl.refactor(&good).unwrap();
            assert!(ldl.is_valid());
            let after = ldl.solve(&b).unwrap();
            assert!(
                before
                    .iter()
                    .zip(&after)
                    .all(|(p, q)| (p - q).abs() < 1e-12),
                "case {case} {ordering:?}: nach Erholung andere Lösung"
            );
        }
    }
}

#[test]
fn non_finite_and_mismatched_inputs_are_errors_not_panics() {
    let mut rng = Rng(17);
    for _ in 0..50 {
        let n = 2 + rng.below(10);
        let a = dominant(&mut rng, n, true);
        let mut nan = a.clone();
        nan.values_mut()[0] = f64::NAN;
        for ordering in ORDERINGS {
            assert!(SparseLdl::new(&nan, ordering).is_err());
            assert!(SparseLu::new(&nan, ordering).is_err());
            let f = SparseLdl::new(&a, ordering).unwrap();
            assert!(f.solve(&vec![0.0; n + 1]).is_err());
            let mut b = rhs(&mut rng, n);
            b[0] = f64::INFINITY;
            // darf Fehler liefern oder nicht endliche Werte — aber nie paniken
            let _ = f.solve(&b);
        }
    }
}
