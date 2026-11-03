//! Tests für `SparseLu` und die Minimum-Degree-Umordnung.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::linalg::LinalgError;
use azmath::scalar::ScalarLu;
use azmath::sparse::{
    CooMatrix, CsrMatrix, Ordering, SparseCholesky, SparseLdl, SparseLu, minimum_degree_ordering,
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

fn residual(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let mut y = vec![0.0; b.len()];
    a.matvec(x, &mut y).unwrap();
    y.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max)
}

fn transpose(a: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let n = a.rows();
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        for (j, v) in a.row(i) {
            coo.push(j, i, v).unwrap();
        }
    }
    CsrMatrix::from_coo(&coo)
}

const ALL: [Ordering; 3] = [
    Ordering::Natural,
    Ordering::ReverseCuthillMcKee,
    Ordering::MinimumDegree,
];

fn random_nonsymmetric(n: usize, seed: u64) -> CsrMatrix<f64> {
    let mut rng = Lcg(seed);
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        for j in 0..n {
            if i != j && rng.next() < 0.15 {
                coo.push(i, j, rng.next() - 0.5).unwrap();
            }
        }
        // Diagonale teils klein oder fehlend: erzwingt Pivotisierung
        if i % 3 != 0 {
            coo.push(i, i, 1.0 + rng.next()).unwrap();
        }
    }
    // Permutationsanteil garantiert Regularität
    for i in 0..n {
        coo.push(i, (i + 1) % n, 2.0 + rng.next()).unwrap();
    }
    CsrMatrix::from_coo(&coo)
}

#[test]
fn needs_pivoting() {
    let mut coo = CooMatrix::new(2, 2);
    coo.push(0, 1, 2.0f64).unwrap();
    coo.push(1, 0, 1.0).unwrap();
    coo.push(1, 1, 1.0).unwrap();
    let a = CsrMatrix::from_coo(&coo);
    for ord in ALL {
        let lu = SparseLu::new(&a, ord).unwrap();
        let x = lu.solve(&[4.0, 3.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-14 && (x[1] - 2.0).abs() < 1e-14);
    }
}

#[test]
fn random_nonsymmetric_matrices() {
    for (n, seed) in [(10usize, 1u64), (25, 2), (60, 3)] {
        let a = random_nonsymmetric(n, seed);
        let at = transpose(&a);
        let mut rng = Lcg(seed + 100);
        let b: Vec<f64> = (0..n).map(|_| rng.next() - 0.5).collect();
        for ord in ALL {
            let lu = SparseLu::new(&a, ord).unwrap();
            let x = lu.solve(&b).unwrap();
            assert!(residual(&a, &x, &b) < 1e-9, "n = {n}, {ord:?}");
            let xt = lu.solve_transposed(&b).unwrap();
            assert!(
                residual(&at, &xt, &b) < 1e-9,
                "transposed, n = {n}, {ord:?}"
            );
        }
    }
}

#[test]
fn determinant_matches_dense() {
    let n = 12;
    let a = random_nonsymmetric(n, 7);
    let dense = ScalarLu::new(n, a.to_dense().as_slice()).unwrap();
    let det = dense.determinant();
    for ord in ALL {
        let lu = SparseLu::new(&a, ord).unwrap();
        assert!(
            (lu.log_abs_determinant() - det.abs().ln()).abs() < 1e-9,
            "{ord:?}"
        );
        assert_eq!(lu.determinant_sign(), det.signum(), "{ord:?}");
    }
}

#[test]
fn convection_diffusion_grid() {
    let m = 12;
    let n = m * m;
    let mut coo = CooMatrix::new(n, n);
    let eps = 0.05;
    for i in 0..m {
        for j in 0..m {
            let p = i * m + j;
            coo.push(p, p, 4.0 * eps + 1.0).unwrap();
            if i > 0 {
                coo.push(p, p - m, -eps - 1.0).unwrap();
            }
            if i + 1 < m {
                coo.push(p, p + m, -eps).unwrap();
            }
            if j > 0 {
                coo.push(p, p - 1, -eps - 0.5).unwrap();
            }
            if j + 1 < m {
                coo.push(p, p + 1, -eps).unwrap();
            }
        }
    }
    let a = CsrMatrix::from_coo(&coo);
    let b: Vec<f64> = (0..n).map(|i| 1.0 + (i % 5) as f64).collect();
    for ord in ALL {
        let lu = SparseLu::new(&a, ord).unwrap();
        assert!(residual(&a, &lu.solve(&b).unwrap(), &b) < 1e-9, "{ord:?}");
        assert!(lu.nnz_l() > 0 && lu.nnz_u() >= n);
    }
}

#[test]
fn errors() {
    // singulär: Nullspalte
    let mut coo = CooMatrix::new(3, 3);
    coo.push(0, 0, 1.0f64).unwrap();
    coo.push(1, 0, 1.0).unwrap();
    coo.push(2, 2, 1.0).unwrap();
    assert!(matches!(
        SparseLu::new(&CsrMatrix::from_coo(&coo), Ordering::Natural),
        Err(LinalgError::Singular)
    ));
    let r = CsrMatrix::from_coo(&CooMatrix::<f64>::new(2, 3));
    assert!(matches!(
        SparseLu::new(&r, Ordering::Natural),
        Err(LinalgError::DimensionMismatch)
    ));
    let mut coo = CooMatrix::new(1, 1);
    coo.push(0, 0, f64::INFINITY).unwrap();
    assert!(matches!(
        SparseLu::new(&CsrMatrix::from_coo(&coo), Ordering::Natural),
        Err(LinalgError::NotFinite)
    ));
    let mut coo = CooMatrix::new(2, 2);
    coo.push(0, 0, 1.0f64).unwrap();
    coo.push(1, 1, 1.0).unwrap();
    let lu = SparseLu::new(&CsrMatrix::from_coo(&coo), Ordering::Natural).unwrap();
    assert!(lu.solve(&[1.0]).is_err() && lu.solve_transposed(&[1.0]).is_err());
}

fn scrambled_grid(m: usize) -> CsrMatrix<f64> {
    let n = m * m;
    let mut map: Vec<usize> = (0..n).collect();
    let mut rng = Lcg(77);
    for i in (1..n).rev() {
        let j = (rng.next() * (i + 1) as f64) as usize % (i + 1);
        map.swap(i, j);
    }
    let mut coo = CooMatrix::new(n, n);
    for i in 0..m {
        for j in 0..m {
            let p = map[i * m + j];
            coo.push(p, p, 4.0).unwrap();
            if i + 1 < m {
                let q = map[(i + 1) * m + j];
                coo.push(p, q, -1.0).unwrap();
                coo.push(q, p, -1.0).unwrap();
            }
            if j + 1 < m {
                let q = map[i * m + j + 1];
                coo.push(p, q, -1.0).unwrap();
                coo.push(q, p, -1.0).unwrap();
            }
        }
    }
    CsrMatrix::from_coo(&coo)
}

#[test]
fn minimum_degree_reduces_fill() {
    let a = scrambled_grid(12);
    let n = a.rows();
    let perm = minimum_degree_ordering(&a);
    let mut seen = vec![false; n];
    for &p in &perm {
        assert!(p < n && !seen[p]);
        seen[p] = true;
    }
    let nat = SparseLdl::new(&a, Ordering::Natural).unwrap();
    let md = SparseLdl::new(&a, Ordering::MinimumDegree).unwrap();
    assert!(
        md.nnz_l() < nat.nnz_l(),
        "{} !< {}",
        md.nnz_l(),
        nat.nnz_l()
    );
    let b = vec![1.0; n];
    assert!(residual(&a, &md.solve(&b).unwrap(), &b) < 1e-10);
    let ch = SparseCholesky::new(&a, Ordering::MinimumDegree).unwrap();
    assert!(residual(&a, &ch.solve(&b).unwrap(), &b) < 1e-10);
    let lu_nat = SparseLu::new(&a, Ordering::Natural).unwrap();
    let lu_md = SparseLu::new(&a, Ordering::MinimumDegree).unwrap();
    assert!(lu_md.nnz_l() + lu_md.nnz_u() < lu_nat.nnz_l() + lu_nat.nnz_u());
}
