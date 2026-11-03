//! Tests für die direkten dünn besetzten Löser (LDLᵀ, Cholesky, RCM-Umordnung).
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::linalg::LinalgError;
use azmath::sparse::{
    CooMatrix, CsrMatrix, Ordering, SparseCholesky, SparseLdl, permute_symmetric, rcm_ordering,
};

fn residual(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let mut y = vec![0.0; b.len()];
    a.matvec(x, &mut y).unwrap();
    y.iter()
        .zip(b)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0, f64::max)
}

/// 2D-Poisson auf einem m×m-Gitter (5-Punkte-Stern), optional mit Knoten-Umnummerierung `map[alt] = neu`.
fn poisson2d(m: usize, map: &[usize]) -> CsrMatrix<f64> {
    let n = m * m;
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
fn tridiagonal_poisson() {
    let n = 50;
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, 2.0).unwrap();
        if i + 1 < n {
            coo.push(i, i + 1, -1.0).unwrap();
            coo.push(i + 1, i, -1.0).unwrap();
        }
    }
    let a = CsrMatrix::from_coo(&coo);
    let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
    for ord in [Ordering::Natural, Ordering::ReverseCuthillMcKee] {
        let ldl = SparseLdl::new(&a, ord).unwrap();
        let x = ldl.solve(&b).unwrap();
        assert!(residual(&a, &x, &b) < 1e-12);
        assert_eq!(ldl.inertia(), (n, 0));
        let ch = SparseCholesky::new(&a, ord).unwrap();
        let x = ch.solve(&b).unwrap();
        assert!(residual(&a, &x, &b) < 1e-12);
        // det des (n×n)-Tridiagonalen = n+1
        assert!((ch.log_determinant() - ((n + 1) as f64).ln()).abs() < 1e-10);
    }
}

#[test]
fn poisson_2d_solves() {
    let m = 10;
    let id: Vec<usize> = (0..m * m).collect();
    let a = poisson2d(m, &id);
    let b: Vec<f64> = (0..m * m).map(|i| 1.0 + (i % 7) as f64).collect();
    for ord in [Ordering::Natural, Ordering::ReverseCuthillMcKee] {
        let x = SparseCholesky::new(&a, ord).unwrap().solve(&b).unwrap();
        assert!(residual(&a, &x, &b) < 1e-11);
        let x = SparseLdl::new(&a, ord).unwrap().solve(&b).unwrap();
        assert!(residual(&a, &x, &b) < 1e-11);
    }
}

#[test]
fn random_spd_matrices() {
    let mut rng = Lcg(12345);
    for n in [5usize, 12, 30] {
        // diagonal dominante symmetrische Matrix mit zufälliger Struktur
        let mut coo = CooMatrix::new(n, n);
        let mut rowsum = vec![0.0; n];
        for i in 0..n {
            for j in 0..i {
                if rng.next() < 0.25 {
                    let v = rng.next() - 0.5;
                    coo.push(i, j, v).unwrap();
                    coo.push(j, i, v).unwrap();
                    rowsum[i] += v.abs();
                    rowsum[j] += v.abs();
                }
            }
        }
        for i in 0..n {
            coo.push(i, i, rowsum[i] + 1.0).unwrap();
        }
        let a = CsrMatrix::from_coo(&coo);
        let b: Vec<f64> = (0..n).map(|_| rng.next() - 0.5).collect();
        for ord in [Ordering::Natural, Ordering::ReverseCuthillMcKee] {
            let x = SparseCholesky::new(&a, ord).unwrap().solve(&b).unwrap();
            assert!(residual(&a, &x, &b) < 1e-11, "n = {n}");
        }
    }
}

#[test]
fn rcm_reduces_fill_on_scrambled_grid() {
    let m = 12;
    let n = m * m;
    // deterministische Zufallspermutation (Fisher-Yates)
    let mut map: Vec<usize> = (0..n).collect();
    let mut rng = Lcg(7);
    for i in (1..n).rev() {
        let j = (rng.next() * (i + 1) as f64) as usize % (i + 1);
        map.swap(i, j);
    }
    let a = poisson2d(m, &map);
    let nat = SparseLdl::new(&a, Ordering::Natural).unwrap();
    let rcm = SparseLdl::new(&a, Ordering::ReverseCuthillMcKee).unwrap();
    assert!(
        rcm.nnz_l() < nat.nnz_l(),
        "{} !< {}",
        rcm.nnz_l(),
        nat.nnz_l()
    );
    let b = vec![1.0; n];
    let x = rcm.solve(&b).unwrap();
    assert!(residual(&a, &x, &b) < 1e-10);
}

#[test]
fn rcm_is_permutation_and_permute_symmetric_consistent() {
    let m = 6;
    let n = m * m;
    let id: Vec<usize> = (0..n).collect();
    let a = poisson2d(m, &id);
    let perm = rcm_ordering(&a);
    let mut seen = vec![false; n];
    for &p in &perm {
        assert!(p < n && !seen[p]);
        seen[p] = true;
    }
    let c = permute_symmetric(&a, &perm);
    let (da, dc) = (a.to_dense(), c.to_dense());
    for i in 0..n {
        for j in 0..n {
            assert_eq!(dc[(i, j)], da[(perm[i], perm[j])]);
        }
    }
}

#[test]
fn indefinite_kkt_matrix() {
    let mut coo = CooMatrix::new(3, 3);
    for &(i, j, v) in &[
        (0, 0, 4.0),
        (1, 1, 3.0),
        (0, 2, 1.0),
        (2, 0, 1.0),
        (1, 2, 1.0),
        (2, 1, 1.0),
        (2, 2, -2.0),
    ] {
        coo.push(i, j, v).unwrap();
    }
    let a = CsrMatrix::from_coo(&coo);
    let ldl = SparseLdl::new(&a, Ordering::Natural).unwrap();
    assert_eq!(ldl.inertia(), (2, 1));
    let b = [1.0, 2.0, 3.0];
    let x = ldl.solve(&b).unwrap();
    assert!(residual(&a, &x, &b) < 1e-12);
    assert!(matches!(
        SparseCholesky::new(&a, Ordering::Natural),
        Err(LinalgError::NotPositiveDefinite)
    ));
}

#[test]
fn error_cases() {
    // singulär: Nullmatrix
    let coo = CooMatrix::<f64>::new(2, 2);
    let z = CsrMatrix::from_coo(&coo);
    assert!(matches!(
        SparseLdl::new(&z, Ordering::Natural),
        Err(LinalgError::Singular)
    ));
    // nicht quadratisch
    let r = CsrMatrix::from_coo(&CooMatrix::<f64>::new(2, 3));
    assert!(matches!(
        SparseLdl::new(&r, Ordering::Natural),
        Err(LinalgError::DimensionMismatch)
    ));
    // nicht endlich
    let mut coo = CooMatrix::new(1, 1);
    coo.push(0, 0, f64::NAN).unwrap();
    assert!(matches!(
        SparseLdl::new(&CsrMatrix::from_coo(&coo), Ordering::Natural),
        Err(LinalgError::NotFinite)
    ));
    // falsche rechte Seite
    let mut coo = CooMatrix::new(2, 2);
    coo.push(0, 0, 1.0).unwrap();
    coo.push(1, 1, 1.0).unwrap();
    let ldl = SparseLdl::new(&CsrMatrix::from_coo(&coo), Ordering::Natural).unwrap();
    assert!(ldl.solve(&[1.0]).is_err());
}

#[test]
fn f32_smoke() {
    let mut coo = CooMatrix::new(3, 3);
    for i in 0..3 {
        coo.push(i, i, 2.0f32).unwrap();
    }
    coo.push(0, 1, -1.0f32).unwrap();
    coo.push(1, 0, -1.0f32).unwrap();
    let a = CsrMatrix::from_coo(&coo);
    let x = SparseCholesky::new(&a, Ordering::ReverseCuthillMcKee)
        .unwrap()
        .solve(&[1.0, 1.0, 1.0])
        .unwrap();
    let mut y = [0.0f32; 3];
    a.matvec(&x, &mut y).unwrap();
    for v in y {
        assert!((v - 1.0).abs() < 1e-5);
    }
}
