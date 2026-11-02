//! Tests für Bandmatrizen, Band-LU und Band-Cholesky.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm` und brauchen `alloc`.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::linalg::{DynMatrix, DynVector, LinalgError, LinearOperator, solve_tridiagonal};
use azmath::sparse::{BandMatrix, SparseError};

/// Deterministische Pseudozufallszahlen in [-1, 1).
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Zufällige Bandmatrix (Diagonale schwach dominant, damit gut konditioniert).
fn random_band(n: usize, kl: usize, ku: usize, seed: u64, dominant: bool) -> BandMatrix<f64> {
    let mut s = seed;
    let mut a = BandMatrix::new(n, kl, ku);
    for i in 0..n {
        for j in i.saturating_sub(kl)..=(i + ku).min(n - 1) {
            let mut v = lcg(&mut s);
            if dominant && i == j {
                v += (kl + ku + 1) as f64 * v.signum() + 0.5;
            }
            a.set(i, j, v).unwrap();
        }
    }
    a
}

fn max_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn storage_access_and_conversion() {
    let mut a = BandMatrix::new(4, 1, 2);
    a.set(0, 2, 5.0f64).unwrap();
    a.set(3, 2, -1.0).unwrap();
    a.set(1, 1, 2.0).unwrap();
    assert_eq!(a.get(0, 2), Some(5.0));
    assert_eq!(a.get(0, 3), Some(0.0)); // außerhalb des Bands, aber in der Matrix
    assert_eq!(a.get(4, 0), None);
    assert_eq!((a.n(), a.kl(), a.ku()), (4, 1, 2));
    assert_eq!(a.set(0, 3, 1.0), Err(SparseError::IndexOutOfBounds));
    assert_eq!(a.set(2, 0, 1.0), Err(SparseError::IndexOutOfBounds));
    assert_eq!(a.set(0, 0, f64::NAN), Err(SparseError::NotFinite));
    let d = a.to_dense();
    assert_eq!(d[(0, 2)], 5.0);
    assert_eq!(d[(3, 2)], -1.0);
    assert_eq!(BandMatrix::from_dense(&d, 1, 2).unwrap(), a);
    // Eintrag außerhalb des Bands
    assert_eq!(
        BandMatrix::from_dense(&d, 0, 2),
        Err(SparseError::IndexOutOfBounds)
    );
    let rect = DynMatrix::<f64>::zeros(2, 3);
    assert_eq!(
        BandMatrix::from_dense(&rect, 1, 1),
        Err(SparseError::DimensionMismatch)
    );
    let t = BandMatrix::from_tridiagonal(&[1.0, 2.0], &[3.0, 4.0, 5.0], &[6.0, 7.0]).unwrap();
    assert_eq!(t.get(1, 0), Some(1.0));
    assert_eq!(t.get(1, 2), Some(7.0));
    assert_eq!(
        BandMatrix::from_tridiagonal(&[1.0], &[3.0, 4.0, 5.0], &[6.0, 7.0]),
        Err(SparseError::DimensionMismatch)
    );
}

#[test]
fn matvec_matches_the_dense_product() {
    let a = random_band(9, 2, 3, 3, false);
    let x: Vec<f64> = (0..9).map(|k| (k as f64).sin()).collect();
    let mut y = vec![0.0; 9];
    a.matvec(&x, &mut y).unwrap();
    let want = a.to_dense().mul_vec(&DynVector::new(x.clone())).unwrap();
    assert!(max_diff(&y, want.as_slice()) < 1e-14);
    let mut y2 = vec![0.0; 9];
    LinearOperator::apply(&a, &x[..], &mut y2);
    assert_eq!(y, y2);
    assert_eq!(
        a.matvec(&x[..3], &mut y),
        Err(SparseError::DimensionMismatch)
    );
}

#[test]
fn lu_solves_like_the_dense_solver() {
    for (n, kl, ku, seed) in [
        (1, 0, 0, 1),
        (6, 1, 1, 2),
        (10, 2, 1, 3),
        (10, 1, 3, 4),
        (15, 4, 4, 5),
        (8, 7, 7, 6),
        (7, 0, 2, 7),
        (7, 3, 0, 8),
    ] {
        // nicht diagonaldominant: erzwingt Zeilenvertauschungen
        let a = random_band(n, kl, ku, seed, false);
        let mut s = seed * 13;
        let b: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
        let lu = a.lu().unwrap();
        let x = lu.solve(&b).unwrap();
        let dense = a.to_dense();
        let want = dense.solve(&DynVector::new(b.clone())).unwrap();
        assert!(
            max_diff(&x, want.as_slice()) < 1e-9,
            "n = {n}, kl = {kl}, ku = {ku}"
        );
        // Residuum
        let mut r = vec![0.0; n];
        a.matvec(&x, &mut r).unwrap();
        assert!(max_diff(&r, &b) < 1e-10);
        let det = dense.determinant().unwrap();
        assert!((lu.determinant() - det).abs() < 1e-8 * (1.0 + det.abs()));
        // über BandMatrix::solve
        assert!(max_diff(a.solve(&DynVector::new(b)).unwrap().as_slice(), &x) < 1e-15);
    }
}

#[test]
fn lu_needs_pivoting_and_detects_singularity() {
    // [[0, 1], [1, 0]] mit kl = ku = 1
    let a = BandMatrix::from_tridiagonal(&[1.0f64], &[0.0, 0.0], &[1.0]).unwrap();
    let lu = a.lu().unwrap();
    assert_eq!(lu.solve(&[3.0, 4.0]).unwrap(), vec![4.0, 3.0]);
    assert!((lu.determinant() + 1.0).abs() < 1e-15);
    let reg = BandMatrix::from_tridiagonal(&[1.0f64, 1.0], &[1.0, 1.0, 1.0], &[1.0, 1.0]).unwrap();
    assert!(reg.lu().is_ok()); // [[1,1,0],[1,1,1],[0,1,1]] ist regulär
    assert_eq!(
        BandMatrix::<f64>::new(3, 1, 1).lu(),
        Err(LinalgError::Singular)
    );
    let dup = BandMatrix::from_tridiagonal(&[0.0f64, 0.0], &[1.0, 1.0, 0.0], &[0.0, 0.0]).unwrap();
    assert_eq!(dup.lu(), Err(LinalgError::Singular));
    let lu = BandMatrix::<f64>::from_tridiagonal(&[], &[2.0], &[])
        .unwrap()
        .lu()
        .unwrap();
    assert_eq!(lu.solve(&[4.0]).unwrap(), vec![2.0]);
    assert_eq!(lu.solve(&[1.0, 2.0]), Err(LinalgError::DimensionMismatch));
}

#[test]
fn cholesky_solves_spd_bands() {
    // Poisson 1D gegen den Thomas-Algorithmus
    let n = 50;
    let mut p = BandMatrix::new(n, 1, 1);
    for i in 0..n {
        p.set(i, i, 2.0).unwrap();
        if i + 1 < n {
            p.set(i, i + 1, -1.0).unwrap();
            p.set(i + 1, i, -1.0).unwrap();
        }
    }
    let b = vec![1.0; n];
    let x = p.cholesky().unwrap().solve(&b).unwrap();
    let mut want = vec![0.0; n];
    let mut scratch = vec![0.0; n];
    let mut lower = vec![-1.0; n];
    lower[0] = 0.0;
    let mut upper = vec![-1.0; n];
    upper[n - 1] = 0.0;
    solve_tridiagonal(&lower, &vec![2.0; n], &upper, &b, &mut want, &mut scratch).unwrap();
    assert!(max_diff(&x, &want) < 1e-10);

    // zufällige SPD-Matrix B = AᵀA + I mit tridiagonalem A (Bandbreite 2)
    let a = random_band(12, 1, 1, 9, true);
    let ad = a.to_dense();
    let mut spd = ad.transpose().matmul(&ad).unwrap();
    for i in 0..12 {
        spd[(i, i)] += 1.0;
    }
    let band = BandMatrix::from_dense(&spd, 2, 2).unwrap();
    let mut s = 4;
    let rhs: Vec<f64> = (0..12).map(|_| lcg(&mut s)).collect();
    let chol = band.cholesky().unwrap();
    let got = chol.solve(&rhs).unwrap();
    let want = spd.solve(&DynVector::new(rhs.clone())).unwrap();
    assert!(max_diff(&got, want.as_slice()) < 1e-10);
    assert!((chol.determinant() / spd.determinant().unwrap() - 1.0).abs() < 1e-9);
}

#[test]
fn cholesky_errors() {
    // nicht symmetrisch
    let ns = BandMatrix::from_tridiagonal(&[1.0f64], &[4.0, 4.0], &[2.0]).unwrap();
    assert_eq!(ns.cholesky(), Err(LinalgError::NotSymmetric));
    // symmetrisch, aber indefinit
    let indef = BandMatrix::from_tridiagonal(&[2.0f64], &[1.0, 1.0], &[2.0]).unwrap();
    assert_eq!(indef.cholesky(), Err(LinalgError::NotPositiveDefinite));
    // kl != ku
    assert_eq!(
        BandMatrix::<f64>::new(3, 1, 2).cholesky(),
        Err(LinalgError::DimensionMismatch)
    );
    let ok = BandMatrix::from_tridiagonal(&[1.0f64], &[4.0, 4.0], &[1.0])
        .unwrap()
        .cholesky()
        .unwrap();
    assert_eq!(ok.solve(&[1.0]), Err(LinalgError::DimensionMismatch));
}

#[test]
fn a_tridiagonal_system_with_100_000_unknowns() {
    let n = 100_000;
    let a =
        BandMatrix::from_tridiagonal(&vec![-1.0; n - 1], &vec![2.0 + 1e-3; n], &vec![-1.0; n - 1])
            .unwrap();
    let b = vec![1.0; n];
    let x = a.lu().unwrap().solve(&b).unwrap();
    let xc = a.cholesky().unwrap().solve(&b).unwrap();
    let mut r = vec![0.0; n];
    a.matvec(&x, &mut r).unwrap();
    assert!(max_diff(&r, &b) < 1e-8);
    assert!(max_diff(&x, &xc) < 1e-8);
}

#[test]
fn works_for_f32() {
    let a = BandMatrix::from_tridiagonal(&[1.0f32, 1.0], &[4.0, 4.0, 4.0], &[1.0, 1.0]).unwrap();
    let x = a.lu().unwrap().solve(&[6.0, 12.0, 14.0]).unwrap();
    for (got, want) in x.iter().zip([1.0f32, 2.0, 3.0]) {
        assert!((got - want).abs() < 1e-5);
    }
}
