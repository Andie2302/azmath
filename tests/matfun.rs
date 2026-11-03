//! Tests für die Matrixfunktionen.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::linalg::DynMatrix;
use azmath::matfun::{expm, logm, lyapunov, powm, sqrtm, sylvester};

fn mat(rows: usize, cols: usize, d: &[f64]) -> DynMatrix<f64> {
    DynMatrix::new(rows, cols, d.to_vec()).unwrap()
}

fn max_diff(a: &DynMatrix<f64>, b: &DynMatrix<f64>) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn expm_zero_and_diagonal() {
    let z = DynMatrix::<f64>::zeros(3, 3);
    assert!(max_diff(&expm(&z).unwrap(), &DynMatrix::identity(3)) < 1e-14);
    let d = mat(2, 2, &[1.0, 0.0, 0.0, -2.0]);
    let e = expm(&d).unwrap();
    assert!((e[(0, 0)] - 1f64.exp()).abs() < 1e-13);
    assert!((e[(1, 1)] - (-2f64).exp()).abs() < 1e-13);
    assert!(e[(0, 1)].abs() < 1e-14);
}

#[test]
fn expm_rotation_generator() {
    let t = 1.3f64;
    let g = mat(2, 2, &[0.0, -t, t, 0.0]);
    let e = expm(&g).unwrap();
    let r = mat(2, 2, &[t.cos(), -t.sin(), t.sin(), t.cos()]);
    assert!(max_diff(&e, &r) < 1e-13);
}

#[test]
fn expm_large_norm_and_nilpotent() {
    // nilpotent: exp(N) = I + N
    let n = mat(2, 2, &[0.0, 5.0, 0.0, 0.0]);
    assert!(max_diff(&expm(&n).unwrap(), &mat(2, 2, &[1.0, 5.0, 0.0, 1.0])) < 1e-13);
    // exp(A)·exp(−A) = I auch bei großer Norm
    let a = mat(2, 2, &[10.0, 3.0, -4.0, 8.0]);
    let neg = mat(2, 2, &[-10.0, -3.0, 4.0, -8.0]);
    let p = expm(&a).unwrap().matmul(&expm(&neg).unwrap()).unwrap();
    assert!(max_diff(&p, &DynMatrix::identity(2)) < 1e-8);
}

#[test]
fn sqrtm_squares_back() {
    let a = mat(2, 2, &[4.0, 1.0, 1.0, 3.0]);
    let s = sqrtm(&a).unwrap();
    assert!(max_diff(&s.matmul(&s).unwrap(), &a) < 1e-12);
    let d = mat(2, 2, &[9.0, 0.0, 0.0, 16.0]);
    assert!(max_diff(&sqrtm(&d).unwrap(), &mat(2, 2, &[3.0, 0.0, 0.0, 4.0])) < 1e-12);
}

#[test]
fn logm_inverts_expm() {
    let a = mat(2, 2, &[4.0, 1.0, 1.0, 3.0]);
    let l = logm(&a).unwrap();
    assert!(max_diff(&expm(&l).unwrap(), &a) < 1e-10);
    let g = mat(2, 2, &[0.0, -0.4, 0.4, 0.0]);
    assert!(max_diff(&logm(&expm(&g).unwrap()).unwrap(), &g) < 1e-10);
    assert!(
        max_diff(
            &logm(&DynMatrix::identity(3)).unwrap(),
            &DynMatrix::zeros(3, 3)
        ) < 1e-14
    );
}

#[test]
fn powm_matches_repeated_product() {
    let a = mat(2, 2, &[1.0, 1.0, 1.0, 0.0]);
    // Fibonacci: A^10 = [[89,55],[55,34]]
    assert!(
        max_diff(
            &powm(&a, 10).unwrap(),
            &mat(2, 2, &[89.0, 55.0, 55.0, 34.0])
        ) < 1e-12
    );
    assert_eq!(powm(&a, 0).unwrap(), DynMatrix::identity(2));
}

#[test]
fn sylvester_and_lyapunov_residuals() {
    let a = mat(2, 2, &[3.0, 1.0, 0.0, 2.0]);
    let b = mat(3, 3, &[1.0, 0.0, 1.0, 0.0, 4.0, 1.0, 1.0, 0.0, 5.0]);
    let c = mat(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let x = sylvester(&a, &b, &c).unwrap();
    let r = a.matmul(&x).unwrap();
    let r2 = x.matmul(&b).unwrap();
    let sum: Vec<f64> = r
        .as_slice()
        .iter()
        .zip(r2.as_slice())
        .map(|(p, q)| p + q)
        .collect();
    let res = DynMatrix::new(2, 3, sum).unwrap();
    assert!(max_diff(&res, &c) < 1e-11);

    let a = mat(2, 2, &[-1.0, 0.5, 0.0, -2.0]);
    let q = mat(2, 2, &[2.0, 0.0, 0.0, 1.0]);
    let x = lyapunov(&a, &q).unwrap();
    let t1 = a.matmul(&x).unwrap();
    let t2 = x.matmul(&a.transpose()).unwrap();
    for i in 0..2 {
        for j in 0..2 {
            assert!((t1[(i, j)] + t2[(i, j)] + q[(i, j)]).abs() < 1e-11);
        }
    }
    assert!((x[(0, 1)] - x[(1, 0)]).abs() < 1e-11);
}

#[test]
fn errors() {
    let r = DynMatrix::<f64>::zeros(2, 3);
    assert!(expm(&r).is_err() && sqrtm(&r).is_err() && logm(&r).is_err() && powm(&r, 2).is_err());
    let nan = mat(1, 1, &[f64::NAN]);
    assert!(expm(&nan).is_err());
    // sqrt von −1 (negative Achse) konvergiert nicht
    assert!(sqrtm(&mat(1, 1, &[-1.0])).is_err());
}
