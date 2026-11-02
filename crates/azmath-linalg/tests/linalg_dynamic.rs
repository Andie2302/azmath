//! Tests für DynMatrix, DynVector und DynLu (Feature `alloc`).
#![cfg(all(feature = "alloc", any(feature = "std", feature = "libm")))]

extern crate alloc;

use alloc::vec;
use azmath_linalg::{DynLu, DynMatrix, DynVector, LinalgError};

fn close(a: f32, b: f32) {
    assert!(
        (a - b).abs() <= 1e-4 * b.abs().max(1.0),
        "erwartet {b}, erhalten {a}"
    );
}

fn mat(rows: usize, cols: usize, data: &[f32]) -> DynMatrix<f32> {
    DynMatrix::new(rows, cols, data.to_vec()).unwrap()
}

#[test]
fn construction_and_dimensions() {
    let m = mat(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!((m.rows(), m.cols()), (2, 3));
    assert_eq!(m[(1, 2)], 6.0);
    assert_eq!(
        DynMatrix::<f32>::new(2, 2, vec![1.0, 2.0, 3.0]),
        Err(LinalgError::DimensionMismatch)
    );
    let z = DynMatrix::<f32>::zeros(2, 2);
    assert_eq!(z.as_slice(), &[0.0; 4]);
    let i = DynMatrix::<f32>::identity(3);
    assert_eq!(i[(0, 0)] + i[(1, 1)] + i[(2, 2)], 3.0);
    assert_eq!(i[(0, 1)], 0.0);
}

#[test]
fn products_and_transpose() {
    let a = mat(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let v = DynVector::new(vec![1.0f32, 0.0, -1.0]);
    assert_eq!(a.mul_vec(&v).unwrap().as_slice(), &[-2.0, -2.0]);
    assert_eq!(
        a.mul_vec(&DynVector::new(vec![1.0f32])),
        Err(LinalgError::DimensionMismatch)
    );

    let t = a.transpose();
    assert_eq!((t.rows(), t.cols()), (3, 2));
    assert_eq!(t[(2, 1)], 6.0);

    let p = a.matmul(&t).unwrap();
    assert_eq!(p.as_slice(), &[14.0, 32.0, 32.0, 77.0]);
    assert_eq!(a.matmul(&a), Err(LinalgError::DimensionMismatch));
}

#[test]
fn vector_ops() {
    let a = DynVector::new(vec![1.0f32, 2.0, 3.0]);
    let b = DynVector::new(vec![4.0f32, 5.0, 6.0]);
    assert_eq!(a.dot(&b), Ok(32.0));
    assert_eq!(
        a.dot(&DynVector::zeros(2)),
        Err(LinalgError::DimensionMismatch)
    );
    assert_eq!(a.max_abs(), 3.0);
    assert_eq!(a.len(), 3);
    assert!(DynVector::<f32>::zeros(0).is_empty());
}

#[test]
fn solve_matches_known_solution() {
    let a = mat(3, 3, &[3.0, 2.0, -1.0, 2.0, -2.0, 4.0, -1.0, 0.5, -1.0]);
    let b = DynVector::new(vec![1.0f32, -2.0, 0.0]);
    let x = a.solve(&b).unwrap();
    close(x[0], 1.0);
    close(x[1], -2.0);
    close(x[2], -2.0);
    let r = a.mul_vec(&x).unwrap();
    for i in 0..3 {
        close(r[i], b[i]);
    }
}

#[test]
fn pivoting_and_determinant() {
    let a = mat(2, 2, &[0.0, 1.0, 1.0, 0.0]);
    let x = a.solve(&DynVector::new(vec![3.0f32, 7.0])).unwrap();
    close(x[0], 7.0);
    close(x[1], 3.0);
    close(a.determinant().unwrap(), -1.0);
    close(
        mat(2, 2, &[1.0, 2.0, 3.0, 4.0]).determinant().unwrap(),
        -2.0,
    );
}

#[test]
fn inverse_roundtrip() {
    let a = mat(3, 3, &[4.0, 7.0, 2.0, 3.0, 6.0, 1.0, 2.0, 5.0, 3.0]);
    let prod = a.matmul(&a.inverse().unwrap()).unwrap();
    let id = DynMatrix::<f32>::identity(3);
    for i in 0..3 {
        for j in 0..3 {
            close(prod[(i, j)], id[(i, j)]);
        }
    }
}

#[test]
fn error_cases() {
    let singular = mat(2, 2, &[1.0, 2.0, 2.0, 4.0]);
    assert_eq!(singular.lu().unwrap_err(), LinalgError::Singular);
    assert_eq!(singular.determinant(), Ok(0.0));

    let rect = mat(2, 3, &[1.0; 6]);
    assert_eq!(rect.lu().unwrap_err(), LinalgError::DimensionMismatch);
    assert_eq!(rect.determinant(), Err(LinalgError::DimensionMismatch));

    let nan = mat(2, 2, &[1.0, f32::NAN, 0.0, 1.0]);
    assert_eq!(nan.lu().unwrap_err(), LinalgError::NotFinite);
    assert!(nan.determinant().unwrap().is_nan());

    let ok = mat(2, 2, &[2.0, 0.0, 0.0, 2.0]);
    assert_eq!(
        ok.solve(&DynVector::new(vec![1.0f32])),
        Err(LinalgError::DimensionMismatch)
    );
}

#[test]
fn lu_reuse_and_permutation() {
    let a = mat(3, 3, &[2.0, 1.0, 1.0, 1.0, 3.0, 2.0, 1.0, 0.0, 0.0]);
    let lu = DynLu::new(&a).unwrap();
    for k in 0..4 {
        let b = DynVector::new(vec![k as f32, 1.0, 2.0 * k as f32]);
        let x = lu.solve(&b).unwrap();
        let r = a.mul_vec(&x).unwrap();
        for i in 0..3 {
            close(r[i], b[i]);
        }
    }
    let mut p = lu.permutation().to_vec();
    p.sort();
    assert_eq!(p, vec![0, 1, 2]);
}

#[test]
fn fixed_and_dynamic_agree() {
    use azmath_linalg::{Matrix, Vector};
    let data = [[4.0f32, 1.0, 2.0], [1.0, 5.0, 3.0], [2.0, 3.0, 6.0]];
    let fixed = Matrix::new(data);
    let dynamic = mat(3, 3, &[4.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 6.0]);
    let xf = fixed.solve(&Vector::new([1.0, 2.0, 3.0])).unwrap();
    let xd = dynamic
        .solve(&DynVector::new(vec![1.0f32, 2.0, 3.0]))
        .unwrap();
    for i in 0..3 {
        assert_eq!(xf[i], xd[i]);
    }
    close(fixed.determinant(), dynamic.determinant().unwrap());
}

#[test]
fn larger_system_f64() {
    // Diagonaldominante 20x20-Matrix, bekannte Lösung (1, 2, ..., n)
    let n = 20;
    let mut data = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i * n + j] = if i == j {
                25.0
            } else {
                1.0 / (1 + i + j) as f64
            };
        }
    }
    let a = DynMatrix::new(n, n, data).unwrap();
    let x_true = DynVector::new((1..=n).map(|k| k as f64).collect());
    let b = a.mul_vec(&x_true).unwrap();
    let x = a.solve(&b).unwrap();
    for i in 0..n {
        assert!((x[i] - x_true[i]).abs() < 1e-10);
    }
}
