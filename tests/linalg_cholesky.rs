//! Tests für die Cholesky-Zerlegung (fest und dynamisch).
//!
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`; `micromath` ist eine bewusst
//! ungenaue Näherung (Fehler akkumulieren), daher werden damit keine Zahlenwerte geprüft.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::linalg::{LinalgError, Matrix, Vector};

/// Relative Toleranz für f32-Vergleiche.
const TOL: f32 = 1e-4;

fn close(a: f32, b: f32) {
    assert!(
        (a - b).abs() <= TOL * b.abs().max(1.0),
        "erwartet {b}, erhalten {a}"
    );
}

/// Klassisches Beispiel: A = L·Lᵀ mit L = [[2,0,0],[6,1,0],[-8,5,3]].
fn example() -> Matrix<f32, 3, 3> {
    Matrix::new([
        [4.0, 12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0],
    ])
}

#[test]
fn factor_matches_known_l() {
    let c = example().cholesky().unwrap();
    let l = c.l();
    let expect = [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]];
    for i in 0..3 {
        for j in 0..3 {
            close(l[(i, j)], expect[i][j]);
        }
    }
}

#[test]
fn l_times_l_transpose_reconstructs_a() {
    let a = example();
    let c = a.cholesky().unwrap();
    let rec = *c.l() * c.l().transpose();
    for i in 0..3 {
        for j in 0..3 {
            close(rec[(i, j)], a[(i, j)]);
        }
    }
}

#[test]
fn solve_spd_system() {
    let a = example();
    let x_true = Vector::new([1.0f32, -2.0, 0.5]);
    let b = a * x_true;
    let x = a.solve_spd(&b).unwrap();
    for i in 0..3 {
        close(x[i], x_true[i]);
    }
}

#[test]
fn determinant_and_inverse() {
    let a = example();
    let c = a.cholesky().unwrap();
    close(c.determinant(), 36.0);
    let prod = a * c.inverse();
    let id = Matrix::<f32, 3, 3>::identity();
    for i in 0..3 {
        for j in 0..3 {
            close(prod[(i, j)], id[(i, j)]);
        }
    }
}

#[test]
fn agrees_with_lu() {
    let a = Matrix::new([[6.0f32, 2.0], [2.0, 5.0]]);
    let b = Vector::new([1.0f32, 3.0]);
    let x1 = a.solve_spd(&b).unwrap();
    let x2 = a.solve(&b).unwrap();
    for i in 0..2 {
        close(x1[i], x2[i]);
    }
}

#[test]
fn gram_matrix_is_spd() {
    // BᵀB ist für Matrizen mit vollem Spaltenrang symmetrisch positiv definit
    let b = Matrix::new([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.5]]);
    let g = b.transpose() * b;
    assert!(g.cholesky().is_ok());
}

#[test]
fn rejects_non_symmetric() {
    let a = Matrix::new([[4.0f32, 1.0], [2.0, 3.0]]);
    assert_eq!(a.cholesky().unwrap_err(), LinalgError::NotSymmetric);
}

#[test]
fn rejects_non_positive_definite() {
    // Eigenwerte 3 und -1
    let a = Matrix::new([[1.0f32, 2.0], [2.0, 1.0]]);
    assert_eq!(a.cholesky().unwrap_err(), LinalgError::NotPositiveDefinite);
    // negative Diagonale
    let b = Matrix::new([[-1.0f32, 0.0], [0.0, 1.0]]);
    assert_eq!(b.cholesky().unwrap_err(), LinalgError::NotPositiveDefinite);
    // Nullmatrix
    let z = Matrix::<f32, 2, 2>::zeros();
    assert_eq!(z.cholesky().unwrap_err(), LinalgError::NotPositiveDefinite);
}

#[test]
fn rejects_non_finite() {
    let a = Matrix::new([[1.0f32, f32::NAN], [f32::NAN, 1.0]]);
    assert_eq!(a.cholesky().unwrap_err(), LinalgError::NotFinite);
}

#[test]
fn one_by_one_and_empty() {
    let a = Matrix::new([[9.0f32]]);
    close(a.cholesky().unwrap().l()[(0, 0)], 3.0);
    close(a.solve_spd(&Vector::new([18.0])).unwrap()[0], 2.0);
    assert!(Matrix::<f32, 0, 0>::zeros().cholesky().is_ok());
}

#[test]
fn hilbert_5x5_f64() {
    // Hilbert-Matrizen sind SPD, aber schlecht konditioniert
    let h = Matrix::<f64, 5, 5>::from_fn(|i, j| 1.0 / (i + j + 1) as f64);
    let x_true = Vector::new([1.0f64; 5]);
    let x = h.solve_spd(&(h * x_true)).unwrap();
    for i in 0..5 {
        assert!((x[i] - 1.0).abs() < 1e-8, "x[{i}] = {}", x[i]);
    }
}

#[cfg(feature = "alloc")]
mod dynamic {
    extern crate alloc;
    use alloc::vec;
    use azmath::linalg::{DynMatrix, DynVector, LinalgError};

    use super::close;

    fn example() -> DynMatrix<f32> {
        DynMatrix::new(
            3,
            3,
            vec![4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
        )
        .unwrap()
    }

    #[test]
    fn factor_solve_det_inverse() {
        let a = example();
        let c = a.cholesky().unwrap();
        let l = c.l();
        close(l[(0, 0)], 2.0);
        close(l[(2, 0)], -8.0);
        close(l[(2, 2)], 3.0);
        assert_eq!(l[(0, 2)], 0.0);
        close(c.determinant(), 36.0);

        let x_true = DynVector::new(vec![1.0f32, -2.0, 0.5]);
        let b = a.mul_vec(&x_true).unwrap();
        let x = a.solve_spd(&b).unwrap();
        for i in 0..3 {
            close(x[i], x_true[i]);
        }

        let prod = a.matmul(&c.inverse()).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                close(prod[(i, j)], if i == j { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn errors() {
        let rect = DynMatrix::new(2, 3, vec![1.0f32; 6]).unwrap();
        assert_eq!(rect.cholesky().unwrap_err(), LinalgError::DimensionMismatch);

        let asym = DynMatrix::new(2, 2, vec![4.0f32, 1.0, 2.0, 3.0]).unwrap();
        assert_eq!(asym.cholesky().unwrap_err(), LinalgError::NotSymmetric);

        let indef = DynMatrix::new(2, 2, vec![1.0f32, 2.0, 2.0, 1.0]).unwrap();
        assert_eq!(
            indef.cholesky().unwrap_err(),
            LinalgError::NotPositiveDefinite
        );

        let c = example().cholesky().unwrap();
        assert_eq!(
            c.solve(&DynVector::new(vec![1.0f32])),
            Err(LinalgError::DimensionMismatch)
        );
    }

    #[test]
    fn agrees_with_fixed() {
        use azmath::linalg::{Matrix, Vector};
        let fixed = Matrix::new([[6.0f32, 2.0], [2.0, 5.0]]);
        let dynamic = DynMatrix::new(2, 2, vec![6.0f32, 2.0, 2.0, 5.0]).unwrap();
        let xf = fixed.solve_spd(&Vector::new([1.0, 3.0])).unwrap();
        let xd = dynamic
            .solve_spd(&DynVector::new(vec![1.0f32, 3.0]))
            .unwrap();
        for i in 0..2 {
            assert_eq!(xf[i], xd[i]);
        }
    }
}
