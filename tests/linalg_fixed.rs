//! Tests für Vector, Matrix und LuDecomposition fester Größe.
#![cfg(any(feature = "std", feature = "libm", feature = "micromath"))]

use azmath::linalg::{LinalgError, LuDecomposition, Matrix, Vector};

fn close(a: f32, b: f32) {
    assert!(
        (a - b).abs() <= 1e-4 * b.abs().max(1.0),
        "erwartet {b}, erhalten {a}"
    );
}

fn assert_matrix_close<const N: usize>(a: &Matrix<f32, N, N>, b: &Matrix<f32, N, N>) {
    for i in 0..N {
        for j in 0..N {
            close(a[(i, j)], b[(i, j)]);
        }
    }
}

#[test]
fn vector_basics() {
    let a = Vector::new([1.0f32, 2.0, 3.0]);
    let b = Vector::new([4.0f32, 5.0, 6.0]);
    assert_eq!(a + b, Vector::new([5.0, 7.0, 9.0]));
    assert_eq!(b - a, Vector::new([3.0, 3.0, 3.0]));
    assert_eq!(-a, Vector::new([-1.0, -2.0, -3.0]));
    assert_eq!(a * 2.0, Vector::new([2.0, 4.0, 6.0]));
    assert_eq!(a.dot(&b), 32.0);
    assert_eq!(a.max_abs(), 3.0);
    assert_eq!(a.len(), 3);
    assert!(!a.is_empty());
    assert_eq!(Vector::<f32, 2>::zeros(), Vector::new([0.0, 0.0]));
}

#[test]
fn vector_norm() {
    let v = Vector::new([3.0f32, 4.0]);
    assert_eq!(v.norm_squared(), 25.0);
    // sqrt hängt vom Backend ab (micromath ist nur eine Näherung)
    let tol = if cfg!(any(feature = "std", feature = "libm")) {
        1e-5
    } else {
        0.2
    };
    assert!((v.norm() - 5.0).abs() <= tol);
}

#[test]
fn matrix_basics() {
    let a = Matrix::new([[1.0f32, 2.0], [3.0, 4.0]]);
    let b = Matrix::new([[5.0f32, 6.0], [7.0, 8.0]]);
    assert_eq!(a + b, Matrix::new([[6.0, 8.0], [10.0, 12.0]]));
    assert_eq!(b - a, Matrix::new([[4.0, 4.0], [4.0, 4.0]]));
    assert_eq!(-a, Matrix::new([[-1.0, -2.0], [-3.0, -4.0]]));
    assert_eq!(a * 2.0, Matrix::new([[2.0, 4.0], [6.0, 8.0]]));
    assert_eq!(a * b, Matrix::new([[19.0, 22.0], [43.0, 50.0]]));
    assert_eq!(a.trace(), 5.0);
    assert_eq!(a.max_abs(), 4.0);
    assert_eq!((a.rows(), a.cols()), (2, 2));
}

#[test]
fn matrix_vector_product_and_transpose() {
    let a = Matrix::new([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let v = Vector::new([1.0f32, 0.0, -1.0]);
    assert_eq!(a * v, Vector::new([-2.0, -2.0]));

    let t = a.transpose();
    assert_eq!(t, Matrix::new([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]));
    assert_eq!(t.transpose(), a);

    // (2x3) * (3x2) = (2x2)
    assert_eq!(a * t, Matrix::new([[14.0, 32.0], [32.0, 77.0]]));
}

#[test]
fn identity_and_from_fn() {
    let i3 = Matrix::<f32, 3, 3>::identity();
    let a = Matrix::<f32, 3, 3>::from_fn(|i, j| (i * 3 + j) as f32);
    assert_eq!(i3 * a, a);
    assert_eq!(a * i3, a);
    assert_eq!(
        Matrix::<f32, 2, 2>::zeros(),
        Matrix::new([[0.0, 0.0], [0.0, 0.0]])
    );
}

#[test]
fn solve_2x2() {
    // 2x + y = 5, x + 3y = 10  ->  x = 1, y = 3
    let a = Matrix::new([[2.0f32, 1.0], [1.0, 3.0]]);
    let x = a.solve(&Vector::new([5.0, 10.0])).unwrap();
    close(x[0], 1.0);
    close(x[1], 3.0);
}

#[test]
fn solve_3x3_with_residual() {
    let a = Matrix::new([[3.0f32, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]]);
    let b = Vector::new([1.0f32, -2.0, 0.0]);
    let x = a.solve(&b).unwrap();
    close(x[0], 1.0);
    close(x[1], -2.0);
    close(x[2], -2.0);
    let r = a * x - b;
    assert!(r.max_abs() < 1e-4);
}

#[test]
fn solve_needs_pivoting() {
    // Diagonale ist 0 -> ohne Zeilentausch nicht lösbar
    let a = Matrix::new([[0.0f32, 1.0], [1.0, 0.0]]);
    let x = a.solve(&Vector::new([3.0, 7.0])).unwrap();
    close(x[0], 7.0);
    close(x[1], 3.0);

    let b = Matrix::new([[1e-8f32, 1.0], [1.0, 1.0]]);
    let y = b.solve(&Vector::new([1.0, 2.0])).unwrap();
    // exakte Lösung ≈ (1, 1)
    close(y[0], 1.0);
    close(y[1], 1.0);
}

#[test]
fn determinant_values() {
    close(Matrix::new([[1.0f32, 2.0], [3.0, 4.0]]).determinant(), -2.0);
    close(Matrix::<f32, 3, 3>::identity().determinant(), 1.0);
    close(
        Matrix::new([[2.0f32, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]]).determinant(),
        24.0,
    );
    // Zeilentausch ändert das Vorzeichen
    close(Matrix::new([[0.0f32, 1.0], [1.0, 0.0]]).determinant(), -1.0);
    close(Matrix::new([[5.0f32]]).determinant(), 5.0);
}

#[test]
fn inverse_roundtrip() {
    let a = Matrix::new([[4.0f32, 7.0, 2.0], [3.0, 6.0, 1.0], [2.0, 5.0, 3.0]]);
    let inv = a.inverse().unwrap();
    assert_matrix_close(&(a * inv), &Matrix::identity());
    assert_matrix_close(&(inv * a), &Matrix::identity());
}

#[test]
fn inverse_2x2_known() {
    let a = Matrix::new([[4.0f32, 7.0], [2.0, 6.0]]);
    let inv = a.inverse().unwrap();
    assert_matrix_close(&inv, &Matrix::new([[0.6, -0.7], [-0.2, 0.4]]));
}

#[test]
fn singular_matrices() {
    let a = Matrix::new([[1.0f32, 2.0], [2.0, 4.0]]);
    assert_eq!(a.lu().unwrap_err(), LinalgError::Singular);
    assert_eq!(
        a.solve(&Vector::new([1.0, 2.0])),
        Err(LinalgError::Singular)
    );
    assert_eq!(a.inverse(), Err(LinalgError::Singular));
    assert_eq!(a.determinant(), 0.0);

    let zero = Matrix::<f32, 3, 3>::zeros();
    assert_eq!(zero.lu().unwrap_err(), LinalgError::Singular);
    assert_eq!(zero.determinant(), 0.0);
}

#[test]
fn not_finite_input() {
    let a = Matrix::new([[1.0f32, f32::NAN], [0.0, 1.0]]);
    assert_eq!(a.lu().unwrap_err(), LinalgError::NotFinite);
    assert!(a.determinant().is_nan());

    let b = Matrix::new([[f32::INFINITY, 0.0], [0.0, 1.0]]);
    assert_eq!(b.lu().unwrap_err(), LinalgError::NotFinite);
}

#[test]
fn lu_reuse_for_many_right_hand_sides() {
    let a = Matrix::new([[2.0f32, 1.0, 1.0], [1.0, 3.0, 2.0], [1.0, 0.0, 0.0]]);
    let lu = LuDecomposition::new(&a).unwrap();
    for k in 0..5 {
        let b = Vector::new([k as f32, 1.0, 2.0 * k as f32]);
        let x = lu.solve(&b);
        assert!((a * x - b).max_abs() < 1e-4);
    }
    // Permutation ist eine gültige Permutation von 0..N
    let mut seen = [false; 3];
    for &p in lu.permutation() {
        seen[p] = true;
    }
    assert!(seen.iter().all(|&s| s));
}

#[test]
fn empty_matrix() {
    let a = Matrix::<f32, 0, 0>::zeros();
    assert!(a.lu().is_ok());
    assert_eq!(a.determinant(), 1.0);
}

#[test]
fn error_display() {
    extern crate std;
    use std::string::ToString;
    assert!(!LinalgError::Singular.to_string().is_empty());
    assert!(!LinalgError::NotFinite.to_string().is_empty());
    assert!(!LinalgError::DimensionMismatch.to_string().is_empty());
}

#[cfg(any(feature = "std", feature = "libm"))]
mod f64_tests {
    use azmath::linalg::{Matrix, Vector};

    #[test]
    fn solve_hilbert_4x4_f64() {
        // Schlecht konditioniert, aber mit f64 gut lösbar: Lösung (1, 1, 1, 1)
        let h = Matrix::<f64, 4, 4>::from_fn(|i, j| 1.0 / (i + j + 1) as f64);
        let x_true = Vector::new([1.0f64; 4]);
        let b = h * x_true;
        let x = h.solve(&b).unwrap();
        for i in 0..4 {
            assert!((x[i] - 1.0).abs() < 1e-10, "x[{i}] = {}", x[i]);
        }
    }

    #[test]
    fn determinant_and_inverse_f64() {
        let a = Matrix::new([[2.0f64, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 4.0]]);
        assert!((a.determinant() - 18.0).abs() < 1e-12);
        let prod = a * a.inverse().unwrap();
        let id = Matrix::<f64, 3, 3>::identity();
        for i in 0..3 {
            for j in 0..3 {
                assert!((prod[(i, j)] - id[(i, j)]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn near_singular_is_detected_f64() {
        let a = Matrix::new([[1.0f64, 2.0], [1.0, 2.0 + 1e-18]]);
        assert!(a.lu().is_err());
    }
}
