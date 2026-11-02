//! Tests für die Dreieckslöser (Vorwärts-/Rückwärtssubstitution), fest und dynamisch.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::linalg::{LinalgError, Matrix, Vector};

fn close(a: f64, b: f64) {
    assert!(
        (a - b).abs() <= 1e-12 * b.abs().max(1.0),
        "erwartet {b}, erhalten {a}"
    );
}

fn assert_vec_close<const N: usize>(a: &Vector<f64, N>, b: &Vector<f64, N>) {
    for i in 0..N {
        close(a[i], b[i]);
    }
}

/// Untere Dreiecksmatrix, im oberen Dreieck mit Müll (99) gefüllt, der ignoriert werden muss.
fn lower_with_garbage() -> Matrix<f64, 3, 3> {
    Matrix::new([[2.0, 99.0, 99.0], [1.0, 3.0, 99.0], [4.0, 5.0, 6.0]])
}

/// Dieselbe Matrix ohne Müll.
fn lower_clean() -> Matrix<f64, 3, 3> {
    Matrix::new([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [4.0, 5.0, 6.0]])
}

fn upper_with_garbage() -> Matrix<f64, 3, 3> {
    Matrix::new([[2.0, 1.0, 4.0], [99.0, 3.0, 5.0], [99.0, 99.0, 6.0]])
}

fn upper_clean() -> Matrix<f64, 3, 3> {
    Matrix::new([[2.0, 1.0, 4.0], [0.0, 3.0, 5.0], [0.0, 0.0, 6.0]])
}

const X: [f64; 3] = [1.0, -2.0, 0.5];

#[test]
fn solve_lower_forward_substitution() {
    let x_true = Vector::new(X);
    let b = lower_clean() * x_true;
    let x = lower_with_garbage().solve_lower(&b).unwrap();
    assert_vec_close(&x, &x_true);
}

#[test]
fn solve_upper_backward_substitution() {
    let x_true = Vector::new(X);
    let b = upper_clean() * x_true;
    let x = upper_with_garbage().solve_upper(&b).unwrap();
    assert_vec_close(&x, &x_true);
}

#[test]
fn unit_variants_ignore_the_diagonal() {
    let x_true = Vector::new(X);

    // Untere Matrix mit Einheitsdiagonale; in der Testmatrix steht auf der Diagonale (und im
    // oberen Dreieck) beliebiger Müll, der nicht gelesen werden darf.
    let unit_lower = Matrix::new([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [4.0, 5.0, 1.0]]);
    let garbage_lower = Matrix::new([[7.0, 99.0, 99.0], [1.0, -3.0, 99.0], [4.0, 5.0, 0.0]]);
    assert_vec_close(
        &garbage_lower.solve_lower_unit(&(unit_lower * x_true)),
        &x_true,
    );

    let unit_upper = Matrix::new([[1.0, 1.0, 4.0], [0.0, 1.0, 5.0], [0.0, 0.0, 1.0]]);
    let garbage_upper = Matrix::new([[7.0, 1.0, 4.0], [99.0, 0.0, 5.0], [99.0, 99.0, -2.0]]);
    assert_vec_close(
        &garbage_upper.solve_upper_unit(&(unit_upper * x_true)),
        &x_true,
    );
}

#[test]
fn transposed_variants_match_explicit_transpose() {
    let b = Vector::new([3.0, -1.0, 2.0]);

    let l = lower_with_garbage();
    let via_transposed = l.solve_lower_transposed(&b).unwrap();
    let explicit = lower_clean().transpose().solve_upper(&b).unwrap();
    assert_vec_close(&via_transposed, &explicit);
    // Probe: Lᵀ·x = b
    assert_vec_close(&(lower_clean().transpose() * via_transposed), &b);

    let u = upper_with_garbage();
    let via_transposed = u.solve_upper_transposed(&b).unwrap();
    let explicit = upper_clean().transpose().solve_lower(&b).unwrap();
    assert_vec_close(&via_transposed, &explicit);
    assert_vec_close(&(upper_clean().transpose() * via_transposed), &b);
}

#[test]
fn agrees_with_lu_on_triangular_matrices() {
    let b = Vector::new([1.0, 2.0, 3.0]);
    assert_vec_close(
        &lower_clean().solve_lower(&b).unwrap(),
        &lower_clean().solve(&b).unwrap(),
    );
    assert_vec_close(
        &upper_clean().solve_upper(&b).unwrap(),
        &upper_clean().solve(&b).unwrap(),
    );
}

#[test]
fn cholesky_in_two_triangular_steps() {
    // A·x = b  <=>  L·y = b, Lᵀ·x = y
    let a = Matrix::new([
        [4.0, 12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0],
    ]);
    let b = Vector::new([1.0, 2.0, 3.0]);
    let l = *a.cholesky().unwrap().l();
    let y = l.solve_lower(&b).unwrap();
    let x = l.solve_lower_transposed(&y).unwrap();
    assert_vec_close(&x, &a.solve_spd(&b).unwrap());
    assert_vec_close(&(a * x), &b);
}

#[test]
fn zero_or_nan_diagonal_is_singular() {
    let b = Vector::new([1.0, 1.0]);
    let zero = Matrix::new([[1.0, 0.0], [1.0, 0.0]]);
    assert_eq!(zero.solve_lower(&b), Err(LinalgError::Singular));
    assert_eq!(zero.solve_upper(&b), Err(LinalgError::Singular));
    assert_eq!(zero.solve_lower_transposed(&b), Err(LinalgError::Singular));
    assert_eq!(zero.solve_upper_transposed(&b), Err(LinalgError::Singular));

    let nan = Matrix::new([[f64::NAN, 0.0], [1.0, 1.0]]);
    assert_eq!(nan.solve_lower(&b), Err(LinalgError::Singular));

    // Mit Einheitsdiagonale spielt die Diagonale keine Rolle
    let x = zero.solve_lower_unit(&b);
    assert_vec_close(&x, &Vector::new([1.0, 0.0]));
    let y = zero.solve_upper_unit(&b);
    assert_vec_close(&y, &Vector::new([1.0, 1.0]));
}

#[test]
fn one_by_one_and_empty() {
    close(
        Matrix::new([[4.0]])
            .solve_lower(&Vector::new([8.0]))
            .unwrap()[0],
        2.0,
    );
    close(
        Matrix::new([[4.0]])
            .solve_upper(&Vector::new([8.0]))
            .unwrap()[0],
        2.0,
    );
    close(
        Matrix::new([[4.0]]).solve_lower_unit(&Vector::new([8.0]))[0],
        8.0,
    );
    let e = Matrix::<f64, 0, 0>::zeros();
    assert!(e.solve_lower(&Vector::new([])).is_ok());
}

#[test]
fn f32_works_too() {
    let l = Matrix::new([[2.0f32, 0.0], [1.0, 4.0]]);
    let x = l.solve_lower(&Vector::new([4.0, 9.0])).unwrap();
    assert!((x[0] - 2.0).abs() < 1e-6 && (x[1] - 1.75).abs() < 1e-6);
}

#[test]
fn larger_system_residual() {
    const N: usize = 12;
    let l = Matrix::<f64, N, N>::from_fn(|i, j| {
        if i == j {
            2.0 + i as f64
        } else if j < i {
            1.0 / (1 + i + j) as f64
        } else {
            0.0
        }
    });
    let x_true = Vector::<f64, N>::new(core::array::from_fn(|i| i as f64 - 5.0));
    let b = l * x_true;
    let x = l.solve_lower(&b).unwrap();
    for i in 0..N {
        assert!((x[i] - x_true[i]).abs() < 1e-10);
    }
    let xt = l.solve_lower_transposed(&b).unwrap();
    assert!((l.transpose() * xt - b).max_abs() < 1e-10);
}

#[cfg(feature = "alloc")]
mod dynamic {
    extern crate alloc;
    use azmath::linalg::{DynMatrix, DynVector, LinalgError};

    fn m(rows: usize, cols: usize, d: &[f64]) -> DynMatrix<f64> {
        DynMatrix::new(rows, cols, d.to_vec()).unwrap()
    }

    fn v(d: &[f64]) -> DynVector<f64> {
        DynVector::new(d.to_vec())
    }

    fn close_vec(a: &DynVector<f64>, b: &DynVector<f64>) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert!((a[i] - b[i]).abs() < 1e-12, "{} vs {}", a[i], b[i]);
        }
    }

    #[test]
    fn lower_upper_and_residual() {
        let l = m(3, 3, &[2.0, 99.0, 99.0, 1.0, 3.0, 99.0, 4.0, 5.0, 6.0]);
        let l_clean = m(3, 3, &[2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 5.0, 6.0]);
        let x_true = v(&[1.0, -2.0, 0.5]);
        let b = l_clean.mul_vec(&x_true).unwrap();
        close_vec(&l.solve_lower(&b).unwrap(), &x_true);

        let u = m(3, 3, &[2.0, 1.0, 4.0, 99.0, 3.0, 5.0, 99.0, 99.0, 6.0]);
        let u_clean = m(3, 3, &[2.0, 1.0, 4.0, 0.0, 3.0, 5.0, 0.0, 0.0, 6.0]);
        let bu = u_clean.mul_vec(&x_true).unwrap();
        close_vec(&u.solve_upper(&bu).unwrap(), &x_true);
    }

    #[test]
    fn unit_and_transposed() {
        let diag_garbage = m(2, 2, &[7.0, 0.0, 3.0, -9.0]);
        // Einheitsdiagonale: x0 = 1, x1 = 5 - 3*1 = 2
        close_vec(
            &diag_garbage.solve_lower_unit(&v(&[1.0, 5.0])).unwrap(),
            &v(&[1.0, 2.0]),
        );

        let l = m(2, 2, &[2.0, 99.0, 1.0, 4.0]);
        let b = v(&[3.0, 8.0]);
        let explicit = m(2, 2, &[2.0, 1.0, 0.0, 4.0]).solve_upper(&b).unwrap();
        close_vec(&l.solve_lower_transposed(&b).unwrap(), &explicit);

        let u = m(2, 2, &[2.0, 1.0, 99.0, 4.0]);
        let explicit = m(2, 2, &[2.0, 0.0, 1.0, 4.0]).solve_lower(&b).unwrap();
        close_vec(&u.solve_upper_transposed(&b).unwrap(), &explicit);
        close_vec(
            &u.solve_upper_unit(&v(&[1.0, 2.0])).unwrap(),
            &v(&[-1.0, 2.0]),
        );
    }

    #[test]
    fn errors() {
        let singular = m(2, 2, &[1.0, 0.0, 1.0, 0.0]);
        assert_eq!(
            singular.solve_lower(&v(&[1.0, 1.0])),
            Err(LinalgError::Singular)
        );
        assert_eq!(
            singular.solve_upper(&v(&[1.0, 1.0])),
            Err(LinalgError::Singular)
        );
        assert!(singular.solve_lower_unit(&v(&[1.0, 1.0])).is_ok());

        let rect = m(2, 3, &[1.0; 6]);
        assert_eq!(
            rect.solve_lower(&v(&[1.0, 1.0])),
            Err(LinalgError::DimensionMismatch)
        );
        assert_eq!(
            rect.solve_upper_unit(&v(&[1.0, 1.0])),
            Err(LinalgError::DimensionMismatch)
        );

        let ok = m(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            ok.solve_lower(&v(&[1.0])),
            Err(LinalgError::DimensionMismatch)
        );
    }

    #[test]
    fn agrees_with_fixed_and_cholesky() {
        use azmath::linalg::{Matrix, Vector};
        let fixed = Matrix::new([[2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [4.0, 5.0, 6.0]]);
        let dynamic = m(3, 3, &[2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 5.0, 6.0]);
        let xf = fixed.solve_lower(&Vector::new([1.0, 2.0, 3.0])).unwrap();
        let xd = dynamic.solve_lower(&v(&[1.0, 2.0, 3.0])).unwrap();
        for i in 0..3 {
            assert_eq!(xf[i], xd[i]);
        }

        // Cholesky in zwei Dreiecks-Schritten
        let a = m(2, 2, &[6.0, 2.0, 2.0, 5.0]);
        let b = v(&[1.0, 3.0]);
        let l = a.cholesky().unwrap().l();
        let y = l.solve_lower(&b).unwrap();
        let x = l.solve_lower_transposed(&y).unwrap();
        close_vec(&x, &a.solve_spd(&b).unwrap());
    }

    #[test]
    fn empty_system() {
        let e = DynMatrix::<f64>::zeros(0, 0);
        assert_eq!(e.solve_lower(&DynVector::zeros(0)).unwrap().len(), 0);
    }
}
