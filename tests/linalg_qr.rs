//! Tests für die QR-Zerlegung und Kleinste-Quadrate-Probleme (fest und dynamisch).
//!
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`; `micromath` ist eine bewusst
//! ungenaue Näherung (Fehler akkumulieren), daher werden damit keine Zahlenwerte geprüft.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::linalg::{LinalgError, Matrix, Vector};

const TOL: f32 = 1e-4;

fn close(a: f32, b: f32) {
    assert!(
        (a - b).abs() <= TOL * b.abs().max(1.0),
        "erwartet {b}, erhalten {a}"
    );
}

#[test]
fn q_times_r_reconstructs_a() {
    let a = Matrix::new([[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 9.0]]);
    let qr = a.qr().unwrap();
    let q = qr.q();
    let r = qr.r();
    // Q (4x4) mal R (2x2) um Nullzeilen erweitert
    let r_full = Matrix::<f32, 4, 2>::from_fn(|i, j| if i < 2 { r[(i, j)] } else { 0.0 });
    let rec = q * r_full;
    for i in 0..4 {
        for j in 0..2 {
            close(rec[(i, j)], a[(i, j)]);
        }
    }
}

#[test]
fn q_is_orthogonal() {
    let a = Matrix::new([
        [2.0f32, -1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 4.0],
        [1.0, 1.0, 1.0],
    ]);
    let q = a.qr().unwrap().q();
    let qtq = q.transpose() * q;
    for i in 0..4 {
        for j in 0..4 {
            close(qtq[(i, j)], if i == j { 1.0 } else { 0.0 });
        }
    }
}

#[test]
fn r_is_upper_triangular() {
    let a = Matrix::new([[2.0f32, 1.0], [1.0, 3.0], [0.0, 1.0]]);
    let r = a.qr().unwrap().r();
    assert_eq!(r[(1, 0)], 0.0);
}

#[test]
fn exact_line_fit() {
    // y = 1 + 2x durch (0,1), (1,3), (2,5), (3,7)
    let a = Matrix::new([[1.0f32, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]);
    let b = Vector::new([1.0f32, 3.0, 5.0, 7.0]);
    let qr = a.qr().unwrap();
    let x = qr.solve(&b);
    close(x[0], 1.0);
    close(x[1], 2.0);
    assert!(qr.residual_norm(&b) < TOL);
}

#[test]
fn noisy_line_fit_matches_normal_equations() {
    // Punkte (1,1), (2,2), (3,2), (4,5): Kleinste-Quadrate-Gerade y = -0.5 + 1.2x
    let a = Matrix::new([[1.0f32, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]);
    let b = Vector::new([1.0f32, 2.0, 2.0, 5.0]);
    let x = a.least_squares(&b).unwrap();
    close(x[0], -0.5);
    close(x[1], 1.2);

    // Gegenprobe über die Normalengleichungen AᵀA·x = Aᵀb (per Cholesky)
    let at = a.transpose();
    let x2 = (at * a).solve_spd(&(at * b)).unwrap();
    close(x[0], x2[0]);
    close(x[1], x2[1]);
}

#[test]
fn residual_is_orthogonal_to_columns() {
    let a = Matrix::new([[1.0f32, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]);
    let b = Vector::new([1.0f32, 2.0, 2.0, 5.0]);
    let qr = a.qr().unwrap();
    let x = qr.solve(&b);
    let r = b - a * x;
    // Aᵀ·r ≈ 0 charakterisiert die Kleinste-Quadrate-Lösung
    let atr = a.transpose() * r;
    assert!(atr.max_abs() < 1e-4);
    // und residual_norm entspricht ‖r‖
    close(qr.residual_norm(&b), r.norm());
}

#[test]
fn square_system_matches_lu() {
    let a = Matrix::new([[3.0f32, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]]);
    let b = Vector::new([1.0f32, -2.0, 0.0]);
    let x = a.qr().unwrap().solve(&b);
    let y = a.solve(&b).unwrap();
    for i in 0..3 {
        close(x[i], y[i]);
    }
}

#[test]
fn square_system_needs_no_pivot_order() {
    // Diagonale 0: bei Householder kein Problem
    let a = Matrix::new([[0.0f32, 1.0], [1.0, 0.0]]);
    let x = a.least_squares(&Vector::new([3.0, 7.0])).unwrap();
    close(x[0], 7.0);
    close(x[1], 3.0);
}

#[test]
fn underdetermined_is_rejected() {
    let a = Matrix::new([[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(a.qr().unwrap_err(), LinalgError::DimensionMismatch);
}

#[test]
fn rank_deficient_is_rejected() {
    // zweite Spalte = 2 · erste Spalte
    let a = Matrix::new([[1.0f32, 2.0], [2.0, 4.0], [3.0, 6.0]]);
    assert_eq!(a.qr().unwrap_err(), LinalgError::Singular);
    // Nullspalte
    let z = Matrix::new([[1.0f32, 0.0], [2.0, 0.0], [3.0, 0.0]]);
    assert_eq!(z.qr().unwrap_err(), LinalgError::Singular);
}

#[test]
fn zero_matrix_is_rejected() {
    // Der Spaltenvektor ist exakt 0 -> keine Näherung nötig, in jedem Backend Singular
    let z = Matrix::<f32, 3, 2>::zeros();
    assert_eq!(z.qr().unwrap_err(), LinalgError::Singular);
}

#[test]
fn rejects_non_finite() {
    let a = Matrix::new([[1.0f32, f32::INFINITY], [0.0, 1.0]]);
    assert_eq!(a.qr().unwrap_err(), LinalgError::NotFinite);
}

mod f64_tests {
    use azmath::linalg::{Matrix, Vector};

    #[test]
    fn polynomial_fit_f64() {
        // Parabel y = 2 - 3x + 0.5x² durch 6 exakte Punkte, Vandermonde-Matrix
        let xs = [0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let a = Matrix::<f64, 6, 3>::from_fn(|i, j| xs[i].powi(j as i32));
        let b = Vector::new(xs.map(|x| 2.0 - 3.0 * x + 0.5 * x * x));
        let c = a.least_squares(&b).unwrap();
        assert!((c[0] - 2.0).abs() < 1e-10);
        assert!((c[1] + 3.0).abs() < 1e-10);
        assert!((c[2] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn hilbert_least_squares_f64() {
        // 8x4-Ausschnitt der Hilbert-Matrix mit exakter Lösung (1,1,1,1)
        let a = Matrix::<f64, 8, 4>::from_fn(|i, j| 1.0 / (i + j + 1) as f64);
        let b = a * Vector::new([1.0f64; 4]);
        let x = a.least_squares(&b).unwrap();
        for i in 0..4 {
            assert!((x[i] - 1.0).abs() < 1e-8, "x[{i}] = {}", x[i]);
        }
    }

    #[test]
    fn qr_is_more_accurate_than_normal_equations() {
        // Schlecht konditioniert: QR löst genauer als AᵀA mit Cholesky (Konditionszahl wird quadriert)
        let a = Matrix::<f64, 8, 4>::from_fn(|i, j| 1.0 / (i + j + 1) as f64);
        let x_true = Vector::new([1.0f64; 4]);
        let b = a * x_true;
        let xq = a.least_squares(&b).unwrap();
        let at = a.transpose();
        let xn = (at * a).solve_spd(&(at * b));
        let err_q = (xq - x_true).max_abs();
        if let Ok(xn) = xn {
            let err_n = (xn - x_true).max_abs();
            assert!(
                err_q <= err_n,
                "QR {err_q:e} vs. Normalengleichung {err_n:e}"
            );
        }
    }
}

#[cfg(feature = "alloc")]
mod dynamic {
    extern crate alloc;
    use alloc::vec;
    use azmath::linalg::{DynMatrix, DynVector, LinalgError};

    use super::{TOL, close};

    fn line_fit_matrix() -> DynMatrix<f32> {
        DynMatrix::new(4, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0]).unwrap()
    }

    #[test]
    fn least_squares_line() {
        let a = line_fit_matrix();
        let b = DynVector::new(vec![1.0f32, 2.0, 2.0, 5.0]);
        let x = a.least_squares(&b).unwrap();
        close(x[0], -0.5);
        close(x[1], 1.2);
        let qr = a.qr().unwrap();
        let rn = qr.residual_norm(&b).unwrap();
        // ‖r‖ = sqrt(1.8) ≈ 1.3416 für diese Punkte
        assert!((rn - 1.8f32.sqrt()).abs() < 5.0 * TOL, "residual {rn}");
    }

    #[test]
    fn reconstruction_and_orthogonality() {
        let a = DynMatrix::new(4, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0]).unwrap();
        let qr = a.qr().unwrap();
        let q = qr.q();
        let r = qr.r();
        assert_eq!((q.rows(), q.cols()), (4, 4));
        assert_eq!((r.rows(), r.cols()), (2, 2));
        let qtq = q.transpose().matmul(&q).unwrap();
        for i in 0..4 {
            for j in 0..4 {
                close(qtq[(i, j)], if i == j { 1.0 } else { 0.0 });
            }
        }
        let mut r_full = DynMatrix::<f32>::zeros(4, 2);
        for i in 0..2 {
            for j in 0..2 {
                r_full[(i, j)] = r[(i, j)];
            }
        }
        let rec = q.matmul(&r_full).unwrap();
        for i in 0..4 {
            for j in 0..2 {
                close(rec[(i, j)], a[(i, j)]);
            }
        }
    }

    #[test]
    fn errors() {
        let wide = DynMatrix::new(2, 3, vec![1.0f32; 6]).unwrap();
        assert_eq!(wide.qr().unwrap_err(), LinalgError::DimensionMismatch);

        let zero = DynMatrix::<f32>::zeros(3, 2);
        assert_eq!(zero.qr().unwrap_err(), LinalgError::Singular);

        let a = line_fit_matrix();
        let qr = a.qr().unwrap();
        assert_eq!(
            qr.solve(&DynVector::new(vec![1.0f32; 3])),
            Err(LinalgError::DimensionMismatch)
        );
        assert_eq!(
            qr.residual_norm(&DynVector::new(vec![1.0f32; 3])),
            Err(LinalgError::DimensionMismatch)
        );
    }

    #[test]
    fn agrees_with_fixed() {
        use azmath::linalg::{Matrix, Vector};
        let fixed = Matrix::new([[1.0f32, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]);
        let xf = fixed
            .least_squares(&Vector::new([1.0, 2.0, 2.0, 5.0]))
            .unwrap();
        let xd = line_fit_matrix()
            .least_squares(&DynVector::new(vec![1.0f32, 2.0, 2.0, 5.0]))
            .unwrap();
        for i in 0..2 {
            assert_eq!(xf[i], xd[i]);
        }
    }
}
