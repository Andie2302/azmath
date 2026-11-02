//! Tests für LinearOperator, CG und vorkonditioniertes CG (fest, matrixfrei und dynamisch).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath_linalg::iterative::{Identity, IterOptions, Jacobi, cg, pcg};
use azmath_linalg::{FnOperator, LinalgError, LinearOperator, Matrix, Vector};

fn opts() -> IterOptions<f64> {
    IterOptions::new(1e-12, 1000)
}

/// 1D-Poisson-Operator (tridiagonal 2, -1) matrixfrei: y = T·x.
fn poisson(x: &[f64], y: &mut [f64]) {
    let n = x.len();
    for i in 0..n {
        let left = if i > 0 { x[i - 1] } else { 0.0 };
        let right = if i + 1 < n { x[i + 1] } else { 0.0 };
        y[i] = 2.0 * x[i] - left - right;
    }
}

fn spd3() -> Matrix<f64, 3, 3> {
    Matrix::new([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
}

#[test]
fn matrix_is_a_linear_operator() {
    let a = Matrix::new([[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!((a.rows(), a.cols()), (2, 3));
    let mut y = [0.0; 2];
    a.apply(&[1.0, 0.0, -1.0], &mut y);
    assert_eq!(y, [-2.0, -2.0]);
    // auch über eine Referenz
    let r = &a;
    let mut y2 = [0.0; 2];
    r.apply(&[1.0, 1.0, 1.0], &mut y2);
    assert_eq!(y2, [6.0, 15.0]);
}

#[test]
fn fn_operator_applies_closure() {
    let op = FnOperator::new(4, 4, poisson);
    assert_eq!((op.rows(), op.cols()), (4, 4));
    let mut y = [0.0; 4];
    op.apply(&[1.0, 1.0, 1.0, 1.0], &mut y);
    assert_eq!(y, [1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn cg_matches_cholesky_on_small_system() {
    let a = spd3();
    let b = Vector::new([1.0, 2.0, 3.0]);
    let (x, res) = a.solve_cg(&b, opts()).unwrap();
    let x_ref = a.solve_spd(&b).unwrap();
    for i in 0..3 {
        assert!((x[i] - x_ref[i]).abs() < 1e-10, "{} vs {}", x[i], x_ref[i]);
    }
    // In exakter Arithmetik genügen n Schritte
    assert!(res.iterations <= 4, "{} Iterationen", res.iterations);
    assert!(res.residual_norm < 1e-10);
}

#[test]
fn cg_slice_api_with_work_buffer() {
    let a = spd3();
    let b = [1.0, 2.0, 3.0];
    let mut x = [0.0; 3];
    let mut work = [0.0; 12];
    let r = cg(&a, &b, &mut x, opts(), &mut work).unwrap();
    let res = (a * Vector::new(x) - Vector::new(b)).max_abs();
    assert!(res < 1e-10 && r.residual_norm < 1e-10);
}

#[test]
fn matrix_free_poisson_1d() {
    const N: usize = 50;
    let op = FnOperator::new(N, N, poisson);
    // exakte Lösung vorgeben: x_i = sin-ähnliche glatte Funktion
    let x_true: [f64; N] = core::array::from_fn(|i| {
        let t = (i + 1) as f64 / (N + 1) as f64;
        t * (1.0 - t)
    });
    let mut b = [0.0; N];
    poisson(&x_true, &mut b);

    let mut x = [0.0; N];
    let mut work = [0.0; 4 * N];
    let r = cg(&op, &b, &mut x, opts(), &mut work).unwrap();
    for i in 0..N {
        assert!((x[i] - x_true[i]).abs() < 1e-9, "x[{i}]");
    }
    // Residuum stimmt
    let mut ax = [0.0; N];
    poisson(&x, &mut ax);
    let res = ax
        .iter()
        .zip(&b)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(res < 1e-9);
    assert!(r.iterations <= 2 * N);
}

#[test]
fn start_guess_is_used() {
    let a = spd3();
    let b = Vector::new([1.0, 2.0, 3.0]);
    let exact = a.solve_spd(&b).unwrap();
    // Exakte Startnäherung: keine Iteration nötig
    let mut x = exact.into_array();
    let mut work = [0.0; 12];
    let r = cg(&a, b.as_slice(), &mut x, opts(), &mut work).unwrap();
    assert_eq!(r.iterations, 0);
}

#[test]
fn zero_right_hand_side_gives_zero() {
    let a = spd3();
    let mut x = [5.0, 5.0, 5.0];
    let mut work = [0.0; 12];
    let r = cg(&a, &[0.0; 3], &mut x, opts(), &mut work).unwrap();
    assert_eq!(x, [0.0; 3]);
    assert_eq!(r.iterations, 0);
    assert_eq!(r.residual_norm, 0.0);
}

#[test]
fn no_convergence_with_too_few_iterations() {
    const N: usize = 50;
    let op = FnOperator::new(N, N, poisson);
    let b = [1.0; N];
    let mut x = [0.0; N];
    let mut work = [0.0; 4 * N];
    let r = cg(&op, &b, &mut x, IterOptions::new(1e-12, 3), &mut work);
    assert_eq!(r, Err(LinalgError::NoConvergence));
    // x enthält die letzte Näherung (nicht mehr 0)
    assert!(x.iter().any(|&v| v != 0.0));
}

#[test]
fn jacobi_reduces_iterations_for_badly_scaled_matrix() {
    // A = S·B·S mit B = tridiagonal(4, 1) und stark unterschiedlichen Skalen s_i
    let s = [1.0f64, 10.0, 100.0, 1000.0, 10.0, 1.0];
    let a = Matrix::<f64, 6, 6>::from_fn(|i, j| {
        let b = if i == j {
            4.0
        } else if i.abs_diff(j) == 1 {
            1.0
        } else {
            0.0
        };
        s[i] * s[j] * b
    });
    let b = Vector::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let o = IterOptions::new(1e-10, 2000);

    let (x_plain, plain) = a.solve_cg(&b, o).unwrap();
    let jac = Jacobi::new(&a).unwrap();
    let (x_pre, pre) = a.solve_pcg(&jac, &b, o).unwrap();

    assert!(
        pre.iterations < plain.iterations,
        "Jacobi {} vs. ohne {}",
        pre.iterations,
        plain.iterations
    );
    let x_ref = a.solve_spd(&b).unwrap();
    for i in 0..6 {
        assert!((x_pre[i] - x_ref[i]).abs() <= 1e-8 * x_ref[i].abs().max(1e-3));
        assert!((x_plain[i] - x_ref[i]).abs() <= 1e-6 * x_ref[i].abs().max(1e-3));
    }
}

#[test]
fn pcg_with_identity_equals_cg() {
    let a = spd3();
    let b = [1.0, 2.0, 3.0];
    let (mut x1, mut x2) = ([0.0; 3], [0.0; 3]);
    let mut w1 = [0.0; 12];
    let mut w2 = [0.0; 12];
    let r1 = cg(&a, &b, &mut x1, opts(), &mut w1).unwrap();
    let r2 = pcg(&a, &Identity, &b, &mut x2, opts(), &mut w2).unwrap();
    assert_eq!(x1, x2);
    assert_eq!(r1, r2);
}

#[test]
fn jacobi_rejects_zero_diagonal() {
    let a = Matrix::new([[1.0f64, 2.0], [2.0, 0.0]]);
    assert_eq!(Jacobi::new(&a).unwrap_err(), LinalgError::Singular);
}

#[test]
fn indefinite_matrix_is_detected() {
    // Eigenwerte 3 und -1; b = (1, -1) liefert pᵀAp < 0
    let a = Matrix::new([[1.0f64, 2.0], [2.0, 1.0]]);
    let b = Vector::new([1.0, -1.0]);
    assert_eq!(
        a.solve_cg(&b, opts()),
        Err(LinalgError::NotPositiveDefinite)
    );
}

#[test]
fn dimension_errors() {
    let a = spd3();
    let b = [1.0, 2.0, 3.0];
    let mut x = [0.0; 3];
    // Arbeitsspeicher zu klein
    let mut short = [0.0; 11];
    assert_eq!(
        cg(&a, &b, &mut x, opts(), &mut short),
        Err(LinalgError::DimensionMismatch)
    );
    // falsche Länge von x
    let mut x2 = [0.0; 2];
    let mut work = [0.0; 12];
    assert_eq!(
        cg(&a, &b, &mut x2, opts(), &mut work),
        Err(LinalgError::DimensionMismatch)
    );
    // nicht quadratischer Operator
    let rect = Matrix::new([[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let mut work = [0.0; 12];
    assert_eq!(
        cg(&rect, &b, &mut x, opts(), &mut work),
        Err(LinalgError::DimensionMismatch)
    );
}

#[test]
fn non_finite_input_is_rejected() {
    let a = spd3();
    let mut x = [0.0; 3];
    let mut work = [0.0; 12];
    assert_eq!(
        cg(&a, &[1.0, f64::NAN, 0.0], &mut x, opts(), &mut work),
        Err(LinalgError::NotFinite)
    );
}

#[test]
fn default_options_are_usable() {
    let o = IterOptions::<f64>::default();
    assert!(o.tol > 0.0 && o.tol < 1e-6);
    assert_eq!(o.max_iter, 1000);
    let a = spd3();
    let (x, _) = a.solve_cg(&Vector::new([1.0, 2.0, 3.0]), o).unwrap();
    let r = (a * x - Vector::new([1.0, 2.0, 3.0])).max_abs();
    assert!(r < 1e-6);
}

#[test]
fn f32_converges() {
    let a = Matrix::new([[4.0f32, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]);
    let b = Vector::new([1.0f32, 2.0, 3.0]);
    let (x, res) = a.solve_cg(&b, IterOptions::new(1e-5, 100)).unwrap();
    assert!((a * x - b).max_abs() < 1e-4);
    assert!(res.iterations <= 10);
}

#[test]
fn vector_axpy() {
    let mut y = Vector::new([1.0f64, 2.0, 3.0]);
    y.axpy(2.0, &Vector::new([1.0, 1.0, 1.0]));
    assert_eq!(y, Vector::new([3.0, 4.0, 5.0]));
}

#[cfg(feature = "alloc")]
mod dynamic {
    extern crate alloc;
    use alloc::vec;
    use azmath_linalg::iterative::{DynJacobi, IterOptions};
    use azmath_linalg::{DynMatrix, DynVector, LinalgError};

    fn o() -> IterOptions<f64> {
        IterOptions::new(1e-12, 2000)
    }

    fn poisson_matrix(n: usize) -> DynMatrix<f64> {
        let mut a = DynMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 2.0;
            if i > 0 {
                a[(i, i - 1)] = -1.0;
            }
            if i + 1 < n {
                a[(i, i + 1)] = -1.0;
            }
        }
        a
    }

    #[test]
    fn solve_cg_matches_cholesky() {
        let a = poisson_matrix(30);
        let b = DynVector::new((0..30).map(|i| (i % 5) as f64 - 1.0).collect());
        let (x, res) = a.solve_cg(&b, o()).unwrap();
        let x_ref = a.solve_spd(&b).unwrap();
        for i in 0..30 {
            assert!((x[i] - x_ref[i]).abs() < 1e-8, "x[{i}]");
        }
        assert!(res.iterations <= 60);
    }

    #[test]
    fn large_matrix_free_with_dyn_vectors() {
        use azmath_linalg::FnOperator;
        use azmath_linalg::iterative::cg;
        let n = 500;
        let op = FnOperator::new(n, n, |x: &[f64], y: &mut [f64]| {
            for i in 0..x.len() {
                let l = if i > 0 { x[i - 1] } else { 0.0 };
                let r = if i + 1 < x.len() { x[i + 1] } else { 0.0 };
                y[i] = 2.0 * x[i] - l - r;
            }
        });
        let b = vec![1.0; n];
        let mut x = vec![0.0; n];
        let mut work = vec![0.0; 4 * n];
        let r = cg(&op, &b, &mut x, IterOptions::new(1e-10, 5000), &mut work).unwrap();
        // Lösung von -u'' = 1 mit u(0)=u(1)=0 ist parabelförmig: Maximum in der Mitte
        let mid = x[n / 2];
        assert!(mid > x[0] && mid > x[n - 1]);
        assert!(r.residual_norm < 1e-8 * (n as f64).sqrt() * 10.0);
    }

    #[test]
    fn dyn_jacobi_helps() {
        // stark unterschiedlich skalierte Diagonale
        let n = 8;
        let s: alloc::vec::Vec<f64> = (0..n).map(|i| 10f64.powi(i as i32 / 2)).collect();
        let mut a = DynMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let base = if i == j {
                    4.0
                } else if i.abs_diff(j) == 1 {
                    1.0
                } else {
                    0.0
                };
                a[(i, j)] = s[i] * s[j] * base;
            }
        }
        let b = DynVector::new(vec![1.0; n]);
        let opt = IterOptions::new(1e-10, 5000);
        let (_, plain) = a.solve_cg(&b, opt).unwrap();
        let jac = DynJacobi::new(&a).unwrap();
        let (x, pre) = a.solve_pcg(&jac, &b, opt).unwrap();
        assert!(pre.iterations < plain.iterations);
        let x_ref = a.solve_spd(&b).unwrap();
        for i in 0..n {
            assert!((x[i] - x_ref[i]).abs() <= 1e-8 * x_ref[i].abs().max(1e-6));
        }
    }

    #[test]
    fn errors_and_edge_cases() {
        let a = poisson_matrix(4);
        // falsche Länge von b
        assert_eq!(
            a.solve_cg(&DynVector::new(vec![1.0; 3]), o()).unwrap_err(),
            LinalgError::DimensionMismatch
        );
        // zu wenig Iterationen
        let big = poisson_matrix(60);
        assert_eq!(
            big.solve_cg(&DynVector::new(vec![1.0; 60]), IterOptions::new(1e-12, 2))
                .unwrap_err(),
            LinalgError::NoConvergence
        );
        // DynJacobi: nicht quadratisch / Nulldiagonale
        let rect = DynMatrix::new(2, 3, vec![1.0; 6]).unwrap();
        assert_eq!(
            DynJacobi::new(&rect).unwrap_err(),
            LinalgError::DimensionMismatch
        );
        let zero_diag = DynMatrix::new(2, 2, vec![0.0, 1.0, 1.0, 1.0]).unwrap();
        assert_eq!(
            DynJacobi::new(&zero_diag).unwrap_err(),
            LinalgError::Singular
        );
    }

    #[test]
    fn dyn_vector_axpy() {
        let mut y = DynVector::new(vec![1.0f64, 2.0]);
        y.axpy(3.0, &DynVector::new(vec![1.0, 1.0])).unwrap();
        assert_eq!(y.as_slice(), &[4.0, 5.0]);
        assert_eq!(
            y.axpy(1.0, &DynVector::new(vec![1.0])),
            Err(LinalgError::DimensionMismatch)
        );
    }

    #[test]
    fn dyn_matrix_is_operator() {
        use azmath_linalg::LinearOperator;
        let a = DynMatrix::new(2, 3, vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut y = [0.0; 2];
        a.apply(&[1.0, 0.0, -1.0], &mut y);
        assert_eq!(y, [-2.0, -2.0]);
    }
}

#[test]
fn vector_slice_accessors() {
    let mut v = Vector::new([1.0f64, 2.0, 3.0]);
    assert_eq!(v.as_slice(), &[1.0, 2.0, 3.0]);
    v.as_mut_slice()[1] = 9.0;
    assert_eq!(v.into_array(), [1.0, 9.0, 3.0]);
}
