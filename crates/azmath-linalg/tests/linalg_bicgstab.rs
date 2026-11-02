//! Tests für BiCGSTAB und vorkonditioniertes BiCGSTAB (fest, matrixfrei und dynamisch).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath_linalg::iterative::{Identity, IterOptions, Jacobi, bicgstab, cg, pbicgstab};
use azmath_linalg::{FnOperator, LinalgError, Matrix, Vector};

fn opts() -> IterOptions<f64> {
    IterOptions::new(1e-12, 1000)
}

fn nonsym3() -> Matrix<f64, 3, 3> {
    Matrix::new([[4.0, 1.0, 0.0], [2.0, 5.0, 1.0], [0.0, 3.0, 6.0]])
}

/// 1D-Konvektions-Diffusions-Operator (nichtsymmetrisch): y_i = −(1+c)·x_{i-1} + 2·x_i − (1−c)·x_{i+1}.
///
/// Die Matrix ist ähnlich zu einer symmetrischen mit Wachstumsfaktor √((1+c)/(1−c)) pro Zeile; ihre
/// Konditionszahl wächst also exponentiell mit n. Deshalb nur kleine `c` und moderate n verwenden.
fn convection_diffusion(c: f64) -> impl Fn(&[f64], &mut [f64]) {
    move |x: &[f64], y: &mut [f64]| {
        let n = x.len();
        for i in 0..n {
            let left = if i > 0 { x[i - 1] } else { 0.0 };
            let right = if i + 1 < n { x[i + 1] } else { 0.0 };
            y[i] = -(1.0 + c) * left + 2.0 * x[i] - (1.0 - c) * right;
        }
    }
}

#[test]
fn nonsymmetric_system_matches_lu() {
    let a = nonsym3();
    let b = Vector::new([1.0, 2.0, 3.0]);
    let (x, res) = a.solve_bicgstab(&b, opts()).unwrap();
    let x_ref = a.solve(&b).unwrap();
    for i in 0..3 {
        assert!((x[i] - x_ref[i]).abs() < 1e-10, "{} vs {}", x[i], x_ref[i]);
    }
    assert!(res.iterations <= 6, "{} Iterationen", res.iterations);
    assert!(res.residual_norm < 1e-10);
}

#[test]
fn also_works_for_spd_and_agrees_with_cg() {
    let a = Matrix::new([[4.0f64, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]);
    let b = [1.0, 2.0, 3.0];
    let (mut x1, mut x2) = ([0.0; 3], [0.0; 3]);
    let mut w1 = [0.0; 12];
    let mut w2 = [0.0; 24];
    cg(&a, &b, &mut x1, opts(), &mut w1).unwrap();
    bicgstab(&a, &b, &mut x2, opts(), &mut w2).unwrap();
    for i in 0..3 {
        assert!((x1[i] - x2[i]).abs() < 1e-10);
    }
}

#[test]
fn matrix_free_convection_diffusion() {
    const N: usize = 100;
    let op = FnOperator::new(N, N, convection_diffusion(0.05));
    let x_true: [f64; N] = core::array::from_fn(|i| {
        let t = (i + 1) as f64 / (N + 1) as f64;
        t * (1.0 - t)
    });
    let mut b = [0.0; N];
    convection_diffusion(0.05)(&x_true, &mut b);

    let mut x = [0.0; N];
    let mut work = [0.0; 8 * N];
    let r = bicgstab(&op, &b, &mut x, opts(), &mut work).unwrap();
    // Residuum und Lösung
    let mut ax = [0.0; N];
    convection_diffusion(0.05)(&x, &mut ax);
    let res = ax
        .iter()
        .zip(&b)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(res < 1e-10, "Residuum {res:e}");
    for i in 0..N {
        assert!((x[i] - x_true[i]).abs() < 1e-7, "x[{i}]");
    }
    assert!(r.iterations <= 4 * N, "{} Iterationen", r.iterations);
}

#[test]
fn start_guess_is_used() {
    let a = nonsym3();
    let b = Vector::new([1.0, 2.0, 3.0]);
    let mut x = a.solve(&b).unwrap().into_array();
    let mut work = [0.0; 24];
    let r = bicgstab(&a, b.as_slice(), &mut x, opts(), &mut work).unwrap();
    assert_eq!(r.iterations, 0);
}

#[test]
fn zero_right_hand_side_gives_zero() {
    let a = nonsym3();
    let mut x = [5.0, 5.0, 5.0];
    let mut work = [0.0; 24];
    let r = bicgstab(&a, &[0.0; 3], &mut x, opts(), &mut work).unwrap();
    assert_eq!(x, [0.0; 3]);
    assert_eq!((r.iterations, r.residual_norm), (0, 0.0));
}

#[test]
fn preconditioner_identity_equals_plain() {
    let a = nonsym3();
    let b = [1.0, 2.0, 3.0];
    let (mut x1, mut x2) = ([0.0; 3], [0.0; 3]);
    let (mut w1, mut w2) = ([0.0; 24], [0.0; 24]);
    let r1 = bicgstab(&a, &b, &mut x1, opts(), &mut w1).unwrap();
    let r2 = pbicgstab(&a, &Identity, &b, &mut x2, opts(), &mut w2).unwrap();
    assert_eq!(x1, x2);
    assert_eq!(r1, r2);
}

#[test]
fn jacobi_helps_for_badly_scaled_nonsymmetric_matrix() {
    // A = S·B mit nichtsymmetrischem B (Zeilenskalierung) und stark unterschiedlichen Skalen
    let s = [1.0f64, 100.0, 1.0e4, 10.0, 1.0e3, 1.0];
    let a = Matrix::<f64, 6, 6>::from_fn(|i, j| {
        let basis = if i == j {
            5.0
        } else if j + 1 == i {
            2.0
        } else if i + 1 == j {
            -1.0
        } else {
            0.0
        };
        s[i] * basis
    });
    let b = Vector::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let o = IterOptions::new(1e-10, 5000);

    let (x_plain, plain) = a.solve_bicgstab(&b, o).unwrap();
    let jac = Jacobi::new(&a).unwrap();
    let (x_pre, pre) = a.solve_pbicgstab(&jac, &b, o).unwrap();

    assert!(
        pre.iterations <= plain.iterations,
        "Jacobi {} vs. ohne {}",
        pre.iterations,
        plain.iterations
    );
    let x_ref = a.solve(&b).unwrap();
    for i in 0..6 {
        assert!((x_pre[i] - x_ref[i]).abs() <= 1e-7 * x_ref[i].abs().max(1e-6));
        assert!((x_plain[i] - x_ref[i]).abs() <= 1e-5 * x_ref[i].abs().max(1e-6));
    }
}

#[test]
fn no_convergence_with_too_few_iterations() {
    const N: usize = 100;
    let op = FnOperator::new(N, N, convection_diffusion(0.3));
    let b = [1.0; N];
    let mut x = [0.0; N];
    let mut work = [0.0; 8 * N];
    let r = bicgstab(&op, &b, &mut x, IterOptions::new(1e-12, 3), &mut work);
    assert_eq!(r, Err(LinalgError::NoConvergence));
    assert!(x.iter().any(|&v| v != 0.0));
}

#[test]
fn breakdown_is_reported() {
    // Schiefsymmetrische Matrix: für r̂ = r = b gilt r̂ᵀ·A·r̂ = 0 -> Division durch 0
    let a = Matrix::new([[0.0f64, 1.0], [-1.0, 0.0]]);
    let r = a.solve_bicgstab(&Vector::new([1.0, 0.0]), opts());
    assert_eq!(r, Err(LinalgError::Breakdown));
}

#[test]
fn dimension_errors() {
    let a = nonsym3();
    let b = [1.0, 2.0, 3.0];
    let mut x = [0.0; 3];
    let mut short = [0.0; 23];
    assert_eq!(
        bicgstab(&a, &b, &mut x, opts(), &mut short),
        Err(LinalgError::DimensionMismatch)
    );
    let mut x2 = [0.0; 2];
    let mut work = [0.0; 24];
    assert_eq!(
        bicgstab(&a, &b, &mut x2, opts(), &mut work),
        Err(LinalgError::DimensionMismatch)
    );
    let rect = Matrix::new([[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(
        bicgstab(&rect, &b, &mut x, opts(), &mut work),
        Err(LinalgError::DimensionMismatch)
    );
}

#[test]
fn non_finite_input_is_rejected() {
    let a = nonsym3();
    let mut x = [0.0; 3];
    let mut work = [0.0; 24];
    assert_eq!(
        bicgstab(&a, &[1.0, f64::NAN, 0.0], &mut x, opts(), &mut work),
        Err(LinalgError::NotFinite)
    );
    let bad = Matrix::new([[f64::NAN, 0.0], [0.0, 1.0]]);
    let mut x2 = [0.0; 2];
    let mut w2 = [0.0; 16];
    assert_eq!(
        bicgstab(&bad, &[1.0, 1.0], &mut x2, opts(), &mut w2),
        Err(LinalgError::NotFinite)
    );
}

#[test]
fn f32_converges() {
    let a = Matrix::new([[4.0f32, 1.0, 0.0], [2.0, 5.0, 1.0], [0.0, 3.0, 6.0]]);
    let b = Vector::new([1.0f32, 2.0, 3.0]);
    let (x, res) = a.solve_bicgstab(&b, IterOptions::new(1e-5, 100)).unwrap();
    assert!((a * x - b).max_abs() < 1e-4);
    assert!(res.iterations <= 10);
}

#[test]
fn default_options_work() {
    let a = nonsym3();
    let b = Vector::new([1.0, 2.0, 3.0]);
    let (x, _) = a.solve_bicgstab(&b, IterOptions::default()).unwrap();
    assert!((a * x - b).max_abs() < 1e-6);
}

#[cfg(feature = "alloc")]
mod dynamic {
    extern crate alloc;
    use alloc::vec;
    use azmath_linalg::iterative::{DynJacobi, IterOptions, bicgstab};
    use azmath_linalg::{DynMatrix, DynVector, FnOperator, LinalgError};

    fn conv_diff_matrix(n: usize, c: f64) -> DynMatrix<f64> {
        let mut a = DynMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 2.0;
            if i > 0 {
                a[(i, i - 1)] = -(1.0 + c);
            }
            if i + 1 < n {
                a[(i, i + 1)] = -(1.0 - c);
            }
        }
        a
    }

    #[test]
    fn solve_bicgstab_matches_lu() {
        let a = conv_diff_matrix(40, 0.2);
        let b = DynVector::new((0..40).map(|i| (i % 7) as f64 - 2.0).collect());
        let (x, res) = a.solve_bicgstab(&b, IterOptions::new(1e-12, 2000)).unwrap();
        let x_ref = a.solve(&b).unwrap();
        for i in 0..40 {
            assert!((x[i] - x_ref[i]).abs() < 1e-7, "x[{i}]");
        }
        assert!(res.iterations <= 120);
    }

    #[test]
    fn large_matrix_free() {
        let n = 400;
        let c = 0.02;
        let op = FnOperator::new(n, n, |x: &[f64], y: &mut [f64]| {
            for i in 0..x.len() {
                let l = if i > 0 { x[i - 1] } else { 0.0 };
                let r = if i + 1 < x.len() { x[i + 1] } else { 0.0 };
                y[i] = -(1.0 + c) * l + 2.0 * x[i] - (1.0 - c) * r;
            }
        });
        let b = vec![1.0; n];
        let mut x = vec![0.0; n];
        let mut work = vec![0.0; 8 * n];
        let r = bicgstab(&op, &b, &mut x, IterOptions::new(1e-10, 5000), &mut work).unwrap();
        // Residuum prüfen (die Lösung selbst ist wegen der Kondition nur grob aussagekräftig)
        let mut ax = vec![0.0; n];
        use azmath_linalg::LinearOperator;
        op.apply(&x, &mut ax);
        let res = ax
            .iter()
            .zip(&b)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(res < 1e-8, "Residuum {res:e}");
        assert!(r.residual_norm < 1e-8 * (n as f64).sqrt() * 10.0);
        assert!(x.iter().all(|v| v.is_finite()) && x[n / 2] > 0.0);
    }

    #[test]
    fn dyn_jacobi_preconditioner() {
        let n = 8;
        let s: alloc::vec::Vec<f64> = (0..n).map(|i| 10f64.powi(i as i32 / 2)).collect();
        let mut a = DynMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 5.0 * s[i];
            if i > 0 {
                a[(i, i - 1)] = 2.0 * s[i];
            }
            if i + 1 < n {
                a[(i, i + 1)] = -s[i];
            }
        }
        let b = DynVector::new(vec![1.0; n]);
        let o = IterOptions::new(1e-10, 5000);
        let (_, plain) = a.solve_bicgstab(&b, o).unwrap();
        let (x, pre) = a
            .solve_pbicgstab(&DynJacobi::new(&a).unwrap(), &b, o)
            .unwrap();
        assert!(pre.iterations <= plain.iterations);
        let x_ref = a.solve(&b).unwrap();
        for i in 0..n {
            assert!((x[i] - x_ref[i]).abs() <= 1e-7 * x_ref[i].abs().max(1e-6));
        }
    }

    #[test]
    fn errors() {
        let a = conv_diff_matrix(4, 0.1);
        assert_eq!(
            a.solve_bicgstab(&DynVector::new(vec![1.0; 3]), IterOptions::default())
                .unwrap_err(),
            LinalgError::DimensionMismatch
        );
        let big = conv_diff_matrix(80, 0.3);
        assert_eq!(
            big.solve_bicgstab(&DynVector::new(vec![1.0; 80]), IterOptions::new(1e-12, 2))
                .unwrap_err(),
            LinalgError::NoConvergence
        );
        let skew = DynMatrix::new(2, 2, vec![0.0, 1.0, -1.0, 0.0]).unwrap();
        assert_eq!(
            skew.solve_bicgstab(&DynVector::new(vec![1.0, 0.0]), IterOptions::default())
                .unwrap_err(),
            LinalgError::Breakdown
        );
    }

    #[test]
    fn fixed_and_dynamic_agree() {
        use azmath_linalg::{Matrix, Vector};
        let fixed = Matrix::new([[4.0f64, 1.0, 0.0], [2.0, 5.0, 1.0], [0.0, 3.0, 6.0]]);
        let dynamic =
            DynMatrix::new(3, 3, vec![4.0, 1.0, 0.0, 2.0, 5.0, 1.0, 0.0, 3.0, 6.0]).unwrap();
        let o = IterOptions::new(1e-12, 100);
        let (xf, _) = fixed
            .solve_bicgstab(&Vector::new([1.0, 2.0, 3.0]), o)
            .unwrap();
        let (xd, _) = dynamic
            .solve_bicgstab(&DynVector::new(vec![1.0, 2.0, 3.0]), o)
            .unwrap();
        for i in 0..3 {
            assert_eq!(xf[i], xd[i]);
        }
    }
}
