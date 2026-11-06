#![cfg(any(feature = "std", feature = "libm"))]
use azmath::optimize::{
    OptimizeError, OptimizeOptions, gauss_newton, gauss_newton_fd, gauss_newton_work_len,
};

#[test]
fn full_rank_step_unpermutes_a_three_column_cycle() {
    let residual = |x: &[f64], r: &mut [f64]| {
        r[0] = x[0] - 2.0;
        r[1] = 3.0 * (x[1] + 1.0);
        r[2] = 2.0 * (x[2] - 4.0);
        r[3] = 1.0; // irreducible residual, zero gradient at the optimum
    };
    let jacobian = |_x: &[f64], j: &mut [f64]| {
        j.copy_from_slice(&[1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]);
    };
    let mut x = [0.0; 3];
    let mut permutation = [usize::MAX; 4];
    let mut work = [f64::NAN; gauss_newton_work_len(4, 3)];
    let result = gauss_newton(
        residual,
        jacobian,
        &mut x,
        4,
        OptimizeOptions::default(),
        &mut work,
        &mut permutation,
    )
    .unwrap();
    assert_eq!(x, [2.0, -1.0, 4.0]);
    assert_eq!(result.residual_norm, 1.0);
    assert_eq!(permutation, [1, 2, 0, usize::MAX]);
}

#[test]
fn redundant_nonlinear_parameters_converge_with_analytic_and_fd_jacobians() {
    // Only x0 + 2*x1 is identifiable; x2 is completely inactive.
    let residual = |x: &[f64], r: &mut [f64]| {
        let v = x[0] + 2.0 * x[1];
        r[0] = v * v - 9.0;
        r[1] = 2.0 * (v * v - 9.0);
        r[2] = 0.0;
    };
    let jacobian = |x: &[f64], j: &mut [f64]| {
        let v = x[0] + 2.0 * x[1];
        j.copy_from_slice(&[2.0 * v, 4.0 * v, 0.0, 4.0 * v, 8.0 * v, 0.0, 0.0, 0.0, 0.0]);
    };
    let mut work = [0.0; gauss_newton_work_len(3, 3)];
    let mut permutation = [usize::MAX; 3];
    for fd in [false, true] {
        let mut x = [1.0, 0.0, 7.0];
        let result = if fd {
            gauss_newton_fd(
                residual,
                &mut x,
                3,
                OptimizeOptions::default(),
                &mut work,
                &mut permutation,
            )
        } else {
            gauss_newton(
                residual,
                jacobian,
                &mut x,
                3,
                OptimizeOptions::default(),
                &mut work,
                &mut permutation,
            )
        }
        .unwrap();
        assert!((x[0] + 2.0 * x[1] - 3.0).abs() < 1e-8, "{x:?}");
        assert_eq!(x[2], 7.0);
        assert!(result.residual_norm < 1e-8);
    }
}

#[test]
fn nearly_dependent_columns_recover_identifiable_parameters() {
    let eps = 1e-8;
    let residual = |x: &[f64], r: &mut [f64]| {
        r[0] = x[0] + x[1] - 2.0;
        r[1] = eps * (x[1] - 2.0);
    };
    let jacobian = |_x: &[f64], j: &mut [f64]| j.copy_from_slice(&[1.0, 1.0, 0.0, eps]);
    let mut work = [0.0; gauss_newton_work_len(2, 2)];
    let mut x = [0.0, 0.0];
    let options = OptimizeOptions {
        tol_f: 1e-20,
        tol_g: 0.0,
        tol_x: 0.0,
        max_iter: 20,
    };
    gauss_newton(
        residual,
        jacobian,
        &mut x,
        2,
        options,
        &mut work,
        &mut [0; 2],
    )
    .unwrap();
    assert!(x[0].abs() < 1e-8 && (x[1] - 2.0).abs() < 1e-8, "{x:?}");
}

#[test]
fn insufficient_index_workspace_is_rejected_before_callbacks() {
    let residual = |_x: &[f64], _r: &mut [f64]| panic!("must validate workspace first");
    let jacobian = |_x: &[f64], _j: &mut [f64]| panic!("must validate workspace first");
    let mut work = [0.0; gauss_newton_work_len(2, 2)];
    assert_eq!(
        gauss_newton(
            residual,
            jacobian,
            &mut [1.0, 2.0],
            2,
            OptimizeOptions::default(),
            &mut work,
            &mut [0; 1]
        ),
        Err(OptimizeError::InvalidInput)
    );
    assert_eq!(
        gauss_newton_fd(
            residual,
            &mut [1.0, 2.0],
            2,
            OptimizeOptions::default(),
            &mut work,
            &mut []
        ),
        Err(OptimizeError::InvalidInput)
    );
}

#[test]
fn rank_deficient_f32_problem_uses_the_identifiable_column() {
    let residual = |x: &[f32], r: &mut [f32]| {
        r[0] = x[0] + 2.0 * x[1] - 3.0;
        r[1] = 2.0 * r[0];
    };
    let jacobian = |_x: &[f32], j: &mut [f32]| j.copy_from_slice(&[1.0, 2.0, 2.0, 4.0]);
    let mut work = [0.0; gauss_newton_work_len(2, 2)];
    let mut x = [0.0, 0.0];
    gauss_newton(
        residual,
        jacobian,
        &mut x,
        2,
        OptimizeOptions::default(),
        &mut work,
        &mut [0; 2],
    )
    .unwrap();
    assert!(x[0].abs() < 1e-5 && (x[1] - 1.5).abs() < 1e-5);
}
