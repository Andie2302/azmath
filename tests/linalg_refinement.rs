#![cfg(any(feature = "std", feature = "libm"))]
use azmath::linalg::{LinalgError, Matrix, RefinementOptions, RefinementTermination, Vector};

#[test]
fn refines_perturbed_solution_with_row_pivots() {
    let a = Matrix::new([[0.0f64, 2.0, 1.0], [3.0, -1.0, 2.0], [1.0, 4.0, -3.0]]);
    let target = Vector::new([1.0, -2.0, 3.0]);
    let b = a * target;
    let lu = a.lu().unwrap();
    let mut x = Vector::new([1.1, -2.1, 2.9]);
    let report = lu.refine(&b, &mut x, RefinementOptions::default()).unwrap();
    assert_eq!(report.termination, RefinementTermination::Tolerance);
    assert!(report.iterations > 0);
    assert!(report.backward_error < report.initial_backward_error * 1e-10);
    for i in 0..3 {
        assert!((x[i] - target[i]).abs() < 1e-13);
    }
}

#[test]
fn actual_single_precision_solve_improves_using_compensated_residual() {
    let a = Matrix::new([[3.0f32, 1.0, 7.0], [2.0, -8.0, 1.0], [5.0, 3.0, -2.0]]);
    let b = Vector::new([1.0f32, 2.0, -3.0]);
    let lu = a.lu().unwrap();
    let initial = lu.solve(&b);
    let mut x = initial;
    let report = lu
        .refine(
            &b,
            &mut x,
            RefinementOptions {
                tolerance: 0.0,
                max_iter: 8,
            },
        )
        .unwrap();
    let error = |x: Vector<f32, 3>| {
        (0..3)
            .map(|i| {
                let r = b[i] as f64 - (0..3).map(|j| a[(i, j)] as f64 * x[j] as f64).sum::<f64>();
                r.abs()
            })
            .fold(0.0f64, f64::max)
    };
    assert!(
        error(x) < error(initial),
        "before {} after {}, {report:?}",
        error(initial),
        error(x)
    );
    assert!(report.backward_error < report.initial_backward_error);
}

#[test]
fn reciprocal_condition_matches_diagonal_and_is_scale_invariant() {
    for scale in [1e-200f64, 1.0, 1e200] {
        let a = Matrix::new([
            [scale, 0.0, 0.0],
            [0.0, 4.0 * scale, 0.0],
            [0.0, 0.0, 10.0 * scale],
        ]);
        assert!((a.lu().unwrap().rcond_inf().unwrap() - 0.1).abs() < 1e-14);
        let a = Matrix::new([[0.0, scale], [2.0 * scale, 0.0]]);
        assert!((a.lu().unwrap().rcond_inf().unwrap() - 0.5).abs() < 1e-14);
    }
    for v in [f64::MAX, f64::MIN_POSITIVE, f64::from_bits(1)] {
        assert_eq!(Matrix::new([[v]]).lu().unwrap().rcond_inf().unwrap(), 1.0);
    }
}

#[test]
fn estimate_tracks_explicit_inverse_for_dense_nonsymmetric_matrices() {
    for k in 1..20 {
        let t = k as f64;
        let a = Matrix::new([[0.01 * t, 2.0, -3.0], [4.0, -t, 1.0], [2.0, 0.5, t + 1.0]]);
        let lu = a.lu().unwrap();
        let reference = 1.0 / (a.norm_inf() * lu.inverse().norm_inf());
        let estimated = lu.rcond_inf().unwrap();
        assert!(
            estimated >= reference * 0.99 && estimated <= reference * 5.0,
            "{estimated} vs {reference}"
        );
    }
    let bad = Matrix::<f64, 6, 6>::from_fn(|i, j| 1.0 / ((i + j + 1) as f64));
    assert!(bad.lu().unwrap().rcond_inf().unwrap() < 1e-6);
}

#[test]
fn stopping_statuses_validation_and_empty_system() {
    let lu = Matrix::new([[2.0f64]]).lu().unwrap();
    let b = Vector::new([2.0]);
    let mut x = Vector::new([0.0]);
    let r = lu
        .refine(
            &b,
            &mut x,
            RefinementOptions {
                tolerance: 0.0,
                max_iter: 0,
            },
        )
        .unwrap();
    assert_eq!(r.termination, RefinementTermination::MaxIterations);
    assert_eq!(x[0], 0.0);
    for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            lu.refine(
                &b,
                &mut x,
                RefinementOptions {
                    tolerance,
                    max_iter: 5
                }
            ),
            Err(LinalgError::InvalidInput)
        );
    }
    assert_eq!(
        lu.solve_refined(&Vector::new([f64::NAN]), RefinementOptions::default()),
        Err(LinalgError::NotFinite)
    );
    let lu = Matrix::<f64, 0, 0>::zeros().lu().unwrap();
    assert_eq!(lu.rcond_inf().unwrap(), 1.0);
    let (_, r) = lu
        .solve_refined(&Vector::zeros(), RefinementOptions::default())
        .unwrap();
    assert_eq!(r.backward_error, 0.0);
    assert_eq!(r.termination, RefinementTermination::Tolerance);
}

#[cfg(feature = "alloc")]
#[test]
fn dynamic_and_fixed_match_and_dimensions_are_checked() {
    use azmath::linalg::{DynMatrix, DynVector};
    let a = Matrix::new([[0.0f64, 2.0], [3.0, 1.0]]);
    let b = Vector::new([4.0, 5.0]);
    let fixed = a.lu().unwrap();
    let d = DynMatrix::new(2, 2, a.as_array().as_flattened().to_vec())
        .unwrap()
        .lu()
        .unwrap();
    assert_eq!(fixed.rcond_inf(), d.rcond_inf());
    let (x, r) = fixed
        .solve_refined(&b, RefinementOptions::default())
        .unwrap();
    let (y, s) = d
        .solve_refined(
            &DynVector::new(b.as_array().to_vec()),
            RefinementOptions::default(),
        )
        .unwrap();
    assert_eq!(x.as_array(), y.as_slice());
    assert_eq!(r, s);
    assert_eq!(
        d.refine(
            &DynVector::zeros(1),
            &mut DynVector::zeros(2),
            RefinementOptions::default()
        ),
        Err(LinalgError::DimensionMismatch)
    );
}

#[test]
fn compensated_residual_detects_rounding_hidden_by_an_ordinary_dot_product() {
    let lu = Matrix::new([[3.0f64]]).lu().unwrap();
    let (x, r) = lu
        .solve_refined(
            &Vector::new([1.0]),
            RefinementOptions {
                tolerance: 0.0,
                max_iter: 5,
            },
        )
        .unwrap();
    assert_eq!(1.0 - 3.0 * x[0], 0.0); // ordinary arithmetic hides the residual
    assert!(r.backward_error > 0.0);
    assert_eq!(r.termination, RefinementTermination::Stagnation);
    assert_eq!(r.iterations, 0);
}

#[test]
fn refinement_and_condition_work_for_extreme_global_scales() {
    for scale in [1e-200f64, 1.0, 1e200] {
        let a = Matrix::new([[0.0, 2.0 * scale], [3.0 * scale, scale]]);
        let b = Vector::new([4.0 * scale, 5.0 * scale]);
        let mut x = Vector::new([0.9, 2.1]);
        let r = a
            .lu()
            .unwrap()
            .refine(&b, &mut x, RefinementOptions::default())
            .unwrap();
        assert!(r.backward_error <= 8.0 * f64::EPSILON);
        assert!((x[0] - 1.0).abs() < 1e-14 && (x[1] - 2.0).abs() < 1e-14);
    }
    for scale in [1e-20f32, 1.0, 1e20] {
        let a = Matrix::new([[0.0, 2.0 * scale], [3.0 * scale, scale]]);
        let b = Vector::new([4.0 * scale, 5.0 * scale]);
        let mut x = Vector::new([0.9, 2.1]);
        let lu = a.lu().unwrap();
        let r = lu.refine(&b, &mut x, RefinementOptions::default()).unwrap();
        assert!(r.backward_error <= 8.0 * f32::EPSILON);
        assert!((x[0] - 1.0).abs() < 1e-5 && (x[1] - 2.0).abs() < 1e-5);
        assert!((lu.rcond_inf().unwrap() - 0.5).abs() < 1e-5);
    }
}
