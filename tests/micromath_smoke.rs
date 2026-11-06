//! Kleine gültige Probleme mit bewusst groben Erwartungen und Iterationsgrenzen.
#![cfg(all(feature = "micromath", not(any(feature = "std", feature = "libm"))))]
use azmath::{
    Real,
    linalg::{
        LinalgError, Matrix,
        iterative::{IterOptions, cg, gmres, gmres_work_len},
    },
};
#[test]
fn elementary_valid_domain_is_finite() {
    for x in [0.1f32, 0.5, 1.0, 2.0] {
        for v in [
            Real::sqrt(x),
            Real::ln(x),
            Real::exp(x),
            Real::sin(x),
            Real::cos(x),
        ] {
            assert!(v.is_finite());
        }
        assert!((Real::sqrt(x) - x.sqrt()).abs() < 0.15);
    }
}
#[test]
fn bounded_iterative_solvers_produce_finite_approximations() {
    let a = Matrix::new([[4.0f32, 1.0], [1.0, 3.0]]);
    let b = [1.0f32, 2.0];
    let mut x = [0.0; 2];
    let opts = IterOptions::new(0.2, 30);
    let result = cg(&a, &b, &mut x, opts, &mut [0.0; 8]);
    assert!(result.is_ok(), "{result:?}");
    assert!(x.iter().all(|v| v.is_finite()));
    assert!((4.0 * x[0] + x[1] - 1.0).abs() < 0.5);
    assert!((x[0] + 3.0 * x[1] - 2.0).abs() < 0.5);
    let mut x = [0.0; 2];
    let result = gmres(&a, &b, &mut x, 2, opts, &mut [0.0; gmres_work_len(2, 2)]);
    assert!(result.is_ok(), "{result:?}");
    assert!(x.iter().all(|v| v.is_finite()));
}
#[test]
fn invalid_tolerance_and_iteration_limit_are_checked() {
    let a = Matrix::<f32, 2, 2>::identity();
    let mut x = [0.0; 2];
    assert_eq!(
        cg(
            &a,
            &[1.0, 2.0],
            &mut x,
            IterOptions::new(f32::NAN, 4),
            &mut [0.0; 8]
        ),
        Err(LinalgError::InvalidInput)
    );
    assert_eq!(
        cg(
            &a,
            &[1.0, 2.0],
            &mut x,
            IterOptions::new(0.01, 0),
            &mut [0.0; 8]
        ),
        Err(LinalgError::NoConvergence)
    );
}
