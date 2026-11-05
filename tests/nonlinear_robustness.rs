//! Nullstellenerfolg wird unabhängig durch erneutes Einsetzen geprüft.
#![cfg(any(feature = "std", feature = "libm"))]
use azmath::optimize::*;

macro_rules! checks {
    ($module:ident, $t:ty, $large:expr, $tiny:expr) => {
        mod $module {
            use super::*;
            type T = $t;
            fn options() -> OptimizeOptions<T> {
                OptimizeOptions {
                    tol_x: T::EPSILON,
                    tol_f: 64.0 * T::EPSILON,
                    tol_g: T::EPSILON,
                    max_iter: 200,
                }
            }
            fn solve(
                which: usize,
                f: impl FnMut(&[T], &mut [T]),
                j: impl FnMut(&[T], &mut [T]),
                x: &mut [T],
                o: OptimizeOptions<T>,
            ) -> Result<OptimizeResult<T>, OptimizeError> {
                let mut w = vec![
                    T::NAN;
                    dogleg_work_len(x.len())
                        .max(broyden_work_len(x.len()))
                        .max(newton_work_len(x.len()))
                ];
                match which {
                    0 => newton_system(f, j, x, o, &mut w),
                    1 => newton_system_fd(f, x, o, &mut w),
                    2 => dogleg(f, j, x, o, &mut w),
                    3 => dogleg_fd(f, x, o, &mut w),
                    4 => broyden(f, j, x, o, &mut w),
                    _ => broyden_fd(f, x, o, &mut w),
                }
            }
            #[test]
            fn stationary_nonroot_is_not_success() {
                for k in 0..6 {
                    let f = |x: &[T], r: &mut [T]| r[0] = 1.0 + x[0] * x[0];
                    let j = |x: &[T], j: &mut [T]| j[0] = 2.0 * x[0];
                    let mut x = [0.0];
                    assert!(solve(k, f, j, &mut x, options()).is_err(), "solver {k}");
                }
            }
            #[test]
            fn tiny_step_with_large_residual_is_not_success() {
                for k in [0, 2, 4] {
                    let mut x = [1.0];
                    let f = |_: &[T], r: &mut [T]| r[0] = 1.0;
                    // Inkonsequente Jacobi-Matrix darf keinen falschen Erfolg erzwingen.
                    let j = |_: &[T], j: &mut [T]| j[0] = $large;
                    assert!(solve(k, f, j, &mut x, options()).is_err());
                }
            }
            #[test]
            fn uniform_large_and_small_residual_scales() {
                for scale in [$large, $tiny] {
                    for k in 0..6 {
                        let mut x = [0.0];
                        let mut o = options();
                        o.tol_f *= scale;
                        let f = |x: &[T], r: &mut [T]| r[0] = scale * (x[0] - 1.0);
                        let j = |_: &[T], j: &mut [T]| j[0] = scale;
                        let result = solve(k, f, j, &mut x, o)
                            .unwrap_or_else(|e| panic!("solver {k}, scale {scale}: {e:?}"));
                        assert_eq!(result.termination, Termination::Residual);
                        assert!((x[0] - 1.0).abs() < 128.0 * T::EPSILON);
                        assert!((scale * (x[0] - 1.0)).abs() <= o.tol_f);
                        assert!(result.residual_norm.is_finite());
                    }
                }
            }
            #[test]
            fn success_on_last_allowed_step_and_zero_budget() {
                for k in 0..6 {
                    let f = |x: &[T], r: &mut [T]| r[0] = x[0];
                    let j = |_: &[T], j: &mut [T]| j[0] = 1.0;
                    let mut o = options();
                    o.max_iter = 1;
                    let mut x = [1.0];
                    let r = solve(k, f, j, &mut x, o).unwrap();
                    assert_eq!(r.iterations, 1);
                    assert_eq!(r.termination, Termination::Residual);
                    o.max_iter = 0;
                    assert_eq!(solve(k, f, j, &mut x, o).unwrap().iterations, 0);
                    x = [1.0];
                    assert_eq!(solve(k, f, j, &mut x, o), Err(OptimizeError::NoConvergence));
                }
            }
            #[test]
            fn invalid_tolerances_are_rejected_before_callbacks() {
                for k in 0..6 {
                    for value in [-1.0, T::NAN, T::INFINITY] {
                        for field in 0..3 {
                            let mut o = options();
                            match field {
                                0 => o.tol_f = value,
                                1 => o.tol_x = value,
                                _ => o.tol_g = value,
                            }
                            let mut x = [0.0];
                            let result = solve(
                                k,
                                |_, _| panic!("invalid options reached residual"),
                                |_, _| panic!("invalid options reached Jacobian"),
                                &mut x,
                                o,
                            );
                            assert_eq!(result, Err(OptimizeError::InvalidInput));
                        }
                    }
                }
            }
            #[test]
            fn invalid_trials_preserve_last_accepted_point() {
                for k in [0, 2, 4] {
                    let mut x = [0.0];
                    let f = |x: &[T], r: &mut [T]| r[0] = if x[0] == 0.0 { 1.0 } else { T::NAN };
                    let j = |_: &[T], j: &mut [T]| j[0] = 1.0;
                    assert!(solve(k, f, j, &mut x, options()).is_err());
                    assert_eq!(x, [0.0]);
                }
            }
            #[test]
            fn nonfinite_jacobian_cannot_masquerade_as_stationarity() {
                for k in [0, 2, 4] {
                    let mut x = [0.0];
                    assert_eq!(
                        solve(
                            k,
                            |_, r| r[0] = 1.0,
                            |_, j| j[0] = T::NAN,
                            &mut x,
                            options()
                        ),
                        Err(OptimizeError::NotFinite)
                    );
                    assert_eq!(x, [0.0]);
                }
            }
            #[test]
            fn scaled_coupled_physical_equations_have_known_solution() {
                // p=a*T*rho, h=b*T+c*rho. Die Gleichungen werden durch ihre Zielgrößen skaliert.
                // a=2, b=3, c=5, T=4, rho=2 => p=16, h=22.
                for k in 0..6 {
                    let f = |x: &[T], r: &mut [T]| {
                        r[0] = (2.0 * x[0] * x[1] - 16.0) / 16.0;
                        r[1] = (3.0 * x[0] + 5.0 * x[1] - 22.0) / 22.0;
                    };
                    let j = |x: &[T], j: &mut [T]| {
                        j.copy_from_slice(&[
                            2.0 * x[1] / 16.0,
                            2.0 * x[0] / 16.0,
                            3.0 / 22.0,
                            5.0 / 22.0,
                        ])
                    };
                    let mut x = [4.5, 1.5];
                    let result = solve(k, f, j, &mut x, options())
                        .unwrap_or_else(|e| panic!("solver {k}: {e:?}"));
                    let mut r = [0.0; 2];
                    f(&x, &mut r);
                    let rn = r[0].hypot(r[1]);
                    assert!(rn <= options().tol_f);
                    assert!((rn - result.residual_norm).abs() <= T::EPSILON);
                    assert!(
                        (x[0] - 4.0).abs() < 1024.0 * T::EPSILON
                            && (x[1] - 2.0).abs() < 1024.0 * T::EPSILON
                    );
                }
            }
            #[test]
            fn nearly_singular_system_never_returns_an_unverified_root() {
                for k in 0..6 {
                    let d = 4.0 * T::EPSILON;
                    let f = |x: &[T], r: &mut [T]| {
                        r[0] = x[0] + x[1] - 2.0;
                        r[1] = x[0] + (1.0 + d) * x[1] - (2.0 + d);
                    };
                    let j = |_: &[T], j: &mut [T]| j.copy_from_slice(&[1.0, 1.0, 1.0, 1.0 + d]);
                    let mut x = [0.0; 2];
                    if let Ok(result) = solve(k, f, j, &mut x, options()) {
                        let mut r = [0.0; 2];
                        f(&x, &mut r);
                        assert!(r[0].hypot(r[1]) <= options().tol_f);
                        assert_eq!(result.termination, Termination::Residual);
                    }
                }
            }
            #[test]
            fn small_representable_step_is_evaluated_before_stopping() {
                for k in [0, 2, 4] {
                    let target = 1.0 + 2.0 * T::EPSILON;
                    let f = |x: &[T], r: &mut [T]| r[0] = (x[0] - target) / T::EPSILON;
                    let j = |_: &[T], j: &mut [T]| j[0] = 1.0 / T::EPSILON;
                    let mut o = options();
                    o.tol_x = 1e-4;
                    o.tol_f = 0.0;
                    let mut x = [1.0];
                    assert_eq!(
                        solve(k, f, j, &mut x, o).unwrap().termination,
                        Termination::Residual
                    );
                    assert_eq!(x, [target]);
                    x = [1.0];
                    let invalid =
                        |x: &[T], r: &mut [T]| r[0] = if x[0] == 1.0 { 2.0 } else { T::NAN };
                    assert!(solve(k, invalid, j, &mut x, o).is_err());
                    assert_eq!(x, [1.0]);
                }
            }
            #[test]
            fn broyden_inverse_handles_differently_scaled_rows() {
                let mut x = [0.0, 0.0];
                let f = |x: &[T], r: &mut [T]| {
                    r[0] = $large * (x[0] - 1.0);
                    r[1] = $tiny * (x[1] - 2.0);
                };
                let j = |_: &[T], j: &mut [T]| j.copy_from_slice(&[$large, 0.0, 0.0, $tiny]);
                let mut o = options();
                o.tol_f = 0.0;
                let r = solve(4, f, j, &mut x, o).unwrap();
                assert_eq!(x, [1.0, 2.0]);
                assert_eq!(r.residual_norm, 0.0);
            }
            #[test]
            fn numerical_jacobian_stays_finite_at_largest_coordinate() {
                let x = [T::MAX];
                let mut j = [0.0];
                let mut work = [T::NAN; 2];
                numerical_jacobian(
                    |x, r| {
                        assert!(x[0].is_finite());
                        r[0] = x[0];
                    },
                    &x,
                    &x,
                    &mut j,
                    &mut work,
                );
                assert!((j[0] - 1.0).abs() <= 8.0 * T::EPSILON);
            }
            #[test]
            fn numerical_jacobian_tries_other_side_at_domain_boundary() {
                let mut j = [0.0];
                let mut work = [T::NAN; 2];
                numerical_jacobian(
                    |x: &[T], r: &mut [T]| r[0] = if x[0] <= 0.0 { x[0] } else { T::NAN },
                    &[0.0],
                    &[0.0],
                    &mut j,
                    &mut work,
                );
                assert_eq!(j, [1.0]);
            }
        }
    };
}
checks!(double, f64, 1e200_f64, 1e-200_f64);
checks!(single, f32, 1e20_f32, 1e-20_f32);
