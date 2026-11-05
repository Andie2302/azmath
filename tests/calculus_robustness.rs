//! Analytische Gegenproben und Randfälle für Integration/Differentiation.
#![cfg(any(feature = "std", feature = "libm"))]
use azmath::{diff::*, integrate::*};

macro_rules! checks {
    ($module:ident,$t:ty) => {
        mod $module {
            use super::*;
            #[test]
            fn finite_integrals_do_not_overflow_in_weighted_sums() {
                let value = <$t>::MAX;
                let exact = value * 0.25;
                let tol = 256.0 * <$t>::EPSILON;
                for result in [
                    trapezoid(|_: $t| value, 0.0, 0.25, 4),
                    simpson(|_: $t| value, 0.0, 0.25, 4),
                    gauss_legendre(|_: $t| value, 0.0, 0.25, 5, 2),
                    trapezoid_samples(&[value; 5], 0.0625),
                    simpson_samples(&[value; 5], 0.0625),
                    trapezoid_xy(&[0.0, 0.125, 0.25], &[value; 3]),
                ] {
                    let r = result.unwrap();
                    assert!((r / exact - 1.0).abs() < tol, "{r} vs {exact}");
                }
                let r = adaptive(
                    |_: $t| value,
                    0.0,
                    0.25,
                    IntegrateOptions::new(0.0, 1e-4 as $t),
                )
                .unwrap();
                assert!((r.value / exact - 1.0).abs() < tol);
                assert!(r.error_estimate.is_finite());
            }
            #[test]
            fn extreme_coordinates_never_reach_callbacks_as_infinity() {
                let max = <$t>::MAX;
                let small = 1.0 / max;
                let f = |x: $t| {
                    assert!(x.is_finite());
                    small
                };
                for r in [
                    trapezoid(f, -max, max, 4),
                    simpson(f, -max, max, 4),
                    gauss_legendre(f, -max, max, 3, 2),
                ] {
                    assert!((r.unwrap() - 2.0).abs() < 512.0 * <$t>::EPSILON);
                }
                let r =
                    adaptive(f, -max, max, IntegrateOptions::new(1e-4 as $t, 1e-4 as $t)).unwrap();
                assert!((r.value - 2.0).abs() < 512.0 * <$t>::EPSILON);
            }
            #[test]
            fn adaptive_obeys_initial_evaluation_budget_and_options() {
                let empty_options = IntegrateOptions {
                    max_evals: 0,
                    ..IntegrateOptions::<$t>::default()
                };
                assert_eq!(
                    adaptive(|_: $t| panic!("leeres Intervall"), 1.0, 1.0, empty_options)
                        .unwrap()
                        .evaluations,
                    0
                );
                assert_eq!(
                    adaptive(
                        |_: $t| panic!("keine Kapazität"),
                        0.0,
                        1.0,
                        IntegrateOptions {
                            max_intervals: 0,
                            ..empty_options
                        }
                    ),
                    Err(IntegrateError::InvalidInput)
                );
                assert_eq!(
                    trapezoid(|_: $t| panic!("leeres Intervall"), 1.0, 1.0, 1),
                    Ok(0.0)
                );
                assert_eq!(
                    simpson(|_: $t| panic!("leeres Intervall"), 1.0, 1.0, 2),
                    Ok(0.0)
                );
                assert_eq!(
                    gauss_legendre(|_: $t| panic!("leeres Intervall"), 1.0, 1.0, 3, 1),
                    Ok(0.0)
                );
                for budget in [0, 1, 14] {
                    let opts = IntegrateOptions {
                        max_evals: budget,
                        ..IntegrateOptions::<$t>::default()
                    };
                    assert_eq!(
                        adaptive(|_: $t| panic!("Budget vor Callback prüfen"), 0.0, 1.0, opts),
                        Err(IntegrateError::NoConvergence)
                    );
                }
                for bad in [-1.0 as $t, <$t>::NAN, <$t>::INFINITY] {
                    for opts in [
                        IntegrateOptions::new(bad, 0.0),
                        IntegrateOptions::new(0.0, bad),
                    ] {
                        assert_eq!(
                            adaptive(
                                |_: $t| panic!("Optionen vor Callback prüfen"),
                                0.0,
                                1.0,
                                opts
                            ),
                            Err(IntegrateError::InvalidInput)
                        );
                    }
                }
            }
            #[test]
            fn adaptive_does_not_succeed_above_its_requested_error() {
                let opts = IntegrateOptions::new(<$t>::EPSILON * <$t>::EPSILON, 0.0);
                if let Ok(r) = adaptive(|x: $t| x.exp(), 0.0, 1.0, opts) {
                    assert!(r.error_estimate <= opts.abs_tol);
                }
            }
            #[test]
            fn collapsed_difference_points_are_not_zero_derivatives() {
                let h = <$t>::MIN_POSITIVE;
                let f = |_: $t| -> $t { panic!("zusammengefallene Punkte nicht auswerten") };
                assert!(central_difference(f, 1.0, h).is_nan());
                assert!(forward_difference(f, 1.0, h).is_nan());
                assert!(backward_difference(f, 1.0, h).is_nan());
                let (d, e) = derivative_ridders(f, 1.0, h);
                assert!(d.is_nan() && e.is_infinite());
            }
            #[test]
            fn constant_large_function_has_zero_derivatives() {
                let f = |_: $t| <$t>::MAX;
                assert_eq!(central_difference4(f, 0.0, 0.1), 0.0);
                assert_eq!(second_derivative(f, 0.0, 0.1), 0.0);
                assert_eq!(second_derivative(|x: $t| x, 0.0, <$t>::MIN_POSITIVE), 0.0);
                let mut h = [0.0; 4];
                let mut w = [0.0; 2];
                hessian(|_: &[$t]| <$t>::MAX, &[0.0, 0.0], &mut h, &mut w);
                assert_eq!(h, [0.0; 4]);
            }
            #[test]
            fn central_difference_handles_large_numerator_and_denominator() {
                let h = <$t>::MAX * 0.75;
                assert!((central_difference(|x: $t| x, 0.0, h) - 1.0).abs() < 16.0 * <$t>::EPSILON);
            }
            #[test]
            fn actual_grid_spacing_is_used_after_rounding() {
                let x = 2.0 / <$t>::EPSILON;
                let h = 3.0;
                let tol = 32.0 * <$t>::EPSILON;
                for d in [
                    forward_difference(|t: $t| t, x, h),
                    backward_difference(|t: $t| t, x, h),
                    central_difference(|t: $t| t, x, h),
                    central_difference4(|t: $t| t, x, h),
                ] {
                    assert!((d - 1.0).abs() <= tol);
                }
                let d = second_derivative(|t: $t| (t - x) * (t - x), x, h);
                assert!((d - 2.0).abs() <= tol);
            }

            #[test]
            fn ridders_refines_large_derivatives_without_an_absolute_error_cap() {
                let scale = <$t>::MAX * 0.001;
                let (d, error) = derivative_ridders(|x: $t| scale * x.exp(), 0.0, 0.2);
                assert!(d.is_finite() && error.is_finite() && error >= 0.0);
                assert!((d / scale - 1.0).abs() < 512.0 * <$t>::EPSILON);
                assert!((d - scale).abs() <= 16.0 * error + 128.0 * <$t>::EPSILON * scale);
                let mut out = [0.0];
                let mut work = [0.0; 3];
                gradient(
                    |_: &[$t]| panic!("keine nichtendlichen Stützpunkte"),
                    &[<$t>::MAX],
                    &mut out,
                    &mut work,
                );
                assert!(out[0].is_nan());
            }

            #[test]
            fn nonfinite_integrands_are_not_silently_ignored() {
                assert_eq!(
                    tanh_sinh(|_: $t| <$t>::NAN, 0.0, 1.0, 1e-4 as $t),
                    Err(IntegrateError::NotFinite)
                );
            }
        }
    };
}
checks!(double, f64);
checks!(single, f32);
