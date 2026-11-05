//! Analytische Gegenproben und Randfälle für Interpolation und Splines.
#![cfg(any(feature = "std", feature = "libm"))]
use azmath::interp::*;
macro_rules! checks {
    ($module:ident,$t:ty,$large:expr,$small:expr) => {
        mod $module {
            use super::*;
            #[test]
            fn linear_and_neville_handle_extreme_coordinates_and_values() {
                let max = <$t>::MAX;
                let x = [-max, max];
                let y = x;
                let mut w = [0.0; 2];
                assert_eq!(linear(&x, &y, 0.0), Ok(0.0));
                assert_eq!(lagrange(&x, &y, 0.0, &mut w), Ok(0.0));
                assert_eq!(linear(&[0.0, 1.0], &y, 0.5), Ok(0.0));
                assert_eq!(lagrange(&[0.0, 1.0], &y, 0.5, &mut w), Ok(0.0));
                assert_eq!(linear(&x, &y, max), Ok(max));
            }
            #[test]
            fn invalid_queries_never_return_successful_nonfinite_results() {
                for q in [<$t>::NAN, <$t>::INFINITY, <$t>::NEG_INFINITY] {
                    assert_eq!(
                        linear(&[0.0, 1.0], &[0.0, 1.0], q),
                        Err(InterpError::NotFinite)
                    );
                    assert_eq!(
                        lagrange(&[0.0, 1.0], &[0.0, 1.0], q, &mut [0.0; 2]),
                        Err(InterpError::NotFinite)
                    );
                }
                assert_eq!(
                    linear(&[0.0, 1.0], &[0.0, <$t>::MAX], 2.0),
                    Err(InterpError::NotFinite)
                );
            }
            #[test]
            fn pchip_reproduces_lines_with_tiny_or_large_widths() {
                for h in [$small, $large, <$t>::MAX * 0.25] {
                    let x = [0.0, h, 2.0 * h, 3.0 * h];
                    let y = x;
                    let mut d = [0.0; 4];
                    let p = Pchip::new(&x, &y, &mut d).unwrap();
                    for i in 0..4 {
                        assert_eq!(p.value(x[i]), y[i]);
                        assert!((p.slopes()[i] - 1.0).abs() < 128.0 * <$t>::EPSILON);
                    }
                    for i in 0..=24 {
                        let t = i as $t / 8.0;
                        let q = t * h;
                        assert!((p.value(q) / h - t).abs() < 256.0 * <$t>::EPSILON);
                        assert!((p.derivative(q) - 1.0).abs() < 256.0 * <$t>::EPSILON);
                    }
                }
            }
            #[test]
            fn pchip_harmonic_mean_does_not_erase_small_slopes() {
                let slope = <$t>::MIN_POSITIVE;
                let x = [0.0, 1.0, 2.0, 3.0];
                let y = x.map(|x| x * slope);
                let mut d = [0.0; 4];
                let p = Pchip::new(&x, &y, &mut d).unwrap();
                for d in p.slopes() {
                    assert!((*d / slope - 1.0).abs() < 128.0 * <$t>::EPSILON);
                }
                let mut last = 0.0;
                for i in 0..=60 {
                    let t = i as $t / 20.0;
                    let v = p.value(t);
                    assert!(v >= last);
                    assert!(v >= 0.0 && v <= 3.0 * slope);
                    last = v;
                }
            }
            #[test]
            fn cubic_reproduces_scaled_quadratics_and_boundary_slopes() {
                for h in [$small, $large] {
                    let x = [0.0, h, 2.0 * h, 3.0 * h];
                    let y = [0.0, h, 4.0 * h, 9.0 * h];
                    for bc in [
                        BoundaryCondition::NotAKnot,
                        BoundaryCondition::Clamped {
                            left: 0.0,
                            right: 6.0,
                        },
                    ] {
                        let mut m = [0.0; 4];
                        let mut work = [0.0; cubic_spline_work_len(4)];
                        let s = CubicSpline::new(&x, &y, bc, &mut m, &mut work).unwrap();
                        for i in 0..=12 {
                            let t = i as $t / 4.0;
                            let q = t * h;
                            assert!((s.value(q) / h - t * t).abs() < 1024.0 * <$t>::EPSILON);
                            assert!((s.derivative(q) - 2.0 * t).abs() < 1024.0 * <$t>::EPSILON);
                            assert!(
                                (s.second_derivative(q) * h - 2.0).abs() < 1024.0 * <$t>::EPSILON
                            );
                        }
                    }
                }
            }
            #[test]
            fn periodic_minimal_spline_has_known_curvature_at_all_scales() {
                for h in [1.0 as $t, $small, $large] {
                    let x = [0.0, h, 2.0 * h];
                    let y = [0.0, h, 0.0];
                    let mut m = [0.0; 3];
                    let mut work = [0.0; cubic_spline_work_len(3)];
                    let s =
                        CubicSpline::new(&x, &y, BoundaryCondition::Periodic, &mut m, &mut work)
                            .unwrap();
                    let tol = 512.0 * <$t>::EPSILON;
                    assert!((s.second_derivative(0.0) * h - 6.0).abs() < tol);
                    assert!((s.second_derivative(h) * h + 6.0).abs() < tol);
                    assert!(s.derivative(0.0).abs() < tol && s.derivative(2.0 * h).abs() < tol);
                    assert!((s.value(0.5 * h) / h - 0.5).abs() < tol);
                }
            }
            #[test]
            fn integrals_of_short_intervals_avoid_antiderivative_cancellation() {
                let x = [0.0, 1.0];
                let value = <$t>::MAX * 0.125;
                let y = [value, value];
                let a = 0.75;
                let b = a + <$t>::EPSILON;
                let expected = value * (b - a);
                let mut d = [0.0; 2];
                let p = Pchip::new(&x, &y, &mut d).unwrap();
                let mut m = [0.0; 2];
                let mut w = [0.0; cubic_spline_work_len(2)];
                let s =
                    CubicSpline::new(&x, &y, BoundaryCondition::Natural, &mut m, &mut w).unwrap();
                for area in [
                    p.integral(a, b),
                    s.integral(a, b),
                    -p.integral(b, a),
                    -s.integral(b, a),
                ] {
                    assert!((area / expected - 1.0).abs() < 128.0 * <$t>::EPSILON);
                }
                #[cfg(feature = "alloc")]
                {
                    let owned = PchipBuf::new(&x, &y).unwrap();
                    assert_eq!(owned.integral(a, b), p.integral(a, b));
                    let owned = CubicSplineBuf::new(&x, &y, BoundaryCondition::Natural).unwrap();
                    assert_eq!(owned.integral(a, b), s.integral(a, b));
                }
            }
            #[test]
            fn constructors_reject_nonrepresentable_derivatives() {
                let u = <$t>::from_bits(1);
                let x = [0.0, u, 2.0 * u];
                let y = [0.0, 1.0, 0.0];
                let mut d = [0.0; 3];
                assert!(matches!(
                    Pchip::new(&x, &y, &mut d),
                    Err(InterpError::NotFinite)
                ));
                let mut m = [0.0; 3];
                let mut w = [0.0; cubic_spline_work_len(3)];
                assert!(matches!(
                    CubicSpline::new(&x, &y, BoundaryCondition::NotAKnot, &mut m, &mut w),
                    Err(InterpError::NotFinite)
                ));
            }
            #[test]
            fn spline_nonfinite_queries_and_limits_are_nan() {
                let x = [0.0, 1.0];
                let y = [0.0, 1.0];
                let mut d = [0.0; 2];
                let p = Pchip::new(&x, &y, &mut d).unwrap();
                let mut m = [0.0; 2];
                let mut w = [0.0; cubic_spline_work_len(2)];
                let s =
                    CubicSpline::new(&x, &y, BoundaryCondition::Natural, &mut m, &mut w).unwrap();
                for q in [<$t>::NAN, <$t>::INFINITY, <$t>::NEG_INFINITY] {
                    assert!(p.value(q).is_nan());
                    assert!(p.derivative(q).is_nan());
                    assert!(s.value(q).is_nan());
                    assert!(s.derivative(q).is_nan());
                    assert!(s.second_derivative(q).is_nan());
                    assert!(p.integral(q, q).is_nan());
                    assert!(s.integral(q, q).is_nan());
                }
            }
        }
    };
}
checks!(double, f64, 1e200_f64, 1e-200_f64);
checks!(single, f32, 1e20_f32, 1e-20_f32);
