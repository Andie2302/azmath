//! Randfälle für eingeschlossene und offene skalare Nullstellenverfahren.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::roots::{RootError, RootOptions, bisection, brent, newton, secant};

macro_rules! checks {
    ($module:ident, $t:ty) => {
        mod $module {
            use super::*;

            #[test]
            fn brackets_cover_extreme_finite_endpoints() {
                let max = <$t>::MAX;
                for reverse in [false, true] {
                    let (a, b) = if reverse { (max, -max) } else { (-max, max) };
                    let f = |x: $t| {
                        assert!(x.is_finite());
                        x / max - 0.25
                    };
                    let opts = RootOptions::new(32.0 * <$t>::EPSILON, 300);
                    for result in [bisection(f, a, b, opts), brent(f, a, b, opts)] {
                        let r = result.unwrap();
                        assert!((r.root / max - 0.25).abs() <= 64.0 * <$t>::EPSILON);
                        assert_eq!(r.f_value, f(r.root));
                    }
                }
            }

            #[test]
            fn secant_avoids_overflow_in_function_and_coordinate_differences() {
                let scale = <$t>::MAX * 0.75;
                let r =
                    secant(|x: $t| scale * (x - 1.0), 0.0, 2.0, RootOptions::default()).unwrap();
                assert_eq!(r.root, 1.0);
                let max = <$t>::MAX;
                let r = secant(|x: $t| x / max, -max, max, RootOptions::default()).unwrap();
                assert_eq!(r.root, 0.0);
            }

            #[test]
            fn newton_never_returns_a_nonfinite_final_function_value() {
                let f = |x: $t| {
                    if x == 0.0 {
                        <$t>::MIN_POSITIVE
                    } else {
                        <$t>::NAN
                    }
                };
                assert_eq!(
                    newton(f, |_| 1.0, 0.0, RootOptions::default()),
                    Err(RootError::NotFinite)
                );
            }

            #[test]
            fn last_allowed_step_can_reach_an_exact_root() {
                let r = newton(|x: $t| x - 1.0, |_| 1.0, 0.0, RootOptions::new(0.0, 1)).unwrap();
                assert_eq!((r.root, r.f_value, r.iterations), (1.0, 0.0, 1));
                let r = secant(|x: $t| x - 1.0, 0.0, 2.0, RootOptions::new(0.0, 1)).unwrap();
                assert_eq!((r.root, r.f_value, r.iterations), (1.0, 0.0, 1));
            }

            #[test]
            fn zero_budget_checks_all_starting_points() {
                let opts = RootOptions::new(0.0, 0);
                assert_eq!(
                    newton(
                        |x: $t| x - 1.0,
                        |_| panic!("keine Ableitung nötig"),
                        1.0,
                        opts
                    )
                    .unwrap()
                    .iterations,
                    0
                );
                assert_eq!(
                    secant(|x: $t| x - 1.0, 0.0, 1.0, opts).unwrap().iterations,
                    0
                );
                assert_eq!(
                    newton(|x: $t| x - 1.0, |_| 1.0, 0.0, opts),
                    Err(RootError::NoConvergence)
                );
            }

            #[test]
            fn invalid_tolerances_are_rejected_before_callbacks() {
                for tol in [-1.0 as $t, <$t>::NAN, <$t>::INFINITY] {
                    let opts = RootOptions::new(tol, 20);
                    let f = |_: $t| -> $t { panic!("ungültige Optionen vor Callback prüfen") };
                    assert_eq!(bisection(f, 0.0, 1.0, opts), Err(RootError::InvalidInput));
                    assert_eq!(brent(f, 0.0, 1.0, opts), Err(RootError::InvalidInput));
                    assert_eq!(newton(f, f, 0.0, opts), Err(RootError::InvalidInput));
                    assert_eq!(secant(f, 0.0, 1.0, opts), Err(RootError::InvalidInput));
                }
            }

            #[test]
            fn invalid_coordinates_are_rejected_before_callbacks() {
                let f = |_: $t| -> $t { panic!("nichtendliche Startwerte nicht auswerten") };
                for invalid in [<$t>::NAN, <$t>::INFINITY, <$t>::NEG_INFINITY] {
                    let opts = RootOptions::default();
                    assert_eq!(bisection(f, invalid, 1.0, opts), Err(RootError::NotFinite));
                    assert_eq!(brent(f, 0.0, invalid, opts), Err(RootError::NotFinite));
                    assert_eq!(newton(f, f, invalid, opts), Err(RootError::NotFinite));
                    assert_eq!(secant(f, 0.0, invalid, opts), Err(RootError::NotFinite));
                }
            }

            #[test]
            fn rounded_stagnation_is_not_an_exact_root() {
                let opts = RootOptions::new(0.0, 5);
                assert_eq!(
                    newton(|_: $t| 1.0, |_| <$t>::MAX, 1.0, opts),
                    Err(RootError::NoConvergence)
                );
                let f = |x: $t| if x == 0.0 { 1.0 } else { <$t>::MIN_POSITIVE };
                assert_eq!(secant(f, 0.0, 1.0, opts), Err(RootError::NoConvergence));
            }
        }
    };
}
checks!(double, f64);
checks!(single, f32);
