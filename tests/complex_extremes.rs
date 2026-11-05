//! Regressionstests für den Wertebereich der komplexen Grundoperationen.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::complex::Complex;

macro_rules! extreme_tests {
    ($module:ident, $t:ty, $large:expr, $small:expr) => {
        mod $module {
            use super::Complex;
            type C = Complex<$t>;

            fn relative(actual: $t, expected: $t) {
                assert!(actual.is_finite(), "nicht endlich: {actual}");
                if expected == 0.0 {
                    assert_eq!(actual, expected);
                } else {
                    assert!(
                        ((actual - expected) / expected).abs() <= 16.0 * <$t>::EPSILON,
                        "erwartet {expected}, erhalten {actual}"
                    );
                }
            }

            #[test]
            fn division_extreme_scales() {
                for v in [
                    $large,
                    $small,
                    <$t>::MAX,
                    <$t>::MIN_POSITIVE,
                    <$t>::from_bits(1),
                ] {
                    for z in [
                        C::new(v, v),
                        C::new(v, -v),
                        C::new(v, v / 2.0),
                        C::new(v / 2.0, -v),
                    ] {
                        let q = z / z;
                        relative(q.re, 1.0);
                        relative(q.im, 0.0);
                    }
                }
                for z in [C::new($large, $large), C::new($large, -$large)] {
                    let q = z / C::new(2.0, 2.0);
                    if z.im > 0.0 {
                        relative(q.re, $large / 2.0);
                        relative(q.im, 0.0);
                    } else {
                        relative(q.re, 0.0);
                        relative(q.im, -$large / 2.0);
                    }
                }
            }

            #[test]
            fn sqrt_extreme_scales() {
                for v in [
                    $large,
                    $small,
                    <$t>::MAX,
                    <$t>::MIN_POSITIVE,
                    <$t>::from_bits(1),
                ] {
                    let root = v.sqrt();
                    let w = C::new(v, 0.0).sqrt();
                    relative(w.re, root);
                    relative(w.im, 0.0);
                    let w = C::new(-v, -0.0).sqrt();
                    relative(w.re, 0.0);
                    relative(w.im, -root);
                    // Analytisch: sqrt(v + iv), ohne den Betrag v*sqrt(2) zu bilden.
                    for sign in [-1.0, 1.0] {
                        let w = C::new(v, sign * v).sqrt();
                        relative(
                            w.re,
                            root * ((2.0 as $t).sqrt() + 1.0).sqrt() / (2.0 as $t).sqrt(),
                        );
                        relative(
                            w.im,
                            sign * root / (((2.0 as $t).sqrt() + 1.0).sqrt() * (2.0 as $t).sqrt()),
                        );
                    }
                }
            }

            #[test]
            fn sqrt_signed_zero_and_nonfinite() {
                for re in [-0.0, 0.0] {
                    let w = C::new(re, -0.0).sqrt();
                    assert!(!w.re.is_sign_negative());
                    assert!(w.im.is_sign_negative());
                }
                let w = C::new(-4.0, -0.0).sqrt();
                relative(w.im, -2.0);
                assert_eq!(
                    C::new(1.0, <$t>::INFINITY).sqrt(),
                    C::new(<$t>::INFINITY, <$t>::INFINITY)
                );
                assert_eq!(
                    C::new(<$t>::NEG_INFINITY, -1.0).sqrt(),
                    C::new(0.0, <$t>::NEG_INFINITY)
                );
                assert!(C::new(<$t>::NAN, 1.0).sqrt().is_nan());
                assert!(C::new(1.0, <$t>::NAN).sqrt().is_nan());
            }
        }
    };
}

extreme_tests!(double, f64, 1e308_f64, 1e-308_f64);
extreme_tests!(single, f32, 3e38_f32, 1e-38_f32);
