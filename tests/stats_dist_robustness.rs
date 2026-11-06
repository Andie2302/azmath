#![cfg(any(feature = "std", feature = "libm"))]
use azmath::{special, stats::dist::*};
#[test]
fn lower_tail_quantiles_remain_finite_and_resolve_small_values() {
    assert!(special::normal_quantile(1e-100f64).is_finite());
    let g = gamma_quantile(1e-100f64, 1., 1.);
    assert!((g / 1e-100 - 1.).abs() < 1e-12, "gamma={g}");
    let b = beta_quantile(1e-100f64, 1., 1.);
    assert!((b / 1e-100 - 1.).abs() < 1e-12, "beta={b}");
    assert!(special::normal_quantile(1e-30f32).is_finite());
}
#[test]
fn invalid_and_infinite_arguments_have_defined_limits() {
    assert!(uniform_cdf(f64::NAN, 0., 1.).is_nan());
    assert!(beta_cdf(f64::NAN, 2., 3.).is_nan());
    assert_eq!(gamma_pdf(f64::INFINITY, 2., 1.), 0.);
    assert_eq!(f_pdf(f64::INFINITY, 3., 7.), 0.);
    assert_eq!(f_cdf(f64::INFINITY, 3., 7.), 1.);
}
macro_rules! check {
    ($mod:ident,$t:ty,$tiny:expr) => {
        mod $mod {
            use super::*;
            #[test]
            fn direct_tails_survive_cdf_rounding_and_invert() {
                let tiny = $tiny as $t;
                let tol = 4096. * <$t>::EPSILON;
                for (x, q) in [
                    (normal_isf_with(tiny, 0., 1.), tiny),
                    (gamma_isf(tiny, 2.5, 1.5), tiny),
                    (student_t_isf(tiny, 1.), tiny),
                    (f_isf(tiny, 2., 2.), tiny),
                ]
                .into_iter()
                .enumerate()
                {
                    let value = match x {
                        0 => normal_sf_with(q.0, 0., 1.),
                        1 => gamma_sf(q.0, 2.5, 1.5),
                        2 => student_t_sf(q.0, 1.),
                        _ => f_sf(q.0, 2., 2.),
                    };
                    assert!(q.0.is_finite());
                    assert!((value / q.1 - 1.).abs() < tol, "tail {x}: {value:?}");
                }
                assert!(normal_sf_with(10. as $t, 0., 1.) > 0.);
                assert_eq!(normal_cdf_with(10. as $t, 0., 1.), 1.);
                for x in [0.01 as $t, 0.1, 0.5, 1., 5., 20.] {
                    for (c, s) in [
                        (gamma_cdf(x, 2.5, 1.), gamma_sf(x, 2.5, 1.)),
                        (beta_cdf(x, 2., 5.), beta_sf(x, 2., 5.)),
                        (student_t_cdf(x, 3.), student_t_sf(x, 3.)),
                        (f_cdf(x, 3., 7.), f_sf(x, 3., 7.)),
                    ] {
                        assert!((c + s - 1.).abs() < tol);
                        assert!((0. ..=1.).contains(&c));
                        assert!((0. ..=1.).contains(&s));
                    }
                }
            }
            #[test]
            fn boundaries_invalid_parameters_and_probabilities() {
                for x in [<$t>::NEG_INFINITY, <$t>::INFINITY] {
                    let expected = if x < 0. { 0. } else { 1. };
                    for c in [
                        normal_cdf_with(x, 0., 1.),
                        exponential_cdf(x, 1.),
                        uniform_cdf(x, 0., 1.),
                        gamma_cdf(x, 2., 1.),
                        beta_cdf(x, 2., 3.),
                        student_t_cdf(x, 3.),
                        f_cdf(x, 3., 7.),
                    ] {
                        assert_eq!(c, expected);
                    }
                    for s in [
                        normal_sf_with(x, 0., 1.),
                        exponential_sf(x, 1.),
                        uniform_sf(x, 0., 1.),
                        gamma_sf(x, 2., 1.),
                        beta_sf(x, 2., 3.),
                        student_t_sf(x, 3.),
                        f_sf(x, 3., 7.),
                    ] {
                        assert_eq!(s, 1. - expected);
                    }
                    for p in [
                        normal_pdf(x, 0., 1.),
                        exponential_pdf(x, 1.),
                        uniform_pdf(x, 0., 1.),
                        gamma_pdf(x, 2., 1.),
                        beta_pdf(x, 2., 3.),
                        student_t_pdf(x, 3.),
                        f_pdf(x, 3., 7.),
                    ] {
                        assert_eq!(p, 0.);
                    }
                }
                for p in [-0.1 as $t, 1.1, <$t>::NAN, <$t>::INFINITY] {
                    for v in [
                        normal_quantile_with(p, 0., 1.),
                        normal_isf_with(p, 0., 1.),
                        gamma_quantile(p, 2., 1.),
                        gamma_isf(p, 2., 1.),
                        beta_quantile(p, 2., 3.),
                        beta_isf(p, 2., 3.),
                        student_t_quantile(p, 3.),
                        student_t_isf(p, 3.),
                        f_quantile(p, 3., 7.),
                        f_isf(p, 3., 7.),
                        uniform_quantile(p, 0., 1.),
                        uniform_isf(p, 0., 1.),
                        exponential_quantile(p, 1.),
                        exponential_isf(p, 1.),
                    ] {
                        assert!(v.is_nan());
                    }
                }
                for a in [0. as $t, -1., <$t>::NAN, <$t>::INFINITY] {
                    for v in [
                        gamma_cdf(1., a, 1.),
                        gamma_sf(1., a, 1.),
                        gamma_quantile(0.5, a, 1.),
                        beta_cdf(0.5, a, 2.),
                        beta_sf(0.5, a, 2.),
                        beta_quantile(0.5, a, 2.),
                        student_t_cdf(1., a),
                        student_t_quantile(0.5, a),
                        f_cdf(1., a, 2.),
                        f_quantile(0.5, a, 2.),
                        normal_cdf_with(1., 0., a),
                        exponential_cdf(1., a),
                    ] {
                        assert!(v.is_nan());
                    }
                }
            }
            #[test]
            fn monotone_quantiles_and_bounded_endpoints() {
                let mut last = [<$t>::NEG_INFINITY; 5];
                for p in [0. as $t, 1e-20, 0.01, 0.5, 0.99, 1.] {
                    let current = [
                        normal_quantile_with(p, 0., 1.),
                        gamma_quantile(p, 0.25, 1.),
                        beta_quantile(p, 0.25, 3.),
                        student_t_quantile(p, 1.),
                        f_quantile(p, 3., 7.),
                    ];
                    for (x, l) in current.into_iter().zip(&mut last) {
                        assert!(x >= *l);
                        *l = x;
                    }
                }
                assert_eq!(beta_quantile(0. as $t, 2., 3.), 0.);
                assert_eq!(beta_quantile(1. as $t, 2., 3.), 1.);
                assert_eq!(beta_isf(0. as $t, 2., 3.), 1.);
                assert_eq!(beta_isf(1. as $t, 2., 3.), 0.);
                assert_eq!(uniform_quantile(0.5 as $t, -<$t>::MAX, <$t>::MAX), 0.);
                assert_eq!(uniform_cdf(0. as $t, -<$t>::MAX, <$t>::MAX), 0.5);
            }
        }
    };
}
check!(single, f32, 1e-30);
check!(double, f64, 1e-100);
#[test]
fn hypothesis_tests_keep_small_p_values() {
    let data: Vec<f64> = (0..20).map(|i| 1000. + (i as f64).sin()).collect();
    let t = azmath::stats::t_test_one_sample(&data, 0.);
    assert!(t.p_value > 0. && t.p_value < 1e-40);
    let c = azmath::stats::chi_square_gof(&[1000.0f64, 0.], &[500., 500.], 0);
    assert!(c.p_value > 0. && c.p_value < 1e-200);
}
