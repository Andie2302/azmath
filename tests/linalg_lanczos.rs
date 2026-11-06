#![cfg(any(feature = "std", feature = "libm"))]
use azmath::linalg::{
    LinalgError, Matrix,
    eigen::{lanczos, lanczos_ritz, lanczos_ritz_work_len, lanczos_work_len},
};
macro_rules! checks {
    ($module:ident, $t:ty, $large:expr, $tiny:expr) => {
        mod $module {
            use super::*;
            fn operator() -> Matrix<$t, 5, 5> {
                Matrix::new([
                    [2., 1., 0., 0., 0.],
                    [1., -1., 0.5, 0., 0.],
                    [0., 0.5, 4., 1., 0.],
                    [0., 0., 1., 3., 2.],
                    [0., 0., 0., 2., 6.],
                ])
            }
            #[test]
            fn reconstruction_projection_orthogonality_and_ritz_residuals() {
                let a = operator();
                let mut alpha = [<$t>::NAN; 3];
                let mut beta = alpha;
                let mut w = [<$t>::NAN; lanczos_work_len(5, 3)];
                let m = lanczos(
                    &a,
                    &[1., 2., -1., 0.5, 3.],
                    3,
                    &mut alpha,
                    &mut beta,
                    &mut w,
                )
                .unwrap();
                assert_eq!(m, 3);
                let tol = 512. * <$t>::EPSILON;
                for i in 0..m {
                    for j in 0..m {
                        let dot = (0..5).map(|r| w[5 * i + r] * w[5 * j + r]).sum::<$t>();
                        assert!((dot - if i == j { 1. } else { 0. }).abs() < tol);
                        let projected = (0..5)
                            .map(|r| {
                                w[5 * i + r] * (0..5).map(|c| a[(r, c)] * w[5 * j + c]).sum::<$t>()
                            })
                            .sum::<$t>();
                        let expected = if i == j {
                            alpha[i]
                        } else if i.abs_diff(j) == 1 {
                            beta[i.min(j)]
                        } else {
                            0.
                        };
                        assert!((projected - expected).abs() < tol * 10.);
                    }
                }
                for j in 0..m {
                    for r in 0..5 {
                        let aq = (0..5).map(|c| a[(r, c)] * w[j * 5 + c]).sum::<$t>();
                        let mut qt = alpha[j] * w[j * 5 + r];
                        if j > 0 {
                            qt += beta[j - 1] * w[(j - 1) * 5 + r];
                        }
                        if j + 1 < m {
                            qt += beta[j] * w[(j + 1) * 5 + r];
                        } else {
                            qt += w[15 + r];
                        }
                        assert!((aq - qt).abs() < tol * 10.);
                    }
                }
                let tridiag = Matrix::new([
                    [alpha[0], beta[0], 0.],
                    [beta[0], alpha[1], beta[1]],
                    [0., beta[1], alpha[2]],
                ]);
                let eig = tridiag.symmetric_eigen().unwrap();
                let mut values = [0.; 3];
                let mut residuals = [0.; 3];
                let mut rw = [0.; lanczos_ritz_work_len(3)];
                lanczos_ritz(&alpha, &beta, m, &mut values, &mut residuals, &mut rw).unwrap();
                assert!(values.windows(2).all(|p| p[0] <= p[1]));
                for i in 0..3 {
                    assert!((values[i] - eig.values[i]).abs() < tol * 10.);
                    let y: [$t; 5] = core::array::from_fn(|r| {
                        (0..3).map(|j| w[j * 5 + r] * eig.vectors[(j, i)]).sum()
                    });
                    let actual = (0..5)
                        .map(|r| {
                            let e = (0..5).map(|c| a[(r, c)] * y[c]).sum::<$t>() - values[i] * y[r];
                            e * e
                        })
                        .sum::<$t>()
                        .sqrt();
                    assert!((actual - residuals[i]).abs() < tol * 10.);
                }
            }
            #[test]
            fn full_diagonal_spectrum_and_invariant_subspaces() {
                let a = Matrix::new([[-3. as $t, 0., 0.], [0., 1., 0.], [0., 0., 7.]]);
                let mut alpha = [0.; 3];
                let mut beta = [0.; 3];
                let mut w = [<$t>::NAN; lanczos_work_len(3, 3)];
                let m = lanczos(&a, &[1., 1., 1.], 3, &mut alpha, &mut beta, &mut w).unwrap();
                assert_eq!(m, 3);
                let mut vals = [0.; 3];
                let mut residuals = [0.; 3];
                let mut rw = [0.; 18];
                lanczos_ritz(&alpha, &beta, m, &mut vals, &mut residuals, &mut rw).unwrap();
                for (v, e) in vals.iter().zip([-3., 1., 7.]) {
                    assert!((*v - e).abs() < 1024. * <$t>::EPSILON);
                }
                assert!(residuals.iter().all(|r| *r < 1024. * <$t>::EPSILON));
                let m = lanczos(&a, &[0., 1., 0.], 3, &mut alpha, &mut beta, &mut w).unwrap();
                assert_eq!(m, 1);
                assert_eq!(alpha[0], 1.);
                assert_eq!(beta[0], 0.);
                let zero = Matrix::<$t, 3, 3>::zeros();
                assert_eq!(
                    lanczos(&zero, &[1., 1., 1.], 3, &mut alpha, &mut beta, &mut w).unwrap(),
                    1
                );
            }
            #[test]
            fn large_and_small_start_vectors() {
                let a = Matrix::<$t, 2, 2>::identity();
                let mut alpha = [0.; 2];
                let mut beta = [0.; 2];
                let mut w = [0.; 6];
                for s in [$large as $t, $tiny as $t] {
                    assert_eq!(
                        lanczos(&a, &[s, s], 2, &mut alpha, &mut beta, &mut w).unwrap(),
                        1
                    );
                    assert!((w[0] * w[0] + w[1] * w[1] - 1.).abs() < 16. * <$t>::EPSILON);
                }
            }
            #[test]
            fn large_and_small_operator_residual_norms() {
                let mut alpha = [0.; 2];
                let mut beta = [0.; 2];
                let mut w = [0.; 6];
                for s in [$large as $t, $tiny as $t] {
                    let a = Matrix::new([[0., s], [s, 0.]]);
                    assert_eq!(
                        lanczos(&a, &[1., 0.], 2, &mut alpha, &mut beta, &mut w).unwrap(),
                        2
                    );
                    assert!((beta[0] / s - 1.).abs() < 16. * <$t>::EPSILON);
                }
            }
            #[test]
            fn ritz_extreme_tridiagonal_scaling() {
                let mut v = [0. as $t; 2];
                let mut r = [0.; 2];
                let mut w = [0.; 8];
                for scale in [$large as $t, $tiny as $t] {
                    lanczos_ritz(&[0., 0.], &[scale, 0.], 2, &mut v, &mut r, &mut w).unwrap();
                    assert!((v[0] / scale + 1.).abs() < 32. * <$t>::EPSILON);
                    assert!((v[1] / scale - 1.).abs() < 32. * <$t>::EPSILON);
                    assert_eq!(r, [0., 0.]);
                }
            }
            #[test]
            fn invalid_inputs_and_nonfinite_operators() {
                let a = Matrix::<$t, 2, 2>::identity();
                let mut alpha = [0.; 2];
                let mut beta = [0.; 2];
                let mut w = [0.; 6];
                for bad in [<$t>::NAN, <$t>::INFINITY, <$t>::NEG_INFINITY] {
                    assert_eq!(
                        lanczos(&a, &[bad, 1.], 2, &mut alpha, &mut beta, &mut w),
                        Err(LinalgError::NotFinite)
                    );
                    let op = Matrix::new([[1., bad], [bad, 1.]]);
                    assert_eq!(
                        lanczos(&op, &[1., 1.], 2, &mut alpha, &mut beta, &mut w),
                        Err(LinalgError::NotFinite)
                    );
                }
                assert_eq!(
                    lanczos(&a, &[0., 0.], 2, &mut alpha, &mut beta, &mut w),
                    Err(LinalgError::Breakdown)
                );
                for k in [0, 3] {
                    assert_eq!(
                        lanczos(&a, &[1., 1.], k, &mut alpha, &mut beta, &mut w),
                        Err(LinalgError::DimensionMismatch)
                    );
                }
                assert_eq!(
                    lanczos(&a, &[1.], 1, &mut alpha, &mut beta, &mut w),
                    Err(LinalgError::DimensionMismatch)
                );
                assert_eq!(
                    lanczos(&a, &[1., 1.], 2, &mut alpha[..1], &mut beta, &mut w),
                    Err(LinalgError::DimensionMismatch)
                );
                assert_eq!(
                    lanczos(&a, &[1., 1.], 2, &mut alpha, &mut beta[..1], &mut w),
                    Err(LinalgError::DimensionMismatch)
                );
                assert_eq!(
                    lanczos(&a, &[1., 1.], 2, &mut alpha, &mut beta, &mut w[..5]),
                    Err(LinalgError::DimensionMismatch)
                );
            }
            #[test]
            fn ritz_empty_singleton_dimensions_and_nonfinite_input() {
                let mut v = [0. as $t; 2];
                let mut r = [0.; 2];
                let mut w = [0.; 8];
                assert_eq!(
                    lanczos_ritz::<$t>(&[], &[], 0, &mut [], &mut [], &mut []),
                    Ok(())
                );
                lanczos_ritz(&[3.], &[0.25], 1, &mut v, &mut r, &mut w).unwrap();
                assert_eq!(v[0], 3.);
                assert_eq!(r[0], 0.25);
                lanczos_ritz(&[0., 0.], &[0., 0.], 2, &mut v, &mut r, &mut w).unwrap();
                assert_eq!(v, [0., 0.]);
                assert_eq!(r, [0., 0.]);
                assert_eq!(
                    lanczos_ritz(
                        &[<$t>::MAX, <$t>::MAX],
                        &[<$t>::MAX, 0.],
                        2,
                        &mut v,
                        &mut r,
                        &mut w
                    ),
                    Err(LinalgError::NotFinite)
                );
                assert_eq!(
                    lanczos_ritz(&[1.], &[1., 1.], 2, &mut v, &mut r, &mut w),
                    Err(LinalgError::DimensionMismatch)
                );
                assert_eq!(
                    lanczos_ritz(&[1., 1.], &[1.], 2, &mut v, &mut r, &mut w),
                    Err(LinalgError::DimensionMismatch)
                );
                assert_eq!(
                    lanczos_ritz(&[1., 1.], &[1., 1.], 2, &mut v[..1], &mut r, &mut w),
                    Err(LinalgError::DimensionMismatch)
                );
                assert_eq!(
                    lanczos_ritz(&[1., 1.], &[1., 1.], 2, &mut v, &mut r[..1], &mut w),
                    Err(LinalgError::DimensionMismatch)
                );
                assert_eq!(
                    lanczos_ritz(&[1., 1.], &[1., 1.], 2, &mut v, &mut r, &mut w[..7]),
                    Err(LinalgError::DimensionMismatch)
                );
                for bad in [<$t>::NAN, <$t>::INFINITY, <$t>::NEG_INFINITY] {
                    assert_eq!(
                        lanczos_ritz(&[bad], &[0.], 1, &mut v, &mut r, &mut w),
                        Err(LinalgError::NotFinite)
                    );
                    assert_eq!(
                        lanczos_ritz(&[1.], &[bad], 1, &mut v, &mut r, &mut w),
                        Err(LinalgError::NotFinite)
                    );
                }
            }
        }
    };
}
checks!(single, f32, 1e30, 1e-30);
checks!(double, f64, 1e200, 1e-200);
