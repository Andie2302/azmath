//! Wertebereich und Fehlerdiagnosen für Cholesky und Dreieckslöser.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::linalg::{LinalgError, Matrix, Vector};

macro_rules! robustness_tests {
    ($module:ident, $t:ty, $large:expr, $tiny:expr) => {
        mod $module {
            use super::*;

            #[test]
            fn cholesky_scaled_reconstruction_and_known_solution() {
                for scale in [1.0 as $t, $large, $tiny] {
                    let a = Matrix::new([[4.0 * scale, 2.0 * scale], [2.0 * scale, 5.0 * scale]]);
                    let c = a.cholesky().unwrap();
                    let l = c.l();
                    let root = scale.sqrt();
                    let tol = 128.0 * <$t>::EPSILON;
                    assert!((l[(0, 0)] / root - 2.0).abs() < tol);
                    assert!((l[(1, 0)] / root - 1.0).abs() < tol);
                    assert!((l[(1, 1)] / root - 2.0).abs() < tol);
                    let rec = *l * l.transpose();
                    for i in 0..2 {
                        for j in 0..2 {
                            assert!((rec[(i, j)] / scale - a[(i, j)] / scale).abs() < tol);
                        }
                    }
                    let b = Vector::new([0.0, -8.0 * scale]); // x=(1,-2)
                    let x = a.solve_spd(&b).unwrap();
                    assert!((x[0] - 1.0).abs() < tol && (x[1] + 2.0).abs() < tol);
                    #[cfg(feature = "alloc")]
                    {
                        use azmath::linalg::{DynMatrix, DynVector};
                        let d = DynMatrix::new(
                            2,
                            2,
                            vec![4.0 * scale, 2.0 * scale, 2.0 * scale, 5.0 * scale],
                        )
                        .unwrap();
                        let x = d.solve_spd(&DynVector::new(b.as_slice().to_vec())).unwrap();
                        assert!((x[0] - 1.0).abs() < tol && (x[1] + 2.0).abs() < tol);
                    }
                }
            }

            #[test]
            fn cholesky_subnormal_entries_keep_factor_accuracy() {
                let u = <$t>::from_bits(1);
                let a = Matrix::new([[2.0 * u, u], [u, u]]);
                let c = a.cholesky().unwrap();
                let l = c.l();
                let root = u.sqrt();
                let tol = 64.0 * <$t>::EPSILON;
                assert!((l[(0, 0)] / root - (2.0 as $t).sqrt()).abs() < tol);
                assert!((l[(1, 0)] / root - 1.0 / (2.0 as $t).sqrt()).abs() < tol);
                assert!((l[(1, 1)] / root - 1.0 / (2.0 as $t).sqrt()).abs() < tol);
            }

            #[test]
            fn cholesky_rejects_invalid_matrices_and_rhs() {
                for scale in [1.0 as $t, $large, $tiny] {
                    let asym = Matrix::new([[4.0 * scale, scale], [2.0 * scale, 4.0 * scale]]);
                    assert_eq!(asym.cholesky().unwrap_err(), LinalgError::NotSymmetric);
                    for a in [
                        Matrix::new([[scale, 2.0 * scale], [2.0 * scale, scale]]),
                        Matrix::new([[scale, scale], [scale, scale]]),
                    ] {
                        assert_eq!(a.cholesky().unwrap_err(), LinalgError::NotPositiveDefinite);
                    }
                }
                let a = Matrix::<$t, 2, 2>::identity();
                for invalid in [<$t>::NAN, <$t>::INFINITY, <$t>::NEG_INFINITY] {
                    assert_eq!(
                        a.solve_spd(&Vector::new([1.0, invalid])),
                        Err(LinalgError::NotFinite)
                    );
                    #[cfg(feature = "alloc")]
                    {
                        use azmath::linalg::{DynMatrix, DynVector};
                        let d = DynMatrix::new(2, 2, vec![1.0 as $t, 0.0, 0.0, 1.0]).unwrap();
                        assert_eq!(
                            d.solve_spd(&DynVector::new(vec![1.0, invalid])),
                            Err(LinalgError::NotFinite)
                        );
                    }
                }
                let tiny = Matrix::new([[<$t>::MIN_POSITIVE]]);
                assert_eq!(
                    tiny.solve_spd(&Vector::new([<$t>::MAX])),
                    Err(LinalgError::NotFinite)
                );
            }

            #[test]
            fn triangular_solve_avoids_intermediate_overflow() {
                let v = <$t>::MAX;
                let lower = Matrix::new([[v, 0.0], [-v, v]]);
                let x = lower.solve_lower(&Vector::new([v, v])).unwrap();
                assert_eq!(x, Vector::new([1.0, 2.0]));
                let upper = lower.transpose();
                assert_eq!(
                    upper.solve_upper(&Vector::new([v, v])).unwrap(),
                    Vector::new([2.0, 1.0])
                );
                assert_eq!(
                    upper.solve_upper_transposed(&Vector::new([v, v])).unwrap(),
                    Vector::new([1.0, 2.0])
                );
                assert_eq!(
                    lower.solve_lower_transposed(&Vector::new([v, v])).unwrap(),
                    Vector::new([2.0, 1.0])
                );
            }

            #[test]
            fn triangular_checks_only_the_used_entries() {
                let b = Vector::new([2.0 as $t, 7.0]);
                let lower = Matrix::new([[2.0, <$t>::NAN], [1.0, 3.0]]);
                assert_eq!(lower.solve_lower(&b).unwrap(), Vector::new([1.0, 2.0]));
                let upper = lower.transpose();
                assert_eq!(
                    upper.solve_upper_transposed(&b).unwrap(),
                    Vector::new([1.0, 2.0])
                );
                for invalid in [<$t>::NAN, <$t>::INFINITY, <$t>::NEG_INFINITY] {
                    let used = Matrix::new([[2.0, 0.0], [invalid, 3.0]]);
                    assert_eq!(used.solve_lower(&b), Err(LinalgError::NotFinite));
                    assert_eq!(used.solve_lower_transposed(&b), Err(LinalgError::NotFinite));
                    assert_eq!(
                        lower.solve_lower(&Vector::new([invalid, 1.0])),
                        Err(LinalgError::NotFinite)
                    );
                }
                let infinite_diagonal = Matrix::new([[<$t>::INFINITY]]);
                assert_eq!(
                    infinite_diagonal.solve_lower(&Vector::new([1.0])),
                    Err(LinalgError::NotFinite)
                );
                let tiny = Matrix::new([[<$t>::MIN_POSITIVE]]);
                assert_eq!(
                    tiny.solve_upper(&Vector::new([<$t>::MAX])),
                    Err(LinalgError::NotFinite)
                );
            }

            #[test]
            fn unit_diagonal_checks_preserve_ignored_values() {
                let lower = Matrix::new([[<$t>::NAN, <$t>::INFINITY], [2.0, <$t>::NAN]]);
                let b = Vector::new([1.0, 5.0]);
                assert_eq!(
                    lower.try_solve_lower_unit(&b).unwrap(),
                    Vector::new([1.0, 3.0])
                );
                assert_eq!(
                    lower
                        .transpose()
                        .try_solve_upper_unit(&Vector::new([7.0, 3.0]))
                        .unwrap(),
                    Vector::new([1.0, 3.0])
                );
                assert_eq!(
                    lower.try_solve_lower_unit(&Vector::new([1.0, <$t>::NAN])),
                    Err(LinalgError::NotFinite)
                );
                let bad = Matrix::new([[1.0, 0.0], [<$t>::INFINITY, 1.0]]);
                assert_eq!(bad.try_solve_lower_unit(&b), Err(LinalgError::NotFinite));
                #[cfg(feature = "alloc")]
                {
                    use azmath::linalg::{DynMatrix, DynVector};
                    let d =
                        DynMatrix::new(2, 2, lower.as_array().iter().flatten().copied().collect())
                            .unwrap();
                    let x = d.solve_lower_unit(&DynVector::new(vec![1.0, 5.0])).unwrap();
                    assert_eq!(x.as_slice(), &[1.0, 3.0]);
                    assert_eq!(
                        d.solve_lower_unit(&DynVector::new(vec![1.0, <$t>::INFINITY])),
                        Err(LinalgError::NotFinite)
                    );
                }
            }

            #[test]
            fn triangular_known_solutions_all_orientations() {
                for scale in [1.0 as $t, $large, $tiny] {
                    let l = Matrix::new([
                        [2.0 * scale, 0.0, 0.0],
                        [-scale, 3.0 * scale, 0.0],
                        [scale, 2.0 * scale, 4.0 * scale],
                    ]);
                    let b = Vector::new([2.0 * scale, -7.0 * scale, -scale]);
                    let x = l.solve_lower(&b).unwrap();
                    let expected = Vector::new([1.0, -2.0, 0.5]);
                    assert!((x - expected).max_abs() < 64.0 * <$t>::EPSILON);
                    let bt = Vector::new([4.5 * scale, -5.0 * scale, 2.0 * scale]);
                    assert!(
                        (l.solve_lower_transposed(&bt).unwrap() - expected).max_abs()
                            < 64.0 * <$t>::EPSILON
                    );
                    assert!(
                        (l.transpose().solve_upper(&bt).unwrap() - expected).max_abs()
                            < 64.0 * <$t>::EPSILON
                    );
                    #[cfg(feature = "alloc")]
                    {
                        use azmath::linalg::{DynMatrix, DynVector};
                        let d =
                            DynMatrix::new(3, 3, l.as_array().iter().flatten().copied().collect())
                                .unwrap();
                        let x = d
                            .solve_lower(&DynVector::new(b.as_slice().to_vec()))
                            .unwrap();
                        for i in 0..3 {
                            assert!((x[i] - expected[i]).abs() < 64.0 * <$t>::EPSILON);
                        }
                    }
                }
            }
        }
    };
}

robustness_tests!(double, f64, 1e200_f64, 1e-200_f64);
robustness_tests!(single, f32, 1e20_f32, 1e-30_f32);
