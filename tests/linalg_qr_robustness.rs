//! Unabhängige Prüfungen für Householder-QR und kleinste Quadrate.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::linalg::{LinalgError, Matrix, Vector};

macro_rules! qr_tests {
    ($module:ident, $t:ty, $large:expr, $tiny:expr) => {
        mod $module {
            use super::*;

            #[test]
            fn decomposition_and_fit_are_invariant_under_scaling() {
                for scale in [1.0 as $t, $large, $tiny] {
                    let data = [[1.0, -2.0], [2.0, 1.0], [-1.0, 3.0], [3.0, 2.0]];
                    let a = Matrix::new(data.map(|row| row.map(|x| x * scale)));
                    let qr = a.qr().unwrap();
                    let q = qr.q();
                    let qtq = q.transpose() * q;
                    let rec = qr.thin_q() * qr.r();
                    let tol = 256.0 * <$t>::EPSILON;
                    for i in 0..4 {
                        for j in 0..4 {
                            assert!((qtq[(i, j)] - if i == j { 1.0 } else { 0.0 }).abs() < tol);
                        }
                        for j in 0..2 {
                            assert!((rec[(i, j)] / scale - data[i][j]).abs() < tol);
                        }
                    }
                    let b = Vector::new(data.map(|row| (row[0] * 2.0 - row[1]) * scale));
                    let x = a.least_squares(&b).unwrap();
                    assert!((x[0] - 2.0).abs() < tol);
                    assert!((x[1] + 1.0).abs() < tol);
                    #[cfg(feature = "alloc")]
                    {
                        use azmath::linalg::{DynMatrix, DynVector};
                        let d = DynMatrix::new(
                            4,
                            2,
                            data.into_iter().flatten().map(|x| x * scale).collect(),
                        )
                        .unwrap();
                        let dx = d
                            .least_squares(&DynVector::new(b.as_slice().to_vec()))
                            .unwrap();
                        assert!((dx[0] - 2.0).abs() < tol);
                        assert!((dx[1] + 1.0).abs() < tol);
                    }
                }
            }

            #[test]
            fn noisy_fit_has_known_solution_and_stationary_residual() {
                // A^T r = 0 exakt: r=(1,-2,1), x=(2,-1), b=A*x+r.
                for scale in [1.0 as $t, $large, $tiny] {
                    let a = Matrix::new([[scale, -scale], [scale, 0.0], [scale, scale]]);
                    let b = Vector::new([4.0 * scale, 0.0, 2.0 * scale]);
                    let qr = a.qr().unwrap();
                    let x = qr.solve(&b);
                    let tol = 256.0 * <$t>::EPSILON;
                    assert!((x[0] - 2.0).abs() < tol);
                    assert!((x[1] + 1.0).abs() < tol);
                    let residual = [4.0 - x[0] + x[1], -x[0], 2.0 - x[0] - x[1]];
                    assert!((residual[0] + residual[1] + residual[2]).abs() < tol);
                    assert!((-residual[0] + residual[2]).abs() < tol);
                    assert!((qr.residual_norm(&b) / scale - (6.0 as $t).sqrt()).abs() < tol);
                }
            }

            #[test]
            fn residual_norm_does_not_square_extreme_values() {
                let a = Matrix::new([[1.0 as $t], [0.0], [0.0]]);
                let qr = a.qr().unwrap();
                for scale in [$large, $tiny] {
                    let b = Vector::new([0.0, 3.0 * scale, -4.0 * scale]);
                    assert!((qr.residual_norm(&b) / scale - 5.0).abs() < 32.0 * <$t>::EPSILON);
                }
            }

            #[test]
            fn one_by_one_handles_extreme_and_subnormal_values() {
                for value in [<$t>::MAX, <$t>::MIN_POSITIVE, <$t>::from_bits(1)] {
                    for sign in [-1.0, 1.0] {
                        let value = sign * value;
                        let a = Matrix::new([[value]]);
                        let qr = a.qr().unwrap();
                        let x = qr.solve(&Vector::new([value]));
                        assert!((x[0] - 1.0).abs() <= 16.0 * <$t>::EPSILON);
                        assert_eq!((qr.q() * qr.r())[(0, 0)], value);
                    }
                }
            }

            #[test]
            fn reflector_avoids_overflow_near_maximum() {
                let max = <$t>::MAX;
                let a = Matrix::new([[max, max / 2.0], [0.0, max / 2.0]]);
                let qr = a.qr().unwrap();
                let x = qr.try_solve(&Vector::new([max, 0.0])).unwrap();
                assert_eq!(x[0], 1.0);
                assert_eq!(x[1], 0.0);
                let rec = qr.q() * qr.r();
                for i in 0..2 {
                    for j in 0..2 {
                        assert_eq!(rec[(i, j)], a[(i, j)]);
                    }
                }
            }

            #[test]
            fn checked_solvers_reject_invalid_rhs_and_overflow() {
                let a = Matrix::new([[1.0 as $t], [0.0], [0.0]]);
                let qr = a.qr().unwrap();
                for invalid in [<$t>::NAN, <$t>::INFINITY, <$t>::NEG_INFINITY] {
                    for i in 0..3 {
                        let mut b = [1.0; 3];
                        b[i] = invalid;
                        let b = Vector::new(b);
                        assert_eq!(qr.try_solve(&b), Err(LinalgError::NotFinite));
                        assert_eq!(a.least_squares(&b), Err(LinalgError::NotFinite));
                        assert_eq!(qr.try_residual_norm(&b), Err(LinalgError::NotFinite));
                        #[cfg(feature = "alloc")]
                        {
                            use azmath::linalg::{DynMatrix, DynVector};
                            let d = DynMatrix::new(3, 1, vec![1.0 as $t, 0.0, 0.0]).unwrap();
                            let db = DynVector::new(b.as_slice().to_vec());
                            assert_eq!(d.least_squares(&db), Err(LinalgError::NotFinite));
                            assert_eq!(
                                d.qr().unwrap().residual_norm(&db),
                                Err(LinalgError::NotFinite)
                            );
                        }
                    }
                }
                let tiny = Matrix::new([[<$t>::MIN_POSITIVE]]);
                assert_eq!(
                    tiny.least_squares(&Vector::new([<$t>::MAX])),
                    Err(LinalgError::NotFinite)
                );
                let b = Vector::new([0.0, <$t>::MAX, <$t>::MAX]);
                assert_eq!(qr.try_residual_norm(&b), Err(LinalgError::NotFinite));
                let x = qr.try_solve(&Vector::new([2.0, 3.0, 4.0])).unwrap();
                assert_eq!(x[0], 2.0);
                assert_eq!(qr.try_residual_norm(&Vector::new([2.0, 3.0, 4.0])), Ok(5.0));
            }

            #[test]
            fn rejects_unrepresentable_factor_and_rank_deficiency() {
                let a = Matrix::new([[<$t>::MAX], [<$t>::MAX]]);
                assert_eq!(a.qr().unwrap_err(), LinalgError::NotFinite);
                for scale in [1.0 as $t, $large, $tiny] {
                    let a = Matrix::new([
                        [scale, scale],
                        [2.0 * scale, 2.0 * scale],
                        [3.0 * scale, 3.0 * scale],
                    ]);
                    assert_eq!(a.qr().unwrap_err(), LinalgError::Singular);
                }
            }
        }
    };
}

qr_tests!(double, f64, 1e200_f64, 1e-200_f64);
qr_tests!(single, f32, 1e20_f32, 1e-30_f32);
