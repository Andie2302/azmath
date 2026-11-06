//! Rank, reconstruction and least-squares checks for column-pivoted QR.
#![cfg(any(feature = "std", feature = "libm"))]
use azmath::linalg::{LinalgError, Matrix, Vector};

macro_rules! checks {
    ($module:ident, $t:ty, $small:expr, $large:expr, $gap:expr) => {
        mod $module {
            use super::*;
            fn close(a: $t, b: $t) {
                assert!(
                    (a - b).abs() <= 128.0 * <$t>::EPSILON * b.abs().max(1.0),
                    "{a:e} != {b:e}"
                );
            }

            #[test]
            fn factors_reconstruct_permuted_matrix_and_q_is_orthogonal() {
                let a = Matrix::new([
                    [1.0 as $t, 3.0, -2.0],
                    [2.0, -4.0, 5.0],
                    [-1.0, 7.0, 2.0],
                    [4.0, 1.0, 3.0],
                ]);
                let qr = a.qr_pivoted().unwrap();
                let q = qr.q();
                let r = qr.r();
                for i in 0..4 {
                    for j in 0..3 {
                        let mut value = 0.0;
                        for k in 0..3 {
                            value += q[(i, k)] * r[(k, j)];
                        }
                        close(value, a[(i, qr.permutation()[j])]);
                    }
                }
                let qtq = q.transpose() * q;
                for i in 0..4 {
                    for j in 0..4 {
                        close(qtq[(i, j)], if i == j { 1.0 } else { 0.0 });
                    }
                }
                assert_eq!(qr.rank(), 3);
                // Unpivoting must also work for a permutation cycle of length three.
                let d = Matrix::new([[1.0 as $t, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]]);
                let dq = d.qr_pivoted().unwrap();
                assert_eq!(dq.permutation(), &[1, 2, 0]);
                let x = dq.solve(&Vector::new([2.0, -3.0, 8.0])).unwrap();
                for (v, w) in x.as_array().iter().zip([2.0, -1.0, 4.0]) {
                    close(*v, w);
                }
            }

            #[test]
            fn rank_deficient_basic_solution_matches_svd_residual() {
                let a = Matrix::new([[1.0 as $t, 2.0], [2.0, 4.0], [0.0, 0.0]]);
                let b = Vector::new([3.0 as $t, 6.0, 2.0]);
                let qr = a.qr_pivoted().unwrap();
                assert_eq!(qr.rank(), 1);
                let x = qr.solve(&b).unwrap();
                assert_eq!(x[0], 0.0);
                close(x[1], 1.5);
                let min_norm = a.svd().unwrap().solve(&b);
                let ax = a * x;
                let am = a * min_norm;
                for i in 0..3 {
                    close(ax[i], am[i]);
                }
                close(qr.residual_norm(&b).unwrap(), 2.0);
                // The CPQR basis solution deliberately has a larger norm than SVD's solution.
                assert!(x[0].hypot(x[1]) > min_norm[0].hypot(min_norm[1]));
            }

            #[test]
            fn near_dependence_and_custom_absolute_rank_threshold() {
                let eps: $t = $gap;
                let a = Matrix::new([[1.0 as $t, 1.0], [0.0, eps], [0.0, 0.0]]);
                let qr = a.qr_pivoted().unwrap();
                assert_eq!(qr.rank(), 2);
                assert_eq!(qr.rank_with_tolerance(2.0 * eps).unwrap(), 1);
                assert_eq!(qr.rank_with_tolerance(qr.r()[(1, 1)].abs()).unwrap(), 1);
                let x = qr.solve(&Vector::new([2.0, 2.0 * eps, 0.0])).unwrap();
                close(x[0], 0.0);
                close(x[1], 2.0);
                for tol in [-1.0, <$t>::NAN, <$t>::INFINITY] {
                    assert_eq!(qr.rank_with_tolerance(tol), Err(LinalgError::InvalidInput));
                    assert_eq!(
                        qr.solve_with_tolerance(&Vector::zeros(), tol),
                        Err(LinalgError::InvalidInput)
                    );
                }
            }

            #[test]
            fn full_rank_noisy_fit_is_stationary_and_scale_invariant() {
                for scale in [$small as $t, 1.0, $large] {
                    let a = Matrix::new([[scale, -scale], [scale, 0.0], [scale, scale]]);
                    let b = Vector::new([4.0 * scale, 0.0, 2.0 * scale]);
                    let qr = a.qr_pivoted().unwrap();
                    assert_eq!(qr.rank(), 2);
                    let x = a.least_squares(&b).unwrap();
                    close(x[0], 2.0);
                    close(x[1], -1.0);
                    close(qr.residual_norm(&b).unwrap() / scale, (6.0 as $t).sqrt());
                    let r = [x[0] - x[1] - 4.0, x[0], x[0] + x[1] - 2.0];
                    close(r[0] + r[1] + r[2], 0.0);
                    close(-r[0] + r[2], 0.0);
                }
            }

            #[test]
            fn column_scaling_affects_rank_and_is_explicitly_controllable() {
                let tiny = <$t>::EPSILON / 4.0;
                let a = Matrix::new([[tiny, 0.0], [0.0, 1.0]]);
                let qr = a.qr_pivoted().unwrap();
                assert_eq!(qr.permutation(), &[1, 0]);
                assert_eq!(qr.rank(), 1);
                assert_eq!(qr.rank_with_tolerance(0.0).unwrap(), 2);
                let x = qr
                    .solve_with_tolerance(&Vector::new([2.0 * tiny, 0.0]), 0.0)
                    .unwrap();
                close(x[0], 2.0);
                close(x[1], 0.0);
            }

            #[test]
            fn zero_empty_and_extreme_one_by_one() {
                let z = Matrix::<$t, 3, 2>::zeros().qr_pivoted().unwrap();
                assert_eq!(z.rank(), 0);
                assert_eq!(
                    z.solve(&Vector::new([3.0, 4.0, 0.0])).unwrap(),
                    Vector::zeros()
                );
                close(z.residual_norm(&Vector::new([3.0, 4.0, 0.0])).unwrap(), 5.0);
                let empty = Matrix::<$t, 0, 0>::zeros().qr_pivoted().unwrap();
                assert_eq!(empty.rank(), 0);
                assert_eq!(empty.solve(&Vector::zeros()).unwrap(), Vector::zeros());
                let no_columns = Matrix::<$t, 2, 0>::zeros().qr_pivoted().unwrap();
                close(
                    no_columns.residual_norm(&Vector::new([3.0, 4.0])).unwrap(),
                    5.0,
                );
                for v in [<$t>::MAX, <$t>::MIN_POSITIVE, <$t>::from_bits(1)] {
                    let qr = Matrix::new([[v]]).qr_pivoted().unwrap();
                    assert_eq!(qr.rank(), 1);
                    close(qr.solve(&Vector::new([v])).unwrap()[0], 1.0);
                }
            }

            #[test]
            fn errors_are_checked_in_factors_rhs_and_solution() {
                assert_eq!(
                    Matrix::<$t, 1, 2>::zeros().qr_pivoted(),
                    Err(LinalgError::DimensionMismatch)
                );
                for v in [<$t>::NAN, <$t>::INFINITY] {
                    assert_eq!(Matrix::new([[v]]).qr_pivoted(), Err(LinalgError::NotFinite));
                    let qr = Matrix::new([[1.0 as $t]]).qr_pivoted().unwrap();
                    assert_eq!(qr.solve(&Vector::new([v])), Err(LinalgError::NotFinite));
                    assert_eq!(
                        qr.residual_norm(&Vector::new([v])),
                        Err(LinalgError::NotFinite)
                    );
                }
                assert_eq!(
                    Matrix::new([[<$t>::MAX], [<$t>::MAX]]).qr_pivoted(),
                    Err(LinalgError::NotFinite)
                );
                let qr = Matrix::new([[<$t>::MIN_POSITIVE]]).qr_pivoted().unwrap();
                assert_eq!(
                    qr.solve(&Vector::new([<$t>::MAX])),
                    Err(LinalgError::NotFinite)
                );
            }
        }
    };
}
checks!(single, f32, 1e-20, 1e20, 1e-4);
checks!(double, f64, 1e-200, 1e200, 1e-8);

#[cfg(feature = "alloc")]
mod dynamic {
    use super::*;
    use azmath::linalg::{DynMatrix, DynVector};
    #[test]
    fn dynamic_and_fixed_agree_including_permutation_factors_and_errors() {
        let a = Matrix::new([
            [1.0f64, 2.0, 0.0],
            [2.0, 4.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
        ]);
        let fixed = a.qr_pivoted().unwrap();
        let matrix = DynMatrix::new(4, 3, a.as_array().as_flattened().to_vec()).unwrap();
        let qr = matrix.qr_pivoted().unwrap();
        assert_eq!(qr.rank(), fixed.rank());
        assert_eq!(qr.permutation(), fixed.permutation());
        assert_eq!(qr.q().as_slice(), fixed.q().as_array().as_flattened());
        assert_eq!(qr.r().as_slice(), fixed.r().as_array().as_flattened());
        let b = Vector::new([3.0, 6.0, 9.0, 2.0]);
        let db = DynVector::new(b.as_array().to_vec());
        assert_eq!(
            qr.solve(&db).unwrap().as_slice(),
            fixed.solve(&b).unwrap().as_array()
        );
        assert_eq!(qr.residual_norm(&db), fixed.residual_norm(&b));
        assert_eq!(matrix.least_squares(&db), qr.solve(&db));
        assert_eq!(
            qr.solve(&DynVector::zeros(3)),
            Err(LinalgError::DimensionMismatch)
        );
        assert_eq!(
            qr.residual_norm(&DynVector::zeros(3)),
            Err(LinalgError::DimensionMismatch)
        );
        assert_eq!(
            qr.rank_with_tolerance(f64::NAN),
            Err(LinalgError::InvalidInput)
        );
        assert_eq!(
            qr.solve_with_tolerance(&db, -1.0),
            Err(LinalgError::InvalidInput)
        );
        assert_eq!(
            DynMatrix::<f64>::zeros(1, 2).qr_pivoted(),
            Err(LinalgError::DimensionMismatch)
        );
        let empty = DynMatrix::<f64>::zeros(0, 0).qr_pivoted().unwrap();
        assert_eq!(empty.rank(), 0);
        assert!(
            empty
                .solve(&DynVector::zeros(0))
                .unwrap()
                .as_slice()
                .is_empty()
        );
    }
}
