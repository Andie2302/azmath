#![cfg(any(feature = "std", feature = "libm"))]
use azmath::linalg::{
    LinalgError, Matrix,
    eigen::{arnoldi, arnoldi_work_len},
};
macro_rules! checks {
    ($module:ident,$t:ty,$large:expr,$tiny:expr) => {
        mod $module {
            use super::*;
            #[test]
            fn relation_orthogonality_and_hessenberg_structure() {
                let a = Matrix::new([
                    [1.0 as $t, 2.0, 0.0, 1.0],
                    [0.0, 3.0, 1.0, 0.0],
                    [2.0, 0.0, 4.0, 1.0],
                    [0.0, 1.0, 0.0, 5.0],
                ]);
                let mut h = [<$t>::NAN; 12];
                let mut work = [<$t>::NAN; arnoldi_work_len(4, 3)];
                let m = arnoldi(&a, &[1.0, 2.0, -1.0, 0.5], 3, &mut h, &mut work).unwrap();
                assert_eq!(m, 3);
                let tol = 256.0 * <$t>::EPSILON;
                for i in 0..4 {
                    for j in 0..4 {
                        let dot = (0..4)
                            .map(|r| work[i * 4 + r] * work[j * 4 + r])
                            .sum::<$t>();
                        assert!((dot - if i == j { 1.0 } else { 0.0 }).abs() < tol);
                    }
                }
                for j in 0..m {
                    for r in 0..4 {
                        let av = (0..4).map(|c| a[(r, c)] * work[j * 4 + c]).sum::<$t>();
                        let vh = (0..=m).map(|i| work[i * 4 + r] * h[i * 3 + j]).sum::<$t>();
                        assert!((av - vh).abs() < tol * 10.0);
                    }
                }
                for j in 0..3 {
                    for i in j + 2..4 {
                        assert_eq!(h[i * 3 + j], 0.0);
                    }
                }
            }
            #[test]
            fn invariant_subspace_early_exit_and_fresh_buffers() {
                let a = Matrix::<$t, 3, 3>::identity();
                let mut h = [<$t>::NAN; 12];
                let mut w = [<$t>::NAN; arnoldi_work_len(3, 3)];
                assert_eq!(arnoldi(&a, &[1.0, 0.0, 0.0], 3, &mut h, &mut w).unwrap(), 1);
                assert_eq!(h[0], 1.0);
                assert_eq!(h[3], 0.0);
                assert!(w[..6].iter().all(|v| v.is_finite()));
            }
            #[test]
            fn extreme_start_normalization_and_errors() {
                let a = Matrix::<$t, 2, 2>::identity();
                let mut h = [0.0; 6];
                let mut w = [0.0; arnoldi_work_len(2, 2)];
                for scale in [$large as $t, $tiny] {
                    assert_eq!(arnoldi(&a, &[scale, scale], 2, &mut h, &mut w).unwrap(), 1);
                    assert!(w[..2].iter().all(|v| v.is_finite()));
                }
                assert_eq!(
                    arnoldi(&a, &[0.0, 0.0], 2, &mut h, &mut w),
                    Err(LinalgError::Breakdown)
                );
                assert_eq!(
                    arnoldi(&a, &[<$t>::NAN, 0.0], 2, &mut h, &mut w),
                    Err(LinalgError::NotFinite)
                );
                assert_eq!(
                    arnoldi(&a, &[1.0, 0.0], 0, &mut h, &mut w),
                    Err(LinalgError::DimensionMismatch)
                );
                let bad = Matrix::new([[<$t>::INFINITY, 0.0], [0.0, 1.0]]);
                assert_eq!(
                    arnoldi(&bad, &[1.0, 0.0], 2, &mut h, &mut w),
                    Err(LinalgError::NotFinite)
                );
            }
        }
    };
}
checks!(single, f32, 1e30, 1e-30);
checks!(double, f64, 1e200, 1e-200);
