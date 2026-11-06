#![cfg(any(feature = "std", feature = "libm"))]
use azmath::linalg::{
    LinalgError, Matrix, Vector,
    iterative::{IterOptions, Preconditioner, cg, minres, minres_work_len, pminres},
};
macro_rules! tests {
    ($module:ident,$t:ty,$large:expr,$tiny:expr) => {
        mod $module {
            use super::*;
            fn options() -> IterOptions<$t> {
                IterOptions::new(128. * <$t>::EPSILON, 100)
            }
            fn residual<const N: usize>(a: &Matrix<$t, N, N>, b: &[$t; N], x: &[$t; N]) -> $t {
                (0..N)
                    .map(|i| (b[i] - (0..N).map(|j| a[(i, j)] * x[j]).sum::<$t>()).abs())
                    .fold(0., <$t>::max)
            }
            #[test]
            fn indefinite_zero_diagonal_where_cg_is_inapplicable() {
                let a = Matrix::new([[0. as $t, 1.], [1., 0.]]);
                let b = [1., 0.];
                let mut x = [0.; 2];
                let mut work = [<$t>::NAN; 16];
                assert_eq!(
                    cg(&a, &b, &mut x, options(), &mut work),
                    Err(LinalgError::NotPositiveDefinite)
                );
                x.fill(0.);
                let result = minres(&a, &b, &mut x, options(), &mut work).unwrap();
                assert_eq!(result.iterations, 2);
                assert_eq!(x, [0., 1.]);
                assert_eq!(result.residual_norm, 0.);
                let (wrapped, r) = a.solve_minres(&Vector::new(b), options()).unwrap();
                assert_eq!(wrapped.as_slice(), &x);
                assert_eq!(r, result);
            }
            #[test]
            fn symmetric_indefinite_problems_match_lu() {
                let mut state = 17u64;
                for case in 0..32 {
                    let mut data = [[0. as $t; 4]; 4];
                    for i in 0..4 {
                        data[i][i] = if i % 2 == 0 {
                            -(i as $t + 2.)
                        } else {
                            i as $t + 2.
                        };
                        for j in 0..i {
                            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                            let v = ((state >> 48) as $t / 65536. - 0.5) * 0.25;
                            data[i][j] = v;
                            data[j][i] = v;
                        }
                    }
                    let a = Matrix::new(data);
                    let b = [1., 2., -1., 0.5];
                    let expected = a.solve(&Vector::new(b)).unwrap();
                    let mut x = [0.; 4];
                    let mut work = [0.; 32];
                    let result = minres(&a, &b, &mut x, options(), &mut work).unwrap();
                    let actual = residual(&a, &b, &x);
                    assert!(actual < 512. * <$t>::EPSILON, "case {case}: {actual}");
                    assert!(result.residual_norm >= actual);
                    assert!(result.residual_norm <= options().tol * 2.5);
                    for i in 0..4 {
                        assert!((x[i] - expected[i]).abs() < 1024. * <$t>::EPSILON);
                    }
                }
            }
            struct AbsDiagonal;
            impl Preconditioner<$t> for AbsDiagonal {
                fn apply(&self, r: &[$t], z: &mut [$t]) {
                    for (i, (r, z)) in r.iter().zip(z).enumerate() {
                        *z = *r / [0.01, 0.01, 1., 1., 100., 100.][i];
                    }
                }
            }
            #[test]
            fn spd_preconditioner_reduces_iterations_and_checks_original_residual() {
                let mut data = [[0. as $t; 6]; 6];
                for (i, d) in [-0.01, 0.01, -1., 1., -100., 100.].into_iter().enumerate() {
                    data[i][i] = d;
                }
                let a = Matrix::new(data);
                let b = [1.; 6];
                let mut cold = [0.; 6];
                let mut warm = [0.; 6];
                let mut work = [0.; 48];
                let plain = minres(&a, &b, &mut cold, options(), &mut work).unwrap();
                let pre = pminres(&a, &AbsDiagonal, &b, &mut warm, options(), &mut work).unwrap();
                println!(
                    "MINRES {} iterations; preconditioned {}",
                    plain.iterations, pre.iterations
                );
                assert!(pre.iterations < plain.iterations);
                assert!(pre.iterations <= 3);
                assert!(residual(&a, &b, &warm) <= pre.residual_norm);
                assert!(pre.residual_norm <= options().tol * (6. as $t).sqrt());
            }
            #[test]
            fn extreme_rhs_and_operator_scales() {
                for scale in [$large as $t, $tiny as $t] {
                    for (op, rhs) in [(1., scale), (scale, scale)] {
                        let a = Matrix::new([[2. * op, op], [op, -2. * op]]);
                        let b = [rhs, -2. * rhs];
                        let mut x = [0.; 2];
                        let mut work = [0.; minres_work_len(2)];
                        let result = minres(&a, &b, &mut x, options(), &mut work).unwrap();
                        assert!(x.iter().all(|v| v.is_finite()));
                        assert!((x[1] / (rhs / op) - 1.).abs() < 512. * <$t>::EPSILON);
                        assert!(result.residual_norm / rhs.abs() <= options().tol * 3.);
                    }
                }
            }
            #[test]
            fn starts_zero_rhs_empty_and_singletons() {
                let a = Matrix::new([[0. as $t, 1.], [1., 0.]]);
                let b = [1., 0.];
                let mut x = [0., 1.];
                let mut work = [0.; minres_work_len(2)];
                assert_eq!(
                    minres(&a, &b, &mut x, IterOptions::new(0., 0), &mut work)
                        .unwrap()
                        .iterations,
                    0
                );
                assert_eq!(
                    minres(&a, &b, &mut [0.; 2], IterOptions::new(0., 0), &mut work),
                    Err(LinalgError::NoConvergence)
                );
                x.fill(7.);
                let result = minres(&a, &[0., 0.], &mut x, options(), &mut work).unwrap();
                assert_eq!(x, [0., 0.]);
                assert_eq!(result.iterations, 0);
                let empty = Matrix::<$t, 0, 0>::zeros();
                assert_eq!(
                    minres(&empty, &[], &mut [], options(), &mut [])
                        .unwrap()
                        .residual_norm,
                    0.
                );
                let one = Matrix::new([[-2. as $t]]);
                let mut one_x = [3.];
                assert_eq!(
                    minres(&one, &[4.], &mut one_x, options(), &mut work)
                        .unwrap()
                        .iterations,
                    1
                );
                assert_eq!(one_x, [-2.]);
                // A non-exact initial guess is corrected, not overwritten by a zero start.
                x = [0.3, -0.2];
                minres(&a, &b, &mut x, options(), &mut work).unwrap();
                assert!(residual(&a, &b, &x) < 512. * <$t>::EPSILON);
            }
            struct Negative;
            impl Preconditioner<$t> for Negative {
                fn apply(&self, r: &[$t], z: &mut [$t]) {
                    for (r, z) in r.iter().zip(z) {
                        *z = -*r;
                    }
                }
            }
            struct Invalid;
            impl Preconditioner<$t> for Invalid {
                fn apply(&self, _: &[$t], z: &mut [$t]) {
                    z.fill(<$t>::NAN);
                }
            }
            struct Zero;
            impl Preconditioner<$t> for Zero {
                fn apply(&self, _: &[$t], z: &mut [$t]) {
                    z.fill(0.);
                }
            }
            #[test]
            fn wrong_preconditioners_and_nonfinite_data() {
                let a = Matrix::<$t, 2, 2>::identity();
                let mut x = [0.; 2];
                let mut work = [0.; minres_work_len(2)];
                assert_eq!(
                    pminres(&a, &Negative, &[1., 1.], &mut x, options(), &mut work),
                    Err(LinalgError::NotPositiveDefinite)
                );
                assert_eq!(
                    pminres(&a, &Zero, &[1., 1.], &mut x, options(), &mut work),
                    Err(LinalgError::NotPositiveDefinite)
                );
                assert_eq!(
                    pminres(&a, &Invalid, &[1., 1.], &mut x, options(), &mut work),
                    Err(LinalgError::NotFinite)
                );
                for bad in [<$t>::NAN, <$t>::INFINITY, <$t>::NEG_INFINITY] {
                    assert_eq!(
                        minres(&a, &[bad, 1.], &mut x, options(), &mut work),
                        Err(LinalgError::NotFinite)
                    );
                    assert_eq!(
                        minres(&a, &[1., 1.], &mut [bad, 0.], options(), &mut work),
                        Err(LinalgError::NotFinite)
                    );
                    let op = Matrix::new([[bad, 0.], [0., 1.]]);
                    assert_eq!(
                        minres(&op, &[1., 1.], &mut x, options(), &mut work),
                        Err(LinalgError::NotFinite)
                    );
                }
            }
            #[test]
            fn errors_iteration_limit_and_singular_systems() {
                let a = Matrix::new([[1. as $t, 0.], [0., 2.]]);
                let b = [1., 1.];
                let mut x = [0.; 2];
                let mut work = [0.; minres_work_len(2)];
                for tol in [-1., <$t>::NAN, <$t>::INFINITY] {
                    assert_eq!(
                        minres(&a, &b, &mut x, IterOptions::new(tol, 10), &mut work),
                        Err(LinalgError::InvalidInput)
                    );
                }
                assert_eq!(
                    minres(&a, &b, &mut x, options(), &mut work[..15]),
                    Err(LinalgError::DimensionMismatch)
                );
                assert_eq!(
                    minres(&a, &b, &mut x[..1], options(), &mut work),
                    Err(LinalgError::DimensionMismatch)
                );
                assert_eq!(
                    minres(&a, &b[..1], &mut x, options(), &mut work),
                    Err(LinalgError::DimensionMismatch)
                );
                assert_eq!(
                    minres(&a, &b, &mut x, IterOptions::new(0., 1), &mut work),
                    Err(LinalgError::NoConvergence)
                );
                assert!(x.iter().all(|v| v.is_finite()));
                assert!(x.iter().any(|v| *v != 0.));
                let singular = Matrix::new([[1. as $t, 0.], [0., 0.]]);
                x.fill(0.);
                minres(&singular, &[1., 0.], &mut x, options(), &mut work).unwrap();
                assert_eq!(x, [1., 0.]);
                x.fill(0.);
                assert_eq!(
                    minres(&singular, &[0., 1.], &mut x, options(), &mut work),
                    Err(LinalgError::Breakdown)
                );
            }
        }
    };
}
tests!(single, f32, 1e30, 1e-30);
tests!(double, f64, 1e200, 1e-200);
#[cfg(feature = "alloc")]
#[test]
fn csr_and_dynamic_convenience_match_fixed_matrix() {
    use azmath::{
        linalg::{DynMatrix, DynVector},
        sparse::CsrMatrix,
    };
    let a = Matrix::new([[0.0f64, 1.], [1., 0.]]);
    let b = DynVector::new(vec![1., 2.]);
    let opts = IterOptions::new(1e-12, 10);
    let csr = CsrMatrix::<f64>::from_dense_slice(2, 2, &[0., 1., 1., 0.], 0.).unwrap();
    let mut x = [0.; 2];
    let mut work = [0.; minres_work_len(2)];
    minres(&csr, b.as_slice(), &mut x, opts, &mut work).unwrap();
    assert!((x[0] - 2.).abs() < 1e-12);
    let d = DynMatrix::new(2, 2, vec![0., 1., 1., 0.]).unwrap();
    let (y, _) = d.solve_minres(&b, opts).unwrap();
    let (z, _) = a.solve_minres(&Vector::new([1., 2.]), opts).unwrap();
    for i in 0..2 {
        assert!((y[i] - z[i]).abs() < 1e-12);
    }
    assert_eq!(
        d.solve_minres(&DynVector::zeros(1), opts),
        Err(LinalgError::DimensionMismatch)
    );
}
