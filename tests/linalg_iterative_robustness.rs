//! Unabhängige Residuen- und Wertebereichsprüfung für die Krylov-Löser.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::linalg::iterative::{IterOptions, IterResult, bicgstab, cg, gmres, gmres_work_len};
use azmath::linalg::{LinalgError, Matrix};

macro_rules! checks {
    ($module:ident, $t:ty, $large:expr, $tiny:expr) => {
        mod $module {
            use super::*;
            fn solve(
                which: usize,
                a: &Matrix<$t, 3, 3>,
                b: &[$t; 3],
                x: &mut [$t; 3],
                opts: IterOptions<$t>,
            ) -> Result<IterResult<$t>, LinalgError> {
                let mut work = [0.0; gmres_work_len(3, 3) + 24];
                match which {
                    0 => cg(a, b, x, opts, &mut work),
                    1 => bicgstab(a, b, x, opts, &mut work),
                    _ => gmres(a, b, x, 3, opts, &mut work),
                }
            }
            fn actual_residual(a: &Matrix<$t, 3, 3>, b: &[$t; 3], x: &[$t; 3]) -> $t {
                let r: [$t; 3] =
                    core::array::from_fn(|i| b[i] - (0..3).map(|j| a[(i, j)] * x[j]).sum::<$t>());
                r.iter().fold(0.0 as $t, |s, v| s.hypot(*v))
            }
            #[test]
            fn nonzero_rhs_is_not_lost_when_squaring() {
                let a = Matrix::<$t, 3, 3>::identity();
                for which in 0..3 {
                    for scale in [$large, $tiny] {
                        let b = [scale, -2.0 * scale, 3.0 * scale];
                        let mut x = [0.0; 3];
                        let result = solve(
                            which,
                            &a,
                            &b,
                            &mut x,
                            IterOptions::new(128.0 * <$t>::EPSILON, 30),
                        )
                        .unwrap();
                        for i in 0..3 {
                            assert!((x[i] / scale - b[i] / scale).abs() < 256.0 * <$t>::EPSILON);
                        }
                        let rn = actual_residual(&a, &b, &x);
                        let bn = b.iter().fold(0.0 as $t, |s, v| s.hypot(*v));
                        assert!(rn / bn <= 128.0 * <$t>::EPSILON);
                        assert!((result.residual_norm - rn).abs() / bn <= 8.0 * <$t>::EPSILON);
                    }
                }
            }
            #[test]
            fn successful_exit_matches_true_residual_even_at_tight_tolerance() {
                let a = Matrix::new([[4.0 as $t, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]);
                let b = [1.0, 2.0, -0.7];
                let bn = b.iter().fold(0.0 as $t, |s, v| s.hypot(*v));
                for which in 0..3 {
                    for tol in [128.0 * <$t>::EPSILON, <$t>::EPSILON * 0.01] {
                        let mut x = [0.0; 3];
                        if let Ok(result) = solve(which, &a, &b, &mut x, IterOptions::new(tol, 40))
                        {
                            let rn = actual_residual(&a, &b, &x);
                            assert!(
                                rn / bn <= tol,
                                "solver {which}: true relative residual {} > {tol}",
                                rn / bn
                            );
                            assert!((result.residual_norm - rn).abs() / bn <= 8.0 * <$t>::EPSILON);
                        }
                    }
                }
            }
            #[test]
            fn invalid_initial_guess_is_rejected_even_for_zero_rhs() {
                let a = Matrix::<$t, 3, 3>::identity();
                for which in 0..3 {
                    for invalid in [<$t>::NAN, <$t>::INFINITY, <$t>::NEG_INFINITY] {
                        let mut x = [invalid, 0.0, 0.0];
                        assert_eq!(
                            solve(which, &a, &[0.0; 3], &mut x, IterOptions::default()),
                            Err(LinalgError::NotFinite)
                        );
                    }
                }
            }
            #[test]
            fn invalid_tolerances_are_rejected() {
                let a = Matrix::<$t, 3, 3>::identity();
                for which in 0..3 {
                    for tol in [-1.0 as $t, <$t>::NAN, <$t>::INFINITY, <$t>::NEG_INFINITY] {
                        let mut x = [0.0; 3];
                        assert_eq!(
                            solve(which, &a, &[1.0; 3], &mut x, IterOptions::new(tol, 10)),
                            Err(LinalgError::InvalidInput)
                        );
                    }
                }
            }

            #[test]
            fn scaled_operator_does_not_overflow_givens_rotations() {
                for scale in [$large, $tiny] {
                    let a = Matrix::<$t, 3, 3>::identity() * scale;
                    for which in 0..3 {
                        let mut x = [0.0; 3];
                        let result = solve(
                            which,
                            &a,
                            &[1.0, 0.0, 0.0],
                            &mut x,
                            IterOptions::new(128.0 * <$t>::EPSILON, 20),
                        )
                        .unwrap();
                        assert!((x[0] * scale - 1.0).abs() <= 128.0 * <$t>::EPSILON);
                        assert!(result.residual_norm <= 128.0 * <$t>::EPSILON);
                    }
                }
            }

            struct BadPreconditioner($t);
            impl azmath::linalg::iterative::Preconditioner<$t> for BadPreconditioner {
                fn apply(&self, _r: &[$t], z: &mut [$t]) {
                    z.fill(self.0);
                }
            }
            #[test]
            fn preconditioner_nonfinite_output_is_rejected() {
                use azmath::linalg::iterative::{
                    pbicgstab, pcg, pgmres, pgmres_left, pgmres_left_work_len,
                };
                let a = Matrix::<$t, 3, 3>::identity();
                for invalid in [<$t>::NAN, <$t>::INFINITY] {
                    let m = BadPreconditioner(invalid);
                    let mut work = [0.0; pgmres_left_work_len(3, 3) + 24];
                    let opts = IterOptions::default();
                    assert_eq!(
                        pcg(&a, &m, &[1.0; 3], &mut [0.0; 3], opts, &mut work),
                        Err(LinalgError::NotFinite)
                    );
                    assert_eq!(
                        pbicgstab(&a, &m, &[1.0; 3], &mut [0.0; 3], opts, &mut work),
                        Err(LinalgError::NotFinite)
                    );
                    assert_eq!(
                        pgmres(&a, &m, &[1.0; 3], &mut [0.0; 3], 3, opts, &mut work),
                        Err(LinalgError::NotFinite)
                    );
                    assert_eq!(
                        pgmres_left(&a, &m, &[1.0; 3], &mut [0.0; 3], 3, opts, &mut work),
                        Err(LinalgError::NotFinite)
                    );
                }
            }

            #[test]
            fn breakdown_and_restarted_stagnation_are_errors() {
                use azmath::linalg::iterative::{pgmres_left, pgmres_left_work_len};
                let a = Matrix::<$t, 3, 3>::zeros();
                let mut work = [0.0; pgmres_left_work_len(3, 3) + 24];
                let opts = IterOptions::new(128.0 * <$t>::EPSILON, 10);
                assert_eq!(
                    cg(&a, &[1.0; 3], &mut [0.0; 3], opts, &mut work),
                    Err(LinalgError::NotPositiveDefinite)
                );
                assert_eq!(
                    bicgstab(&a, &[1.0; 3], &mut [0.0; 3], opts, &mut work),
                    Err(LinalgError::Breakdown)
                );
                assert_eq!(
                    gmres(&a, &[1.0; 3], &mut [0.0; 3], 3, opts, &mut work),
                    Err(LinalgError::Singular)
                );
                let rotation =
                    Matrix::new([[0.0 as $t, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
                assert_eq!(
                    gmres(
                        &rotation,
                        &[1.0, 0.0, 0.0],
                        &mut [0.0; 3],
                        1,
                        opts,
                        &mut work
                    ),
                    Err(LinalgError::NoConvergence)
                );
                let identity = Matrix::<$t, 3, 3>::identity();
                assert_eq!(
                    pgmres_left(
                        &identity,
                        &BadPreconditioner(0.0),
                        &[1.0; 3],
                        &mut [0.0; 3],
                        3,
                        opts,
                        &mut work
                    ),
                    Err(LinalgError::Breakdown)
                );
                // Vor Aufruf des Vorkonditionierers müssen die Dimensionen geprüft sein.
                assert_eq!(
                    pgmres_left(
                        &identity,
                        &BadPreconditioner(<$t>::NAN),
                        &[1.0; 3],
                        &mut [0.0; 2],
                        3,
                        opts,
                        &mut work
                    ),
                    Err(LinalgError::DimensionMismatch)
                );
            }

            #[test]
            fn zero_budget_accepts_only_an_already_solved_system() {
                let a = Matrix::<$t, 3, 3>::identity();
                let b = [1.0, 2.0, 3.0];
                for which in 0..3 {
                    // Bei tol=0 darf ein unterlaufender relativer Quotient nicht als
                    // exaktes Nullresiduum gelten.
                    let mixed = [$large, $tiny, 0.0];
                    let mut nearly = [$large, 0.0, 0.0];
                    assert_eq!(
                        solve(which, &a, &mixed, &mut nearly, IterOptions::new(0.0, 0)),
                        Err(LinalgError::NoConvergence)
                    );
                    let mut x = b;
                    assert_eq!(
                        solve(which, &a, &b, &mut x, IterOptions::new(1e-5 as $t, 0))
                            .unwrap()
                            .iterations,
                        0
                    );
                    let mut x = [0.0; 3];
                    assert_eq!(
                        solve(which, &a, &b, &mut x, IterOptions::new(1e-5 as $t, 0)),
                        Err(LinalgError::NoConvergence)
                    );
                }
            }
        }
    };
}
checks!(double, f64, 1e200_f64, 1e-200_f64);
checks!(single, f32, 1e20_f32, 1e-30_f32);
