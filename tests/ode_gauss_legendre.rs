#![cfg(any(feature = "std", feature = "libm"))]

use azmath::ode::{OdeError, gauss_legendre4, gauss_legendre4_fd, gauss_legendre4_work_len};
use azmath::optimize::OptimizeOptions;

macro_rules! cases {
    ($module:ident, $t:ty, $tol:expr, $error:expr) => {
        mod $module {
            use super::*;
            type T = $t;
            fn opts() -> OptimizeOptions<T> {
                OptimizeOptions {
                    tol_x: 0.0,
                    ..OptimizeOptions::with_tolerance($tol)
                }
            }
            fn rhs(_: T, y: &[T], out: &mut [T]) {
                out[0] = y[1];
                out[1] = -y[0];
            }
            fn jac(_: T, _: &[T], j: &mut [T]) {
                j.copy_from_slice(&[0.0, 1.0, -1.0, 0.0]);
            }
            fn run(y: &mut [T], start: T, end: T, steps: usize) {
                let mut work = [0.0; gauss_legendre4_work_len(2)];
                gauss_legendre4(rhs, jac, start, end, y, steps, opts(), &mut work, |_, _| {})
                    .unwrap();
            }
            #[test]
            fn fourth_order() {
                let exact = [<T>::cos(1.0), -<T>::sin(1.0)];
                let mut errors = [0.0; 2];
                for (i, steps) in [1, 2].into_iter().enumerate() {
                    let mut y = [1.0, 0.0];
                    run(&mut y, 0.0, 1.0, steps);
                    errors[i] = (y[0] - exact[0]).hypot(y[1] - exact[1]);
                }
                let ratio = errors[0] / errors[1];
                assert!(
                    (14.0..17.0).contains(&ratio),
                    "ratio={ratio}, errors={errors:?}"
                );
            }
            #[test]
            fn time_reversal_and_quadratic_energy() {
                let mut y = [1.0, 0.3];
                let initial = y;
                let e0 = y[0] * y[0] + y[1] * y[1];
                let mut work = [0.0; gauss_legendre4_work_len(2)];
                let mut max_error: T = 0.0;
                gauss_legendre4(
                    rhs,
                    jac,
                    0.0,
                    100.0,
                    &mut y,
                    1000,
                    opts(),
                    &mut work,
                    |_, z| {
                        max_error = max_error.max((z[0] * z[0] + z[1] * z[1] - e0).abs());
                    },
                )
                .unwrap();
                assert!(max_error < $error * 20.0, "energy={max_error}");
                run(&mut y, 100.0, 0.0, 1000);
                assert!((y[0] - initial[0]).abs() < $error * 40.0);
                assert!((y[1] - initial[1]).abs() < $error * 40.0);
            }
            #[test]
            fn nonseparable_hamiltonian() {
                // H=(q²+p²)/2+q*p/2, so q'=p+q/2, p'=-q-p/2.
                let mut y = [1.0, 0.0];
                let mut work = [0.0; gauss_legendre4_work_len(2)];
                gauss_legendre4(
                    |_, z, out| {
                        out[0] = z[1] + 0.5 * z[0];
                        out[1] = -z[0] - 0.5 * z[1];
                    },
                    |_, _, j| j.copy_from_slice(&[0.5, 1.0, -1.0, -0.5]),
                    0.0,
                    1.0,
                    &mut y,
                    20,
                    opts(),
                    &mut work,
                    |_, z| {
                        let h = (z[0] * z[0] + z[1] * z[1] + z[0] * z[1]) * 0.5;
                        assert!((h - 0.5).abs() < $error * 4.0);
                    },
                )
                .unwrap();
                let w = (0.75 as T).sqrt();
                assert!((y[0] - (w.cos() + 0.5 * w.sin() / w)).abs() < $error * 4.0);
                assert!((y[1] + w.sin() / w).abs() < $error * 4.0);
            }
            #[test]
            fn numerical_jacobian_and_nonlinearity() {
                let mut y = [0.5];
                let mut work = [0.0; gauss_legendre4_work_len(1)];
                gauss_legendre4_fd(
                    |_, z, out| out[0] = z[0] * z[0],
                    0.0,
                    1.0,
                    &mut y,
                    20,
                    opts(),
                    &mut work,
                    |_, _| {},
                )
                .unwrap();
                assert!((y[0] - 1.0).abs() < $error * 8.0, "y={y:?}");
            }
            #[test]
            fn analytic_nonlinear_jacobian() {
                let mut y = [0.5];
                let mut work = [0.0; gauss_legendre4_work_len(1)];
                gauss_legendre4(
                    |_, z, out| out[0] = z[0] * z[0],
                    |_, z, j| j[0] = 2.0 * z[0],
                    0.0,
                    1.0,
                    &mut y,
                    20,
                    opts(),
                    &mut work,
                    |_, _| {},
                )
                .unwrap();
                assert!((y[0] - 1.0).abs() < $error * 8.0);
            }
            #[test]
            fn nonautonomous_stage_times_and_observer() {
                let mut y = [0.0];
                let mut work = [0.0; gauss_legendre4_work_len(1)];
                let mut calls = 0;
                gauss_legendre4(
                    |t, _, out| out[0] = t * t * t,
                    |_, _, j| j[0] = 0.0,
                    0.0,
                    1.0,
                    &mut y,
                    4,
                    opts(),
                    &mut work,
                    |t, z| {
                        calls += 1;
                        assert!((z[0] - t.powi(4) / 4.0).abs() < $error);
                    },
                )
                .unwrap();
                assert_eq!(calls, 5);
                assert!((y[0] - 0.25).abs() < $error);
            }
            #[test]
            fn empty_interval_has_no_callbacks() {
                let mut y = [1.0];
                let mut work = [0.0; gauss_legendre4_work_len(1)];
                let mut observed = 0;
                gauss_legendre4(
                    |_, _, _| panic!("rhs"),
                    |_, _, _| panic!("jac"),
                    2.0,
                    2.0,
                    &mut y,
                    3,
                    opts(),
                    &mut work,
                    |t, z| {
                        observed += 1;
                        assert_eq!(t, 2.0);
                        assert_eq!(z, &[1.0]);
                    },
                )
                .unwrap();
                assert_eq!(observed, 1);
            }
            #[test]
            fn input_validation_precedes_callbacks() {
                let mut work = [0.0; gauss_legendre4_work_len(1)];
                for (state, start, end, steps, length, options, expected) in [
                    (1.0, 0.0, 1.0, 0, work.len(), opts(), OdeError::InvalidInput),
                    (
                        1.0,
                        0.0,
                        1.0,
                        1,
                        work.len() - 1,
                        opts(),
                        OdeError::InvalidInput,
                    ),
                    (T::NAN, 0.0, 1.0, 1, work.len(), opts(), OdeError::NotFinite),
                    (
                        1.0,
                        T::INFINITY,
                        1.0,
                        1,
                        work.len(),
                        opts(),
                        OdeError::NotFinite,
                    ),
                    (
                        1.0,
                        0.0,
                        1.0,
                        1,
                        work.len(),
                        OptimizeOptions::with_tolerance(-1.0),
                        OdeError::InvalidInput,
                    ),
                    (
                        1.0,
                        0.0,
                        1.0,
                        1,
                        work.len(),
                        OptimizeOptions::with_tolerance(T::NAN),
                        OdeError::InvalidInput,
                    ),
                ] {
                    let result = gauss_legendre4(
                        |_, _, _| panic!("rhs"),
                        |_, _, _| panic!("jac"),
                        start,
                        end,
                        &mut [state],
                        steps,
                        options,
                        &mut work[..length],
                        |_, _| panic!("observer"),
                    );
                    assert_eq!(result, Err(expected));
                }
                assert_eq!(
                    gauss_legendre4(
                        |_, _, _| {},
                        |_, _, _| {},
                        0.0 as T,
                        1.0,
                        &mut [],
                        1,
                        opts(),
                        &mut work,
                        |_, _| {}
                    ),
                    Err(OdeError::InvalidInput)
                );
            }
            #[test]
            fn nonfinite_rhs_and_jacobian() {
                let mut work = [0.0; gauss_legendre4_work_len(1)];
                for bad_jac in [false, true] {
                    let mut y = [1.0];
                    let result = gauss_legendre4(
                        |_, _, out| out[0] = if bad_jac { 1.0 } else { T::INFINITY },
                        |_, _, j| j[0] = T::NAN,
                        0.0,
                        1.0,
                        &mut y,
                        1,
                        opts(),
                        &mut work,
                        |_, _| {},
                    );
                    assert_eq!(result, Err(OdeError::NotFinite));
                    assert_eq!(y, [1.0]);
                }
            }
            #[test]
            fn newton_limit_preserves_state() {
                let mut y = [1.0, 0.0];
                let mut work = [0.0; gauss_legendre4_work_len(2)];
                let mut options = opts();
                options.max_iter = 0;
                assert_eq!(
                    gauss_legendre4(rhs, jac, 0.0, 1.0, &mut y, 1, options, &mut work, |_, _| {}),
                    Err(OdeError::NewtonFailed)
                );
                assert_eq!(y, [1.0, 0.0]);
            }
            #[test]
            fn later_failure_preserves_last_complete_step() {
                let mut y = [0.0];
                let mut work = [0.0; gauss_legendre4_work_len(1)];
                let mut observed = 0;
                let result = gauss_legendre4(
                    |t, _, out| out[0] = if t > 0.6 { T::NAN } else { 1.0 },
                    |_, _, j| j[0] = 0.0,
                    0.0,
                    1.0,
                    &mut y,
                    2,
                    opts(),
                    &mut work,
                    |_, _| observed += 1,
                );
                assert_eq!(result, Err(OdeError::NotFinite));
                assert!((y[0] - 0.5).abs() < $error);
                assert_eq!(observed, 2);
            }
            #[test]
            fn unrepresentable_step() {
                let start = 1.0 as T;
                let end = start + T::EPSILON;
                let mut y = [1.0];
                let mut work = [0.0; gauss_legendre4_work_len(1)];
                assert_eq!(
                    gauss_legendre4(
                        |_, _, _| panic!("rhs"),
                        |_, _, _| panic!("jac"),
                        start,
                        end,
                        &mut y,
                        8,
                        opts(),
                        &mut work,
                        |_, _| {}
                    ),
                    Err(OdeError::StepSizeTooSmall)
                );
                assert_eq!(y, [1.0]);
            }
        }
    };
}
cases!(f64_cases, f64, 2e-14, 2e-8);
cases!(f32_cases, f32, 3e-7, 5e-6);

#[test]
fn workspace_size_saturates() {
    assert_eq!(gauss_legendre4_work_len(0), 0);
    assert_eq!(gauss_legendre4_work_len(1), 26);
    assert_eq!(gauss_legendre4_work_len(usize::MAX), usize::MAX);
}
