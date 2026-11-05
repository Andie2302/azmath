//! Randfälle und unabhängige Gegenproben für die grundlegenden ODE-Verfahren.
#![cfg(any(feature = "std", feature = "libm"))]
use azmath::ode::*;
use azmath::optimize::OptimizeOptions;

macro_rules! checks {
    ($module:ident,$t:ty) => {
        mod $module {
            use super::*;
            #[test]
            fn empty_intervals_do_not_evaluate_rhs() {
                let mut y = [1.0 as $t];
                let mut work = [0.0; implicit_work_len(1) + rk45_work_len(1)];
                let f = |_: $t, _: &[$t], _: &mut [$t]| panic!("leeres Intervall");
                let j = f;
                euler(f, 1.0, 1.0, &mut y, 3, &mut work, |_, _| {}).unwrap();
                rk4(f, 1.0, 1.0, &mut y, 3, &mut work, |_, _| {}).unwrap();
                implicit_euler(
                    f,
                    j,
                    1.0,
                    1.0,
                    &mut y,
                    3,
                    OptimizeOptions::default(),
                    &mut work,
                    |_, _| {},
                )
                .unwrap();
                implicit_trapezoid(
                    f,
                    j,
                    1.0,
                    1.0,
                    &mut y,
                    3,
                    OptimizeOptions::default(),
                    &mut work,
                    |_, _| {},
                )
                .unwrap();
                assert_eq!(y, [1.0]);
            }
            #[test]
            fn fixed_time_grid_must_make_representable_progress() {
                let t = 2.0 / <$t>::EPSILON;
                let mut y = [0.0];
                let mut work = [0.0; implicit_work_len(1) + rk45_work_len(1)];
                let max = <$t>::MAX;
                let tiny = |time: $t, _: &[$t], dy: &mut [$t]| {
                    assert!(time.is_finite());
                    dy[0] = 1.0 / max;
                };
                euler(tiny, -max, max, &mut y, 2, &mut work, |_, _| {}).unwrap();
                assert!((y[0] - 2.0).abs() < 512.0 * <$t>::EPSILON);
                y[0] = 0.0;
                rk4(tiny, -max, max, &mut y, 2, &mut work, |_, _| {}).unwrap();
                assert!((y[0] - 2.0).abs() < 512.0 * <$t>::EPSILON);
                y[0] = 0.0;
                let f = |_: $t, _: &[$t], dy: &mut [$t]| dy[0] = 1.0;
                assert_eq!(
                    euler(f, t, t + 4.0, &mut y, 8, &mut work, |_, _| {}),
                    Err(OdeError::StepSizeTooSmall)
                );
                y[0] = 0.0;
                assert_eq!(
                    rk4(f, t, t + 4.0, &mut y, 8, &mut work, |_, _| {}),
                    Err(OdeError::StepSizeTooSmall)
                );
            }
            #[test]
            fn fixed_states_match_observer_times_on_rounded_grids() {
                let t = 2.0 / <$t>::EPSILON;
                let mut work = [0.0; rk45_work_len(1)];
                let f = |_: $t, _: &[$t], dy: &mut [$t]| dy[0] = 1.0;
                let mut y = [0.0];
                euler(f, t, t + 6.0, &mut y, 2, &mut work, |now, state| {
                    assert_eq!(state[0], now - t)
                })
                .unwrap();
                y[0] = 0.0;
                rk4(f, t, t + 6.0, &mut y, 2, &mut work, |now, state| {
                    assert!((state[0] - (now - t)).abs() < 32.0 * <$t>::EPSILON)
                })
                .unwrap();
            }
            #[test]
            fn rk_weighted_sums_can_remain_finite_for_large_rhs() {
                let value = <$t>::MAX * 0.5;
                let f = |_: $t, _: &[$t], dy: &mut [$t]| dy[0] = value;
                let mut y = [0.0];
                let mut work = [0.0; rk45_work_len(1)];
                rk4(f, 0.0, 0.001, &mut y, 1, &mut work, |_, _| {}).unwrap();
                assert!((y[0] / value - 0.001).abs() < 128.0 * <$t>::EPSILON);
                y[0] = 0.0;
                let opts = Rk45Options {
                    h_init: 0.001,
                    h_max: 0.001,
                    ..Rk45Options::with_tolerance(1e-4 as $t, 1e-6 as $t)
                };
                rk45(f, 0.0, 0.001, &mut y, opts, &mut work, |_, _| {}).unwrap();
                assert!((y[0] / value - 0.001).abs() < 128.0 * <$t>::EPSILON);
                y[0] = 0.0;
                rk45(
                    f,
                    0.0,
                    0.001,
                    &mut y,
                    Rk45Options::with_tolerance(1e-4 as $t, 1e-6 as $t),
                    &mut work,
                    |_, _| {},
                )
                .unwrap();
                assert!((y[0] / value - 0.001).abs() < 128.0 * <$t>::EPSILON);
            }
            #[test]
            fn runge_kutta_checks_intermediate_derivatives_before_the_next_callback() {
                let mut calls = 0;
                let mut y = [1.0];
                let mut work = [0.0; rk45_work_len(1)];
                let f = |_: $t, state: &[$t], dy: &mut [$t]| {
                    calls += 1;
                    assert!(state.iter().all(|v| v.is_finite()));
                    dy[0] = if calls == 2 { <$t>::NAN } else { 0.0 };
                };
                let opts = Rk45Options {
                    h_init: 0.1,
                    ..Rk45Options::default()
                };
                assert_eq!(
                    rk45(f, 0.0, 0.1, &mut y, opts, &mut work, |_, _| {}),
                    Err(OdeError::NotFinite)
                );
                assert_eq!(calls, 2);
                assert_eq!(y, [1.0]);
                calls = 0;
                let mut bad = |_: $t, state: &[$t], dy: &mut [$t]| {
                    calls += 1;
                    assert!(state[0].is_finite());
                    dy[0] = if calls == 2 { <$t>::NAN } else { 0.0 };
                };
                assert_eq!(
                    try_rk4_step(&mut bad, 0.0, &mut y, 0.1, &mut work),
                    Err(OdeError::NotFinite)
                );
                assert_eq!(calls, 2);
                assert_eq!(y, [1.0]);
            }
            #[test]
            fn rk45_budget_and_options_are_checked_before_rhs() {
                let mut y = [1.0];
                let mut work = [0.0; rk45_work_len(1)];
                let f = |_: $t, _: &[$t], _: &mut [$t]| panic!("vor Callback prüfen");
                let opts = Rk45Options {
                    max_steps: 0,
                    ..Rk45Options::default()
                };
                assert_eq!(
                    rk45(f, 0.0, 1.0, &mut y, opts, &mut work, |_, _| {}),
                    Err(OdeError::MaxStepsExceeded)
                );
                for bad in [-1.0 as $t, <$t>::NAN, <$t>::INFINITY] {
                    for opts in [
                        Rk45Options {
                            rtol: bad,
                            ..Rk45Options::default()
                        },
                        Rk45Options {
                            atol: bad,
                            ..Rk45Options::default()
                        },
                        Rk45Options {
                            h_init: bad,
                            ..Rk45Options::default()
                        },
                        Rk45Options {
                            h_max: bad,
                            ..Rk45Options::default()
                        },
                    ] {
                        assert_eq!(
                            rk45(f, 0.0, 1.0, &mut y, opts, &mut work, |_, _| {}),
                            Err(OdeError::InvalidInput)
                        );
                    }
                }
            }
            #[test]
            fn rk45_initial_probe_stays_inside_the_requested_interval() {
                let end = 1e-8 as $t;
                let mut y = [1.0];
                let mut work = [0.0; rk45_work_len(1)];
                let f = |t: $t, _: &[$t], dy: &mut [$t]| {
                    assert!(t >= 0.0 && t <= end);
                    dy[0] = 0.0;
                };
                rk45(
                    f,
                    0.0,
                    end,
                    &mut y,
                    Rk45Options::default(),
                    &mut work,
                    |_, _| {},
                )
                .unwrap();
                assert_eq!(y, [1.0]);
            }
            #[test]
            fn purely_relative_tolerance_handles_a_zero_solution() {
                let mut y = [0.0];
                let mut work = [0.0; rk45_work_len(1)];
                let opts = Rk45Options::with_tolerance(1e-4 as $t, 0.0);
                rk45(
                    |_: $t, y: &[$t], dy: &mut [$t]| dy[0] = y[0],
                    0.0,
                    1.0,
                    &mut y,
                    opts,
                    &mut work,
                    |_, _| {},
                )
                .unwrap();
                assert_eq!(y, [0.0]);
                let mut pair = [1.0, 0.0];
                let mut w = [0.0; rk45_work_len(2)];
                rk45(
                    |_: $t, s: &[$t], d: &mut [$t]| {
                        d[0] = -s[0];
                        d[1] = 0.0;
                    },
                    0.0,
                    1.0,
                    &mut pair,
                    opts,
                    &mut w,
                    |_, _| {},
                )
                .unwrap();
                assert!((pair[0] - (-1.0 as $t).exp()).abs() < 1e-3 as $t);
                assert_eq!(pair[1], 0.0);
            }
            #[test]
            fn implicit_euler_does_not_read_unused_old_derivative_workspace() {
                let mut y = [1.0];
                let mut work = [<$t>::NAN; implicit_work_len(1)];
                implicit_euler(
                    |_: $t, y: &[$t], dy: &mut [$t]| dy[0] = -y[0],
                    |_: $t, _: &[$t], j: &mut [$t]| j[0] = -1.0,
                    0.0,
                    0.25,
                    &mut y,
                    1,
                    OptimizeOptions::with_tolerance(64.0 * <$t>::EPSILON),
                    &mut work,
                    |_, _| {},
                )
                .unwrap();
                assert!((y[0] - 0.8).abs() < 128.0 * <$t>::EPSILON);
                y[0] = 1.0;
                work.fill(<$t>::NAN);
                implicit_euler_fd(
                    |_: $t, y: &[$t], dy: &mut [$t]| dy[0] = -y[0],
                    0.0,
                    0.25,
                    &mut y,
                    1,
                    OptimizeOptions::with_tolerance(64.0 * <$t>::EPSILON),
                    &mut work,
                    |_, _| {},
                )
                .unwrap();
                assert!((y[0] - 0.8).abs() < 128.0 * <$t>::EPSILON);
            }
            #[test]
            fn implicit_step_must_satisfy_its_discrete_equation() {
                // g(z)=z-0.5-(z²+0.75)=-(z-0.5)²-1 hat keine Nullstelle,
                // aber bei Start z=0.5 einen stationären Punkt mit g'=0.
                let mut y = [0.5];
                let mut work = [0.0; implicit_work_len(1)];
                let result = implicit_euler(
                    |_: $t, y: &[$t], dy: &mut [$t]| dy[0] = y[0] * y[0] + 0.75,
                    |_: $t, y: &[$t], j: &mut [$t]| j[0] = 2.0 * y[0],
                    0.0,
                    1.0,
                    &mut y,
                    1,
                    OptimizeOptions::default(),
                    &mut work,
                    |_, _| {},
                );
                assert_eq!(result, Err(OdeError::NewtonFailed));
                assert_eq!(y, [0.5]);
                y[0] = 1.0;
                let result = implicit_euler_fd(
                    |_: $t, y: &[$t], d: &mut [$t]| d[0] = y[0] * y[0],
                    0.0,
                    1.0,
                    &mut y,
                    1,
                    OptimizeOptions::default(),
                    &mut work,
                    |_, _| {},
                );
                assert_eq!(result, Err(OdeError::NewtonFailed));
                assert_eq!(y, [1.0]);
            }
            #[test]
            fn stiff_linear_steps_match_their_exact_discrete_amplification() {
                let f = |_: $t, y: &[$t], dy: &mut [$t]| dy[0] = -1000.0 * y[0];
                let j = |_: $t, _: &[$t], j: &mut [$t]| j[0] = -1000.0;
                let mut work = [0.0; implicit_work_len(1)];
                let opts = OptimizeOptions::with_tolerance(64.0 * <$t>::EPSILON);
                let mut y = [1.0];
                implicit_euler(f, j, 0.0, 0.1, &mut y, 1, opts, &mut work, |_, _| {}).unwrap();
                assert!((y[0] - 1.0 / 101.0).abs() < 128.0 * <$t>::EPSILON);
                y[0] = 1.0;
                implicit_trapezoid(f, j, 0.0, 0.1, &mut y, 1, opts, &mut work, |_, _| {}).unwrap();
                assert!((y[0] + 49.0 / 51.0).abs() < 128.0 * <$t>::EPSILON);
            }
        }
    };
}
checks!(double, f64);
checks!(single, f32);

#[test]
fn oscillator_phase_and_energy_improve_with_smaller_rk4_steps() {
    let end = 20.0 * core::f64::consts::PI;
    let run = |steps| {
        let mut y = [0.0, 1.0];
        let mut work = [0.0; rk4_work_len(2)];
        rk4(
            |_, s: &[f64], d: &mut [f64]| {
                d[0] = s[1];
                d[1] = -s[0];
            },
            0.0,
            end,
            &mut y,
            steps,
            &mut work,
            |_, _| {},
        )
        .unwrap();
        (
            (y[0].atan2(y[1]) - end.sin().atan2(end.cos())).abs(),
            (y[0] * y[0] + y[1] * y[1] - 1.0).abs(),
        )
    };
    let (p1, e1) = run(1000);
    let (p2, e2) = run(2000);
    assert!((p1 / p2 - 16.0).abs() < 1.0);
    assert!(e2 < e1 && e2 < 1e-6);
}
