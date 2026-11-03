//! Tests für die adaptiven impliziten Löser (Schrittverdopplung).
#![cfg(any(feature = "std", feature = "libm"))]
#![allow(
    clippy::needless_range_loop,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::excessive_precision
)]

use azmath::ode::{
    AdaptiveOptions, OdeError, adaptive_work_len, radau5_adaptive, radau5_adaptive_fd,
    ros2_adaptive, ros2_adaptive_fd,
};

#[test]
fn radau_stiff_linear() {
    // y' = −1000 (y − cos t)
    let f = |t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -1000.0 * (y[0] - t.cos());
    let jac = |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = -1000.0;
    let mut y = [1.0];
    let mut work = [0.0; adaptive_work_len(1)];
    let mut last_t = 0.0;
    let res = radau5_adaptive(
        f,
        jac,
        0.0,
        2.0,
        &mut y,
        AdaptiveOptions::with_tolerance(1e-8, 1e-10),
        &mut work,
        |t, _| last_t = t,
    )
    .unwrap();
    assert_eq!(last_t, 2.0);
    // Lösung ≈ (1000² cos t + 1000 sin t)/(1000² + 1) für große t
    let exact = (1.0e6 * 2f64.cos() + 1000.0 * 2f64.sin()) / (1.0e6 + 1.0);
    assert!((y[0] - exact).abs() < 1e-6, "{} vs {}", y[0], exact);
    assert!(res.accepted_steps < 2000, "{res:?}");
}

#[test]
fn radau_fd_matches_analytic_and_backward() {
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -y[0];
    let mut y = [1.0];
    let mut work = [0.0; adaptive_work_len(1)];
    radau5_adaptive_fd(
        f,
        0.0,
        1.0,
        &mut y,
        AdaptiveOptions::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    assert!((y[0] - (-1f64).exp()).abs() < 1e-6);
    // rückwärts zurück
    radau5_adaptive_fd(
        f,
        1.0,
        0.0,
        &mut y,
        AdaptiveOptions::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    assert!((y[0] - 1.0).abs() < 1e-5);
}

#[test]
fn ros2_adaptive_works() {
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = y[1];
        dy[1] = -y[0];
    };
    let jac = |_t: f64, _y: &[f64], j: &mut [f64]| {
        j.copy_from_slice(&[0.0, 1.0, -1.0, 0.0]);
    };
    let mut y = [0.0, 1.0];
    let mut work = [0.0; adaptive_work_len(2)];
    ros2_adaptive(
        f,
        jac,
        0.0,
        1.0,
        &mut y,
        AdaptiveOptions::with_tolerance(1e-7, 1e-10),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    assert!((y[0] - 1f64.sin()).abs() < 1e-4);
    assert!((y[1] - 1f64.cos()).abs() < 1e-4);
    let mut y = [1.0];
    let mut work = [0.0; adaptive_work_len(1)];
    ros2_adaptive_fd(
        |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -2.0 * y[0],
        0.0,
        1.0,
        &mut y,
        AdaptiveOptions::with_tolerance(1e-7, 1e-10),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    assert!((y[0] - (-2f64).exp()).abs() < 1e-4);
}

#[test]
fn radau_conserves_linear_invariant() {
    // A → B → C: Summe der Konzentrationen bleibt 1
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = -0.04 * y[0];
        dy[1] = 0.04 * y[0] - 1.0e4 * y[1];
        dy[2] = 1.0e4 * y[1];
    };
    let mut y = [1.0, 0.0, 0.0];
    let mut work = [0.0; adaptive_work_len(3)];
    radau5_adaptive_fd(
        f,
        0.0,
        10.0,
        &mut y,
        AdaptiveOptions::with_tolerance(1e-7, 1e-10),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    assert!((y.iter().sum::<f64>() - 1.0).abs() < 1e-8);
    assert!((y[0] - (-0.4f64).exp()).abs() < 1e-4);
}

#[test]
fn error_cases() {
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -y[0];
    let mut work = [0.0; adaptive_work_len(1)];
    let mut y: [f64; 0] = [];
    assert_eq!(
        radau5_adaptive_fd(
            f,
            0.0,
            1.0,
            &mut y,
            AdaptiveOptions::default(),
            &mut work,
            |_, _| {}
        )
        .unwrap_err(),
        OdeError::InvalidInput
    );
    let mut y = [f64::NAN];
    assert_eq!(
        radau5_adaptive_fd(
            f,
            0.0,
            1.0,
            &mut y,
            AdaptiveOptions::default(),
            &mut work,
            |_, _| {}
        )
        .unwrap_err(),
        OdeError::NotFinite
    );
    let mut y = [1.0];
    let mut opts = AdaptiveOptions::default();
    opts.max_steps = 2;
    opts.h_init = 1e-6;
    assert_eq!(
        radau5_adaptive_fd(f, 0.0, 1.0, &mut y, opts, &mut work, |_, _| {}).unwrap_err(),
        OdeError::MaxStepsExceeded
    );
    let mut short = [0.0; 3];
    assert_eq!(
        radau5_adaptive_fd(
            f,
            0.0,
            1.0,
            &mut y,
            AdaptiveOptions::default(),
            &mut short,
            |_, _| {}
        )
        .unwrap_err(),
        OdeError::InvalidInput
    );
}
