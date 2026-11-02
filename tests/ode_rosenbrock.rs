//! Tests für das linear-implizite Rosenbrock-Verfahren ROS2.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::ode::{OdeError, ros2, ros2_fd, ros2_work_len};

/// y' = −y + sin t, y(0) = 1  ->  y = (sin t − cos t)/2 + 1,5·e^{−t}
fn forced(t: f64, y: &[f64], dy: &mut [f64]) {
    dy[0] = -y[0] + t.sin();
}
fn forced_jac(_t: f64, _y: &[f64], j: &mut [f64]) {
    j[0] = -1.0;
}
fn forced_exact(t: f64) -> f64 {
    (t.sin() - t.cos()) / 2.0 + 1.5 * (-t).exp()
}

fn forced_error(steps: usize) -> f64 {
    let mut y = [1.0];
    let mut work = [0.0; ros2_work_len(1)];
    ros2(
        forced,
        forced_jac,
        0.0,
        3.0,
        &mut y,
        steps,
        &mut work,
        |_, _| {},
    )
    .unwrap();
    (y[0] - forced_exact(3.0)).abs()
}

#[test]
fn second_order_convergence_including_time_dependence() {
    let (e1, e2) = (forced_error(50), forced_error(100));
    let rate = (e1 / e2).log2();
    assert!(rate > 1.8, "Rate {rate} (Fehler {e1:e} → {e2:e})");
    assert!(forced_error(400) < 1e-5);
}

#[test]
fn stiff_decay_is_stable_with_huge_steps() {
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -1000.0 * y[0];
    let jac = |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = -1000.0;
    let mut y = [1.0];
    let mut work = [0.0; ros2_work_len(1)];
    ros2(f, jac, 0.0, 1.0, &mut y, 20, &mut work, |_, _| {}).unwrap();
    assert!(y[0].abs() < 1e-3, "{}", y[0]);
}

#[test]
fn stiff_forced_problem_follows_the_slow_solution() {
    // y' = −10⁴·(y − sin t) + cos t, y(0) = 0  ->  y = sin t
    let lambda = -1.0e4;
    let f = move |t: f64, y: &[f64], dy: &mut [f64]| dy[0] = lambda * (y[0] - t.sin()) + t.cos();
    let jac = move |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = lambda;
    let mut y = [0.0];
    let mut work = [0.0; ros2_work_len(1)];
    ros2(f, jac, 0.0, 1.0, &mut y, 100, &mut work, |_, _| {}).unwrap();
    assert!((y[0] - 1.0f64.sin()).abs() < 1e-3, "{}", y[0]);
}

#[test]
fn finite_difference_jacobian_matches_the_analytic_one() {
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = -50.0 * y[0] + y[1] * y[1];
        dy[1] = y[0] - 2.0 * y[1];
    };
    let jac =
        |_t: f64, y: &[f64], j: &mut [f64]| j.copy_from_slice(&[-50.0, 2.0 * y[1], 1.0, -2.0]);
    let mut work = [0.0; ros2_work_len(2)];
    let mut a = [1.0, 1.0];
    ros2(f, jac, 0.0, 1.0, &mut a, 200, &mut work, |_, _| {}).unwrap();
    let mut b = [1.0, 1.0];
    ros2_fd(f, 0.0, 1.0, &mut b, 200, &mut work, |_, _| {}).unwrap();
    assert!(
        (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6,
        "{a:?} gegen {b:?}"
    );
    assert!(a[0].is_finite() && a[1].is_finite());
}

#[test]
fn observer_times() {
    let mut y = [1.0];
    let mut work = [0.0; ros2_work_len(1)];
    let mut times = Vec::new();
    ros2(
        forced,
        forced_jac,
        0.0,
        1.0,
        &mut y,
        4,
        &mut work,
        |t, _| times.push(t),
    )
    .unwrap();
    assert_eq!(times, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn errors() {
    let mut work = [0.0; ros2_work_len(1)];
    let mut y = [1.0];
    assert_eq!(
        ros2(
            forced,
            forced_jac,
            0.0,
            1.0,
            &mut y,
            0,
            &mut work,
            |_, _| {}
        ),
        Err(OdeError::InvalidInput)
    );
    assert_eq!(
        ros2(
            forced,
            forced_jac,
            0.0,
            1.0,
            &mut y,
            5,
            &mut work[..3],
            |_, _| {}
        ),
        Err(OdeError::InvalidInput)
    );
    assert_eq!(
        ros2(
            forced,
            forced_jac,
            0.0,
            1.0,
            &mut [],
            5,
            &mut work,
            |_, _| {}
        ),
        Err(OdeError::InvalidInput)
    );
    let mut bad = [f64::INFINITY];
    assert_eq!(
        ros2_fd(forced, 0.0, 1.0, &mut bad, 5, &mut work, |_, _| {}),
        Err(OdeError::NotFinite)
    );
}

#[test]
fn works_for_f32() {
    let mut y = [1.0f32];
    let mut work = [0.0f32; ros2_work_len(1)];
    ros2(
        |_t, y: &[f32], dy: &mut [f32]| dy[0] = -y[0],
        |_t, _y: &[f32], j: &mut [f32]| j[0] = -1.0,
        0.0,
        1.0,
        &mut y,
        100,
        &mut work,
        |_, _| {},
    )
    .unwrap();
    assert!((y[0] - (-1.0f32).exp()).abs() < 1e-3);
}
