//! Tests für die BDF-Verfahren (Ordnung 1 bis 5).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::ode::{OdeError, bdf, bdf_fd, bdf_work_len};
use azmath::optimize::OptimizeOptions;

fn decay_error(order: usize, steps: usize) -> f64 {
    let mut y = [1.0];
    let mut work = [0.0; bdf_work_len(1)];
    bdf(
        |_t, y: &[f64], dy: &mut [f64]| dy[0] = -y[0],
        |_t, _y: &[f64], j: &mut [f64]| j[0] = -1.0,
        order,
        0.0,
        1.0,
        &mut y,
        steps,
        OptimizeOptions::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    (y[0] - (-1.0f64).exp()).abs()
}

#[test]
fn convergence_order_matches_the_requested_order() {
    // Schrittzahlen so gewählt, dass der Fehler weit über der Newton-Toleranz bleibt
    for (order, steps) in [(1, 100), (2, 40), (3, 20), (4, 12), (5, 10)] {
        let (e1, e2) = (decay_error(order, steps), decay_error(order, 2 * steps));
        let rate = (e1 / e2).log2();
        assert!(
            rate >= order as f64 - 0.5,
            "BDF{order}: Rate {rate} (Fehler {e1:e} → {e2:e})"
        );
    }
}

#[test]
fn bdf2_is_stable_on_a_stiff_problem() {
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -1000.0 * y[0];
    let jac = |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = -1000.0;
    for order in [1, 2] {
        let mut y = [1.0];
        let mut work = [0.0; bdf_work_len(1)];
        bdf(
            f,
            jac,
            order,
            0.0,
            1.0,
            &mut y,
            20,
            OptimizeOptions::default(),
            &mut work,
            |_, _| {},
        )
        .unwrap();
        assert!(y[0].abs() < 1e-6, "BDF{order}: {}", y[0]);
    }
}

#[test]
fn stiff_forced_problem_tracks_the_slow_solution() {
    // y' = −500·(y − cos t)  ->  y ≈ cos t + sin t/500
    let f = |t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -500.0 * (y[0] - t.cos());
    let jac = |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = -500.0;
    let mut y = [1.0];
    let mut work = [0.0; bdf_work_len(1)];
    bdf(
        f,
        jac,
        3,
        0.0,
        2.0,
        &mut y,
        100,
        OptimizeOptions::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    assert!(
        (y[0] - (2.0f64.cos() + 2.0f64.sin() / 500.0)).abs() < 5e-4,
        "{}",
        y[0]
    );
}

#[test]
fn nonlinear_system_with_finite_differences() {
    // Van-der-Pol-ähnlich, mäßig steif: y₁' = y₂, y₂' = μ(1 − y₁²)y₂ − y₁ mit μ = 5
    let mu = 5.0;
    let f = move |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = y[1];
        dy[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    };
    let jac = move |_t: f64, y: &[f64], j: &mut [f64]| {
        j.copy_from_slice(&[
            0.0,
            1.0,
            -2.0 * mu * y[0] * y[1] - 1.0,
            mu * (1.0 - y[0] * y[0]),
        ]);
    };
    let mut work = [0.0; bdf_work_len(2)];
    let mut a = [2.0, 0.0];
    bdf(
        f,
        jac,
        4,
        0.0,
        5.0,
        &mut a,
        2000,
        OptimizeOptions::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    let mut b = [2.0, 0.0];
    bdf_fd(
        f,
        4,
        0.0,
        5.0,
        &mut b,
        2000,
        OptimizeOptions::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    assert!(a[0].is_finite() && a[1].is_finite());
    assert!(
        (a[0] - b[0]).abs() < 1e-5 && (a[1] - b[1]).abs() < 1e-5,
        "{a:?} gegen {b:?}"
    );
}

#[test]
fn startup_steps_and_observer() {
    let f = |t: f64, _y: &[f64], dy: &mut [f64]| dy[0] = 4.0 * t * t * t; // y = t⁴ + y₀
    let jac = |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = 0.0;
    let mut y = [1.0];
    let mut work = [0.0; bdf_work_len(1)];
    let mut times = Vec::new();
    bdf(
        f,
        jac,
        5,
        0.0,
        1.0,
        &mut y,
        4,
        OptimizeOptions::default(),
        &mut work,
        |t, _| times.push(t),
    )
    .unwrap();
    assert_eq!(times.len(), 5);
    assert!((y[0] - 2.0).abs() < 1e-3, "{}", y[0]); // BDF5 ist für t⁴ nicht exakt, aber sehr nah
}

#[test]
fn errors() {
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -y[0];
    let jac = |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = -1.0;
    let mut work = [0.0; bdf_work_len(1)];
    let o = OptimizeOptions::default();
    let mut y = [1.0];
    assert_eq!(
        bdf(f, jac, 0, 0.0, 1.0, &mut y, 5, o, &mut work, |_, _| {}),
        Err(OdeError::InvalidInput)
    );
    assert_eq!(
        bdf(f, jac, 6, 0.0, 1.0, &mut y, 5, o, &mut work, |_, _| {}),
        Err(OdeError::InvalidInput)
    );
    assert_eq!(
        bdf(f, jac, 2, 0.0, 1.0, &mut y, 0, o, &mut work, |_, _| {}),
        Err(OdeError::InvalidInput)
    );
    assert_eq!(
        bdf(f, jac, 2, 0.0, 1.0, &mut y, 5, o, &mut work[..5], |_, _| {}),
        Err(OdeError::InvalidInput)
    );
    let mut bad = [f64::NAN];
    assert_eq!(
        bdf_fd(f, 2, 0.0, 1.0, &mut bad, 5, o, &mut work, |_, _| {}),
        Err(OdeError::NotFinite)
    );
}

#[test]
fn works_for_f32() {
    let mut y = [1.0f32];
    let mut work = [0.0f32; bdf_work_len(1)];
    bdf(
        |_t, y: &[f32], dy: &mut [f32]| dy[0] = -y[0],
        |_t, _y: &[f32], j: &mut [f32]| j[0] = -1.0,
        2,
        0.0,
        1.0,
        &mut y,
        100,
        OptimizeOptions::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    assert!((y[0] - (-1.0f32).exp()).abs() < 1e-3);
}
