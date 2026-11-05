//! Tests für das Radau-IIA-Verfahren (Ordnung 5).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::ode::{OdeError, radau_work_len, radau5, radau5_fd};
use azmath::optimize::OptimizeOptions;

fn decay(_t: f64, y: &[f64], dy: &mut [f64]) {
    dy[0] = -y[0];
}
fn decay_jac(_t: f64, _y: &[f64], j: &mut [f64]) {
    j[0] = -1.0;
}

fn decay_error(steps: usize) -> f64 {
    let mut y = [1.0];
    let mut work = [0.0; radau_work_len(1)];
    radau5(
        decay,
        decay_jac,
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
fn fifth_order_convergence() {
    let (e1, e2) = (decay_error(2), decay_error(4));
    let rate = (e1 / e2).log2();
    assert!(rate > 4.5, "Rate {rate} (Fehler {e1:e} → {e2:e})");
    assert!(decay_error(10) < 1e-9);
}

#[test]
fn stiff_decay_is_stable_with_huge_steps() {
    // y' = −1000·y mit h = 0,05: explizit instabil, Radau klingt ab (L-stabil)
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -1000.0 * y[0];
    let jac = |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = -1000.0;
    let mut y = [1.0];
    let mut work = [0.0; radau_work_len(1)];
    radau5(
        f,
        jac,
        0.0,
        1.0,
        &mut y,
        20,
        OptimizeOptions::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    assert!(y[0].abs() < 1e-8, "{}", y[0]);
}

#[test]
fn prothero_robinson_is_solved_accurately_despite_extreme_stiffness() {
    // y' = λ(y − sin t) + cos t, y(0) = 0  ->  y = sin t, mit λ = −10⁶
    let lambda = -1.0e6;
    let f = move |t: f64, y: &[f64], dy: &mut [f64]| dy[0] = lambda * (y[0] - t.sin()) + t.cos();
    let jac = move |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = lambda;
    let mut y = [0.0];
    let mut work = [0.0; radau_work_len(1)];
    radau5(
        f,
        jac,
        0.0,
        1.0,
        &mut y,
        10,
        OptimizeOptions::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    assert!((y[0] - 1.0f64.sin()).abs() < 1e-6, "{}", y[0]);
}

#[test]
fn nonlinear_logistic_equation_matches_the_closed_form() {
    // y' = y(1 − y), y(0) = 0,1 -> y = 1/(1 + 9e^{−t})
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = y[0] * (1.0 - y[0]);
    let jac = |_t: f64, y: &[f64], j: &mut [f64]| j[0] = 1.0 - 2.0 * y[0];
    let mut y = [0.1];
    let mut work = [0.0; radau_work_len(1)];
    radau5(
        f,
        jac,
        0.0,
        5.0,
        &mut y,
        20,
        OptimizeOptions::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    let exact = 1.0 / (1.0 + 9.0 * (-5.0f64).exp());
    assert!((y[0] - exact).abs() < 1e-8, "{} gegen {exact}", y[0]);
}

#[test]
fn system_with_two_time_scales_and_finite_difference_jacobian() {
    // y₁' = −1000·y₁ + y₂,  y₂' = −y₂   (y₂ = e^{−t})
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = -1000.0 * y[0] + y[1];
        dy[1] = -y[1];
    };
    let jac = |_t: f64, _y: &[f64], j: &mut [f64]| j.copy_from_slice(&[-1000.0, 1.0, 0.0, -1.0]);
    let mut work = [0.0; radau_work_len(2)];
    let mut a = [0.0, 1.0];
    radau5(
        f,
        jac,
        0.0,
        2.0,
        &mut a,
        20,
        OptimizeOptions::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    let mut b = [0.0, 1.0];
    radau5_fd(
        f,
        0.0,
        2.0,
        &mut b,
        20,
        OptimizeOptions::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    // Lösung: y₂ = e^{−t}, y₁ ≈ y₂/999 nach dem Einschwingen
    assert!((a[1] - (-2.0f64).exp()).abs() < 1e-7);
    assert!((a[0] - a[1] / 999.0).abs() < 1e-6, "{a:?}");
    assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6);
}

#[test]
fn observer_and_time_dependence() {
    // y' = 3t²  ->  y = t³ + y₀ (Polynom vom Grad 3: Ordnung 5 ist exakt)
    let f = |t: f64, _y: &[f64], dy: &mut [f64]| dy[0] = 3.0 * t * t;
    let jac = |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = 0.0;
    let mut y = [2.0];
    let mut work = [0.0; radau_work_len(1)];
    let mut count = 0;
    radau5(
        f,
        jac,
        0.0,
        2.0,
        &mut y,
        3,
        OptimizeOptions::default(),
        &mut work,
        |_, _| count += 1,
    )
    .unwrap();
    assert_eq!(count, 4);
    assert!((y[0] - 10.0).abs() < 1e-10);
}

#[test]
fn errors() {
    let mut work = [0.0; radau_work_len(1)];
    let o = OptimizeOptions::default();
    let mut y = [1.0];
    assert_eq!(
        radau5(
            decay,
            decay_jac,
            0.0,
            1.0,
            &mut y,
            0,
            o,
            &mut work,
            |_, _| {}
        ),
        Err(OdeError::InvalidInput)
    );
    assert_eq!(
        radau5(
            decay,
            decay_jac,
            0.0,
            1.0,
            &mut y,
            5,
            o,
            &mut work[..4],
            |_, _| {}
        ),
        Err(OdeError::InvalidInput)
    );
    assert_eq!(
        radau5(
            decay,
            decay_jac,
            0.0,
            1.0,
            &mut [],
            5,
            o,
            &mut work,
            |_, _| {}
        ),
        Err(OdeError::InvalidInput)
    );
    let mut bad = [f64::NAN];
    assert_eq!(
        radau5_fd(decay, 0.0, 1.0, &mut bad, 5, o, &mut work, |_, _| {}),
        Err(OdeError::NotFinite)
    );
    assert_eq!(
        radau5_fd(decay, f64::NAN, 1.0, &mut y, 5, o, &mut work, |_, _| {}),
        Err(OdeError::NotFinite)
    );
}

#[test]
fn works_for_f32() {
    let mut y = [1.0f32];
    let mut work = [0.0f32; radau_work_len(1)];
    radau5(
        |_t, y: &[f32], dy: &mut [f32]| dy[0] = -y[0],
        |_t, _y: &[f32], j: &mut [f32]| j[0] = -1.0,
        0.0,
        1.0,
        &mut y,
        10,
        OptimizeOptions::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    assert!((y[0] - (-1.0f32).exp()).abs() < 1e-4);
}

#[test]
fn stiff_stage_scaling_also_works_with_numerical_jacobian() {
    // Bekannte Lösung sin(t); Stufen-Jacobi-Einträge sind etwa 10^5.
    let f = |t: f64, y: &[f64], out: &mut [f64]| out[0] = -1e6 * (y[0] - t.sin()) + t.cos();
    let mut y = [0.0];
    let mut work = [f64::NAN; radau_work_len(1)];
    radau5_fd(
        f,
        0.0,
        1.0,
        &mut y,
        10,
        OptimizeOptions::default(),
        &mut work,
        |t, y| {
            assert!((y[0] - t.sin()).abs() < 1e-6);
        },
    )
    .unwrap();
    assert!((y[0] - 1.0f64.sin()).abs() < 1e-6);
}
