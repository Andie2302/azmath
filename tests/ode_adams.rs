//! Tests für das Adams-Bashforth-Moulton-Verfahren.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::ode::{OdeError, adams_bashforth_moulton, adams_work_len};

fn decay(_t: f64, y: &[f64], dy: &mut [f64]) {
    dy[0] = -y[0];
}

/// Fehler bei `y(1)` für `y' = −y` nach `steps` Schritten der Ordnung `order`.
fn decay_error(order: usize, steps: usize) -> f64 {
    let mut y = [1.0];
    let mut work = [0.0; adams_work_len(1)];
    adams_bashforth_moulton(decay, 0.0, 1.0, &mut y, order, steps, &mut work, |_, _| {}).unwrap();
    (y[0] - (-1.0f64).exp()).abs()
}

#[test]
fn convergence_order_matches_the_requested_order() {
    for order in 1..=4 {
        let (e1, e2) = (decay_error(order, 20), decay_error(order, 40));
        let rate = (e1 / e2).log2();
        assert!(
            rate >= order as f64 - 0.4,
            "Ordnung {order}: gemessene Rate {rate} (Fehler {e1:e} → {e2:e})"
        );
    }
}

#[test]
fn order_four_is_accurate_with_few_steps() {
    assert!(decay_error(4, 50) < 1e-8);
}

#[test]
fn harmonic_oscillator_over_several_periods() {
    // y'' = −y als System; Energie bleibt nahezu erhalten
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = y[1];
        dy[1] = -y[0];
    };
    let mut y = [1.0, 0.0];
    let mut work = [0.0; adams_work_len(2)];
    let t_end = 4.0 * core::f64::consts::PI;
    adams_bashforth_moulton(f, 0.0, t_end, &mut y, 4, 2000, &mut work, |_, _| {}).unwrap();
    assert!((y[0] - 1.0).abs() < 1e-6 && y[1].abs() < 1e-6, "{y:?}");
}

#[test]
fn observer_sees_every_step_and_time_dependence_works() {
    // y' = t  ->  y = t²/2 + y₀
    let f = |t: f64, _y: &[f64], dy: &mut [f64]| dy[0] = t;
    let mut y = [1.0];
    let mut work = [0.0; adams_work_len(1)];
    let mut times = Vec::new();
    adams_bashforth_moulton(f, 0.0, 2.0, &mut y, 3, 10, &mut work, |t, _| times.push(t)).unwrap();
    assert_eq!(times.len(), 11);
    assert_eq!(times[0], 0.0);
    assert!((times[10] - 2.0).abs() < 1e-14);
    assert!((y[0] - 3.0).abs() < 1e-12);
}

#[test]
fn integrating_backwards_in_time() {
    let mut y = [(-1.0f64).exp()];
    let mut work = [0.0; adams_work_len(1)];
    adams_bashforth_moulton(decay, 1.0, 0.0, &mut y, 4, 100, &mut work, |_, _| {}).unwrap();
    assert!((y[0] - 1.0).abs() < 1e-8);
}

#[test]
fn errors() {
    let mut work = [0.0; adams_work_len(1)];
    let mut y = [1.0];
    let run = |order, steps, y: &mut [f64], w: &mut [f64]| {
        adams_bashforth_moulton(decay, 0.0, 1.0, y, order, steps, w, |_, _| {})
    };
    assert_eq!(run(0, 10, &mut y, &mut work), Err(OdeError::InvalidInput));
    assert_eq!(run(5, 10, &mut y, &mut work), Err(OdeError::InvalidInput));
    assert_eq!(run(2, 0, &mut y, &mut work), Err(OdeError::InvalidInput));
    assert_eq!(
        run(2, 10, &mut y, &mut work[..3]),
        Err(OdeError::InvalidInput)
    );
    assert_eq!(run(2, 10, &mut [], &mut work), Err(OdeError::InvalidInput));
    let mut bad = [f64::NAN];
    assert_eq!(run(2, 10, &mut bad, &mut work), Err(OdeError::NotFinite));
    // explodierende Lösung y' = y² ab y(0) = 1 erreicht ∞ vor t = 2
    let blow = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = y[0] * y[0];
    let mut y = [1.0];
    let r = adams_bashforth_moulton(blow, 0.0, 2.0, &mut y, 2, 50, &mut work, |_, _| {});
    assert_eq!(r, Err(OdeError::NotFinite));
}

#[test]
fn works_for_f32() {
    let mut y = [1.0f32];
    let mut work = [0.0f32; adams_work_len(1)];
    adams_bashforth_moulton(
        |_t, y: &[f32], dy: &mut [f32]| dy[0] = -y[0],
        0.0,
        1.0,
        &mut y,
        3,
        50,
        &mut work,
        |_, _| {},
    )
    .unwrap();
    assert!((y[0] - (-1.0f32).exp()).abs() < 1e-4);
}
