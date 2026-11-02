//! Tests für die Hermite-Interpolation und die Ereignissuche.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::ode::{EventDirection, OdeError, event_work_len, hermite_interpolate, rk4_event};
use core::f64::consts::{FRAC_PI_2, PI};

fn oscillator(_t: f64, y: &[f64], dy: &mut [f64]) {
    dy[0] = y[1];
    dy[1] = -y[0];
}

#[test]
fn hermite_reproduces_cubics_and_endpoints() {
    // y = t³ − 2t + 1 auf [1, 3]: y' = 3t² − 2
    let y = |t: f64| t * t * t - 2.0 * t + 1.0;
    let dy = |t: f64| 3.0 * t * t - 2.0;
    let (t0, h) = (1.0, 2.0);
    let mut out = [0.0];
    for theta in [0.0, 0.1, 0.5, 0.77, 1.0] {
        hermite_interpolate(
            &[y(t0)],
            &[dy(t0)],
            &[y(t0 + h)],
            &[dy(t0 + h)],
            h,
            theta,
            &mut out,
        );
        assert!((out[0] - y(t0 + theta * h)).abs() < 1e-12, "θ = {theta}");
    }
    // Endpunkte exakt
    hermite_interpolate(
        &[5.0, 6.0],
        &[1.0, 1.0],
        &[7.0, 8.0],
        &[2.0, 2.0],
        0.3,
        0.0,
        &mut [0.0; 2],
    );
    let mut two = [0.0; 2];
    hermite_interpolate(
        &[5.0, 6.0],
        &[1.0, 1.0],
        &[7.0, 8.0],
        &[2.0, 2.0],
        0.3,
        1.0,
        &mut two,
    );
    assert_eq!(two, [7.0, 8.0]);
}

#[test]
fn bouncing_ball_hits_the_ground_at_the_analytic_time() {
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = y[1];
        dy[1] = -9.81;
    };
    let mut y = [10.0, 0.0];
    let mut work = [0.0; event_work_len(2)];
    let ev = rk4_event(
        f,
        |_t, y: &[f64]| y[0],
        0.0,
        5.0,
        &mut y,
        50,
        EventDirection::Falling,
        1e-13,
        &mut work,
    )
    .unwrap()
    .expect("Ereignis erwartet");
    let t_exact = (2.0f64 * 10.0 / 9.81).sqrt();
    assert!((ev.t - t_exact).abs() < 1e-9, "{} gegen {t_exact}", ev.t);
    assert!(ev.g_value.abs() < 1e-9);
    // Zustand zum Ereigniszeitpunkt
    assert!(y[0].abs() < 1e-9);
    assert!((y[1] + 9.81 * t_exact).abs() < 1e-8);
}

#[test]
fn directions_select_the_right_crossing() {
    // y = cos t: fällt durch 0 bei π/2, steigt durch 0 bei 3π/2
    let g = |_t: f64, y: &[f64]| y[0];
    let mut work = [0.0; event_work_len(2)];
    for (dir, expected) in [
        (EventDirection::Falling, FRAC_PI_2),
        (EventDirection::Rising, 3.0 * FRAC_PI_2),
        (EventDirection::Either, FRAC_PI_2),
    ] {
        let mut y = [1.0, 0.0];
        let ev = rk4_event(oscillator, g, 0.0, 6.0, &mut y, 120, dir, 1e-12, &mut work)
            .unwrap()
            .unwrap();
        assert!(
            (ev.t - expected).abs() < 1e-7,
            "{dir:?}: {} gegen {expected}",
            ev.t
        );
    }
}

#[test]
fn no_event_returns_none_and_the_final_state() {
    let mut y = [1.0, 0.0];
    let mut work = [0.0; event_work_len(2)];
    // y = cos t bleibt auf [0, 1] positiv
    let r = rk4_event(
        oscillator,
        |_t, y: &[f64]| y[0],
        0.0,
        1.0,
        &mut y,
        20,
        EventDirection::Either,
        1e-12,
        &mut work,
    )
    .unwrap();
    assert_eq!(r, None);
    assert!((y[0] - 1.0f64.cos()).abs() < 1e-7);
    // falsche Richtung: steigender Nulldurchgang kommt auf [0, 2] nicht vor
    let mut y = [1.0, 0.0];
    let r = rk4_event(
        oscillator,
        |_t, y: &[f64]| y[0],
        0.0,
        2.0,
        &mut y,
        40,
        EventDirection::Rising,
        1e-12,
        &mut work,
    )
    .unwrap();
    assert_eq!(r, None);
}

#[test]
fn event_time_depends_on_the_time_argument_of_g() {
    // y' = 0 (konstant 1), g = t − π: Ereignis bei t = π unabhängig vom Zustand
    let f = |_t: f64, _y: &[f64], dy: &mut [f64]| dy[0] = 0.0;
    let mut y = [1.0];
    let mut work = [0.0; event_work_len(1)];
    let ev = rk4_event(
        f,
        |t, _y: &[f64]| t - PI,
        0.0,
        10.0,
        &mut y,
        7,
        EventDirection::Rising,
        1e-13,
        &mut work,
    )
    .unwrap()
    .unwrap();
    assert!((ev.t - PI).abs() < 1e-10);
    assert_eq!(ev.steps_before, 2); // h = 10/7 ≈ 1,43: π liegt im dritten Schritt
}

#[test]
fn errors() {
    let mut work = [0.0; event_work_len(2)];
    let g = |_t: f64, y: &[f64]| y[0];
    let run = |steps, y: &mut [f64], w: &mut [f64]| {
        rk4_event(
            oscillator,
            g,
            0.0,
            1.0,
            y,
            steps,
            EventDirection::Either,
            1e-12,
            w,
        )
    };
    let mut y = [1.0, 0.0];
    assert_eq!(run(0, &mut y, &mut work), Err(OdeError::InvalidInput));
    assert_eq!(run(10, &mut y, &mut work[..5]), Err(OdeError::InvalidInput));
    assert_eq!(run(10, &mut [], &mut work), Err(OdeError::InvalidInput));
    let mut bad = [f64::NAN, 0.0];
    assert_eq!(run(10, &mut bad, &mut work), Err(OdeError::NotFinite));
    let r = rk4_event(
        oscillator,
        |_t, _y: &[f64]| f64::NAN,
        0.0,
        1.0,
        &mut y,
        10,
        EventDirection::Either,
        1e-12,
        &mut work,
    );
    assert_eq!(r, Err(OdeError::NotFinite));
    assert!(OdeError::EventNotLocated.to_string().contains("Ereignis"));
}

#[test]
fn works_for_f32() {
    let f = |_t: f32, y: &[f32], dy: &mut [f32]| {
        dy[0] = y[1];
        dy[1] = -9.81;
    };
    let mut y = [10.0f32, 0.0];
    let mut work = [0.0f32; event_work_len(2)];
    let ev = rk4_event(
        f,
        |_t, y: &[f32]| y[0],
        0.0,
        5.0,
        &mut y,
        50,
        EventDirection::Falling,
        1e-5,
        &mut work,
    )
    .unwrap()
    .unwrap();
    assert!((ev.t - (2.0f32 * 10.0 / 9.81).sqrt()).abs() < 1e-3);
}
