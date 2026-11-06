//! Tests für das symplektische Verlet-Verfahren (`verlet`, `verlet_separable`).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::ode::{OdeError, verlet, verlet_separable, verlet_work_len};

const TWO_PI: f64 = 2.0 * core::f64::consts::PI;

fn oscillator(_t: f64, q: &[f64], a: &mut [f64]) {
    a[0] = -q[0];
}

fn pendulum(_t: f64, q: &[f64], a: &mut [f64]) {
    a[0] = -q[0].sin();
}

/// Fehlernorm nach einer Periode des Oszillators (Start: q = 1, v = 0).
fn period_error(steps: usize) -> f64 {
    let (mut q, mut v) = ([1.0], [0.0]);
    let mut work = [0.0; verlet_work_len(1)];
    verlet(
        oscillator,
        0.0,
        TWO_PI,
        &mut q,
        &mut v,
        steps,
        &mut work,
        |_, _, _| {},
    )
    .unwrap();
    ((q[0] - 1.0).powi(2) + v[0].powi(2)).sqrt()
}

#[test]
fn second_order_convergence_on_the_oscillator() {
    // Phasenfehler ≈ 2π·h²/24: Verdoppeln der Schrittzahl vierteilt den Fehler.
    let e100 = period_error(100);
    let e200 = period_error(200);
    let e400 = period_error(400);
    assert!(e100 < 2e-3, "{e100}");
    let r1 = e100 / e200;
    let r2 = e200 / e400;
    assert!((3.6..4.4).contains(&r1), "Verhältnis {r1}");
    assert!((3.6..4.4).contains(&r2), "Verhältnis {r2}");
}

/// Größte Energieabweichung `|½(v² + q²) − E₀|` über einen Lauf.
fn max_energy_error(t_end: f64, steps: usize) -> f64 {
    let (mut q, mut v) = ([1.0], [0.0]);
    let mut work = [0.0; verlet_work_len(1)];
    let e0 = 0.5;
    let mut worst = 0.0f64;
    verlet(
        oscillator,
        0.0,
        t_end,
        &mut q,
        &mut v,
        steps,
        &mut work,
        |_, q, v| {
            worst = worst.max((0.5 * (v[0] * v[0] + q[0] * q[0]) - e0).abs());
        },
    )
    .unwrap();
    worst
}

#[test]
fn energy_error_stays_bounded_over_long_times() {
    // h = 0.01; die Energieabweichung schwingt mit etwa h²/8 und driftet nicht.
    let short = max_energy_error(200.0, 20_000);
    let long = max_energy_error(2000.0, 200_000);
    assert!(short < 1e-4, "{short}");
    assert!(long <= 1.2 * short + 1e-10, "kurz {short}, lang {long}");
}

#[test]
fn energy_error_stays_bounded_for_the_pendulum() {
    // E = ½v² − cos q, Amplitude 1 rad, h = 0.01 über 200 Zeiteinheiten.
    let (mut q, mut v) = ([1.0f64], [0.0f64]);
    let mut work = [0.0; verlet_work_len(1)];
    let e0 = -(1.0f64).cos();
    let mut worst = 0.0f64;
    verlet(
        pendulum,
        0.0,
        200.0,
        &mut q,
        &mut v,
        20_000,
        &mut work,
        |_, q, v| {
            worst = worst.max((0.5 * v[0] * v[0] - q[0].cos() - e0).abs());
        },
    )
    .unwrap();
    assert!(worst < 1e-4, "{worst}");
}

#[test]
fn time_reversal_returns_to_the_start() {
    for (accel, q0, v0) in [
        (oscillator as fn(f64, &[f64], &mut [f64]), 1.0, 0.3),
        (pendulum as fn(f64, &[f64], &mut [f64]), 1.2, -0.4),
    ] {
        let (mut q, mut v) = ([q0], [v0]);
        let mut work = [0.0; verlet_work_len(1)];
        verlet(
            accel,
            0.0,
            7.0,
            &mut q,
            &mut v,
            1000,
            &mut work,
            |_, _, _| {},
        )
        .unwrap();
        assert!((q[0] - q0).abs() > 1e-3, "es muss sich etwas bewegt haben");
        // Rückwärts mit vertauschten Zeitgrenzen und derselben Geschwindigkeit.
        verlet(
            accel,
            7.0,
            0.0,
            &mut q,
            &mut v,
            1000,
            &mut work,
            |_, _, _| {},
        )
        .unwrap();
        assert!((q[0] - q0).abs() < 1e-9, "q: {} statt {q0}", q[0]);
        assert!((v[0] - v0).abs() < 1e-9, "v: {} statt {v0}", v[0]);
    }
}

#[test]
fn constant_acceleration_is_integrated_exactly() {
    let g = -9.81;
    let accel = |_t: f64, _q: &[f64], a: &mut [f64]| a[0] = g;
    let (mut q, mut v) = ([1.0], [2.0]);
    let mut work = [0.0; verlet_work_len(1)];
    verlet(accel, 0.0, 3.0, &mut q, &mut v, 7, &mut work, |_, _, _| {}).unwrap();
    let exact_q = 1.0 + 2.0 * 3.0 + 0.5 * g * 9.0;
    let exact_v = 2.0 + g * 3.0;
    assert!((q[0] - exact_q).abs() < 1e-12, "{} vs {exact_q}", q[0]);
    assert!((v[0] - exact_v).abs() < 1e-12, "{} vs {exact_v}", v[0]);
}

#[test]
fn separable_system_with_mass_matches_the_analytic_solution() {
    // H = p²/(2m) + ½k q² mit m = 2, k = 8  →  ω = 2, q = cos 2t, p = −4 sin 2t.
    let (m, k) = (2.0, 8.0);
    let force = |_t: f64, q: &[f64], f: &mut [f64]| f[0] = -k * q[0];
    let drift = |p: &[f64], d: &mut [f64]| d[0] = p[0] / m;
    let (mut q, mut p) = ([1.0], [0.0]);
    let mut work = [0.0; verlet_work_len(1)];
    verlet_separable(
        force,
        drift,
        0.0,
        1.0,
        &mut q,
        &mut p,
        2000,
        &mut work,
        |_, _, _| {},
    )
    .unwrap();
    assert!((q[0] - (2.0f64).cos()).abs() < 1e-6, "{}", q[0]);
    assert!((p[0] + 4.0 * (2.0f64).sin()).abs() < 1e-5, "{}", p[0]);
}

#[test]
fn multiple_degrees_of_freedom_are_independent_for_uncoupled_forces() {
    // Zwei entkoppelte Oszillatoren mit ω = 1 und ω = 2.
    let accel = |_t: f64, q: &[f64], a: &mut [f64]| {
        a[0] = -q[0];
        a[1] = -4.0 * q[1];
    };
    let (mut q, mut v) = ([1.0, 1.0], [0.0, 0.0]);
    let mut work = [0.0; verlet_work_len(2)];
    verlet(
        accel,
        0.0,
        1.0,
        &mut q,
        &mut v,
        2000,
        &mut work,
        |_, _, _| {},
    )
    .unwrap();
    assert!((q[0] - (1.0f64).cos()).abs() < 1e-6, "{}", q[0]);
    assert!((q[1] - (2.0f64).cos()).abs() < 1e-6, "{}", q[1]);
}

#[test]
fn time_dependent_force_uses_the_time_at_both_ends_of_each_step() {
    // q'' = t  →  q = q0 + v0 t + t³/6. Die Geschwindigkeit ist hier exakt (Trapezregel für eine
    // lineare Beschleunigung); q hat einen O(h²)-Fehler: Verdoppeln der Schritte vierteilt ihn.
    let accel = |t: f64, _q: &[f64], a: &mut [f64]| a[0] = t;
    let run = |steps: usize| {
        let (mut q, mut v) = ([0.0], [0.0]);
        let mut work = [0.0; verlet_work_len(1)];
        verlet(
            accel,
            0.0,
            2.0,
            &mut q,
            &mut v,
            steps,
            &mut work,
            |_, _, _| {},
        )
        .unwrap();
        assert!((v[0] - 2.0).abs() < 1e-12, "v = {}", v[0]);
        (q[0] - 8.0 / 6.0).abs()
    };
    let (e1, e2) = (run(10), run(20));
    // Globaler Fehler in q: T·h²/6 (h = 0,2: etwa 0,0133).
    assert!(e1 < 2e-2 && e1 > 0.0, "{e1}");
    assert!((e1 / e2 - 4.0).abs() < 0.2, "Verhältnis {}", e1 / e2);
}

#[test]
fn observer_sees_start_and_every_step() {
    let (mut q, mut v) = ([1.0], [0.0]);
    let mut work = [0.0; verlet_work_len(1)];
    let mut times = Vec::new();
    verlet(
        oscillator,
        0.0,
        1.0,
        &mut q,
        &mut v,
        5,
        &mut work,
        |t, _, _| times.push(t),
    )
    .unwrap();
    assert_eq!(times.len(), 6);
    assert_eq!(times[0], 0.0);
    assert_eq!(times[5], 1.0);
    assert!(times.windows(2).all(|w| w[1] > w[0]));
}

#[test]
fn empty_interval_calls_the_observer_once_and_evaluates_no_force() {
    let (mut q, mut v) = ([1.0], [0.5]);
    let mut work = [0.0; verlet_work_len(1)];
    let (mut calls, mut observed) = (0, 0);
    let accel = |_t: f64, _q: &[f64], a: &mut [f64]| {
        calls += 1;
        a[0] = 0.0;
    };
    verlet(accel, 2.0, 2.0, &mut q, &mut v, 3, &mut work, |_, _, _| {
        observed += 1
    })
    .unwrap();
    assert_eq!((calls, observed), (0, 1));
    assert_eq!((q[0], v[0]), (1.0, 0.5));
}

#[test]
fn one_force_evaluation_per_step_plus_the_initial_one() {
    let (mut q, mut v) = ([1.0], [0.0]);
    let mut work = [0.0; verlet_work_len(1)];
    let mut calls = 0;
    let accel = |_t: f64, q: &[f64], a: &mut [f64]| {
        calls += 1;
        a[0] = -q[0];
    };
    verlet(accel, 0.0, 1.0, &mut q, &mut v, 10, &mut work, |_, _, _| {}).unwrap();
    assert_eq!(calls, 11);
}

#[test]
fn invalid_input_is_rejected() {
    let mut work = [0.0; verlet_work_len(2)];
    let none = |_: f64, _: &[f64], _: &mut [f64]| {};
    let nop = |_: f64, _: &[f64], _: &[f64]| {};
    // leerer Zustand
    assert_eq!(
        verlet(none, 0.0, 1.0, &mut [], &mut [], 3, &mut work, nop),
        Err(OdeError::InvalidInput)
    );
    // q und v verschieden lang
    assert_eq!(
        verlet(
            none,
            0.0,
            1.0,
            &mut [1.0, 2.0],
            &mut [0.0],
            3,
            &mut work,
            nop
        ),
        Err(OdeError::InvalidInput)
    );
    // keine Schritte
    assert_eq!(
        verlet(none, 0.0, 1.0, &mut [1.0], &mut [0.0], 0, &mut work, nop),
        Err(OdeError::InvalidInput)
    );
    // zu wenig Arbeitsspeicher
    let mut short = [0.0; 3];
    assert_eq!(
        verlet(none, 0.0, 1.0, &mut [1.0], &mut [0.0], 3, &mut short, nop),
        Err(OdeError::InvalidInput)
    );
    // nichtendliche Zeiten oder Zustände
    assert_eq!(
        verlet(
            none,
            f64::NAN,
            1.0,
            &mut [1.0],
            &mut [0.0],
            3,
            &mut work,
            nop
        ),
        Err(OdeError::NotFinite)
    );
    assert_eq!(
        verlet(
            none,
            0.0,
            f64::INFINITY,
            &mut [1.0],
            &mut [0.0],
            3,
            &mut work,
            nop
        ),
        Err(OdeError::NotFinite)
    );
    assert_eq!(
        verlet(
            none,
            0.0,
            1.0,
            &mut [f64::NAN],
            &mut [0.0],
            3,
            &mut work,
            nop
        ),
        Err(OdeError::NotFinite)
    );
    assert_eq!(
        verlet(
            none,
            0.0,
            1.0,
            &mut [1.0],
            &mut [f64::INFINITY],
            3,
            &mut work,
            nop
        ),
        Err(OdeError::NotFinite)
    );
}

#[test]
fn non_finite_force_is_an_error_and_keeps_the_last_completed_step() {
    // Die Beschleunigung wird für t > 0.55 zu NaN.
    let accel = |t: f64, q: &[f64], a: &mut [f64]| {
        a[0] = if t > 0.55 { f64::NAN } else { -q[0] };
    };
    let (mut q, mut v) = ([1.0], [0.0]);
    let mut work = [0.0; verlet_work_len(1)];
    let mut last = (1.0, 0.0);
    let r = verlet(accel, 0.0, 1.0, &mut q, &mut v, 10, &mut work, |_, q, v| {
        last = (q[0], v[0]);
    });
    assert_eq!(r, Err(OdeError::NotFinite));
    assert_eq!((q[0], v[0]), last);
    assert!(q[0].is_finite() && v[0].is_finite());
    assert!(
        last != (1.0, 0.0),
        "es muss mindestens ein Schritt gelungen sein"
    );

    // schon am Anfang NaN: Zustand bleibt unverändert
    let bad = |_t: f64, _q: &[f64], a: &mut [f64]| a[0] = f64::NAN;
    let (mut q, mut v) = ([1.0], [0.5]);
    assert_eq!(
        verlet(bad, 0.0, 1.0, &mut q, &mut v, 4, &mut work, |_, _, _| {}),
        Err(OdeError::NotFinite)
    );
    assert_eq!((q[0], v[0]), (1.0, 0.5));
}

#[test]
fn overflow_during_the_drift_is_reported() {
    let force = |_t: f64, _q: &[f64], f: &mut [f64]| f[0] = 0.0;
    let drift = |_p: &[f64], d: &mut [f64]| d[0] = f64::INFINITY;
    let (mut q, mut p) = ([1.0], [1.0]);
    let mut work = [0.0; verlet_work_len(1)];
    assert_eq!(
        verlet_separable(
            force,
            drift,
            0.0,
            1.0,
            &mut q,
            &mut p,
            4,
            &mut work,
            |_, _, _| {}
        ),
        Err(OdeError::NotFinite)
    );
    assert_eq!((q[0], p[0]), (1.0, 1.0));
}

#[test]
fn works_for_f32() {
    let accel = |_t: f32, q: &[f32], a: &mut [f32]| a[0] = -q[0];
    let (mut q, mut v) = ([1.0f32], [0.0f32]);
    let mut work = [0.0f32; verlet_work_len(1)];
    let t_end = core::f32::consts::TAU;
    verlet(
        accel,
        0.0,
        t_end,
        &mut q,
        &mut v,
        1000,
        &mut work,
        |_, _, _| {},
    )
    .unwrap();
    assert!(
        (q[0] - 1.0).abs() < 1e-3 && v[0].abs() < 1e-3,
        "{} {}",
        q[0],
        v[0]
    );
}

#[test]
fn work_len_is_a_const_fn() {
    const LEN: usize = verlet_work_len(3);
    assert_eq!(LEN, 12);
}
