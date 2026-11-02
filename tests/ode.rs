//! Tests für die ODE-Löser: Euler, RK4, RK45 (adaptiv), implizites Euler und Trapezverfahren.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::Real;
use azmath::ode::{
    OdeError, Rk45Options, euler, euler_work_len, implicit_euler, implicit_euler_fd,
    implicit_trapezoid, implicit_trapezoid_fd, implicit_work_len, rk4, rk4_step, rk4_work_len,
    rk45, rk45_work_len,
};
use azmath::optimize::OptimizeOptions;

fn newton() -> OptimizeOptions<f64> {
    OptimizeOptions::default()
}

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "erwartet {b}, erhalten {a} (Δ = {:e})",
        (a - b).abs()
    );
}

fn decay(_t: f64, y: &[f64], dy: &mut [f64]) {
    dy[0] = -y[0];
}

fn oscillator(_t: f64, y: &[f64], dy: &mut [f64]) {
    dy[0] = y[1];
    dy[1] = -y[0];
}

/// y' = −λ(y − cos t): steif für große λ. Exakte Lösung mit y(0) = `y0`.
fn stiff_exact(lambda: f64, y0: f64, t: f64) -> f64 {
    let yp =
        |t: f64| (lambda * lambda * Real::cos(t) + lambda * Real::sin(t)) / (lambda * lambda + 1.0);
    yp(t) + (y0 - yp(0.0)) * Real::exp(-lambda * t)
}

// ---------------------------------------------------------------------------
// Feste Schrittweite
// ---------------------------------------------------------------------------

fn run_euler(steps: usize) -> f64 {
    let mut y = [1.0];
    let mut work = [0.0; euler_work_len(1)];
    euler(decay, 0.0, 1.0, &mut y, steps, &mut work, |_, _| {}).unwrap();
    (y[0] - Real::exp(-1.0f64)).abs()
}

fn run_rk4(steps: usize) -> f64 {
    let mut y = [1.0];
    let mut work = [0.0; rk4_work_len(1)];
    rk4(decay, 0.0, 1.0, &mut y, steps, &mut work, |_, _| {}).unwrap();
    (y[0] - Real::exp(-1.0f64)).abs()
}

#[test]
fn euler_has_order_one() {
    let ratio = run_euler(100) / run_euler(200);
    assert!((ratio - 2.0).abs() < 0.05, "Verhältnis {ratio}");
}

#[test]
fn rk4_has_order_four() {
    let ratio = run_rk4(8) / run_rk4(16);
    assert!((ratio - 16.0).abs() < 1.0, "Verhältnis {ratio}");
    assert!(run_rk4(20) < 1e-7);
}

#[test]
fn rk4_is_exact_for_cubic_right_hand_side() {
    // y' = t³  ->  y = t⁴/4 + y₀; RK4 integriert Polynome bis Grad 3 in t exakt
    let mut y = [1.0];
    let mut work = [0.0; rk4_work_len(1)];
    rk4(
        |t: f64, _y: &[f64], dy: &mut [f64]| dy[0] = t * t * t,
        0.0,
        2.0,
        &mut y,
        1,
        &mut work,
        |_, _| {},
    )
    .unwrap();
    close(y[0], 1.0 + 4.0, 1e-13);
}

#[test]
fn rk4_harmonic_oscillator_period_and_energy() {
    let mut y = [0.0, 1.0];
    let mut work = [0.0; rk4_work_len(2)];
    let two_pi = 2.0 * core::f64::consts::PI;
    let mut max_energy_error = 0.0f64;
    rk4(oscillator, 0.0, two_pi, &mut y, 200, &mut work, |_, s| {
        max_energy_error = max_energy_error.max((s[0] * s[0] + s[1] * s[1] - 1.0).abs());
    })
    .unwrap();
    close(y[0], 0.0, 1e-5);
    close(y[1], 1.0, 1e-5);
    assert!(
        max_energy_error < 1e-6,
        "Energiefehler {max_energy_error:e}"
    );
}

#[test]
fn rk4_step_loop_equals_rk4() {
    let mut a = [0.0, 1.0];
    let mut work = [0.0; rk4_work_len(2)];
    rk4(oscillator, 0.0, 1.0, &mut a, 10, &mut work, |_, _| {}).unwrap();
    let mut b = [0.0, 1.0];
    let mut f = oscillator;
    for i in 0..10 {
        rk4_step(&mut f, 0.1 * i as f64, &mut b, 0.1, &mut work);
    }
    close(a[0], b[0], 1e-14);
    close(a[1], b[1], 1e-14);
}

#[test]
fn observer_sees_initial_state_and_every_step() {
    let mut y = [1.0];
    let mut work = [0.0; rk4_work_len(1)];
    let mut times = Vec::new();
    let mut values = Vec::new();
    rk4(decay, 0.0, 1.0, &mut y, 4, &mut work, |t, s| {
        times.push(t);
        values.push(s[0]);
    })
    .unwrap();
    assert_eq!(times, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    assert_eq!(values[0], 1.0);
    assert_eq!(*values.last().unwrap(), y[0]);
    assert!(values.windows(2).all(|w| w[1] < w[0]), "y muss fallen");
}

#[test]
fn fixed_step_methods_integrate_backwards() {
    // von t = 1 (y = e⁻¹) zurück nach t = 0 (y = 1)
    let mut y = [Real::exp(-1.0f64)];
    let mut work = [0.0; rk4_work_len(1)];
    rk4(decay, 1.0, 0.0, &mut y, 20, &mut work, |_, _| {}).unwrap();
    close(y[0], 1.0, 1e-7);
}

// ---------------------------------------------------------------------------
// RK45 (adaptiv)
// ---------------------------------------------------------------------------

#[test]
fn rk45_exponential_growth() {
    let mut y = [1.0];
    let mut work = [0.0; rk45_work_len(1)];
    let r = rk45(
        |_t, y: &[f64], dy: &mut [f64]| dy[0] = y[0],
        0.0,
        1.0,
        &mut y,
        Rk45Options::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    close(y[0], core::f64::consts::E, 1e-5);
    assert!(r.accepted_steps >= 1 && r.evaluations > 0);
}

#[test]
fn rk45_tighter_tolerance_means_smaller_error_and_more_steps() {
    let run = |rtol: f64| {
        let mut y = [0.0, 1.0];
        let mut work = [0.0; rk45_work_len(2)];
        let r = rk45(
            oscillator,
            0.0,
            10.0,
            &mut y,
            Rk45Options::with_tolerance(rtol, rtol * 1e-3),
            &mut work,
            |_, _| {},
        )
        .unwrap();
        let err = (y[0] - Real::sin(10.0f64))
            .abs()
            .max((y[1] - Real::cos(10.0f64)).abs());
        (err, r.accepted_steps)
    };
    let (e_loose, n_loose) = run(1e-4);
    let (e_tight, n_tight) = run(1e-9);
    assert!(e_tight < e_loose * 1e-2, "{e_loose:e} -> {e_tight:e}");
    assert!(n_tight > n_loose);
    assert!(e_tight < 1e-7, "Fehler {e_tight:e}");
}

#[test]
fn rk45_conserves_lotka_volterra_invariant() {
    // x' = a x − b x y,  y' = d x y − c y;  H = d x − c ln x + b y − a ln y ist konstant
    let (a, b, c, d) = (1.0, 0.5, 0.75, 0.25);
    let f = move |_t: f64, s: &[f64], ds: &mut [f64]| {
        ds[0] = a * s[0] - b * s[0] * s[1];
        ds[1] = d * s[0] * s[1] - c * s[1];
    };
    let h = |s: &[f64]| d * s[0] - c * Real::ln(s[0]) + b * s[1] - a * Real::ln(s[1]);
    let mut s = [4.0, 2.0];
    let h0 = h(&s);
    let mut work = [0.0; rk45_work_len(2)];
    let mut worst = 0.0f64;
    rk45(
        f,
        0.0,
        20.0,
        &mut s,
        Rk45Options::with_tolerance(1e-10, 1e-12),
        &mut work,
        |_, st| worst = worst.max((h(st) - h0).abs()),
    )
    .unwrap();
    assert!(worst < 1e-7, "Abweichung der Erhaltungsgröße {worst:e}");
}

#[test]
fn rk45_integrates_backwards() {
    let mut y = [Real::exp(-1.0f64)];
    let mut work = [0.0; rk45_work_len(1)];
    rk45(
        decay,
        1.0,
        0.0,
        &mut y,
        Rk45Options::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    close(y[0], 1.0, 1e-5);
}

#[test]
fn rk45_observer_times_are_monotone_and_end_exactly_at_t1() {
    let mut y = [1.0];
    let mut work = [0.0; rk45_work_len(1)];
    let mut times = Vec::new();
    rk45(
        decay,
        0.0,
        3.0,
        &mut y,
        Rk45Options::default(),
        &mut work,
        |t, _| times.push(t),
    )
    .unwrap();
    assert_eq!(times[0], 0.0);
    assert_eq!(*times.last().unwrap(), 3.0, "endet exakt bei t1");
    assert!(times.windows(2).all(|w| w[1] > w[0]));
}

#[test]
fn rk45_respects_h_max() {
    let mut y = [1.0];
    let mut work = [0.0; rk45_work_len(1)];
    let mut times = Vec::new();
    let opts = Rk45Options {
        h_max: 0.05,
        ..Rk45Options::default()
    };
    rk45(decay, 0.0, 1.0, &mut y, opts, &mut work, |t, _| {
        times.push(t)
    })
    .unwrap();
    let widest = times.windows(2).map(|w| w[1] - w[0]).fold(0.0f64, f64::max);
    assert!(widest <= 0.05 * 1.0002, "größter Schritt {widest}");
    assert!(times.len() >= 20);
}

#[test]
fn rk45_with_a_bad_initial_step_recovers() {
    let mut y = [1.0];
    let mut work = [0.0; rk45_work_len(1)];
    let opts = Rk45Options {
        h_init: 50.0,
        ..Rk45Options::default()
    };
    let r = rk45(decay, 0.0, 5.0, &mut y, opts, &mut work, |_, _| {}).unwrap();
    close(y[0], Real::exp(-5.0f64), 1e-6);
    assert!(
        r.rejected_steps >= 1,
        "zu großer Startschritt muss abgelehnt werden"
    );
}

#[test]
fn rk45_empty_interval_is_a_noop() {
    let mut y = [1.0];
    let mut work = [0.0; rk45_work_len(1)];
    let mut calls = 0;
    let r = rk45(
        decay,
        2.0,
        2.0,
        &mut y,
        Rk45Options::default(),
        &mut work,
        |_, _| calls += 1,
    )
    .unwrap();
    assert_eq!(
        (y[0], r.accepted_steps, r.evaluations, calls),
        (1.0, 0, 0, 1)
    );
}

#[test]
fn rk45_matches_exact_stiff_solution_but_needs_many_steps() {
    // λ = 1000: RK45 ist stabil, aber die Stabilitätsgrenze erzwingt viele kleine Schritte
    let f = |t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -1000.0 * (y[0] - Real::cos(t));
    let mut y = [0.0];
    let mut work = [0.0; rk45_work_len(1)];
    let r = rk45(
        f,
        0.0,
        2.0,
        &mut y,
        Rk45Options::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    close(y[0], stiff_exact(1000.0, 0.0, 2.0), 1e-5);
    assert!(r.accepted_steps > 150, "nur {} Schritte", r.accepted_steps);
}

#[test]
fn rk45_step_limit() {
    let f = |t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -1000.0 * (y[0] - Real::cos(t));
    let mut y = [0.0];
    let mut work = [0.0; rk45_work_len(1)];
    let opts = Rk45Options {
        max_steps: 20,
        ..Rk45Options::default()
    };
    assert_eq!(
        rk45(f, 0.0, 10.0, &mut y, opts, &mut work, |_, _| {}),
        Err(OdeError::MaxStepsExceeded)
    );
}

#[test]
fn rk45_blow_up_is_reported() {
    // y' = y², y(0) = 1 explodiert bei t = 1
    let mut y = [1.0];
    let mut work = [0.0; rk45_work_len(1)];
    let r = rk45(
        |_t, y: &[f64], dy: &mut [f64]| dy[0] = y[0] * y[0],
        0.0,
        2.0,
        &mut y,
        Rk45Options::default(),
        &mut work,
        |_, _| {},
    );
    assert!(
        matches!(
            r,
            Err(OdeError::StepSizeTooSmall | OdeError::NotFinite | OdeError::MaxStepsExceeded)
        ),
        "{r:?}"
    );
}

// ---------------------------------------------------------------------------
// Implizite Verfahren
// ---------------------------------------------------------------------------

fn decay_jac(_t: f64, _y: &[f64], j: &mut [f64]) {
    j[0] = -1.0;
}

fn run_implicit_euler(steps: usize) -> f64 {
    let mut y = [1.0];
    let mut work = [0.0; implicit_work_len(1)];
    implicit_euler(
        decay,
        decay_jac,
        0.0,
        1.0,
        &mut y,
        steps,
        newton(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    (y[0] - Real::exp(-1.0f64)).abs()
}

fn run_implicit_trapezoid(steps: usize) -> f64 {
    let mut y = [1.0];
    let mut work = [0.0; implicit_work_len(1)];
    implicit_trapezoid(
        decay,
        decay_jac,
        0.0,
        1.0,
        &mut y,
        steps,
        newton(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    (y[0] - Real::exp(-1.0f64)).abs()
}

#[test]
fn implicit_euler_has_order_one() {
    let ratio = run_implicit_euler(100) / run_implicit_euler(200);
    assert!((ratio - 2.0).abs() < 0.05, "Verhältnis {ratio}");
}

#[test]
fn implicit_trapezoid_has_order_two() {
    let ratio = run_implicit_trapezoid(50) / run_implicit_trapezoid(100);
    assert!((ratio - 4.0).abs() < 0.1, "Verhältnis {ratio}");
    assert!(run_implicit_trapezoid(100) < run_implicit_euler(100) * 0.01);
}

#[test]
fn implicit_analytic_and_finite_difference_jacobians_agree() {
    let mut a = [1.0];
    let mut b = [1.0];
    let mut work = [0.0; implicit_work_len(1)];
    implicit_euler(
        decay,
        decay_jac,
        0.0,
        1.0,
        &mut a,
        20,
        newton(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    implicit_euler_fd(decay, 0.0, 1.0, &mut b, 20, newton(), &mut work, |_, _| {}).unwrap();
    close(a[0], b[0], 1e-7);
    let mut c = [1.0];
    let mut d = [1.0];
    implicit_trapezoid(
        decay,
        decay_jac,
        0.0,
        1.0,
        &mut c,
        20,
        newton(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    implicit_trapezoid_fd(decay, 0.0, 1.0, &mut d, 20, newton(), &mut work, |_, _| {}).unwrap();
    close(c[0], d[0], 1e-7);
}

#[test]
fn implicit_methods_are_stable_on_a_stiff_problem_where_explicit_euler_explodes() {
    // λ = 1000, h = 0.05, konsistenter Anfangswert y(0) = 1 (= cos 0): die Lösung folgt cos t.
    // Explizit ist |1 − λh| = 49 > 1: jede Störung wächst um den Faktor 49 pro Schritt.
    let lambda = 1000.0;
    let f = move |t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -lambda * (y[0] - Real::cos(t));
    let jac = move |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = -lambda;

    let mut explicit = [1.0];
    let mut ework = [0.0; euler_work_len(1)];
    let r = euler(f, 0.0, 1.0, &mut explicit, 20, &mut ework, |_, _| {});
    assert!(
        r == Err(OdeError::NotFinite) || explicit[0].abs() > 1e6,
        "explizites Euler soll explodieren: {explicit:?} {r:?}"
    );

    let exact = stiff_exact(lambda, 1.0, 1.0);
    let mut work = [0.0; implicit_work_len(1)];
    let mut ie = [1.0];
    implicit_euler(
        f,
        jac,
        0.0,
        1.0,
        &mut ie,
        20,
        newton(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    close(ie[0], exact, 0.01);
    let mut it = [1.0];
    implicit_trapezoid(
        f,
        jac,
        0.0,
        1.0,
        &mut it,
        20,
        newton(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    close(it[0], exact, 1e-3);
}

#[test]
fn implicit_euler_damps_stiff_transients_but_trapezoid_rings() {
    // Inkonsistenter Anfangswert y(0) = 0 (Quasi-Gleichgewicht wäre cos 0 = 1): ein steifer
    // Einschwingvorgang. Euler (L-stabil) dämpft ihn sofort. Die Trapezregel (nur A-stabil) hat bei
    // hλ = 50 den Verstärkungsfaktor (1 − 25)/(1 + 25) ≈ −0.92: der Fehler klingt nur langsam ab und
    // wechselt dabei das Vorzeichen.
    let lambda = 1000.0;
    let f = move |t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -lambda * (y[0] - Real::cos(t));
    let jac = move |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = -lambda;
    let mut work = [0.0; implicit_work_len(1)];

    let mut ie = [0.0];
    implicit_euler(
        f,
        jac,
        0.0,
        1.0,
        &mut ie,
        20,
        newton(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    let exact = stiff_exact(lambda, 0.0, 1.0);
    close(ie[0], exact, 0.01);

    let mut it = [0.0];
    let mut signs = Vec::new();
    implicit_trapezoid(
        f,
        jac,
        0.0,
        1.0,
        &mut it,
        20,
        newton(),
        &mut work,
        |t, y| {
            signs.push(y[0] - Real::cos(t));
        },
    )
    .unwrap();
    let late_error = (it[0] - exact).abs();
    assert!(
        late_error > 0.1,
        "Trapez sollte hier sichtbar nachschwingen: {late_error}"
    );
    // Vorzeichenwechsel der Abweichung von cos t
    let flips = signs.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
    assert!(flips >= 5, "nur {flips} Vorzeichenwechsel");
}

#[test]
fn implicit_euler_solves_a_nonlinear_equation() {
    // y' = −y³, y(0) = 1  ->  y = 1/√(1 + 2t)
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -y[0] * y[0] * y[0];
    let jac = |_t: f64, y: &[f64], j: &mut [f64]| j[0] = -3.0 * y[0] * y[0];
    let mut y = [1.0];
    let mut work = [0.0; implicit_work_len(1)];
    implicit_euler(
        f,
        jac,
        0.0,
        4.0,
        &mut y,
        4000,
        newton(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    close(y[0], 1.0 / Real::sqrt(9.0f64), 5e-4);
}

#[test]
fn implicit_euler_on_robertson_problem_conserves_mass() {
    // Klassisches steifes Chemie-Problem (Raten von 0.04 bis 3·10⁷)
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
        dy[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
        dy[2] = 3.0e7 * y[1] * y[1];
    };
    let jac = |_t: f64, y: &[f64], j: &mut [f64]| {
        j.copy_from_slice(&[
            -0.04,
            1.0e4 * y[2],
            1.0e4 * y[1],
            0.04,
            -1.0e4 * y[2] - 6.0e7 * y[1],
            -1.0e4 * y[1],
            0.0,
            6.0e7 * y[1],
            0.0,
        ]);
    };
    let mut y = [1.0, 0.0, 0.0];
    let mut work = [0.0; implicit_work_len(3)];
    implicit_euler(
        f,
        jac,
        0.0,
        40.0,
        &mut y,
        4000,
        newton(),
        &mut work,
        |_, s| {
            assert!(s.iter().all(|v| v.is_finite()));
        },
    )
    .unwrap();
    // Referenzwerte bei t = 40: y₁ ≈ 0.7158, y₂ ≈ 9.185e-6, y₃ ≈ 0.2842
    close(y[0], 0.7158, 5e-3);
    close(y[2], 0.2842, 5e-3);
    assert!(y[1] > 0.0 && y[1] < 5e-5, "y₂ = {:e}", y[1]);
    close(y[0] + y[1] + y[2], 1.0, 1e-9);
}

#[test]
fn implicit_methods_integrate_systems_and_backwards() {
    let jac = |_t: f64, _y: &[f64], j: &mut [f64]| j.copy_from_slice(&[0.0, 1.0, -1.0, 0.0]);
    let mut y = [0.0, 1.0];
    let mut work = [0.0; implicit_work_len(2)];
    let two_pi = 2.0 * core::f64::consts::PI;
    implicit_trapezoid(
        oscillator,
        jac,
        0.0,
        two_pi,
        &mut y,
        400,
        newton(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    close(y[0], 0.0, 1e-3);
    close(y[1], 1.0, 1e-3);
    // Trapezregel erhält die Energie y² + v² des Oszillators exakt
    close(y[0] * y[0] + y[1] * y[1], 1.0, 1e-9);

    let mut z = [Real::exp(-1.0f64)];
    let mut w1 = [0.0; implicit_work_len(1)];
    implicit_trapezoid(
        decay,
        decay_jac,
        1.0,
        0.0,
        &mut z,
        100,
        newton(),
        &mut w1,
        |_, _| {},
    )
    .unwrap();
    close(z[0], 1.0, 1e-4);
}

#[test]
fn implicit_newton_failure_is_reported() {
    // y' = y², y(0) = 1, h = 1: y² − y + 1 = 0 hat keine reelle Lösung
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = y[0] * y[0];
    let mut y = [1.0];
    let mut work = [0.0; implicit_work_len(1)];
    let r = implicit_euler_fd(f, 0.0, 1.0, &mut y, 1, newton(), &mut work, |_, _| {});
    assert_eq!(r, Err(OdeError::NewtonFailed));
}

// ---------------------------------------------------------------------------
// Fehlerfälle und Sonstiges
// ---------------------------------------------------------------------------

#[test]
fn invalid_input_is_rejected() {
    let mut big = [0.0; 100];
    let mut empty: [f64; 0] = [];
    let mut y = [1.0];
    let mut none = |_: f64, _: &[f64]| {};

    assert_eq!(
        euler(decay, 0.0, 1.0, &mut empty, 4, &mut big, &mut none),
        Err(OdeError::InvalidInput)
    );
    assert_eq!(
        euler(decay, 0.0, 1.0, &mut y, 0, &mut big, &mut none),
        Err(OdeError::InvalidInput)
    );
    assert_eq!(
        rk4(decay, 0.0, 1.0, &mut y, 0, &mut big, &mut none),
        Err(OdeError::InvalidInput)
    );
    let mut short = [0.0; 2];
    assert_eq!(
        rk4(decay, 0.0, 1.0, &mut y, 4, &mut short, &mut none),
        Err(OdeError::InvalidInput)
    );
    assert_eq!(
        rk45(
            decay,
            0.0,
            1.0,
            &mut y,
            Rk45Options::default(),
            &mut short,
            &mut none
        ),
        Err(OdeError::InvalidInput)
    );
    assert_eq!(
        rk45(
            decay,
            0.0,
            1.0,
            &mut empty,
            Rk45Options::default(),
            &mut big,
            &mut none
        ),
        Err(OdeError::InvalidInput)
    );
    assert_eq!(
        implicit_euler_fd(decay, 0.0, 1.0, &mut y, 4, newton(), &mut short, &mut none),
        Err(OdeError::InvalidInput)
    );
    assert_eq!(
        implicit_euler_fd(decay, 0.0, 1.0, &mut y, 0, newton(), &mut big, &mut none),
        Err(OdeError::InvalidInput)
    );
}

#[test]
fn non_finite_values_are_reported() {
    let mut work = [0.0; 100];
    let mut none = |_: f64, _: &[f64]| {};
    let nan_f = |_t: f64, _y: &[f64], dy: &mut [f64]| dy[0] = f64::NAN;
    let mut y = [1.0];
    assert_eq!(
        rk4(nan_f, 0.0, 1.0, &mut y, 4, &mut work, &mut none),
        Err(OdeError::NotFinite)
    );
    let mut y = [1.0];
    assert_eq!(
        euler(nan_f, 0.0, 1.0, &mut y, 4, &mut work, &mut none),
        Err(OdeError::NotFinite)
    );
    let mut y = [1.0];
    assert_eq!(
        rk45(
            nan_f,
            0.0,
            1.0,
            &mut y,
            Rk45Options::default(),
            &mut work,
            &mut none
        ),
        Err(OdeError::NotFinite)
    );
    let mut y = [f64::NAN];
    assert_eq!(
        rk4(decay, 0.0, 1.0, &mut y, 4, &mut work, &mut none),
        Err(OdeError::NotFinite)
    );
    let mut y = [1.0];
    assert_eq!(
        rk4(decay, 0.0, f64::INFINITY, &mut y, 4, &mut work, &mut none),
        Err(OdeError::NotFinite)
    );
    let mut y = [1.0];
    assert_eq!(
        implicit_euler_fd(nan_f, 0.0, 1.0, &mut y, 2, newton(), &mut work, &mut none),
        Err(OdeError::NotFinite)
    );
}

#[test]
fn f32_works() {
    let f = |_t: f32, y: &[f32], dy: &mut [f32]| dy[0] = -y[0];
    let mut y = [1.0f32];
    let mut work = [0.0f32; rk45_work_len(1)];
    rk45(
        f,
        0.0,
        1.0,
        &mut y,
        Rk45Options::default(),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    assert!((y[0] - (-1.0f32).exp()).abs() < 1e-4, "{}", y[0]);
    let mut z = [1.0f32];
    let mut w = [0.0f32; rk4_work_len(1)];
    rk4(f, 0.0, 1.0, &mut z, 20, &mut w, |_, _| {}).unwrap();
    assert!((z[0] - (-1.0f32).exp()).abs() < 1e-5);
}

#[test]
fn options_and_error_display() {
    let o = Rk45Options::<f64>::default();
    assert_eq!(
        (o.rtol, o.atol, o.h_init, o.h_max, o.max_steps),
        (1e-6, 1e-9, 0.0, 0.0, 100_000)
    );
    let t = Rk45Options::with_tolerance(1e-3, 1e-6);
    assert_eq!((t.rtol, t.atol), (1e-3, 1e-6));
    extern crate std;
    use std::string::ToString;
    for e in [
        OdeError::InvalidInput,
        OdeError::NotFinite,
        OdeError::StepSizeTooSmall,
        OdeError::MaxStepsExceeded,
        OdeError::NewtonFailed,
    ] {
        assert!(!e.to_string().is_empty());
    }
}
