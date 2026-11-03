//! Tests für `bdf_adaptive` (variable Schrittweite und Ordnung).
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::ode::{BdfAdaptiveOptions, OdeError, bdf_adaptive, bdf_adaptive_fd};

#[test]
fn exponential_decay_accuracy_improves_with_tolerance() {
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -y[0];
    let jac = |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = -1.0;
    let run = |rtol: f64| {
        let mut y = [1.0];
        let r = bdf_adaptive(
            f,
            jac,
            0.0,
            2.0,
            &mut y,
            BdfAdaptiveOptions::with_tolerance(rtol, rtol * 1e-3),
            |_, _| {},
        )
        .unwrap();
        ((y[0] - (-2f64).exp()).abs(), r)
    };
    let (e_loose, r_loose) = run(1e-4);
    let (e_tight, r_tight) = run(1e-9);
    assert!(e_loose < 1e-2, "{e_loose}");
    assert!(e_tight < 1e-6, "{e_tight}");
    assert!(e_tight < e_loose);
    assert!(r_tight.accepted_steps > r_loose.accepted_steps);
    assert!(r_tight.last_order >= 2, "{r_tight:?}");
}

#[test]
fn stiff_problem_few_steps() {
    let mut y = [1.0f64];
    let mut times = Vec::new();
    let r = bdf_adaptive(
        |t, y: &[f64], dy: &mut [f64]| dy[0] = -1000.0 * (y[0] - t.cos()),
        |_t, _y: &[f64], j: &mut [f64]| j[0] = -1000.0,
        0.0,
        2.0,
        &mut y,
        BdfAdaptiveOptions::with_tolerance(1e-7, 1e-10),
        |t, _| times.push(t),
    )
    .unwrap();
    let exact = (1.0e6 * 2f64.cos() + 1000.0 * 2f64.sin()) / (1.0e6 + 1.0);
    assert!((y[0] - exact).abs() < 1e-4, "{} vs {}", y[0], exact);
    assert!(r.accepted_steps < 1000, "{r:?}");
    assert_eq!(*times.last().unwrap(), 2.0);
    assert!(times.windows(2).all(|w| w[1] > w[0]));
}

#[test]
fn robertson_conserves_mass() {
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
        dy[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
        dy[2] = 3.0e7 * y[1] * y[1];
    };
    let mut y = [1.0, 0.0, 0.0];
    let r = bdf_adaptive_fd(
        f,
        0.0,
        40.0,
        &mut y,
        BdfAdaptiveOptions::with_tolerance(1e-7, 1e-11),
        |_, _| {},
    )
    .unwrap();
    assert!((y.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    assert!((y[0] - 0.7158).abs() < 5e-3, "{y:?}");
    assert!(r.accepted_steps < 5000, "{r:?}");
}

#[test]
fn van_der_pol_stiff_runs() {
    let mu = 100.0;
    let f = move |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = y[1];
        dy[1] = mu * (1.0 - y[0] * y[0]) * y[1] - y[0];
    };
    let mut y = [2.0, 0.0];
    let r = bdf_adaptive_fd(
        f,
        0.0,
        200.0,
        &mut y,
        BdfAdaptiveOptions::with_tolerance(1e-5, 1e-8),
        |_, _| {},
    )
    .unwrap();
    assert!(y.iter().all(|v| v.is_finite()) && y[0].abs() <= 2.1);
    assert!(r.accepted_steps < 20_000, "{r:?}");
}

#[test]
fn backward_and_order_limit_and_errors() {
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -y[0];
    let mut y = [(-1f64).exp()];
    bdf_adaptive_fd(
        f,
        1.0,
        0.0,
        &mut y,
        BdfAdaptiveOptions::with_tolerance(1e-8, 1e-11),
        |_, _| {},
    )
    .unwrap();
    assert!((y[0] - 1.0).abs() < 1e-5);
    let mut opts = BdfAdaptiveOptions::with_tolerance(1e-8, 1e-11);
    opts.max_order = 1;
    let mut y = [1.0];
    let r = bdf_adaptive_fd(f, 0.0, 1.0, &mut y, opts, |_, _| {}).unwrap();
    assert_eq!(r.last_order, 1);
    opts.max_order = 6;
    assert_eq!(
        bdf_adaptive_fd(f, 0.0, 1.0, &mut y, opts, |_, _| {}).unwrap_err(),
        OdeError::InvalidInput
    );
    let mut e: [f64; 0] = [];
    assert_eq!(
        bdf_adaptive_fd(
            f,
            0.0,
            1.0,
            &mut e,
            BdfAdaptiveOptions::default(),
            |_, _| {}
        )
        .unwrap_err(),
        OdeError::InvalidInput
    );
    let mut nan = [f64::NAN];
    assert_eq!(
        bdf_adaptive_fd(
            f,
            0.0,
            1.0,
            &mut nan,
            BdfAdaptiveOptions::default(),
            |_, _| {}
        )
        .unwrap_err(),
        OdeError::NotFinite
    );
    let mut o = BdfAdaptiveOptions::default();
    o.max_steps = 3;
    o.h_init = 1e-6;
    let mut y = [1.0];
    assert_eq!(
        bdf_adaptive_fd(f, 0.0, 1.0, &mut y, o, |_, _| {}).unwrap_err(),
        OdeError::MaxStepsExceeded
    );
}
