//! Tests für `seulex` (extrapoliertes linear-implizites Euler-Verfahren).
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]
#![allow(
    clippy::needless_range_loop,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::excessive_precision
)]

use azmath::ode::{OdeError, SeulexOptions, seulex, seulex_fd};

#[test]
fn accuracy_scales_with_tolerance() {
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = y[1];
        dy[1] = -y[0];
    };
    let jac = |_t: f64, _y: &[f64], j: &mut [f64]| j.copy_from_slice(&[0.0, 1.0, -1.0, 0.0]);
    let err = |tol: f64| {
        let mut y = [0.0, 1.0];
        seulex(
            f,
            jac,
            0.0,
            3.0,
            &mut y,
            SeulexOptions::with_tolerance(tol, tol * 1e-3),
            |_, _| {},
        )
        .unwrap();
        (y[0] - 3f64.sin()).abs().max((y[1] - 3f64.cos()).abs())
    };
    let (a, b) = (err(1e-4), err(1e-9));
    assert!(a < 1e-2, "{a}");
    assert!(b < 1e-6, "{b}");
    assert!(b < a);
}

#[test]
fn stiff_nonautonomous_problem() {
    let mut y = [1.0f64];
    let r = seulex_fd(
        |t, y: &[f64], dy: &mut [f64]| dy[0] = -1000.0 * (y[0] - t.cos()),
        0.0,
        2.0,
        &mut y,
        SeulexOptions::with_tolerance(1e-7, 1e-10),
        |_, _| {},
    )
    .unwrap();
    let exact = (1.0e6 * 2f64.cos() + 1000.0 * 2f64.sin()) / (1.0e6 + 1.0);
    assert!((y[0] - exact).abs() < 1e-4, "{} vs {}", y[0], exact);
    assert!(r.accepted_steps < 500, "{r:?}");
}

#[test]
fn robertson_and_backward() {
    let f = |_t: f64, y: &[f64], dy: &mut [f64]| {
        dy[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2];
        dy[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] * y[1];
        dy[2] = 3.0e7 * y[1] * y[1];
    };
    let mut y = [1.0, 0.0, 0.0];
    seulex_fd(
        f,
        0.0,
        40.0,
        &mut y,
        SeulexOptions::with_tolerance(1e-7, 1e-11),
        |_, _| {},
    )
    .unwrap();
    assert!((y.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    assert!((y[0] - 0.7158).abs() < 5e-3, "{y:?}");
    let g = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -y[0];
    let mut z = [(-1f64).exp()];
    seulex_fd(
        g,
        1.0,
        0.0,
        &mut z,
        SeulexOptions::with_tolerance(1e-9, 1e-12),
        |_, _| {},
    )
    .unwrap();
    assert!((z[0] - 1.0).abs() < 1e-6);
}

#[test]
fn errors() {
    let g = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -y[0];
    let mut e: [f64; 0] = [];
    assert_eq!(
        seulex_fd(g, 0.0, 1.0, &mut e, SeulexOptions::default(), |_, _| {}).unwrap_err(),
        OdeError::InvalidInput
    );
    let mut y = [1.0];
    let mut o = SeulexOptions::default();
    o.columns = 1;
    assert_eq!(
        seulex_fd(g, 0.0, 1.0, &mut y, o, |_, _| {}).unwrap_err(),
        OdeError::InvalidInput
    );
    let mut nan = [f64::NAN];
    assert_eq!(
        seulex_fd(g, 0.0, 1.0, &mut nan, SeulexOptions::default(), |_, _| {}).unwrap_err(),
        OdeError::NotFinite
    );
    let mut o = SeulexOptions::default();
    o.max_steps = 1;
    o.h_init = 1e-6;
    assert_eq!(
        seulex_fd(g, 0.0, 1.0, &mut y, o, |_, _| {}).unwrap_err(),
        OdeError::MaxStepsExceeded
    );
}
