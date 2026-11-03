//! Tests für Randwertprobleme und DAE.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::ode::{OdeError, Rk45Options, bvp_shooting, bvp_trapezoid, dae_bdf};
use azmath::optimize::OptimizeOptions;
use core::f64::consts::FRAC_PI_2;

fn osc(_t: f64, y: &[f64], dy: &mut [f64]) {
    dy[0] = y[1];
    dy[1] = -y[0];
}

fn bc_sin(ya: &[f64], yb: &[f64], r: &mut [f64]) {
    r[0] = ya[0];
    r[1] = yb[0] - 1.0;
}

#[test]
fn shooting_sine() {
    let r = bvp_shooting(
        osc,
        bc_sin,
        0.0,
        FRAC_PI_2,
        &[0.0, 0.5],
        10,
        Rk45Options::with_tolerance(1e-10, 1e-12),
        OptimizeOptions::default(),
    )
    .unwrap();
    assert_eq!(r.t.len(), 11);
    for (k, &t) in r.t.iter().enumerate() {
        assert!((r.y[2 * k] - t.sin()).abs() < 1e-6, "k = {k}");
        assert!((r.y[2 * k + 1] - t.cos()).abs() < 1e-6);
    }
    assert!(
        bvp_shooting(
            osc,
            bc_sin,
            0.0,
            1.0,
            &[],
            5,
            Rk45Options::default(),
            OptimizeOptions::default()
        )
        .is_err()
    );
}

#[test]
fn trapezoid_sine_and_convergence() {
    let run = |n: usize| {
        let r = bvp_trapezoid(
            osc,
            bc_sin,
            0.0,
            FRAC_PI_2,
            n,
            2,
            |_t, y: &mut [f64]| y.iter_mut().for_each(|v| *v = 0.0),
            OptimizeOptions::default(),
        )
        .unwrap();
        r.t.iter()
            .enumerate()
            .map(|(k, &t)| (r.y[2 * k] - t.sin()).abs())
            .fold(0.0, f64::max)
    };
    let (e1, e2) = (run(20), run(40));
    assert!(e2 < 1e-3 && e1 / e2 > 3.0, "{e1} {e2}");
}

#[test]
fn trapezoid_handles_unstable_shooting() {
    // y'' = 100·y, y(0) = 1, y(1) = 0  ⇒  y = sinh(10(1 − t))/sinh(10)
    let r = bvp_trapezoid(
        |_t, y: &[f64], dy: &mut [f64]| {
            dy[0] = y[1];
            dy[1] = 100.0 * y[0];
        },
        |ya: &[f64], yb: &[f64], res: &mut [f64]| {
            res[0] = ya[0] - 1.0;
            res[1] = yb[0];
        },
        0.0,
        1.0,
        200,
        2,
        |t, y: &mut [f64]| {
            y[0] = 1.0 - t;
            y[1] = -1.0;
        },
        OptimizeOptions::default(),
    )
    .unwrap();
    for (k, &t) in r.t.iter().enumerate().step_by(20) {
        let exact = (10.0 * (1.0 - t)).sinh() / 10f64.sinh();
        assert!((r.y[2 * k] - exact).abs() < 5e-3, "t = {t}");
    }
    assert!(
        bvp_trapezoid(
            osc,
            bc_sin,
            0.0,
            1.0,
            0,
            2,
            |_t, _y: &mut [f64]| {},
            OptimizeOptions::default()
        )
        .is_err()
    );
}

#[test]
fn dae_linear_index1() {
    let f = |_t: f64, y: &[f64], yp: &[f64], r: &mut [f64]| {
        r[0] = yp[0] + y[0] - y[1];
        r[1] = y[1] - 2.0 * y[0];
    };
    let mut y1 = [1.0, 2.0];
    dae_bdf(
        f,
        0.0,
        1.0,
        &mut y1,
        200,
        1,
        OptimizeOptions::default(),
        |_, _| {},
    )
    .unwrap();
    let mut y2 = [1.0, 2.0];
    dae_bdf(
        f,
        0.0,
        1.0,
        &mut y2,
        200,
        2,
        OptimizeOptions::default(),
        |_, _| {},
    )
    .unwrap();
    let e = std::f64::consts::E;
    assert!((y1[0] - e).abs() < 2e-2);
    assert!((y2[0] - e).abs() < 1e-3);
    assert!((y2[1] - 2.0 * y2[0]).abs() < 1e-9);
    assert!((y2[0] - e).abs() < (y1[0] - e).abs());
}

#[test]
fn dae_robertson() {
    let f = |_t: f64, y: &[f64], yp: &[f64], r: &mut [f64]| {
        r[0] = yp[0] + 0.04 * y[0] - 1.0e4 * y[1] * y[2];
        r[1] = yp[1] - 0.04 * y[0] + 1.0e4 * y[1] * y[2] + 3.0e7 * y[1] * y[1];
        r[2] = y[0] + y[1] + y[2] - 1.0;
    };
    let mut y = [1.0, 0.0, 0.0];
    let mut count = 0;
    dae_bdf(
        f,
        0.0,
        1.0,
        &mut y,
        500,
        2,
        OptimizeOptions::default(),
        |_, _| count += 1,
    )
    .unwrap();
    assert_eq!(count, 501);
    assert!((y.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(y[0] > 0.95 && y[0] < 0.98, "{}", y[0]);
    let mut e = [1.0];
    assert_eq!(
        dae_bdf(
            |_t, _y: &[f64], _yp: &[f64], _r: &mut [f64]| {},
            0.0,
            1.0,
            &mut e,
            10,
            3,
            OptimizeOptions::default(),
            |_, _| {}
        )
        .unwrap_err(),
        OdeError::InvalidInput
    );
}
