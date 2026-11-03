//! Tests für ableitungsfreie Minimierer (Goldener Schnitt, Brent, Nelder-Mead).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]
#![allow(
    clippy::needless_range_loop,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::excessive_precision
)]

use azmath::optimize::{
    MinOptions, NelderMeadOptions, OptimizeError, Termination, brent_min, golden_section,
    nelder_mead, nelder_mead_work_len,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} gegen {b}");
}

#[test]
fn one_dimensional_minimizers_find_known_minima() {
    let opts = MinOptions::default();
    let cases: [(&str, fn(f64) -> f64, f64, f64, f64); 4] = [
        ("Parabel", |x| (x - 2.0).powi(2) + 1.0, 0.0, 5.0, 2.0),
        ("cos", |x| x.cos(), 2.0, 4.0, core::f64::consts::PI),
        ("x·e^x", |x| x * x.exp(), -3.0, 2.0, -1.0),
        ("|x − 1|", |x| (x - 1.0).abs(), -2.0, 3.0, 1.0),
    ];
    for (name, f, a, b, want) in cases {
        let g = golden_section(f, a, b, opts).unwrap();
        let r = brent_min(f, a, b, opts).unwrap();
        close(g.x, want, 1e-6);
        close(r.x, want, 1e-6);
        assert!(r.f_value <= g.f_value + 1e-12, "{name}");
    }
}

#[test]
fn brent_needs_far_fewer_evaluations_on_smooth_functions() {
    let mut count_golden = 0;
    let mut count_brent = 0;
    let f = |x: f64| (x - 0.3).powi(2) + (x - 0.3).powi(4);
    golden_section(
        |x| {
            count_golden += 1;
            f(x)
        },
        -2.0,
        2.0,
        MinOptions::default(),
    )
    .unwrap();
    brent_min(
        |x| {
            count_brent += 1;
            f(x)
        },
        -2.0,
        2.0,
        MinOptions::default(),
    )
    .unwrap();
    assert!(
        count_brent < count_golden,
        "Brent {count_brent}, Goldener Schnitt {count_golden}"
    );
}

#[test]
fn interval_order_does_not_matter_and_errors_are_reported() {
    let f = |x: f64| (x - 1.0) * (x - 1.0);
    close(
        brent_min(f, 4.0, -2.0, MinOptions::default()).unwrap().x,
        1.0,
        1e-6,
    );
    assert_eq!(
        golden_section(f, 1.0, 1.0, MinOptions::default()),
        Err(OptimizeError::InvalidInput)
    );
    assert_eq!(
        brent_min(f, f64::NAN, 1.0, MinOptions::default()),
        Err(OptimizeError::NotFinite)
    );
    assert_eq!(
        brent_min(|_| f64::NAN, 0.0, 1.0, MinOptions::default()),
        Err(OptimizeError::NotFinite)
    );
    let tiny = MinOptions {
        max_iter: 2,
        ..MinOptions::default()
    };
    assert_eq!(
        golden_section(f, -50.0, 50.0, tiny),
        Err(OptimizeError::NoConvergence)
    );
}

#[test]
fn nelder_mead_minimizes_standard_test_functions() {
    let mut work = [0.0; nelder_mead_work_len(2)];
    // Rosenbrock
    let mut x = [-1.2, 1.0];
    let r = nelder_mead(
        |x: &[f64]| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2),
        &mut x,
        NelderMeadOptions::default(),
        &mut work,
    )
    .unwrap();
    close(x[0], 1.0, 1e-4);
    close(x[1], 1.0, 1e-4);
    assert!(r.f_value < 1e-8);
    assert_eq!(r.termination, Termination::Step);
    assert!(r.evaluations > r.iterations);
    // Kugel im Vierdimensionalen und eine nicht glatte Funktion
    let mut work4 = [0.0; nelder_mead_work_len(4)];
    let mut y = [1.0, -2.0, 3.0, 0.5];
    nelder_mead(
        |x: &[f64]| x.iter().map(|v| v * v).sum(),
        &mut y,
        NelderMeadOptions::default(),
        &mut work4,
    )
    .unwrap();
    assert!(y.iter().all(|v| v.abs() < 1e-3), "{y:?}");
    let mut z = [3.0, 2.0];
    nelder_mead(
        |x: &[f64]| (x[0] - 1.0).abs() + 2.0 * (x[1] + 1.0).abs(),
        &mut z,
        NelderMeadOptions::default(),
        &mut work,
    )
    .unwrap();
    close(z[0], 1.0, 1e-3);
    close(z[1], -1.0, 1e-3);
}

#[test]
fn nelder_mead_in_one_dimension_and_with_noise_free_ridges() {
    let mut work = [0.0; nelder_mead_work_len(1)];
    let mut x = [10.0];
    nelder_mead(
        |x: &[f64]| (x[0] - 3.0).powi(2),
        &mut x,
        NelderMeadOptions::default(),
        &mut work,
    )
    .unwrap();
    close(x[0], 3.0, 1e-3);
}

#[test]
fn nelder_mead_errors() {
    let mut work = [0.0; nelder_mead_work_len(2)];
    let f = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let mut empty: [f64; 0] = [];
    assert_eq!(
        nelder_mead(f, &mut empty, NelderMeadOptions::default(), &mut work),
        Err(OptimizeError::InvalidInput)
    );
    let mut x = [1.0, 1.0];
    assert_eq!(
        nelder_mead(f, &mut x, NelderMeadOptions::default(), &mut work[..3]),
        Err(OptimizeError::InvalidInput)
    );
    let mut bad = [f64::NAN, 0.0];
    assert_eq!(
        nelder_mead(f, &mut bad, NelderMeadOptions::default(), &mut work),
        Err(OptimizeError::NotFinite)
    );
    let mut y = [5.0, 5.0];
    let few = NelderMeadOptions {
        max_iter: 3,
        ..NelderMeadOptions::default()
    };
    assert_eq!(
        nelder_mead(f, &mut y, few, &mut work),
        Err(OptimizeError::NoConvergence)
    );
    assert!(f(&y) < 50.0, "beste Ecke wurde zurückgegeben");
}

#[test]
fn works_for_f32() {
    let r = brent_min(|x: f32| (x - 1.5).powi(2), 0.0, 4.0, MinOptions::default()).unwrap();
    assert!((r.x - 1.5).abs() < 1e-3);
    let mut work = [0.0f32; nelder_mead_work_len(2)];
    let mut x = [3.0f32, -2.0];
    nelder_mead(
        |x: &[f32]| (x[0] - 1.0).powi(2) + (x[1] + 0.5).powi(2),
        &mut x,
        NelderMeadOptions {
            tol_x: 1e-3,
            tol_f: 1e-5,
            ..NelderMeadOptions::default()
        },
        &mut work,
    )
    .unwrap();
    assert!((x[0] - 1.0).abs() < 1e-2 && (x[1] + 0.5).abs() < 1e-2);
}
