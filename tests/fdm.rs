//! Tests für `fdm`.
#![cfg(any(feature = "std", feature = "libm"))]
#![allow(
    clippy::needless_range_loop,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::excessive_precision
)]

use azmath::fdm::*;
use azmath::ode::{
    AdaptiveOptions, Rk45Options, adaptive_work_len, radau5_adaptive_fd, rk45, rk45_work_len,
};
use core::f64::consts::PI;

#[test]
fn fornberg_known_weights() {
    let h = 0.1f64;
    let mut w = [0.0; 3];
    let mut work = [0.0; fornberg_work_len(3, 2)];
    fornberg_weights(2, 0.0, &[-h, 0.0, h], &mut w, &mut work);
    for (a, b) in w.iter().zip([1.0f64, -2.0, 1.0]) {
        assert!((a - b / (h * h)).abs() < 1e-9);
    }
    let mut w5 = [0.0; 5];
    let mut work = [0.0; fornberg_work_len(5, 1)];
    fornberg_weights(1, 0.0, &[-2.0 * h, -h, 0.0, h, 2.0 * h], &mut w5, &mut work);
    for (a, b) in w5.iter().zip([1.0f64, -8.0, 0.0, 8.0, -1.0]) {
        assert!((a - b / (12.0 * h)).abs() < 1e-8);
    }
    // einseitig: f'(0) aus Knoten 0, 1, 2: (−3, 4, −1)/2
    let mut w3 = [0.0; 3];
    let mut work = [0.0; fornberg_work_len(3, 1)];
    fornberg_weights(1, 0.0, &[0.0, 1.0, 2.0], &mut w3, &mut work);
    for (a, b) in w3.iter().zip([-1.5f64, 2.0, -0.5]) {
        assert!((a - b).abs() < 1e-13);
    }
}

#[test]
fn fornberg_nonuniform_exact_for_polynomials() {
    let nodes = [0.0, 0.3, 0.9, 1.4];
    let x0 = 0.5;
    let mut w = [0.0; 4];
    let mut work = [0.0; fornberg_work_len(4, 2)];
    fornberg_weights(2, x0, &nodes, &mut w, &mut work);
    // f = x³: f'' = 6x
    let approx: f64 = nodes.iter().zip(&w).map(|(x, w)| w * x * x * x).sum();
    assert!((approx - 6.0 * x0).abs() < 1e-11);
    // Summe der Gewichte einer Ableitung ≥ 1 ist 0
    assert!(w.iter().sum::<f64>().abs() < 1e-10);
}

fn grid(n: usize) -> (Vec<f64>, f64) {
    let h = PI / (n as f64 - 1.0);
    ((0..n).map(|i| i as f64 * h).collect(), h)
}

#[test]
fn derivatives_converge_second_order() {
    let mut errs = Vec::new();
    for n in [21usize, 41] {
        let (x, h) = grid(n);
        let u: Vec<f64> = x.iter().map(|v| v.sin()).collect();
        let (mut d1, mut d2) = (vec![0.0; n], vec![0.0; n]);
        first_derivative_1d(&u, h, &mut d1);
        second_derivative_1d(&u, h, &mut d2);
        let e1 = (0..n)
            .map(|i| (d1[i] - x[i].cos()).abs())
            .fold(0.0, f64::max);
        let e2 = (0..n)
            .map(|i| (d2[i] + x[i].sin()).abs())
            .fold(0.0, f64::max);
        errs.push((e1, e2));
    }
    assert!(
        errs[0].0 / errs[1].0 > 3.5 && errs[0].1 / errs[1].1 > 3.5,
        "{errs:?}"
    );
}

#[test]
fn laplacian_of_quadratic_is_exact() {
    let (nx, ny) = (6, 5);
    let (hx, hy) = (0.2, 0.5);
    let mut u = vec![0.0; nx * ny];
    for j in 0..ny {
        for i in 0..nx {
            let (x, y) = (i as f64 * hx, j as f64 * hy);
            u[j * nx + i] = x * x + 3.0 * y * y;
        }
    }
    let mut out = vec![9.0; nx * ny];
    laplacian_2d(&u, nx, ny, hx, hy, &mut out);
    for j in 0..ny {
        for i in 0..nx {
            let want = if i == 0 || j == 0 || i == nx - 1 || j == ny - 1 {
                0.0
            } else {
                8.0
            };
            assert!((out[j * nx + i] - want).abs() < 1e-9);
        }
    }
}

fn heat_initial(n: usize) -> (Vec<f64>, f64) {
    let h = 1.0 / (n as f64 - 1.0);
    ((0..n).map(|i| (PI * i as f64 * h).sin()).collect(), h)
}

#[test]
fn heat_equation_rk45() {
    let n = 41;
    let (mut u, h) = heat_initial(n);
    let mut work = vec![0.0; rk45_work_len(n)];
    rk45(
        |_t, u: &[f64], du: &mut [f64]| heat_rhs_1d(1.0, u, h, du),
        0.0,
        0.05,
        &mut u,
        Rk45Options::with_tolerance(1e-9, 1e-12),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    let decay = (-PI * PI * 0.05).exp();
    for i in 0..n {
        let exact = decay * (PI * i as f64 * h).sin();
        assert!((u[i] - exact).abs() < 2e-3, "i = {i}");
    }
}

#[test]
fn heat_equation_radau_adaptive() {
    let n = 21;
    let (mut u, h) = heat_initial(n);
    let mut work = vec![0.0; adaptive_work_len(n)];
    radau5_adaptive_fd(
        |_t, u: &[f64], du: &mut [f64]| heat_rhs_1d(1.0, u, h, du),
        0.0,
        0.2,
        &mut u,
        AdaptiveOptions::with_tolerance(1e-7, 1e-10),
        &mut work,
        |_, _| {},
    )
    .unwrap();
    let decay = (-PI * PI * 0.2).exp();
    for i in 0..n {
        let exact = decay * (PI * i as f64 * h).sin();
        assert!((u[i] - exact).abs() < 5e-3, "i = {i}");
    }
}
