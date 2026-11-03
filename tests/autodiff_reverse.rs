//! Tests für den Rückwärtsmodus.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::autodiff::{Tape, gradient_ad};
use azmath::diff::gradient;

#[test]
fn product_and_sin() {
    let (v, g) = gradient_ad(&[2.0f64, 3.0], |x| x[0] * x[1] + x[0].sin());
    assert!((v - (6.0 + 2f64.sin())).abs() < 1e-14);
    assert!((g[0] - (3.0 + 2f64.cos())).abs() < 1e-14);
    assert!((g[1] - 2.0).abs() < 1e-14);
}

#[test]
fn matches_finite_differences() {
    let x0 = [0.7f64, 1.3, -0.4, 2.1];
    let (_, g) = gradient_ad(&x0, |x| {
        let a = (x[0] * x[1]).exp() / (x[2] * x[2] + 1.0);
        let b = (x[3] - x[0]).powi(3) + x[1].sqrt().ln();
        a + b * x[2].tanh() - x[3].atan() + (x[0] - x[1]).abs() * 2.0
    });
    let f = |x: &[f64]| {
        let a = (x[0] * x[1]).exp() / (x[2] * x[2] + 1.0);
        let b = (x[3] - x[0]).powi(3) + x[1].sqrt().ln();
        a + b * x[2].tanh() - x[3].atan() + (x[0] - x[1]).abs() * 2.0
    };
    let mut fd = [0.0; 4];
    let mut work = [0.0; 4];
    gradient(f, &x0, &mut fd, &mut work);
    for i in 0..4 {
        assert!(
            (g[i] - fd[i]).abs() < 1e-6,
            "i = {i}: {} vs {}",
            g[i],
            fd[i]
        );
    }
}

#[test]
fn many_inputs_and_reuse() {
    // f = Σ x_i² · x_{i+1}; Variablen mehrfach verwendet
    let n = 200;
    let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.01 * i as f64).collect();
    let (_, g) = gradient_ad(&x, |v| {
        let mut s = v[0] * 0.0;
        for i in 0..v.len() - 1 {
            s = s + v[i] * v[i] * v[i + 1];
        }
        s
    });
    for i in 1..n - 1 {
        let exact = 2.0 * x[i] * x[i + 1] + x[i - 1] * x[i - 1];
        assert!((g[i] - exact).abs() < 1e-12);
    }
}

#[test]
fn tape_basics_and_constants() {
    let tape = Tape::<f64>::new();
    assert!(tape.is_empty());
    let x = tape.var(3.0);
    let c = tape.constant(4.0);
    let y = x * c + x.powf(2.0) / 2.0 - (-x);
    assert_eq!(y.value(), 12.0 + 4.5 + 3.0);
    let adj = tape.adjoints(&y);
    assert!((adj[x.index()] - (4.0 + 3.0 + 1.0)).abs() < 1e-14);
    assert_eq!(adj[c.index()], 3.0);
    assert!(tape.len() > 3);
}
