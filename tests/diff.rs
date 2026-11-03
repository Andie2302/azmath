//! Tests für das numerische Differenzieren.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::diff::{
    backward_difference, central_difference, central_difference4, derivative_ridders,
    forward_difference, gradient, gradient_work_len, hessian, jacobian, jacobian_work_len,
    second_derivative,
};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} gegen {b}");
}

#[test]
fn simple_quotients_have_the_expected_accuracy() {
    let f = |x: f64| x.exp();
    let exact = 1.0f64.exp();
    close(forward_difference(f, 1.0, 1e-7), exact, 1e-6);
    close(backward_difference(f, 1.0, 1e-7), exact, 1e-6);
    close(central_difference(f, 1.0, 1e-5), exact, 1e-9);
    close(central_difference4(f, 1.0, 1e-3), exact, 1e-10);
    close(second_derivative(f, 1.0, 1e-4), exact, 1e-6);
    // Fehlerordnungen: Halbieren von h
    let e1 = (central_difference(f, 1.0, 0.1) - exact).abs();
    let e2 = (central_difference(f, 1.0, 0.05) - exact).abs();
    assert!((e1 / e2 - 4.0).abs() < 0.2, "zentral: Faktor {}", e1 / e2);
    let g1 = (central_difference4(f, 1.0, 0.2) - exact).abs();
    let g2 = (central_difference4(f, 1.0, 0.1) - exact).abs();
    assert!(
        (g1 / g2 - 16.0).abs() < 1.5,
        "vierte Ordnung: Faktor {}",
        g1 / g2
    );
}

#[test]
fn ridders_gives_many_digits_and_an_error_estimate() {
    for (name, f, df, x) in [
        (
            "sin",
            (|x: f64| x.sin()) as fn(f64) -> f64,
            (|x: f64| x.cos()) as fn(f64) -> f64,
            0.7,
        ),
        ("exp", |x: f64| x.exp(), |x: f64| x.exp(), 1.3),
        ("ln", |x: f64| x.ln(), |x: f64| 1.0 / x, 2.0),
        ("x^5", |x: f64| x.powi(5), |x: f64| 5.0 * x.powi(4), 1.5),
    ] {
        let (d, err) = derivative_ridders(f, x, 0.1);
        assert!((d - df(x)).abs() < 1e-10, "{name}: {d} gegen {}", df(x));
        assert!(err < 1e-8, "{name}: Fehlerschätzung {err}");
        assert!(
            (d - df(x)).abs() <= 10.0 * err + 1e-14,
            "{name}: Schätzung unterschätzt den Fehler"
        );
    }
}

#[test]
fn gradient_of_a_function_of_several_variables() {
    let f = |x: &[f64]| x[0] * x[0] * x[1] + (x[1] * x[2]).sin() + x[2].exp();
    let x = [1.2, -0.7, 0.4];
    let mut g = [0.0; 3];
    let mut work = [0.0; gradient_work_len(3)];
    gradient(f, &x, &mut g, &mut work);
    close(g[0], 2.0 * x[0] * x[1], 1e-8);
    close(g[1], x[0] * x[0] + (x[1] * x[2]).cos() * x[2], 1e-8);
    close(g[2], (x[1] * x[2]).cos() * x[1] + x[2].exp(), 1e-8);
}

#[test]
fn hessian_is_symmetric_and_accurate() {
    // f = x²y + y³ + sin(xy)
    let f = |x: &[f64]| x[0] * x[0] * x[1] + x[1].powi(3) + (x[0] * x[1]).sin();
    let x = [0.6, -0.9];
    let mut h = [0.0; 4];
    let mut work = [0.0; gradient_work_len(2)];
    hessian(f, &x, &mut h, &mut work);
    let (xx, yy) = (x[0], x[1]);
    close(h[0], 2.0 * yy - yy * yy * (xx * yy).sin(), 1e-4);
    close(h[3], 6.0 * yy - xx * xx * (xx * yy).sin(), 1e-4);
    let mixed = 2.0 * xx + (xx * yy).cos() - xx * yy * (xx * yy).sin();
    close(h[1], mixed, 1e-4);
    assert_eq!(h[1], h[2]);
}

#[test]
fn jacobian_of_a_vector_function() {
    let f = |x: &[f64], out: &mut [f64]| {
        out[0] = x[0] * x[1];
        out[1] = x[0].sin() + x[1] * x[1];
        out[2] = (x[0] + x[1]).exp();
    };
    let x = [0.5, 1.5];
    let mut j = [0.0; 6];
    let mut work = [0.0; jacobian_work_len(3, 2)];
    jacobian(f, &x, 3, &mut j, &mut work);
    close(j[0], 1.5, 1e-8);
    close(j[1], 0.5, 1e-8);
    close(j[2], 0.5f64.cos(), 1e-8);
    close(j[3], 3.0, 1e-8);
    close(j[4], 2.0f64.exp(), 1e-7);
    close(j[5], 2.0f64.exp(), 1e-7);
}

#[test]
fn central_jacobian_beats_the_forward_difference_one() {
    use azmath::optimize::{numerical_jacobian, numerical_jacobian_work_len};
    let f = |x: &[f64], out: &mut [f64]| out[0] = x[0].exp();
    let x = [1.0];
    let mut f0 = [0.0];
    f(&x, &mut f0);
    let (mut jf, mut jc) = ([0.0], [0.0]);
    let mut w1 = [0.0; numerical_jacobian_work_len(1, 1)];
    numerical_jacobian(f, &x, &f0, &mut jf, &mut w1);
    let mut w2 = [0.0; jacobian_work_len(1, 1)];
    jacobian(f, &x, 1, &mut jc, &mut w2);
    let exact = 1.0f64.exp();
    assert!((jc[0] - exact).abs() < (jf[0] - exact).abs());
}

#[test]
fn works_for_f32() {
    let d = central_difference(|x: f32| x.sin(), 1.0, 1e-2);
    assert!((d - 1.0f32.cos()).abs() < 1e-3);
    let (r, _) = derivative_ridders(|x: f32| x.exp(), 0.5, 0.1);
    assert!((r - 0.5f32.exp()).abs() < 1e-3);
}
