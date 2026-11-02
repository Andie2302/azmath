//! Tests für Autodiff: `Jet` als `Real`, `VectorFn`, Jacobi- und Hessematrizen.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]
#![allow(clippy::needless_range_loop)]

use azmath::Real;
use azmath::autodiff::{Jet, VectorFn, ad_scratch_len, hessian_ad, jacobian_ad};
use azmath::optimize::{
    OptimizeOptions, levenberg_marquardt, levenberg_marquardt_work_len, newton_system,
    newton_work_len, numerical_jacobian, numerical_jacobian_work_len,
};
use azmath::vector_fn;
use std::cell::Cell;

type J1 = Jet<f64, 1>;

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "erwartet {b}, erhalten {a} (Δ = {:e})",
        (a - b).abs()
    );
}

// ---------------------------------------------------------------------------
// 1. Jet erfüllt die Real-Oberfläche – jede Funktion gegen zentrale Differenzen geprüft
// ---------------------------------------------------------------------------

/// Wertet `f` mit f64 und mit einem Jet aus und vergleicht die Ableitung mit einer zentralen Differenz.
macro_rules! check_unary {
    ($name:expr, $x:expr, |$s:ident| $body:expr) => {{
        fn f<S: Real>($s: S) -> S {
            $body
        }
        let x: f64 = $x;
        let jet = f(J1::variable(x, 0));
        let h = 1e-6;
        let fd = (f(x + h) - f(x - h)) / (2.0 * h);
        assert!(
            (jet.v - f(x)).abs() <= 1e-14 * f(x).abs().max(1.0),
            "{}: Wert",
            $name
        );
        assert!(
            (jet.d[0] - fd).abs() <= 1e-6 * fd.abs().max(1.0),
            "{}: Ableitung {} statt {}",
            $name,
            jet.d[0],
            fd
        );
    }};
}

#[test]
fn jet_derivatives_match_central_differences_for_the_whole_real_surface() {
    check_unary!("sqrt", 2.3, |x| x.sqrt());
    check_unary!("sin", 0.7, |x| x.sin());
    check_unary!("cos", 0.7, |x| x.cos());
    check_unary!("tan", 0.7, |x| x.tan());
    check_unary!("asin", 0.4, |x| x.asin());
    check_unary!("acos", 0.4, |x| x.acos());
    check_unary!("atan", 1.3, |x| x.atan());
    check_unary!("exp", 0.9, |x| x.exp());
    check_unary!("ln", 1.9, |x| x.ln());
    check_unary!("log10", 1.9, |x| x.log10());
    check_unary!("log2", 1.9, |x| x.log2());
    check_unary!("powi", 1.7, |x| x.powi(5));
    check_unary!("powi negativ", 1.7, |x| x.powi(-3));
    check_unary!("powf (x^2.5)", 1.7, |x| x.powf(S_from(2.5)));
    check_unary!("powf (2^x)", 1.7, |x| S_from::<S>(2.0).powf(x));
    check_unary!("powf (x^x)", 1.7, |x| x.powf(x));
    check_unary!("abs", -1.4, |x| x.abs());
    check_unary!("hypot", 0.8, |x| x.hypot(S_from(1.5)));
    check_unary!("atan2 (y)", 0.8, |x| x.atan2(S_from(1.5)));
    check_unary!("atan2 (x)", 0.8, |x| S_from::<S>(1.5).atan2(x));
    check_unary!("mul_add", 0.8, |x| x.mul_add(x, S_from(1.0)));
    // aus Standardimplementierungen des Traits (ohne eigenen Code für Jet)
    check_unary!("sinh", 0.8, |x| x.sinh());
    check_unary!("cosh", 0.8, |x| x.cosh());
    check_unary!("tanh", 0.8, |x| x.tanh());
    check_unary!("cbrt", 2.5, |x| x.cbrt());
    check_unary!("recip", 2.5, |x| x.recip());
    check_unary!("to_radians", 33.0, |x| x.to_radians());
    check_unary!("zusammengesetzt", 0.6, |x| (x.sin() * x.exp() + x.powi(2))
        .sqrt()
        / (S_from::<S>(1.0) + x.cos().abs()));
    // stückweise konstant: Ableitung 0
    for (name, jet) in [
        ("floor", J1::variable(2.5, 0).floor()),
        ("ceil", J1::variable(2.5, 0).ceil()),
        ("round", J1::variable(2.5, 0).round()),
        ("trunc", J1::variable(2.5, 0).trunc()),
        ("signum", J1::variable(2.5, 0).signum()),
    ] {
        assert_eq!(jet.d[0], 0.0, "{name}");
    }
}

#[allow(non_snake_case)]
fn S_from<S: Real>(v: f64) -> S {
    S::from_f64(v)
}

#[test]
fn min_max_follow_the_selected_branch() {
    let a = J1::variable(1.0, 0);
    let b = J1::constant(2.0);
    assert!(a.min(b).same(&a) && a.max(b).same(&b));
    assert_eq!(J1::constant(f64::NAN).min(a).v, 1.0);
}

#[test]
fn ring_properties_and_checked_recip() {
    // Nullteiler: (0, d)·(0, d') = (0, 0)
    let eps1 = Jet::<f64, 2> {
        v: 0.0,
        d: [1.0, 0.0],
    };
    let eps2 = Jet::<f64, 2> {
        v: 0.0,
        d: [0.0, 1.0],
    };
    assert!((eps1 * eps2).same(&Jet::constant(0.0)));
    assert!((eps1 * eps1).same(&Jet::constant(0.0)));
    // geprüfter Kehrwert
    assert!(eps1.checked_recip().is_none());
    let three = Jet::<f64, 2> {
        v: 3.0,
        d: [1.0, 2.0],
    };
    let r = three.checked_recip().unwrap();
    close(r.v, 1.0 / 3.0, 1e-15);
    close(r.d[0], -1.0 / 9.0, 1e-15);
    close(r.d[1], -2.0 / 9.0, 1e-15);
    assert!((three * r).v == 1.0);
    // ungeprüfte Division durch einen Nullteiler verhält sich wie bei Fließkommazahlen
    let q = J1::constant(1.0) / J1::variable(0.0, 0);
    assert!(q.v.is_infinite());
    assert!(Real::is_infinite(q) || Real::is_nan(q));
    // Gleichheit/Ordnung nur über den Wert, `same` vergleicht alles
    assert_eq!(J1::variable(1.0, 0), J1::constant(1.0));
    assert!(!J1::variable(1.0, 0).same(&J1::constant(1.0)));
    // nicht endliche Ableitungen werden von is_nan erkannt (Löser prüfen darüber ihre Eingaben)
    let bad = Jet::<f64, 1> {
        v: 1.0,
        d: [f64::NAN],
    };
    assert!(Real::is_nan(bad));
}

// ---------------------------------------------------------------------------
// 2. Funktionsschnittstelle + Adapter: bestehende Löser bekommen exakte Ableitungen
// ---------------------------------------------------------------------------

// Rosenbrock als Residuen: r = (10 (y − x²), 1 − x)
vector_fn!(Rosenbrock, |x, out| {
    let c = |v: f64| S::from_f64(v);
    out[0] = c(10.0) * (x[1] - x[0] * x[0]);
    out[1] = c(1.0) - x[0];
});

fn rosenbrock_jac(x: &[f64]) -> [f64; 4] {
    [-20.0 * x[0], 10.0, -1.0, 0.0]
}

#[test]
fn ad_jacobian_is_exact_where_finite_differences_are_not() {
    let x = [-1.2, 1.0];
    let exact = rosenbrock_jac(&x);

    let mut scratch = [Jet::<f64, 2>::constant(0.0); ad_scratch_len(2, 2)];
    let mut ad = [0.0; 4];
    jacobian_ad::<f64, 2, _>(&Rosenbrock, &x, 2, &mut ad, &mut scratch);
    let ad_err = ad
        .iter()
        .zip(&exact)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    let mut r0 = [0.0; 2];
    Rosenbrock.eval::<f64>(&x, &mut r0);
    let mut fd = [0.0; 4];
    let mut work = [0.0; numerical_jacobian_work_len(2, 2)];
    numerical_jacobian(
        |p: &[f64], r: &mut [f64]| Rosenbrock.eval::<f64>(p, r),
        &x,
        &r0,
        &mut fd,
        &mut work,
    );
    let fd_err = fd
        .iter()
        .zip(&exact)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);

    println!("Jacobi-Fehler: Autodiff {ad_err:e}, Differenzen {fd_err:e}");
    assert!(ad_err < 1e-14);
    assert!(fd_err > 1e-9);
    assert!(fd_err > 1e4 * ad_err.max(1e-16));
}

#[test]
fn strip_mining_handles_runtime_sizes_with_few_evaluations() {
    // f_i(x) = x_i² + x_{i−1}·x_{i+1} (zyklisch), n zur Laufzeit
    struct Cyclic {
        calls: Cell<usize>,
    }
    impl VectorFn for Cyclic {
        fn eval<S: Real>(&self, x: &[S], out: &mut [S]) {
            self.calls.set(self.calls.get() + 1);
            let n = x.len();
            for i in 0..n {
                out[i] = x[i] * x[i] + x[(i + n - 1) % n] * x[(i + 1) % n];
            }
        }
    }
    let n = 10;
    let x: Vec<f64> = (0..n).map(|i| 0.5 + 0.1 * i as f64).collect();
    let analytic = |i: usize, j: usize| -> f64 {
        let mut v = 0.0;
        if j == i {
            v += 2.0 * x[i];
        }
        if j == (i + n - 1) % n {
            v += x[(i + 1) % n];
        }
        if j == (i + 1) % n {
            v += x[(i + n - 1) % n];
        }
        v
    };
    for (k_label, expected_calls) in [(1usize, 10usize), (3, 4), (4, 3), (10, 1)] {
        let f = Cyclic {
            calls: Cell::new(0),
        };
        let mut jac = vec![0.0; n * n];
        match k_label {
            1 => jacobian_ad::<f64, 1, _>(
                &f,
                &x,
                n,
                &mut jac,
                &mut vec![Jet::constant(0.0); ad_scratch_len(n, n)],
            ),
            3 => jacobian_ad::<f64, 3, _>(
                &f,
                &x,
                n,
                &mut jac,
                &mut vec![Jet::constant(0.0); ad_scratch_len(n, n)],
            ),
            4 => jacobian_ad::<f64, 4, _>(
                &f,
                &x,
                n,
                &mut jac,
                &mut vec![Jet::constant(0.0); ad_scratch_len(n, n)],
            ),
            _ => jacobian_ad::<f64, 10, _>(
                &f,
                &x,
                n,
                &mut jac,
                &mut vec![Jet::constant(0.0); ad_scratch_len(n, n)],
            ),
        }
        assert_eq!(
            f.calls.get(),
            expected_calls,
            "K = {k_label}: ⌈n/K⌉ Auswertungen"
        );
        for i in 0..n {
            for j in 0..n {
                close(jac[i * n + j], analytic(i, j), 1e-14);
            }
        }
    }
}

#[test]
fn ad_plugs_into_the_unchanged_levenberg_marquardt() {
    // Modell y = a·exp(b·t) an exakten Daten; die Funktion einmal generisch geschrieben
    struct ExpFit {
        t: [f64; 8],
        y: [f64; 8],
    }
    impl VectorFn for ExpFit {
        fn eval<S: Real>(&self, p: &[S], out: &mut [S]) {
            for i in 0..8 {
                out[i] = p[0] * (p[1] * S::from_f64(self.t[i])).exp() - S::from_f64(self.y[i]);
            }
        }
    }
    let t = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];
    let f = ExpFit {
        t,
        y: t.map(|t| 2.5 * (-1.3f64 * t).exp()),
    };

    // Die beiden Closures für den bestehenden Löser stammen aus derselben Funktion:
    let mut scratch = [Jet::<f64, 2>::constant(0.0); ad_scratch_len(8, 2)];
    let residual = |p: &[f64], r: &mut [f64]| f.eval::<f64>(p, r);
    let jacobian = |p: &[f64], j: &mut [f64]| jacobian_ad::<f64, 2, _>(&f, p, 8, j, &mut scratch);

    let mut p = [1.0, -0.5];
    let mut work = [0.0; levenberg_marquardt_work_len(8, 2)];
    let r = levenberg_marquardt(
        residual,
        jacobian,
        &mut p,
        8,
        OptimizeOptions::default(),
        &mut work,
    )
    .unwrap();
    close(p[0], 2.5, 1e-9);
    close(p[1], -1.3, 1e-9);
    println!(
        "LM mit Autodiff-Jacobi: {} Iterationen, Rest {:e}",
        r.iterations, r.residual_norm
    );
}

#[test]
fn ad_plugs_into_newton_system() {
    vector_fn!(CircleLine, |v, out| {
        out[0] = v[0] * v[0] + v[1] * v[1] - S::from_f64(4.0);
        out[1] = v[1] - v[0];
    });
    let mut scratch = [Jet::<f64, 2>::constant(0.0); ad_scratch_len(2, 2)];
    let mut x = [1.0, 0.5];
    let mut work = [0.0; newton_work_len(2)];
    let r = newton_system(
        |v: &[f64], o: &mut [f64]| CircleLine.eval::<f64>(v, o),
        |v: &[f64], j: &mut [f64]| jacobian_ad::<f64, 2, _>(&CircleLine, v, 2, j, &mut scratch),
        &mut x,
        OptimizeOptions::default(),
        &mut work,
    )
    .unwrap();
    close(x[0], 2f64.sqrt(), 1e-13);
    close(x[1], 2f64.sqrt(), 1e-13);
    println!("Newton mit Autodiff-Jacobi: {} Iterationen", r.iterations);
}

#[test]
fn nested_jets_give_the_hessian() {
    // f(x, y) = (1 − x)² + 100 (y − x²)²
    vector_fn!(RosenbrockScalar, |x, out| {
        let c = |v: f64| S::from_f64(v);
        let a = c(1.0) - x[0];
        let b = x[1] - x[0] * x[0];
        out[0] = a * a + c(100.0) * b * b;
    });
    let x = [-1.2, 1.0];
    let h = hessian_ad::<f64, 2, _>(&RosenbrockScalar, &x);
    close(h[0][0], 2.0 - 400.0 * x[1] + 1200.0 * x[0] * x[0], 1e-10);
    close(h[0][1], -400.0 * x[0], 1e-10);
    close(h[1][0], -400.0 * x[0], 1e-10);
    close(h[1][1], 200.0, 1e-10);
}

// ---------------------------------------------------------------------------
// 4. Wo Autodiff wirklich etwas ändert: zweite Ableitungen
// ---------------------------------------------------------------------------

#[test]
fn second_derivatives_by_finite_differences_are_much_worse_than_nested_jets() {
    vector_fn!(RosenbrockScalar2, |x, out| {
        let c = |v: f64| S::from_f64(v);
        let a = c(1.0) - x[0];
        let b = x[1] - x[0] * x[0];
        out[0] = a * a + c(100.0) * b * b;
    });
    let f = |x: f64, y: f64| {
        let mut o = [0.0];
        RosenbrockScalar2.eval::<f64>(&[x, y], &mut o);
        o[0]
    };
    let x = [-1.2, 1.0];
    let exact = [
        [2.0 - 400.0 * x[1] + 1200.0 * x[0] * x[0], -400.0 * x[0]],
        [-400.0 * x[0], 200.0],
    ];
    let ad = hessian_ad::<f64, 2, _>(&RosenbrockScalar2, &x);
    let ad_err = (0..2)
        .flat_map(|i| (0..2).map(move |j| (i, j)))
        .map(|(i, j)| (ad[i][j] - exact[i][j]).abs())
        .fold(0.0f64, f64::max);

    // zentrale Differenzen für die Hessematrix mit der besten Schrittweite ~ ε^(1/4)
    let h = f64::EPSILON.powf(0.25);
    let fd = [
        [
            (f(x[0] + h, x[1]) - 2.0 * f(x[0], x[1]) + f(x[0] - h, x[1])) / (h * h),
            (f(x[0] + h, x[1] + h) - f(x[0] + h, x[1] - h) - f(x[0] - h, x[1] + h)
                + f(x[0] - h, x[1] - h))
                / (4.0 * h * h),
        ],
        [
            (f(x[0] + h, x[1] + h) - f(x[0] + h, x[1] - h) - f(x[0] - h, x[1] + h)
                + f(x[0] - h, x[1] - h))
                / (4.0 * h * h),
            (f(x[0], x[1] + h) - 2.0 * f(x[0], x[1]) + f(x[0], x[1] - h)) / (h * h),
        ],
    ];
    let fd_err = (0..2)
        .flat_map(|i| (0..2).map(move |j| (i, j)))
        .map(|(i, j)| (fd[i][j] - exact[i][j]).abs())
        .fold(0.0f64, f64::max);
    println!("Hesse-Fehler: geschachtelte Jets {ad_err:e}, zentrale Differenzen {fd_err:e}");
    assert!(ad_err < 1e-10);
    assert!(fd_err > 1e-7);
    assert!(fd_err > 1e3 * ad_err.max(1e-16));
}
