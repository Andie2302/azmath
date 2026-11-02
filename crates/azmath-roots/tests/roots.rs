//! Tests für die Nullstellensuche: Bisektion, Brent, Newton und Sekante.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath_core::Real;
use azmath_roots::{RootError, RootOptions, bisection, brent, newton, secant};

const SQRT_2: f64 = core::f64::consts::SQRT_2;
const DOTTIE: f64 = 0.739_085_133_215_160_7; // Lösung von cos(x) = x

fn opts() -> RootOptions<f64> {
    RootOptions::default()
}

fn sq2(x: f64) -> f64 {
    x * x - 2.0
}

fn cos_minus_x(x: f64) -> f64 {
    Real::cos(x) - x
}

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "erwartet {b}, erhalten {a}");
}

#[test]
fn all_methods_find_sqrt_two() {
    let b = bisection(sq2, 0.0, 2.0, opts()).unwrap();
    let br = brent(sq2, 0.0, 2.0, opts()).unwrap();
    let n = newton(sq2, |x| 2.0 * x, 1.0, opts()).unwrap();
    let s = secant(sq2, 1.0, 2.0, opts()).unwrap();
    for r in [b, br, n, s] {
        close(r.root, SQRT_2, 1e-14);
        assert!(r.f_value.abs() < 1e-13);
    }
}

#[test]
fn cos_x_equals_x() {
    let br = brent(cos_minus_x, 0.0, 1.0, opts()).unwrap();
    let b = bisection(cos_minus_x, 0.0, 1.0, opts()).unwrap();
    let n = newton(cos_minus_x, |x| -Real::sin(x) - 1.0, 0.5, opts()).unwrap();
    let s = secant(cos_minus_x, 0.0, 1.0, opts()).unwrap();
    for r in [br, b, n, s] {
        close(r.root, DOTTIE, 1e-14);
    }
}

#[test]
fn brent_is_much_faster_than_bisection() {
    let br = brent(cos_minus_x, 0.0, 1.0, opts()).unwrap();
    let b = bisection(cos_minus_x, 0.0, 1.0, opts()).unwrap();
    assert!(
        br.iterations * 3 <= b.iterations,
        "Brent {} vs. Bisektion {}",
        br.iterations,
        b.iterations
    );
}

#[test]
fn newton_and_secant_converge_fast() {
    let n = newton(sq2, |x| 2.0 * x, 1.0, opts()).unwrap();
    assert!(n.iterations <= 8, "{} Iterationen", n.iterations);
    let s = secant(sq2, 1.0, 2.0, opts()).unwrap();
    assert!(s.iterations <= 10, "{} Iterationen", s.iterations);
}

#[test]
fn polynomial_and_exponential() {
    // x³ − 2x − 5 = 0 (Wallis' klassisches Beispiel)
    let r = brent(|x| x * x * x - 2.0 * x - 5.0, 2.0, 3.0, opts()).unwrap();
    close(r.root, 2.094_551_481_542_326_5, 1e-13);
    // eˣ = 3
    let r = brent(|x| Real::exp(x) - 3.0, 0.0, 2.0, opts()).unwrap();
    close(r.root, Real::ln(3.0f64), 1e-14);
    // Gleichung mit mehreren Nullstellen: sin(x) auf [3, 4] hat genau π
    let r = bisection(Real::sin, 3.0, 4.0, opts()).unwrap();
    close(r.root, core::f64::consts::PI, 1e-14);
}

#[test]
fn interval_order_does_not_matter() {
    let a = brent(sq2, 0.0, 2.0, opts()).unwrap();
    let b = brent(sq2, 2.0, 0.0, opts()).unwrap();
    close(a.root, b.root, 1e-14);
    let c = bisection(sq2, 2.0, 0.0, opts()).unwrap();
    close(c.root, SQRT_2, 1e-14);
}

#[test]
fn root_at_interval_end_returns_immediately() {
    let r = bisection(|x: f64| x - 1.0, 1.0, 5.0, opts()).unwrap();
    assert_eq!((r.root, r.iterations), (1.0, 0));
    let r = brent(|x: f64| x - 5.0, 1.0, 5.0, opts()).unwrap();
    assert_eq!((r.root, r.iterations), (5.0, 0));
    let r = secant(|x: f64| x - 3.0, 3.0, 4.0, opts()).unwrap();
    assert_eq!((r.root, r.iterations), (3.0, 0));
}

#[test]
fn exact_start_value_needs_no_iteration() {
    let r = newton(|x: f64| x * x - 4.0, |x| 2.0 * x, 2.0, opts()).unwrap();
    assert_eq!((r.root, r.iterations), (2.0, 0));
}

#[test]
fn not_bracketed() {
    assert_eq!(
        bisection(|x: f64| x * x + 1.0, -1.0, 1.0, opts()),
        Err(RootError::NotBracketed)
    );
    assert_eq!(
        brent(|x: f64| x * x + 1.0, -1.0, 1.0, opts()),
        Err(RootError::NotBracketed)
    );
}

#[test]
fn flat_regions_and_zero_derivative() {
    // f' = 0 im Startpunkt
    let r = newton(|x: f64| x * x + 1.0, |x| 2.0 * x, 0.0, opts());
    assert_eq!(r, Err(RootError::ZeroDerivative));
    // Sekante: f(x0) = f(x1)
    let r = secant(|_x: f64| 1.0, 0.0, 1.0, opts());
    assert_eq!(r, Err(RootError::ZeroDerivative));
}

#[test]
fn non_finite_values_are_reported() {
    assert_eq!(
        bisection(|_: f64| f64::NAN, 0.0, 1.0, opts()),
        Err(RootError::NotFinite)
    );
    assert_eq!(
        brent(|_: f64| f64::NAN, 0.0, 1.0, opts()),
        Err(RootError::NotFinite)
    );
    assert_eq!(brent(sq2, f64::NAN, 1.0, opts()), Err(RootError::NotFinite));
    assert_eq!(
        newton(sq2, |x| 2.0 * x, f64::INFINITY, opts()),
        Err(RootError::NotFinite)
    );
    assert_eq!(
        secant(|_: f64| f64::NAN, 0.0, 1.0, opts()),
        Err(RootError::NotFinite)
    );
}

#[test]
fn no_convergence_with_too_few_iterations() {
    let few = RootOptions::new(1e-15, 3);
    assert_eq!(bisection(sq2, 0.0, 2.0, few), Err(RootError::NoConvergence));
    assert_eq!(
        brent(cos_minus_x, 0.0, 1.0, RootOptions::new(1e-15, 2)),
        Err(RootError::NoConvergence)
    );
    assert_eq!(
        newton(sq2, |x| 2.0 * x, 100.0, RootOptions::new(1e-15, 2)),
        Err(RootError::NoConvergence)
    );
    assert_eq!(
        secant(sq2, 100.0, 101.0, RootOptions::new(1e-15, 2)),
        Err(RootError::NoConvergence)
    );
}

#[test]
fn multiple_root_is_found_with_brent_and_bisection() {
    // (x − 1)³ hat eine dreifache Nullstelle, Newton/Sekante wären hier langsam
    let f = |x: f64| (x - 1.0) * (x - 1.0) * (x - 1.0);
    close(brent(f, 0.0, 3.0, opts()).unwrap().root, 1.0, 1e-6);
    close(bisection(f, 0.0, 3.0, opts()).unwrap().root, 1.0, 1e-6);
    // Auch x⁹ hat eine neunfache Nullstelle bei 0
    close(
        brent(|x: f64| x.powi(9), -1.0, 1.5, opts()).unwrap().root,
        0.0,
        1e-9,
    );
}

#[test]
fn steep_function_and_large_magnitudes() {
    // x¹¹ − 1 (sehr flach nahe 0, steil bei 1)
    let r = brent(|x: f64| x.powi(11) - 1.0, 0.0, 3.0, opts()).unwrap();
    close(r.root, 1.0, 1e-12);
    // Nullstelle weit weg: relative Toleranz muss greifen
    let r = brent(|x: f64| x - 1.0e8, 0.0, 1.0e9, opts()).unwrap();
    close(r.root, 1.0e8, 1.0e8 * 1e-14);
    // Kepler-Gleichung E − e·sin(E) = M
    let (e, m) = (0.6, 1.2);
    let r = brent(|x: f64| x - e * Real::sin(x) - m, 0.0, 3.2, opts()).unwrap();
    close(r.root - e * Real::sin(r.root), m, 1e-13);
}

#[test]
fn pole_is_reported_through_large_f_value() {
    // 1/x wechselt bei 0 das Vorzeichen, hat dort aber eine Polstelle statt einer Nullstelle
    let r = bisection(|x: f64| 1.0 / x, -1.0, 2.0, opts()).unwrap();
    assert!(r.f_value.abs() > 1e6, "f_value = {}", r.f_value);
}

#[test]
fn stateful_closures_work() {
    let mut calls = 0usize;
    let r = brent(
        |x: f64| {
            calls += 1;
            cos_minus_x(x)
        },
        0.0,
        1.0,
        opts(),
    )
    .unwrap();
    // Eine Auswertung pro Iteration plus die beiden Randwerte
    assert!(
        calls <= r.iterations + 2,
        "{calls} Aufrufe, {} Iterationen",
        r.iterations
    );
}

#[test]
fn f32_with_default_options() {
    let o = RootOptions::<f32>::default();
    let f = |x: f32| x * x - 2.0;
    let tol = 5e-6f32;
    let b = bisection(f, 0.0, 2.0, o).unwrap();
    let br = brent(f, 0.0, 2.0, o).unwrap();
    let n = newton(f, |x| 2.0 * x, 1.0, o).unwrap();
    let s = secant(f, 1.0, 2.0, o).unwrap();
    for r in [b, br, n, s] {
        assert!(
            (r.root - core::f32::consts::SQRT_2).abs() < tol,
            "{}",
            r.root
        );
    }
}

#[test]
fn default_options() {
    let o = RootOptions::<f64>::default();
    assert_eq!(o.max_iter, 500);
    assert!(o.tol > 0.0 && o.tol < 1e-13);
}

#[test]
fn error_display() {
    extern crate std;
    use std::string::ToString;
    for e in [
        RootError::NotBracketed,
        RootError::NoConvergence,
        RootError::NotFinite,
        RootError::ZeroDerivative,
    ] {
        assert!(!e.to_string().is_empty());
    }
}
