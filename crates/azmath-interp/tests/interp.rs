//! Tests für Interpolation: linear, kubische Splines, PCHIP und Lagrange.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath_core::Real;
use azmath_interp::{
    BoundaryCondition, CubicSpline, InterpError, Pchip, cubic_spline_work_len, lagrange, linear,
    locate,
};
use core::f64::consts::PI;

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "erwartet {b}, erhalten {a} (Δ = {:e})",
        (a - b).abs()
    );
}

/// Das kubische Polynom p(x) = 1 − 2x + 0.5x² + 0.25x³ samt Ableitungen und Integral.
fn p(x: f64) -> f64 {
    1.0 - 2.0 * x + 0.5 * x * x + 0.25 * x * x * x
}
fn dp(x: f64) -> f64 {
    -2.0 + x + 0.75 * x * x
}
fn ddp(x: f64) -> f64 {
    1.0 + 1.5 * x
}
fn ip(x: f64) -> f64 {
    x - x * x + x * x * x / 6.0 + x * x * x * x / 16.0
}

/// Legt die Puffer im aufrufenden Scope an und bindet den Spline an den Namen `$s`.
macro_rules! spline {
    ($s:ident, $x:expr, $y:expr, $bc:expr) => {
        let mut m = vec![0.0f64; $x.len()];
        let mut work = vec![0.0f64; cubic_spline_work_len($x.len())];
        let $s = CubicSpline::new(&$x, &$y, $bc, &mut m, &mut work).unwrap();
    };
}

const X6: [f64; 6] = [-1.0, -0.2, 0.7, 1.3, 2.4, 3.0];

fn y6() -> [f64; 6] {
    X6.map(p)
}

// ---------------------------------------------------------------------------
// locate / linear
// ---------------------------------------------------------------------------

#[test]
fn locate_finds_the_right_interval() {
    let x = [0.0, 1.0, 2.0, 5.0];
    assert_eq!(locate(&x, 0.0), 0);
    assert_eq!(locate(&x, 0.5), 0);
    assert_eq!(locate(&x, 1.0), 1);
    assert_eq!(locate(&x, 1.999), 1);
    assert_eq!(locate(&x, 2.0), 2);
    assert_eq!(locate(&x, 5.0), 2, "rechter Rand gehört zum letzten Stück");
    assert_eq!(locate(&x, -10.0), 0, "links außerhalb: erstes Stück");
    assert_eq!(locate(&x, 99.0), 2, "rechts außerhalb: letztes Stück");
    assert_eq!(locate(&[0.0, 1.0], 0.5), 0);
}

#[test]
fn linear_interpolation() {
    let (x, y) = ([0.0, 1.0, 3.0], [0.0, 2.0, 0.0]);
    assert_eq!(linear(&x, &y, 0.0).unwrap(), 0.0);
    assert_eq!(linear(&x, &y, 0.5).unwrap(), 1.0);
    assert_eq!(linear(&x, &y, 1.0).unwrap(), 2.0);
    assert_eq!(linear(&x, &y, 2.0).unwrap(), 1.0);
    assert_eq!(linear(&x, &y, 3.0).unwrap(), 0.0);
    // Extrapolation setzt die Geraden fort
    assert_eq!(linear(&x, &y, -1.0).unwrap(), -2.0);
    assert_eq!(linear(&x, &y, 4.0).unwrap(), -1.0);
}

#[test]
fn linear_errors() {
    assert_eq!(
        linear(&[0.0, 1.0], &[0.0], 0.5),
        Err(InterpError::InvalidInput)
    );
    assert_eq!(linear(&[0.0], &[0.0], 0.5), Err(InterpError::InvalidInput));
    assert_eq!(
        linear(&[1.0, 0.0], &[0.0, 1.0], 0.5),
        Err(InterpError::NotIncreasing)
    );
    assert_eq!(
        linear(&[0.0, 0.0], &[0.0, 1.0], 0.5),
        Err(InterpError::NotIncreasing)
    );
    assert_eq!(
        linear(&[0.0, f64::NAN], &[0.0, 1.0], 0.5),
        Err(InterpError::NotFinite)
    );
    assert_eq!(
        linear(&[0.0, 1.0], &[0.0, f64::INFINITY], 0.5),
        Err(InterpError::NotFinite)
    );
}

// ---------------------------------------------------------------------------
// Kubischer Spline
// ---------------------------------------------------------------------------

#[test]
fn spline_interpolates_the_nodes_for_every_boundary_condition() {
    let y = [0.3, -1.2, 2.5, 0.0, 1.1, 0.3]; // y[0] == y[5] für die periodische Variante
    let x = [0.0, 0.7, 1.1, 2.0, 3.5, 4.0];
    let conditions = [
        BoundaryCondition::Natural,
        BoundaryCondition::Clamped {
            left: 1.0,
            right: -2.0,
        },
        BoundaryCondition::NotAKnot,
        BoundaryCondition::Periodic,
    ];
    for bc in conditions {
        spline!(s, x, y, bc);
        for i in 0..6 {
            close(s.value(x[i]), y[i], 1e-12);
        }
    }
}

#[test]
fn not_a_knot_reproduces_cubic_polynomials_exactly() {
    let y = y6();
    spline!(s, X6, y, BoundaryCondition::NotAKnot);
    for k in 0..=40 {
        let xq = -1.0 + 4.0 * k as f64 / 40.0;
        close(s.value(xq), p(xq), 1e-12);
        close(s.derivative(xq), dp(xq), 1e-11);
        close(s.second_derivative(xq), ddp(xq), 1e-10);
    }
    close(s.integral(-1.0, 3.0), ip(3.0) - ip(-1.0), 1e-11);
    close(s.integral(0.1, 2.9), ip(2.9) - ip(0.1), 1e-11);
}

#[test]
fn clamped_with_exact_slopes_reproduces_cubic_polynomials() {
    let y = y6();
    spline!(
        s,
        X6,
        y,
        BoundaryCondition::Clamped {
            left: dp(-1.0),
            right: dp(3.0)
        }
    );
    for k in 0..=40 {
        let xq = -1.0 + 4.0 * k as f64 / 40.0;
        close(s.value(xq), p(xq), 1e-12);
        close(s.derivative(xq), dp(xq), 1e-11);
    }
    // Die Randsteigungen werden exakt eingehalten
    close(s.derivative(-1.0), dp(-1.0), 1e-12);
    close(s.derivative(3.0), dp(3.0), 1e-12);
}

#[test]
fn natural_spline_has_zero_curvature_at_the_ends() {
    let y = y6();
    spline!(s, X6, y, BoundaryCondition::Natural);
    close(s.second_derivative(-1.0), 0.0, 1e-12);
    close(s.second_derivative(3.0), 0.0, 1e-12);
    // und ist hier (p'' ≠ 0 am Rand) nicht exakt
    assert!((s.value(0.0) - p(0.0)).abs() > 1e-4);
    assert_eq!(s.second_derivatives()[0], 0.0);
}

#[test]
fn small_node_counts() {
    // n = 2: Gerade (natürlich/not-a-knot)
    let (x2, y2) = ([0.0, 2.0], [1.0, 5.0]);
    for bc in [BoundaryCondition::Natural, BoundaryCondition::NotAKnot] {
        spline!(s, x2, y2, bc);
        close(s.value(1.0), 3.0, 1e-14);
        close(s.derivative(0.3), 2.0, 1e-14);
    }
    // n = 2 eingespannt: kubisches Hermite-Polynom
    spline!(
        s,
        x2,
        y2,
        BoundaryCondition::Clamped {
            left: 0.0,
            right: 0.0
        }
    );
    close(s.value(1.0), 3.0, 1e-14);
    close(s.derivative(0.0), 0.0, 1e-14);
    // n = 3 not-a-knot: die Parabel durch die drei Punkte (y = x² + 1)
    let (x3, y3) = ([0.0, 1.0, 3.0], [1.0, 2.0, 10.0]);
    spline!(s, x3, y3, BoundaryCondition::NotAKnot);
    for xq in [0.0, 0.4, 1.0, 2.2, 3.0] {
        close(s.value(xq), xq * xq + 1.0, 1e-13);
    }
    // n = 3 natürlich funktioniert ebenfalls
    spline!(s, x3, y3, BoundaryCondition::Natural);
    close(s.value(1.0), 2.0, 1e-13);
}

#[test]
fn spline_is_twice_continuously_differentiable() {
    let x = [0.0, 0.8, 1.5, 2.1, 3.3, 4.0, 5.2];
    let y = [0.5, -0.3, 1.2, 0.7, -0.9, 0.4, 1.0];
    for bc in [BoundaryCondition::Natural, BoundaryCondition::NotAKnot] {
        spline!(s, x, y, bc);
        for &k in &x[1..x.len() - 1] {
            let e = 1e-9;
            close(s.value(k - e), s.value(k + e), 1e-7);
            close(s.derivative(k - e), s.derivative(k + e), 1e-7);
            close(s.second_derivative(k - e), s.second_derivative(k + e), 1e-7);
        }
    }
}

#[test]
fn convergence_order_for_a_smooth_function() {
    // Not-a-knot und eingespannt konvergieren mit h⁴: Halbieren der Schrittweite ≈ Faktor 16
    let max_error = |n: usize, clamped: bool| {
        let x: Vec<f64> = (0..n).map(|i| i as f64 * PI / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| Real::sin(v)).collect();
        let mut m = vec![0.0; n];
        let mut work = vec![0.0; cubic_spline_work_len(n)];
        let bc = if clamped {
            BoundaryCondition::Clamped {
                left: 1.0,
                right: -1.0,
            }
        } else {
            BoundaryCondition::NotAKnot
        };
        let s = CubicSpline::new(&x, &y, bc, &mut m, &mut work).unwrap();
        (0..4000)
            .map(|k| {
                let xq = PI * k as f64 / 3999.0;
                (s.value(xq) - Real::sin(xq)).abs()
            })
            .fold(0.0f64, f64::max)
    };
    for clamped in [false, true] {
        // asymptotisch (feine Gitter) ist das Verhältnis 16; bei groben Gittern ist not-a-knot
        // wegen des Randfehlers vorübergehend besser (Verhältnis ≈ 30)
        let ratio = max_error(81, clamped) / max_error(161, clamped);
        assert!(
            (ratio - 16.0).abs() < 1.0,
            "clamped = {clamped}: Verhältnis {ratio}"
        );
    }
}

#[test]
fn periodic_spline_closes_smoothly() {
    let n = 13;
    let x: Vec<f64> = (0..n)
        .map(|i| i as f64 * 2.0 * PI / (n - 1) as f64)
        .collect();
    let mut y: Vec<f64> = x.iter().map(|&v| Real::sin(v)).collect();
    y[n - 1] = y[0]; // 2π: sin ist dort numerisch ≈ −2.4e-16
    let mut m = vec![0.0; n];
    let mut work = vec![0.0; cubic_spline_work_len(n)];
    let s = CubicSpline::new(&x, &y, BoundaryCondition::Periodic, &mut m, &mut work).unwrap();
    let (lo, hi) = s.domain();
    close(s.value(lo), s.value(hi), 1e-13);
    close(s.derivative(lo), s.derivative(hi), 1e-11);
    close(s.second_derivative(lo), s.second_derivative(hi), 1e-9);
    // Genauigkeit
    for k in 0..200 {
        let xq = 2.0 * PI * k as f64 / 199.0;
        close(s.value(xq), Real::sin(xq), 2e-3);
    }
    // die Ableitung am Rand stimmt mit cos(0) = 1 überein
    close(s.derivative(0.0), 1.0, 5e-3);
}

#[test]
fn periodic_requirements() {
    let x = [0.0, 1.0, 2.0, 3.0];
    let mut m = [0.0; 4];
    let mut work = [0.0; cubic_spline_work_len(4)];
    assert!(matches!(
        CubicSpline::new(
            &x,
            &[0.0, 1.0, 2.0, 3.0],
            BoundaryCondition::Periodic,
            &mut m,
            &mut work
        ),
        Err(InterpError::NotPeriodic)
    ));
    let (x2, y2) = ([0.0, 1.0], [1.0, 1.0]);
    let mut m2 = [0.0; 2];
    let mut w2 = [0.0; cubic_spline_work_len(2)];
    assert!(matches!(
        CubicSpline::new(&x2, &y2, BoundaryCondition::Periodic, &mut m2, &mut w2),
        Err(InterpError::InvalidInput)
    ));
    // n = 3 ist erlaubt
    let (x3, y3) = ([0.0, 1.0, 2.0], [1.0, 3.0, 1.0]);
    let mut m3 = [0.0; 3];
    let mut w3 = [0.0; cubic_spline_work_len(3)];
    let s = CubicSpline::new(&x3, &y3, BoundaryCondition::Periodic, &mut m3, &mut w3).unwrap();
    close(s.derivative(0.0), s.derivative(2.0), 1e-12);
}

#[test]
fn integral_properties() {
    let y = y6();
    spline!(s, X6, y, BoundaryCondition::NotAKnot);
    // vertauschte Grenzen, leeres Intervall, Additivität
    close(s.integral(2.0, 0.0), -s.integral(0.0, 2.0), 1e-13);
    assert_eq!(s.integral(1.0, 1.0), 0.0);
    close(
        s.integral(-1.0, 0.5) + s.integral(0.5, 3.0),
        s.integral(-1.0, 3.0),
        1e-12,
    );
    // Ableitung des Integrals nach der oberen Grenze ist der Spline selbst
    for b in [0.3, 1.0, 2.5] {
        let e = 1e-5;
        let numeric = (s.integral(0.0, b + e) - s.integral(0.0, b - e)) / (2.0 * e);
        close(numeric, s.value(b), 1e-6);
    }
}

#[test]
fn accessors_and_helpers() {
    let y = y6();
    spline!(s, X6, y, BoundaryCondition::NotAKnot);
    assert_eq!(s.nodes(), &X6);
    assert_eq!(s.values(), &y);
    assert_eq!(s.second_derivatives().len(), 6);
    assert_eq!(s.domain(), (-1.0, 3.0));
    assert!(s.contains(-1.0) && s.contains(3.0) && s.contains(1.0));
    assert!(!s.contains(-1.01) && !s.contains(3.01) && !s.contains(f64::NAN));
    let xs = [0.0, 1.0, 2.0];
    let mut out = [0.0; 3];
    s.values_at(&xs, &mut out);
    for i in 0..3 {
        close(out[i], s.value(xs[i]), 1e-15);
    }
}

#[test]
fn extrapolation_continues_the_end_polynomial() {
    let y = y6();
    spline!(s, X6, y, BoundaryCondition::NotAKnot);
    // not-a-knot reproduziert das kubische Polynom auch außerhalb des Bereichs
    close(s.value(4.0), p(4.0), 1e-9);
    close(s.value(-2.0), p(-2.0), 1e-9);
    close(s.integral(3.0, 4.0), ip(4.0) - ip(3.0), 1e-9);
}

#[test]
fn spline_errors() {
    let x = [0.0, 1.0, 2.0];
    let y = [0.0, 1.0, 0.0];
    let mut m = [0.0; 3];
    let mut work = [0.0; cubic_spline_work_len(3)];
    let nat = BoundaryCondition::Natural;
    let mut short = [0.0; 2];
    assert!(matches!(
        CubicSpline::new(&x, &y, nat, &mut short, &mut work),
        Err(InterpError::InvalidInput)
    ));
    let mut short_work = [0.0; cubic_spline_work_len(3) - 1];
    assert!(matches!(
        CubicSpline::new(&x, &y, nat, &mut m, &mut short_work),
        Err(InterpError::InvalidInput)
    ));
    assert!(matches!(
        CubicSpline::new(&x, &[0.0, 1.0], nat, &mut m, &mut work),
        Err(InterpError::InvalidInput)
    ));
    assert!(matches!(
        CubicSpline::new(&[0.0, 2.0, 1.0], &y, nat, &mut m, &mut work),
        Err(InterpError::NotIncreasing)
    ));
    assert!(matches!(
        CubicSpline::new(&x, &[0.0, f64::NAN, 0.0], nat, &mut m, &mut work),
        Err(InterpError::NotFinite)
    ));
    assert!(matches!(
        CubicSpline::new(
            &x,
            &y,
            BoundaryCondition::Clamped {
                left: f64::NAN,
                right: 0.0
            },
            &mut m,
            &mut work
        ),
        Err(InterpError::NotFinite)
    ));
}

#[test]
fn spline_f32() {
    let x = [0.0f32, 1.0, 2.0, 3.0, 4.0];
    let y = x.map(|v| v * v);
    let mut m = [0.0f32; 5];
    let mut work = [0.0f32; cubic_spline_work_len(5)];
    let s = CubicSpline::new(&x, &y, BoundaryCondition::NotAKnot, &mut m, &mut work).unwrap();
    assert!((s.value(2.5) - 6.25).abs() < 1e-4);
}

// ---------------------------------------------------------------------------
// PCHIP
// ---------------------------------------------------------------------------

#[test]
fn pchip_interpolates_and_has_continuous_slope() {
    let x = [0.0, 0.8, 1.5, 2.1, 3.3, 4.0];
    let y = [0.5, -0.3, 1.2, 0.7, -0.9, 0.4];
    let mut d = [0.0; 6];
    let p = Pchip::new(&x, &y, &mut d).unwrap();
    for i in 0..6 {
        close(p.value(x[i]), y[i], 1e-13);
    }
    for &k in &x[1..5] {
        close(p.derivative(k - 1e-9), p.derivative(k + 1e-9), 1e-7);
    }
    assert_eq!(p.slopes().len(), 6);
}

#[test]
fn pchip_preserves_monotonicity_where_a_spline_overshoots() {
    // monotone Daten mit einer Stufe
    let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let mut d = [0.0; 6];
    let p = Pchip::new(&x, &y, &mut d).unwrap();
    let mut prev = -1.0;
    for k in 0..=500 {
        let v = p.value(k as f64 * 0.01);
        assert!((-1e-12..=1.0 + 1e-12).contains(&v), "Überschwinger {v}");
        assert!(v >= prev - 1e-12, "nicht monoton bei {k}");
        prev = v;
    }
    // Der kubische Spline überschwingt bei denselben Daten sichtbar
    spline!(s, x, y, BoundaryCondition::NotAKnot);
    let worst = (0..=500)
        .map(|k| s.value(k as f64 * 0.01))
        .fold(0.0f64, |m, v| m.max(-v).max(v - 1.0));
    assert!(worst > 0.01, "Spline-Überschwinger {worst}");
}

#[test]
fn pchip_stays_within_the_data_range_on_every_interval() {
    let x = [0.0, 1.0, 2.5, 3.0, 4.2, 5.0, 6.5];
    let y = [1.0, 2.0, 2.0, 5.0, 4.0, 4.5, 0.0];
    let mut d = [0.0; 7];
    let p = Pchip::new(&x, &y, &mut d).unwrap();
    for i in 0..6 {
        let (lo, hi) = (y[i].min(y[i + 1]), y[i].max(y[i + 1]));
        for k in 0..=50 {
            let xq = x[i] + (x[i + 1] - x[i]) * k as f64 / 50.0;
            let v = p.value(xq);
            assert!(
                v >= lo - 1e-12 && v <= hi + 1e-12,
                "Intervall {i}: {v} außerhalb [{lo}, {hi}]"
            );
        }
    }
    // lokale Extrema der Daten haben Steigung 0
    close(p.slopes()[3], 0.0, 1e-15);
}

#[test]
fn pchip_reproduces_lines_and_handles_two_points() {
    let x = [0.0, 1.0, 3.0, 4.0];
    let y = x.map(|v| 2.0 * v + 1.0);
    let mut d = [0.0; 4];
    let p = Pchip::new(&x, &y, &mut d).unwrap();
    for xq in [0.0, 0.5, 2.0, 3.7, 4.0] {
        close(p.value(xq), 2.0 * xq + 1.0, 1e-13);
        close(p.derivative(xq), 2.0, 1e-13);
    }
    let (x2, y2) = ([0.0, 2.0], [1.0, 5.0]);
    let mut d2 = [0.0; 2];
    let p2 = Pchip::new(&x2, &y2, &mut d2).unwrap();
    close(p2.value(1.0), 3.0, 1e-14);
}

#[test]
fn pchip_integral_matches_numeric_integration() {
    let x = [0.0, 1.0, 2.5, 3.0, 4.2, 5.0];
    let y = [1.0, 2.0, 2.0, 5.0, 4.0, 4.5];
    let mut d = [0.0; 6];
    let p = Pchip::new(&x, &y, &mut d).unwrap();
    // feine Trapezsumme als Referenz
    let n = 200_000;
    let h = 5.0 / n as f64;
    let mut sum = 0.5 * (p.value(0.0) + p.value(5.0));
    for k in 1..n {
        sum += p.value(k as f64 * h);
    }
    close(p.integral(0.0, 5.0), sum * h, 1e-7);
    close(p.integral(5.0, 0.0), -p.integral(0.0, 5.0), 1e-13);
    close(
        p.integral(0.5, 1.5) + p.integral(1.5, 4.0),
        p.integral(0.5, 4.0),
        1e-12,
    );
}

#[test]
fn pchip_errors_and_accessors() {
    let mut d = [0.0; 3];
    assert!(matches!(
        Pchip::new(&[0.0, 1.0, 2.0], &[0.0, 1.0], &mut d),
        Err(InterpError::InvalidInput)
    ));
    assert!(matches!(
        Pchip::new(&[0.0, 2.0, 1.0], &[0.0, 1.0, 2.0], &mut d),
        Err(InterpError::NotIncreasing)
    ));
    assert!(matches!(
        Pchip::new(&[0.0, 1.0, 2.0], &[0.0, f64::NAN, 2.0], &mut d),
        Err(InterpError::NotFinite)
    ));
    let mut short = [0.0; 2];
    assert!(matches!(
        Pchip::new(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &mut short),
        Err(InterpError::InvalidInput)
    ));
    let x = [0.0, 1.0, 2.0];
    let y = [0.0, 1.0, 4.0];
    let p = Pchip::new(&x, &y, &mut d).unwrap();
    assert_eq!(
        (p.domain(), p.nodes(), p.values()),
        ((0.0, 2.0), &x[..], &y[..])
    );
    assert!(p.contains(1.0) && !p.contains(2.5));
}

// ---------------------------------------------------------------------------
// Lagrange
// ---------------------------------------------------------------------------

#[test]
fn lagrange_reproduces_polynomials_exactly() {
    let x = [0.0, 1.0, 3.0, 4.5];
    let y = x.map(p);
    let mut work = [0.0; 4];
    for xq in [-1.0, 0.5, 2.0, 3.3, 6.0] {
        close(lagrange(&x, &y, xq, &mut work).unwrap(), p(xq), 1e-11);
    }
    // Knoten werden exakt getroffen, unsortierte Knoten sind erlaubt
    let (xu, yu) = ([3.0, 0.0, 1.0], [10.0, 1.0, 2.0]);
    let mut w3 = [0.0; 3];
    close(lagrange(&xu, &yu, 0.0, &mut w3).unwrap(), 1.0, 1e-13);
    close(lagrange(&xu, &yu, 2.0, &mut w3).unwrap(), 5.0, 1e-13);
}

#[test]
fn lagrange_shows_the_runge_phenomenon() {
    // 1/(1 + 25x²) mit 11 äquidistanten Punkten: das Polynom schwingt am Rand stark
    let f = |x: f64| 1.0 / (1.0 + 25.0 * x * x);
    let x: Vec<f64> = (0..11).map(|i| -1.0 + 0.2 * i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| f(v)).collect();
    let mut work = vec![0.0; 11];
    let poly_err = (lagrange(&x, &y, 0.95, &mut work).unwrap() - f(0.95)).abs();
    assert!(poly_err > 0.3, "Polynomfehler {poly_err}");

    let mut m = vec![0.0; 11];
    let mut sw = vec![0.0; cubic_spline_work_len(11)];
    let s = CubicSpline::new(&x, &y, BoundaryCondition::NotAKnot, &mut m, &mut sw).unwrap();
    let spline_err = (s.value(0.95) - f(0.95)).abs();
    assert!(spline_err < 0.05, "Splinefehler {spline_err}");
    assert!(poly_err > 10.0 * spline_err);
}

#[test]
fn lagrange_errors() {
    let mut work = [0.0; 3];
    assert_eq!(
        lagrange(&[0.0, 1.0], &[0.0], 0.5, &mut work),
        Err(InterpError::InvalidInput)
    );
    assert_eq!(
        lagrange::<f64>(&[], &[], 0.5, &mut work),
        Err(InterpError::InvalidInput)
    );
    let mut short = [0.0; 1];
    assert_eq!(
        lagrange(&[0.0, 1.0], &[0.0, 1.0], 0.5, &mut short),
        Err(InterpError::InvalidInput)
    );
    assert_eq!(
        lagrange(&[0.0, 1.0, 1.0], &[0.0, 1.0, 2.0], 0.5, &mut work),
        Err(InterpError::NotIncreasing)
    );
    assert_eq!(
        lagrange(&[0.0, 1.0], &[0.0, f64::NAN], 0.5, &mut work),
        Err(InterpError::NotFinite)
    );
    assert_eq!(
        lagrange(&[0.0, 1.0], &[0.0, 1.0], f64::NAN, &mut work),
        Err(InterpError::NotFinite)
    );
    // ein einzelner Punkt: konstante Funktion
    assert_eq!(lagrange(&[2.0], &[7.0], 100.0, &mut work).unwrap(), 7.0);
}

#[test]
fn error_display() {
    extern crate std;
    use std::string::ToString;
    for e in [
        InterpError::InvalidInput,
        InterpError::NotIncreasing,
        InterpError::NotFinite,
        InterpError::NotPeriodic,
        InterpError::Singular,
    ] {
        assert!(!e.to_string().is_empty());
    }
}

// ---------------------------------------------------------------------------
// Besitzende Varianten (alloc)
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
mod owned {
    use super::*;
    use azmath_interp::{CubicSplineBuf, PchipBuf};

    #[test]
    fn spline_buf_matches_the_borrowed_spline() {
        let y = y6();
        let buf = CubicSplineBuf::new(&X6, &y, BoundaryCondition::NotAKnot).unwrap();
        spline!(borrowed, X6, y, BoundaryCondition::NotAKnot);
        for k in 0..=20 {
            let xq = -1.0 + 4.0 * k as f64 / 20.0;
            assert_eq!(buf.value(xq), borrowed.value(xq));
            assert_eq!(buf.derivative(xq), borrowed.derivative(xq));
            assert_eq!(buf.second_derivative(xq), borrowed.second_derivative(xq));
        }
        assert_eq!(buf.integral(0.0, 2.0), borrowed.integral(0.0, 2.0));
        assert_eq!(buf.domain(), (-1.0, 3.0));
        assert!(buf.contains(0.0) && !buf.contains(5.0));
        let vals = buf.values_at(&[0.0, 1.0, 2.0]);
        assert_eq!(vals.len(), 3);
        close(vals[1], p(1.0), 1e-12);
        assert_eq!(buf.as_spline().nodes(), &X6);
    }

    #[test]
    fn pchip_buf_matches_the_borrowed_pchip() {
        let x = [0.0, 1.0, 2.5, 3.0, 4.2];
        let y = [1.0, 2.0, 2.0, 5.0, 4.0];
        let buf = PchipBuf::new(&x, &y).unwrap();
        let mut d = [0.0; 5];
        let borrowed = Pchip::new(&x, &y, &mut d).unwrap();
        for xq in [0.0, 0.7, 1.9, 3.5, 4.2] {
            assert_eq!(buf.value(xq), borrowed.value(xq));
            assert_eq!(buf.derivative(xq), borrowed.derivative(xq));
        }
        assert_eq!(buf.integral(0.0, 4.2), borrowed.integral(0.0, 4.2));
        assert_eq!(buf.values_at(&[1.0, 3.0]), vec![2.0, 5.0]);
        assert_eq!(buf.as_pchip().slopes().len(), 5);
    }

    #[test]
    fn owned_errors() {
        assert!(matches!(
            CubicSplineBuf::new(&[0.0, 1.0], &[0.0], BoundaryCondition::Natural),
            Err(InterpError::InvalidInput)
        ));
        assert!(matches!(
            CubicSplineBuf::new(&[1.0, 0.0], &[0.0, 1.0], BoundaryCondition::Natural),
            Err(InterpError::NotIncreasing)
        ));
        assert!(matches!(
            PchipBuf::new(&[0.0], &[0.0]),
            Err(InterpError::InvalidInput)
        ));
        assert!(matches!(
            PchipBuf::new(&[0.0, 1.0], &[f64::NAN, 1.0]),
            Err(InterpError::NotFinite)
        ));
    }

    #[test]
    fn dynamic_sizes_known_only_at_runtime() {
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        let y: Vec<f64> = x.iter().map(|&v| Real::cos(v)).collect();
        let s = CubicSplineBuf::new(&x, &y, BoundaryCondition::NotAKnot).unwrap();
        for &xq in &[0.123, 4.321, 9.876] {
            close(s.value(xq), Real::cos(xq), 1e-6);
        }
    }
}
