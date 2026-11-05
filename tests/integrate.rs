//! Tests für die numerische Integration: feste Regeln, Gauß-Legendre, adaptiv, Messdaten.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::Real;
use azmath::integrate::{
    IntegrateError, IntegrateOptions, adaptive, gauss_legendre, simpson, simpson_samples,
    trapezoid, trapezoid_samples, trapezoid_xy,
};
use core::f64::consts::{E, FRAC_PI_2, PI};

fn opts() -> IntegrateOptions<f64> {
    IntegrateOptions::default()
}

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "erwartet {b}, erhalten {a} (Δ = {:e})",
        (a - b).abs()
    );
}

/// Kreuzprobe für die Konstanten: eine einzige G7/K15-Auswertung (großzügige Toleranz, also
/// ohne Teilung) muss Polynome bis Grad 22 exakt integrieren (Kronrod) und der Fehlerschätzer
/// |K − G| bis Grad 13 verschwinden (Gauß).
#[test]
fn gauss_kronrod_constants_are_correct() {
    let huge = IntegrateOptions::new(1.0e3, 1.0e3);
    for degree in 0..=22u32 {
        let r = adaptive(|x: f64| x.powi(degree as i32), -1.0, 1.0, huge).unwrap();
        assert_eq!(r.evaluations, 15, "es darf keine Teilung stattfinden");
        let exact = if degree % 2 == 1 {
            0.0
        } else {
            2.0 / (degree as f64 + 1.0)
        };
        close(r.value, exact, 1e-14);
        if degree <= 13 {
            assert!(
                r.error_estimate < 1e-14,
                "Grad {degree}: Fehlerschätzer {:e}",
                r.error_estimate
            );
        }
    }
    // ab Grad 14 sieht der Schätzer den Unterschied zur Gauß-Regel
    let r = adaptive(|x: f64| x.powi(14), -1.0, 1.0, huge).unwrap();
    assert!(r.error_estimate > 1e-6);
}

#[test]
fn trapezoid_exact_for_linear() {
    close(
        trapezoid(|x: f64| 3.0 * x + 1.0, 0.0, 4.0, 1).unwrap(),
        28.0,
        1e-13,
    );
    close(
        trapezoid(|x: f64| 3.0 * x + 1.0, -2.0, 5.0, 7).unwrap(),
        3.0 * 10.5 + 7.0,
        1e-12,
    );
}

#[test]
fn simpson_exact_for_cubic() {
    // ∫₀² x³ − x + 1 = 4 − 2 + 2 = 4
    close(
        simpson(|x: f64| x * x * x - x + 1.0, 0.0, 2.0, 2).unwrap(),
        4.0,
        1e-13,
    );
}

#[test]
fn convergence_orders() {
    // Trapez: Fehler ∝ h² (Faktor ≈ 4 beim Verdoppeln), Simpson: ∝ h⁴ (Faktor ≈ 16)
    let f = |x: f64| Real::exp(x);
    let exact = E - 1.0;
    let t = |n| (trapezoid(f, 0.0, 1.0, n).unwrap() - exact).abs();
    let s = |n| (simpson(f, 0.0, 1.0, n).unwrap() - exact).abs();
    let (rt, rs) = (t(16) / t(32), s(16) / s(32));
    assert!((rt - 4.0).abs() < 0.1, "Trapez-Verhältnis {rt}");
    assert!((rs - 16.0).abs() < 0.5, "Simpson-Verhältnis {rs}");
    assert!(s(32) < t(32) * 1e-3);
}

#[test]
fn gauss_legendre_exactness_per_order() {
    for order in 1..=5usize {
        let max_degree = 2 * order - 1;
        for degree in 0..=max_degree {
            let v = gauss_legendre(|x: f64| x.powi(degree as i32), 0.0, 1.0, order, 1).unwrap();
            close(v, 1.0 / (degree as f64 + 1.0), 1e-15);
        }
        // Grad 2·order ist nicht mehr exakt
        let v = gauss_legendre(|x: f64| x.powi(2 * order as i32), 0.0, 1.0, order, 1).unwrap();
        assert!(
            (v - 1.0 / (2.0 * order as f64 + 1.0)).abs() > 1e-8,
            "Ordnung {order}"
        );
    }
}

#[test]
fn gauss_legendre_converges_fast_with_panels() {
    let exact = 2.0;
    let e1 = (gauss_legendre(Real::sin, 0.0, PI, 3, 1).unwrap() - exact).abs();
    let e4 = (gauss_legendre(Real::sin, 0.0, PI, 3, 4).unwrap() - exact).abs();
    assert!(e4 < e1 * 1e-3, "{e1:e} -> {e4:e}");
    close(
        gauss_legendre(Real::sin, 0.0, PI, 5, 8).unwrap(),
        exact,
        1e-14,
    );
}

#[test]
fn smooth_integrals_agree_across_methods() {
    // ∫₀¹ 1/(1+x²) dx = π/4
    let f = |x: f64| 1.0 / (1.0 + x * x);
    let exact = PI / 4.0;
    close(trapezoid(f, 0.0, 1.0, 2000).unwrap(), exact, 1e-7);
    close(simpson(f, 0.0, 1.0, 200).unwrap(), exact, 1e-10);
    close(gauss_legendre(f, 0.0, 1.0, 5, 4).unwrap(), exact, 1e-14);
    close(adaptive(f, 0.0, 1.0, opts()).unwrap().value, exact, 1e-13);
}

#[test]
fn adaptive_reaches_tolerance_with_error_estimate() {
    let r = adaptive(Real::sin, 0.0, PI, opts()).unwrap();
    close(r.value, 2.0, 1e-13);
    assert!(
        r.error_estimate >= (r.value - 2.0).abs(),
        "glatt: Schätzung ist vorsichtig"
    );
    assert_eq!(r.evaluations, 15, "glatte Funktion: eine Auswertung genügt");

    let tight = IntegrateOptions::new(1e-14, 1e-14);
    close(
        adaptive(Real::exp, 0.0, 1.0, tight).unwrap().value,
        E - 1.0,
        1e-14,
    );
}

#[test]
fn adaptive_handles_endpoint_singularities() {
    // √x hat unendliche Steigung bei 0: ∫₀¹ = 2/3
    let r = adaptive(Real::sqrt, 0.0, 1.0, opts()).unwrap();
    close(r.value, 2.0 / 3.0, 1e-10);
    // ln(x) divergiert bei 0, ist aber integrierbar: ∫₀¹ = −1
    let r = adaptive(Real::ln, 0.0, 1.0, opts()).unwrap();
    close(r.value, -1.0, 1e-10);
    // 1/√x: ∫₀¹ = 2 (braucht rund 80 Halbierungen am Rand)
    let r = adaptive(|x: f64| 1.0 / Real::sqrt(x), 0.0, 1.0, opts()).unwrap();
    close(r.value, 2.0, 1e-9);
    // x^(−3/4): ∫₀¹ = 4, noch stärker singulär; die Schätzung ist hier nur ein Richtwert
    let r = adaptive(|x: f64| x.powf(-0.75), 0.0, 1.0, opts()).unwrap();
    close(r.value, 4.0, 1e-8);
    // Singularität im Inneren, nicht auf einem Auswertungspunkt:
    // ∫₀¹ |x − 0.3|^(−1/2) dx = 2·(√0.3 + √0.7)
    // (mit lockerer Toleranz: die Halbierung endet an der Auflösung der Gleitkommazahlen um 0.3)
    let loose = IntegrateOptions::new(1e-7, 1e-7);
    let r = adaptive(|x: f64| 1.0 / Real::sqrt((x - 0.3).abs()), 0.0, 1.0, loose).unwrap();
    close(r.value, 2.0 * (0.3f64.sqrt() + 0.7f64.sqrt()), 1e-6);
}

#[test]
fn singularity_exactly_on_a_node_is_not_finite() {
    // Die Mitte von [−1, 1] ist ein G7/K15-Knoten; dort ist f(0) = ∞, also gibt es keinen
    // endlichen Wert. Die Grenzen so wählen, dass die Singularität nicht auf einem Knoten liegt.
    assert_eq!(
        adaptive(|x: f64| 1.0 / Real::sqrt(x.abs()), -1.0, 1.0, opts()),
        Err(IntegrateError::NotFinite)
    );
    let r = adaptive(|x: f64| 1.0 / Real::sqrt(x.abs()), -1.0, 1.1, opts()).unwrap();
    close(r.value, 2.0 + 2.0 * 1.1f64.sqrt(), 1e-8);
}

#[test]
fn singular_integral_after_substitution_is_cheap() {
    // ∫₀¹ x^(−1/2) dx mit x = t²: ∫₀¹ 2 dt – ein Bruchteil der Auswertungen
    let direct = adaptive(|x: f64| 1.0 / Real::sqrt(x), 0.0, 1.0, opts()).unwrap();
    let subst = adaptive(|_t: f64| 2.0, 0.0, 1.0, opts()).unwrap();
    close(subst.value, 2.0, 1e-14);
    assert!(subst.evaluations * 20 < direct.evaluations);
}

#[test]
fn adaptive_handles_peaks_and_oscillation() {
    // schmale Gauß-Glocke: ∫ e^{−(x/0.01)²} dx = 0.01·√π über [−1, 1]
    let peak = |x: f64| Real::exp(-(x / 0.01) * (x / 0.01));
    close(
        adaptive(peak, -1.0, 1.0, opts()).unwrap().value,
        0.01 * PI.sqrt(),
        1e-12,
    );
    // hochfrequent (rund 8 Perioden): ∫₀¹ sin(50x) dx = (1 − cos 50)/50
    let r = adaptive(|x: f64| Real::sin(50.0 * x), 0.0, 1.0, opts()).unwrap();
    close(r.value, (1.0 - Real::cos(50.0f64)) / 50.0, 1e-12);
    assert!(
        r.evaluations > 15,
        "die Oszillation muss zu Teilungen führen"
    );
    // ∫₀^{2π} x·sin(x) dx = −2π
    close(
        adaptive(|x: f64| x * Real::sin(x), 0.0, 2.0 * PI, opts())
            .unwrap()
            .value,
        -2.0 * PI,
        1e-11,
    );
}

#[test]
fn antisymmetric_integrand_still_must_meet_the_reported_error_criterion() {
    // Antisymmetrische Beiträge können bis zum Rundungsniveau auslöschen.
    // Erfolg ist nur zulässig, wenn der gemeldete Fehler die Toleranz erfüllt;
    // Rundungsrauschen rechtfertigt sonst NoConvergence.
    let o = opts();
    match adaptive(|x: f64| Real::sin(50.0 * x), 0.0, PI, o) {
        Ok(r) => {
            assert!(r.error_estimate <= o.abs_tol.max(o.rel_tol * r.value.abs()));
            close(r.value, 0.0, 1e-12);
        }
        Err(e) => assert_eq!(e, IntegrateError::NoConvergence),
    }
    // Ohne diese Symmetrie wird dieselbe Funktion korrekt aufgelöst (siehe oben). Abhilfe:
    // das Intervall unsymmetrisch teilen. Für das stark auslöschende Integral
    // eine absolute Toleranz passend zur unten geprüften Genauigkeit anfordern.
    let split_opts = IntegrateOptions::new(1e-13, 1e-12);
    let a = adaptive(|x: f64| Real::sin(50.0 * x), 0.0, 1.0, split_opts)
        .unwrap()
        .value;
    let b = adaptive(|x: f64| Real::sin(50.0 * x), 1.0, PI, split_opts)
        .unwrap()
        .value;
    close(a + b, 0.0, 1e-12);
}

#[test]
fn adaptive_integrates_a_kink() {
    // |x − 0.3| über [0, 1]: Knick bei 0.3, Integral = 0.045 + 0.245 = 0.29
    let r = adaptive(|x: f64| (x - 0.3).abs(), 0.0, 1.0, opts()).unwrap();
    close(r.value, 0.29, 1e-11);
}

#[test]
fn reversed_and_empty_intervals() {
    let fwd = adaptive(Real::sin, 0.0, FRAC_PI_2, opts()).unwrap().value;
    let rev = adaptive(Real::sin, FRAC_PI_2, 0.0, opts()).unwrap().value;
    close(fwd, 1.0, 1e-13);
    close(rev, -1.0, 1e-13);
    let zero = adaptive(Real::sin, 1.0, 1.0, opts()).unwrap();
    assert_eq!((zero.value, zero.evaluations), (0.0, 0));
    close(trapezoid(|x: f64| x, 2.0, 0.0, 4).unwrap(), -2.0, 1e-14);
    close(simpson(|x: f64| x * x, 3.0, 0.0, 4).unwrap(), -9.0, 1e-13);
    close(
        gauss_legendre(|x: f64| x, 2.0, 0.0, 2, 1).unwrap(),
        -2.0,
        1e-14,
    );
}

#[test]
fn zero_integral_converges_via_absolute_tolerance() {
    let r = adaptive(Real::sin, 0.0, 2.0 * PI, opts()).unwrap();
    close(r.value, 0.0, 1e-13);
    let r = adaptive(|_: f64| 0.0, 0.0, 1.0, opts()).unwrap();
    assert_eq!(r.value, 0.0);
}

#[test]
fn non_integrable_singularity_does_not_converge() {
    // ∫₀¹ 1/x dx divergiert
    let small = IntegrateOptions {
        max_intervals: 60,
        ..opts()
    };
    assert_eq!(
        adaptive(|x: f64| 1.0 / x, 0.0, 1.0, small),
        Err(IntegrateError::NoConvergence)
    );
}

#[test]
fn evaluation_budget_is_respected() {
    let tiny = IntegrateOptions {
        max_evals: 100,
        ..IntegrateOptions::new(1e-14, 1e-14)
    };
    let mut calls = 0usize;
    let r = adaptive(
        |x: f64| {
            calls += 1;
            Real::sin(1.0 / (x + 0.001))
        },
        0.0,
        1.0,
        tiny,
    );
    assert_eq!(r, Err(IntegrateError::NoConvergence));
    assert!(calls <= 130, "{calls} Auswertungen trotz Limit 100");
}

#[test]
fn non_finite_values_are_reported() {
    assert_eq!(
        adaptive(|_: f64| f64::NAN, 0.0, 1.0, opts()),
        Err(IntegrateError::NotFinite)
    );
    assert_eq!(
        adaptive(Real::sin, 0.0, f64::INFINITY, opts()),
        Err(IntegrateError::NotFinite)
    );
    assert_eq!(
        adaptive(Real::sin, f64::NAN, 1.0, opts()),
        Err(IntegrateError::NotFinite)
    );
    assert_eq!(
        trapezoid(|_: f64| f64::INFINITY, 0.0, 1.0, 4),
        Err(IntegrateError::NotFinite)
    );
    assert_eq!(
        simpson(|x: f64| x, f64::NAN, 1.0, 4),
        Err(IntegrateError::NotFinite)
    );
    assert_eq!(
        gauss_legendre(|_: f64| f64::NAN, 0.0, 1.0, 3, 2),
        Err(IntegrateError::NotFinite)
    );
}

#[test]
fn invalid_inputs() {
    let f = |x: f64| x;
    assert_eq!(trapezoid(f, 0.0, 1.0, 0), Err(IntegrateError::InvalidInput));
    assert_eq!(simpson(f, 0.0, 1.0, 0), Err(IntegrateError::InvalidInput));
    assert_eq!(simpson(f, 0.0, 1.0, 3), Err(IntegrateError::InvalidInput));
    assert_eq!(
        gauss_legendre(f, 0.0, 1.0, 0, 1),
        Err(IntegrateError::InvalidInput)
    );
    assert_eq!(
        gauss_legendre(f, 0.0, 1.0, 6, 1),
        Err(IntegrateError::InvalidInput)
    );
    assert_eq!(
        gauss_legendre(f, 0.0, 1.0, 3, 0),
        Err(IntegrateError::InvalidInput)
    );
    assert_eq!(
        trapezoid_samples(&[1.0], 0.1),
        Err(IntegrateError::InvalidInput)
    );
    assert_eq!(
        simpson_samples(&[1.0, 2.0], 0.1),
        Err(IntegrateError::InvalidInput)
    );
    assert_eq!(
        simpson_samples(&[1.0, 2.0, 3.0, 4.0], 0.1),
        Err(IntegrateError::InvalidInput)
    );
    assert_eq!(
        trapezoid_xy(&[0.0, 1.0], &[1.0]),
        Err(IntegrateError::InvalidInput)
    );
    assert_eq!(
        trapezoid_xy(&[0.0], &[1.0]),
        Err(IntegrateError::InvalidInput)
    );
}

#[test]
fn sample_data_uniform() {
    // y = x² auf [0, 4] in 8 Intervallen (h = 0.5): Simpson exakt = 64/3
    let y: Vec<f64> = (0..=8).map(|i| (0.5 * i as f64).powi(2)).collect();
    close(simpson_samples(&y, 0.5).unwrap(), 64.0 / 3.0, 1e-12);
    // Trapez überschätzt die konvexe Parabel leicht
    let t = trapezoid_samples(&y, 0.5).unwrap();
    // Fehler der Trapezregel: h²/12·(b−a)·f'' = 0.25/12·4·2 = 1/6
    close(t, 64.0 / 3.0 + 1.0 / 6.0, 1e-12);
    // Genau drei Punkte = ein Simpson-Paar
    close(
        simpson_samples(&[0.0, 1.0, 4.0], 1.0).unwrap(),
        8.0 / 3.0,
        1e-14,
    );
}

#[test]
fn sample_data_non_uniform() {
    // y = 2x + 1 an ungleichmäßig verteilten Stellen: Trapez exakt
    let x = [0.0, 0.1, 0.5, 1.2, 3.0];
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    close(trapezoid_xy(&x, &y).unwrap(), 9.0 + 3.0, 1e-13);
    // absteigende Stützstellen liefern das negative Integral
    let xr: Vec<f64> = x.iter().rev().copied().collect();
    let yr: Vec<f64> = y.iter().rev().copied().collect();
    close(trapezoid_xy(&xr, &yr).unwrap(), -12.0, 1e-13);
}

#[test]
fn stateful_closures_work() {
    let mut calls = 0;
    let v = simpson(
        |x: f64| {
            calls += 1;
            x * x
        },
        0.0,
        3.0,
        6,
    )
    .unwrap();
    close(v, 9.0, 1e-13);
    assert_eq!(calls, 7);
}

#[test]
fn f32_works() {
    let o = IntegrateOptions::<f32>::default();
    let r = adaptive(|x: f32| x.sin(), 0.0, core::f32::consts::PI, o).unwrap();
    assert!((r.value - 2.0).abs() < 1e-5, "{}", r.value);
    let v = gauss_legendre(|x: f32| x * x, 0.0, 3.0, 2, 1).unwrap();
    assert!((v - 9.0).abs() < 1e-5);
    let t = simpson(|x: f32| x * x * x, 0.0, 2.0, 4).unwrap();
    assert!((t - 4.0).abs() < 1e-5);
}

#[test]
fn default_options_and_error_display() {
    let o = IntegrateOptions::<f64>::default();
    assert!(o.abs_tol > 0.0 && o.rel_tol > 1e-14 && o.rel_tol < 1e-9);
    assert_eq!((o.max_intervals, o.max_evals), (200, 200_000));
    extern crate std;
    use std::string::ToString;
    for e in [
        IntegrateError::InvalidInput,
        IntegrateError::NotFinite,
        IntegrateError::NoConvergence,
    ] {
        assert!(!e.to_string().is_empty());
    }
}
