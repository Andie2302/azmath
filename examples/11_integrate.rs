// Beispiel 11 – Numerische Integration
//
// 1. Feste Regeln im Vergleich: Trapez, Simpson und Gauß-Legendre für ∫₀¹ 4/(1+x²) dx = π.
// 2. Das adaptive Verfahren mit Fehlerschätzung, auch bei Randsingularitäten (√x, ln x).
// 3. Integration von Messdaten (gleichmäßig und beliebig verteilt).
//
// Funktionen wie sin und ln werden über `Real::sin(x)` aufgerufen, damit das gewählte Backend
// (std, libm) zum Einsatz kommt.
//
//   cargo run --example 11_integrate

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::{
    Real,
    integrate::{
        IntegrateOptions, adaptive, gauss_legendre, simpson, simpson_samples, trapezoid,
        trapezoid_xy,
    },
};

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    use core::f64::consts::PI;

    // --- 1. Feste Regeln
    let f = |x: f64| 4.0 / (1.0 + x * x);
    println!("∫₀¹ 4/(1+x²) dx = π = {PI:.15}");
    println!(
        "  Trapez, n = 100            : {:.15}",
        trapezoid(f, 0.0, 1.0, 100).unwrap()
    );
    println!(
        "  Simpson, n = 100           : {:.15}",
        simpson(f, 0.0, 1.0, 100).unwrap()
    );
    println!(
        "  Gauß-Legendre, 5 Punkte × 4: {:.15}",
        gauss_legendre(f, 0.0, 1.0, 5, 4).unwrap()
    );

    // --- 2. Adaptiv
    let opts = IntegrateOptions::default();
    let r = adaptive(f, 0.0, 1.0, opts).unwrap();
    println!(
        "  adaptiv                    : {:.15}  (Fehlerschätzung {:.1e}, {} Auswertungen)",
        r.value, r.error_estimate, r.evaluations
    );

    let s = adaptive(Real::sqrt, 0.0, 1.0, opts).unwrap();
    println!(
        "∫₀¹ √x dx   = 2/3 → {:.12}  ({} Auswertungen)",
        s.value, s.evaluations
    );
    let l = adaptive(Real::ln, 0.0, 1.0, opts).unwrap();
    println!(
        "∫₀¹ ln x dx = −1  → {:.12}  ({} Auswertungen)",
        l.value, l.evaluations
    );
    // Oszillation (rund 8 Perioden); bewusst kein um die Mitte antisymmetrisches Intervall wie
    // [0, π], bei dem sich die symmetrischen Knoten aufheben würden (siehe Doku von `adaptive`)
    let o = adaptive(|x: f64| Real::sin(50.0 * x), 0.0, 1.0, opts).unwrap();
    println!(
        "∫₀¹ sin(50x) dx = {:.12} (exakt {:.12}, {} Auswertungen)",
        o.value,
        (1.0 - Real::cos(50.0f64)) / 50.0,
        o.evaluations
    );

    // --- 3. Messdaten
    // Geschwindigkeit v(t) = 2t in m/s, alle 0,5 s gemessen: Weg nach 4 s = 16 m
    let v: Vec<f64> = (0..=8).map(|i| 2.0 * 0.5 * i as f64).collect();
    println!(
        "Weg aus gleichmäßigen Messwerten (Simpson): {:.6} m",
        simpson_samples(&v, 0.5).unwrap()
    );
    let t = [0.0, 0.4, 1.1, 2.0, 3.5, 4.0];
    let vt: Vec<f64> = t.iter().map(|t| 2.0 * t).collect();
    println!(
        "Weg aus unregelmäßigen Messwerten (Trapez): {:.6} m",
        trapezoid_xy(&t, &vt).unwrap()
    );
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    eprintln!("Bitte mit --features std oder --features libm ausführen.");
}
