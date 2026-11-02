// Beispiel 13 – Differentialgleichungen: RK4, adaptives RK45 und implizite Verfahren
//
// 1. Räuber-Beute-Modell (Lotka-Volterra) mit dem adaptiven RK45; der Observer zeichnet den
//    Verlauf auf, ohne dass das Verfahren selbst Speicher belegt.
// 2. Konvergenzordnung: RK4 gegen Euler an y' = -y.
// 3. Steifes Problem: explizites Euler explodiert, implizites Euler bleibt stabil.
//
// Funktionen wie exp und cos werden über `Real::exp(x)` aufgerufen, damit das gewählte Backend
// (std, libm) zum Einsatz kommt.
//
//   cargo run --example 13_ode

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::{
    Real,
    ode::{
        Rk45Options, euler, euler_work_len, implicit_euler, implicit_work_len, rk4, rk4_work_len,
        rk45, rk45_work_len,
    },
    optimize::OptimizeOptions,
};

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    // --- 1. Räuber-Beute: x' = a·x − b·x·y,  y' = d·x·y − c·y
    let (a, b, c, d) = (1.0, 0.5, 0.75, 0.25);
    let lotka = move |_t: f64, s: &[f64], ds: &mut [f64]| {
        ds[0] = a * s[0] - b * s[0] * s[1];
        ds[1] = d * s[0] * s[1] - c * s[1];
    };
    let energie = |s: &[f64]| d * s[0] - c * Real::ln(s[0]) + b * s[1] - a * Real::ln(s[1]);

    let mut s = [4.0, 2.0];
    let h0 = energie(&s);
    let mut work = [0.0; rk45_work_len(2)];
    let mut verlauf = Vec::new();
    let r = rk45(
        lotka,
        0.0,
        20.0,
        &mut s,
        Rk45Options::with_tolerance(1e-9, 1e-12),
        &mut work,
        |t, st| verlauf.push((t, st[0], st[1])),
    )
    .unwrap();
    println!("Lotka-Volterra, t = 0 … 20:");
    println!(
        "  {} Schritte ({} abgelehnt), {} Auswertungen von f",
        r.accepted_steps, r.rejected_steps, r.evaluations
    );
    println!("  Ende: Beute = {:.6}, Räuber = {:.6}", s[0], s[1]);
    println!(
        "  Erhaltungsgröße H: Abweichung {:.2e}",
        (energie(&s) - h0).abs()
    );
    let (tmin, xmin) = verlauf
        .iter()
        .map(|&(t, x, _)| (t, x))
        .fold((0.0, f64::INFINITY), |m, v| if v.1 < m.1 { v } else { m });
    println!("  kleinste Beutepopulation {xmin:.4} bei t ≈ {tmin:.3}");

    // --- 2. Konvergenzordnung an y' = −y, y(0) = 1, y(1) = e⁻¹
    let zerfall = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -y[0];
    let exakt = Real::exp(-1.0f64);
    println!("Fehler bei y' = −y nach t = 1:");
    for n in [10usize, 20, 40] {
        let (mut ye, mut yr) = ([1.0], [1.0]);
        euler(
            zerfall,
            0.0,
            1.0,
            &mut ye,
            n,
            &mut [0.0; euler_work_len(1)],
            |_, _| {},
        )
        .unwrap();
        rk4(
            zerfall,
            0.0,
            1.0,
            &mut yr,
            n,
            &mut [0.0; rk4_work_len(1)],
            |_, _| {},
        )
        .unwrap();
        println!(
            "  {n:>2} Schritte: Euler {:.2e}, RK4 {:.2e}",
            (ye[0] - exakt).abs(),
            (yr[0] - exakt).abs()
        );
    }

    // --- 3. Steif: y' = −1000·(y − cos t), y(0) = 1, mit h = 0.05
    let lambda = 1000.0;
    let steif = move |t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -lambda * (y[0] - Real::cos(t));
    let jac = move |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = -lambda;
    let mut explizit = [1.0];
    let r = euler(
        steif,
        0.0,
        1.0,
        &mut explizit,
        20,
        &mut [0.0; euler_work_len(1)],
        |_, _| {},
    );
    println!(
        "Steifes Problem (λ = 1000, h = 0.05), exakt ≈ cos(1) = {:.4}:",
        Real::cos(1.0f64)
    );
    println!("  explizites Euler: {:.3e}  ({r:?})", explizit[0]);
    let mut implizit = [1.0];
    let mut iwork = [0.0; implicit_work_len(1)];
    implicit_euler(
        steif,
        jac,
        0.0,
        1.0,
        &mut implizit,
        20,
        OptimizeOptions::default(),
        &mut iwork,
        |_, _| {},
    )
    .unwrap();
    println!("  implizites Euler: {:.4}", implizit[0]);
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    eprintln!("Bitte mit --features std oder --features libm ausführen.");
}
