// Beispiel 15 – Interpolation: linear, kubischer Spline, PCHIP und Lagrange
//
// 1. Runge-Phänomen: Ein Polynom durch 11 gleichmäßig verteilte Punkte von 1/(1+25x²) schwingt am Rand
//    stark, ein kubischer Spline nicht.
// 2. Randbedingungen eines Splines im Vergleich (natürlich, eingespannt, not-a-knot).
// 3. Formerhaltung: PCHIP überschwingt bei monotonen Daten mit Stufe nicht, der Spline schon.
// 4. Integral und Ableitung eines Splines.
//
// Die Spline-Typen leihen sich die Daten und schreiben ihre Koeffizienten in einen vom Aufrufer
// übergebenen Slice (kein Allokator nötig). Mit `--features alloc` gibt es besitzende Varianten.
//
//   cargo run --example 15_interpolation

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::{
    Real,
    interp::{BoundaryCondition, CubicSpline, Pchip, cubic_spline_work_len, lagrange, linear},
};

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    use core::f64::consts::PI;

    // --- 1. Runge-Phänomen
    let f = |x: f64| 1.0 / (1.0 + 25.0 * x * x);
    let x: Vec<f64> = (0..11).map(|i| -1.0 + 0.2 * i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&v| f(v)).collect();
    let mut work = vec![0.0; cubic_spline_work_len(11)];
    let mut m = vec![0.0; 11];
    let spline = CubicSpline::new(&x, &y, BoundaryCondition::NotAKnot, &mut m, &mut work).unwrap();
    println!("Runge-Funktion 1/(1+25x²) mit 11 gleichmäßigen Punkten, Fehler bei x = 0.95:");
    println!(
        "  Polynom (Lagrange): {:.4}",
        (lagrange(&x, &y, 0.95, &mut work).unwrap() - f(0.95)).abs()
    );
    println!(
        "  kubischer Spline  : {:.4}",
        (spline.value(0.95) - f(0.95)).abs()
    );
    println!(
        "  linear            : {:.4}",
        (linear(&x, &y, 0.95).unwrap() - f(0.95)).abs()
    );

    // --- 2. Randbedingungen an sin(x) auf [0, π] mit 7 Punkten
    //     Der natürliche Spline nimmt S'' = 0 an den Rändern an – für sin stimmt das (sin'' = 0
    //     bei 0 und π), deshalb schneidet er hier so gut ab wie der eingespannte mit exakten
    //     Steigungen. Not-a-knot ist ein guter Standard, wenn man nichts über den Rand weiß, aber
    //     nicht immer der beste.
    let xs: Vec<f64> = (0..7).map(|i| i as f64 * PI / 6.0).collect();
    let ys: Vec<f64> = xs.iter().map(|&v| Real::sin(v)).collect();
    println!("sin(x) auf [0, π] mit 7 Punkten, größter Fehler:");
    for (name, bc) in [
        ("natürlich  ", BoundaryCondition::Natural),
        (
            "eingespannt",
            BoundaryCondition::Clamped {
                left: 1.0,
                right: -1.0,
            },
        ),
        ("not-a-knot ", BoundaryCondition::NotAKnot),
    ] {
        let mut m = vec![0.0; 7];
        let s = CubicSpline::new(&xs, &ys, bc, &mut m, &mut work).unwrap();
        let err = (0..=1000)
            .map(|k| {
                let xq = PI * k as f64 / 1000.0;
                (s.value(xq) - Real::sin(xq)).abs()
            })
            .fold(0.0f64, f64::max);
        println!("  {name}: {err:.2e}");
    }

    // --- 3. Formerhaltung bei monotonen Daten mit Stufe
    let sx = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let sy = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let (mut ms, mut ds) = (vec![0.0; 6], vec![0.0; 6]);
    let sp = CubicSpline::new(&sx, &sy, BoundaryCondition::NotAKnot, &mut ms, &mut work).unwrap();
    let pc = Pchip::new(&sx, &sy, &mut ds).unwrap();
    let ueberschwingen = |eval: &dyn Fn(f64) -> f64| {
        (0..=500)
            .map(|k| eval(k as f64 * 0.01))
            .fold(0.0f64, |m, v| m.max(-v).max(v - 1.0))
    };
    println!("Stufe 0 → 1, größter Überschwinger außerhalb [0, 1]:");
    println!(
        "  kubischer Spline: {:.4}",
        ueberschwingen(&|v| sp.value(v))
    );
    println!(
        "  PCHIP           : {:.4}",
        ueberschwingen(&|v| pc.value(v))
    );

    // --- 4. Ableitung und Integral eines Splines
    let mut m = vec![0.0; 7];
    let sin_spline =
        CubicSpline::new(&xs, &ys, BoundaryCondition::Natural, &mut m, &mut work).unwrap();
    println!("Spline durch sin:");
    println!(
        "  S'(π/3) = {:.5}  (cos(π/3) = {:.5})",
        sin_spline.derivative(PI / 3.0),
        Real::cos(PI / 3.0)
    );
    println!("  ∫₀^π S  = {:.6}  (exakt 2)", sin_spline.integral(0.0, PI));
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    eprintln!("Bitte mit --features std oder --features libm ausführen.");
}
