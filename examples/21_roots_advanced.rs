// Beispiel 21 – Fortgeschrittene Nullstellensuche und Polynom-Löser
//
// 1. Regula Falsi gegen das Illinois-Verfahren an einer stark konvexen Funktion.
//    Zeigt eindrucksvoll, wie das Illinois-Verfahren die Stagnation bricht.
// 2. Jenkins-Traub (CPOLY) für die komplexen Nullstellen eines Polynoms.
// 3. Aberth-Ehrlich zur simultanen Berechnung aller komplexen Wurzeln.
//
// Ausführen:  cargo run --example 21_roots_advanced --features std

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::{
    complex::Complex,
    roots::{
        RootOptions, aberth, illinois, jenkins_traub, jenkins_traub_work_len, polyval, regula_falsi,
    },
};

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    let opts = RootOptions::default();

    // --- 1. Regula Falsi vs. Illinois an f(x) = x¹⁰ - 1 auf [0.0, 1.3]
    // Eine stark konvexe Funktion. Hier bleibt bei der klassischen Regula Falsi
    // eine Intervallgrenze stur stehen, was die Konvergenz extrem verlangsamt.
    let f_konvex = |x: f64| x.powi(10) - 1.0;

    println!("Lösung von x¹⁰ - 1 = 0 auf [0.0, 1.3] (exakt: 1.0):");

    let rf = regula_falsi(f_konvex, 0.0, 1.3, opts).unwrap();
    println!(
        "  Regula Falsi : {:.12}  ({:>4} Iterationen, Rest: {:e})",
        rf.root, rf.iterations, rf.f_value
    );

    let il = illinois(f_konvex, 0.0, 1.3, opts).unwrap();
    println!(
        "  Illinois     : {:.12}  ({:>4} Iterationen, Rest: {:e})",
        il.root, il.iterations, il.f_value
    );
    println!("  -> Das Illinois-Verfahren braucht nur einen Bruchteil der Schritte!");

    println!("\n--- Polynom-Nullstellen (Komplexer Raum) ---");

    // --- 2. Jenkins-Traub (Standard für hohen Grad)
    // Wir lösen: x³ - 6x² + 11x - 6 = 0  -> Wurzeln bei 1.0, 2.0, 3.0
    // Koeffizienten aufsteigend sortiert: c[0] + c[1]·x + ...
    let coeffs = [-6.0, 11.0, -6.0, 1.0];
    let grad = coeffs.len() - 1;

    let mut roots_jt = vec![Complex::ZERO; grad];
    let mut work_jt = vec![Complex::ZERO; jenkins_traub_work_len(grad)];

    jenkins_traub(&coeffs, &mut roots_jt, &mut work_jt, opts).unwrap();
    println!("Jenkins-Traub (Wurzeln von x³ - 6x² + 11x - 6):");
    for (i, z) in roots_jt.iter().enumerate() {
        println!("  Wurzel {}: {:.6} + {:.6}i", i + 1, z.re, z.im);
    }

    // --- 3. Aberth-Ehrlich (Simultane elektrostatische Abstoßung)
    // Wir lösen ein Polynom mit rein komplexen Wurzeln: x⁴ - 1 = 0
    // Wurzeln liegen bei: 1, -1, i, -i
    let coeffs_unity = [-1.0, 0.0, 0.0, 0.0, 1.0]; // 1·x⁴ + 0·x³ + 0·x² + 0·x - 1
    let grad_unity = coeffs_unity.len() - 1;

    let mut roots_ab = vec![Complex::ZERO; grad_unity];

    let res_ab = aberth(&coeffs_unity, &mut roots_ab, opts).unwrap();
    println!(
        "\nAberth-Ehrlich (Wurzeln von x⁴ - 1, {} Runden):",
        res_ab.iterations
    );
    for (i, z) in roots_ab.iter().enumerate() {
        println!("  Wurzel {}: {:5.1} + {:5.1}i", i + 1, z.re, z.im);
    }

    // Kontrollprobe über dein funktionierendes Horner-Schema (polyval)
    println!("\nKontrollprobe über Horner-Schema (polyval):");
    let test_x = 2.0f64;
    let p_wert = polyval(&coeffs, test_x);
    println!("  Wert von p(2) über polyval: {p_wert} (erwartet: 0.0, weil 2 eine Nullstelle ist)");
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    eprintln!("Bitte mit --features std oder --features libm ausführen.");
}
