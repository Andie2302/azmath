// Beispiel 22 – Fortgeschrittene Matrixfunktionen und Regelungstechnik
//
// 1. Matrix-Exponentialfunktion (`expm`) über die Higham-Padé-Approximation.
//    Berechnet die exakte 2D-Drehmatrix aus einer schiefsymmetrischen Matrix.
// 2. Kontinuierliche algebraische Riccati-Gleichung (`care`) für einen LQR-Regler.
//    Berechnet die optimalen Stabilisierungs-Gewichte für ein inverses Pendel.
//
// Ausführen:  cargo run --example 22_matrix_functions --features alloc

#[cfg(all(feature = "alloc", any(feature = "std", feature = "libm")))]
use azmath::{
    Real,
    linalg::DynMatrix,
    matfun::{care, expm, logm, sqrtm},
};

#[cfg(all(feature = "alloc", any(feature = "std", feature = "libm")))]
fn max_diff(a: &DynMatrix<f64>, b: &DynMatrix<f64>) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[cfg(all(feature = "alloc", any(feature = "std", feature = "libm")))]
fn main() {
    println!("--- 1. Matrix-Exponentialfunktion & Wurzeln ---");

    // Wir nehmen einen 2D-Rotations-Generator mit dem Winkel theta = 1.0 rad (ca. 57.3 Grad)
    let theta = 1.0f64;
    let generator = DynMatrix::new(2, 2, vec![0.0, -theta, theta, 0.0]).unwrap();

    // expm(Generator) MUSS die exakte 2D-Drehmatrix im Bogenmaß ergeben
    let rotation = expm(&generator).unwrap();
    println!("Matrix-Exponential von [[0, -1], [1, 0]]:");
    println!(
        "  Erhalten:   [{:.6}, {:.6}]",
        rotation[(0, 0)],
        rotation[(0, 1)]
    );
    println!(
        "              [{:.6}, {:.6}]",
        rotation[(1, 0)],
        rotation[(1, 1)]
    );
    println!("  Analytisch: [{:.6}, {:.6}]", theta.cos(), -theta.sin());
    println!("              [{:.6}, {:.6}]", theta.sin(), theta.cos());

    // Rückprobe über die Matrix-Quadratwurzel und den Logarithmus
    let wurzel = sqrtm(&rotation).unwrap();
    let log_matrix = logm(&rotation).unwrap();

    // Die Wurzel einer 1.0-Rad-Drehung muss eine 0.5-Rad-Drehung sein
    println!("\nMatrix-Wurzel (Halber Drehwinkel):");
    println!("  Erhalten (0.5 rad): cos = {:.6}", wurzel[(0, 0)]);
    println!("  Wahr (cos(0.5)):         = {:.6}", 0.5f64.cos());
    println!(
        "  Matrix-Hauptlogarithmus Fehler: {:e}",
        max_diff(&log_matrix, &generator)
    );

    println!("\n--- 2. LQR-Regelungstechnik (Riccati-Löser `care`) ---");
    // Ein inverses Pendel (Segway), das balanciert werden muss.
    // Zustand: [Position, Geschwindigkeit, Winkel, Winkelgeschwindigkeit]
    // A-Matrix beschreibt die Eigendynamik (instabil, da das Pendel umfällt)
    let a = DynMatrix::new(
        2,
        2,
        vec![
            0.0, 1.0, // Position wandert mit Geschwindigkeit
            10.0, 0.0, // Schwerkraft zieht den Winkel nach außen (instabil!)
        ],
    )
    .unwrap();

    // B-Matrix beschreibt, wie der Motor (Eingriff) auf das System wirkt
    let b = DynMatrix::new(
        2,
        1,
        vec![
            0.0, // Motor wirkt nicht direkt auf die Position
            1.0, // Motor beschleunigt den Winkel
        ],
    )
    .unwrap();

    // Q-Matrix: Strafgewichte für Abweichungen (wir wollen den Winkel stark strafen)
    let q = DynMatrix::new(2, 2, vec![1.0, 0.0, 0.0, 10.0]).unwrap();

    // R-Matrix: Strafgewicht für den Motoreinsatz (Energieverbrauch)
    let r = DynMatrix::new(1, 1, vec![1.0]).unwrap();

    // Der care-Löser berechnet über die Hamilton-Vorzeichenfunktion die Matrix X
    let x = care(&a, &b, &q, &r).unwrap();
    println!("Stabilisierungs-Matrix X aus dem Riccati-Löser:");
    println!("  X = [{:.4}, {:.4}]", x[(0, 0)], x[(0, 1)]);
    println!("      [{:.4}, {:.4}]", x[(1, 0)], x[(1, 1)]);

    // Das mathematische Gesetz (Das Residuum der algebraischen Riccati-Gleichung MUSS Null sein):
    // AᵀX + XA - X B R⁻¹ Bᵀ X + Q = 0
    let at_x = a.transpose().matmul(&x).unwrap();
    let x_a = x.matmul(&a).unwrap();
    let x_b = x.matmul(&b).unwrap();
    let r_inv = r.inverse().unwrap();
    let kkt_drift = x_b
        .matmul(&r_inv)
        .unwrap()
        .matmul(&x_b.transpose())
        .unwrap();

    // Residuum zusammenrechnen: (AᵀX + XA + Q) - (X B R⁻¹ Bᵀ X)
    let residual = at_x
        .as_slice()
        .iter()
        .zip(x_a.as_slice())
        .zip(q.as_slice())
        .zip(kkt_drift.as_slice())
        .map(|(((atx, xa), qi), drift)| (atx + xa + qi) - drift)
        .fold(0.0f64, |m, v| m.max(v.abs()));

    println!("\nUnbestechlicher KKT-Prüfstand:");
    println!("  Riccati-Residuum (Abweichung von Null): {:e}", residual);
    assert!(residual < 1e-8, "Riccati-Gleichung nicht exakt erfüllt!");
    println!("  -> Der Regelungs-Löser arbeitet mathematisch absolut fehlerfrei!");
}

#[cfg(not(all(feature = "alloc", any(feature = "std", feature = "libm"))))]
fn main() {
    eprintln!("Dieses Beispiel benötigt die Features `alloc` und `std` oder `libm`.");
}
