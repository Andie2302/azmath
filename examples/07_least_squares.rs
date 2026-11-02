// Beispiel 7 – Ausgleichsrechnung (QR) und symmetrisch positiv definite Systeme (Cholesky)
//
// Eine Ausgleichsgerade y = a + b·x wird durch verrauschte Messpunkte gelegt:
//  - per QR-Zerlegung (numerisch stabil, Standardweg),
//  - per Normalengleichungen AᵀA·x = Aᵀy mit Cholesky (schneller, aber ungenauer).
//
// Beide Verfahren brauchen sqrt; mit dem Backend `micromath` sind die Ergebnisse entsprechend
// grob. Für genaue Werte `std` oder `libm` verwenden.
//
//   cargo run --example 07_least_squares

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::linalg::{Matrix, Vector};

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    // Messpunkte (x, y) – etwa y ≈ 0.5 + 2x mit Rauschen
    let xs = [0.0f64, 1.0, 2.0, 3.0, 4.0, 5.0];
    let ys = [0.6f64, 2.4, 4.4, 6.3, 8.6, 10.4];

    // Entwurfsmatrix: eine Spalte mit Einsen (Achsenabschnitt), eine mit den x-Werten
    let a = Matrix::<f64, 6, 2>::from_fn(|i, j| if j == 0 { 1.0 } else { xs[i] });
    let y = Vector::new(ys);

    let qr = a.qr().expect("Spalten sind linear unabhängig");
    let beta = qr.solve(&y);
    println!("QR:        y = {:.4} + {:.4}·x", beta[0], beta[1]);
    println!("Residuum:  ‖Aβ − y‖ = {:.4}", qr.residual_norm(&y));

    // Normalengleichungen: AᵀA ist symmetrisch positiv definit -> Cholesky
    let at = a.transpose();
    let beta_n = (at * a).solve_spd(&(at * y)).expect("AᵀA ist SPD");
    println!("Cholesky:  y = {:.4} + {:.4}·x", beta_n[0], beta_n[1]);

    // Eine Parabel durch 5 Punkte: Vandermonde-Matrix mit Spalten 1, x, x²
    let px = [-2.0f64, -1.0, 0.0, 1.0, 2.0];
    let v = Matrix::<f64, 5, 3>::from_fn(|i, j| px[i].powi(j as i32));
    let py = Vector::new(px.map(|x| 1.0 - x + 0.5 * x * x));
    let c = v.least_squares(&py).unwrap();
    println!(
        "Parabel:   y = {:.4} + {:.4}·x + {:.4}·x²",
        c[0], c[1], c[2]
    );

    #[cfg(feature = "alloc")]
    dynamisch(&xs, &ys);
}

#[cfg(all(feature = "alloc", any(feature = "std", feature = "libm")))]
fn dynamisch(xs: &[f64], ys: &[f64]) {
    use azmath::linalg::{DynMatrix, DynVector};

    // Anzahl der Messpunkte erst zur Laufzeit bekannt
    let m = xs.len();
    let mut data = Vec::with_capacity(m * 2);
    for &x in xs {
        data.push(1.0);
        data.push(x);
    }
    let a = DynMatrix::new(m, 2, data).unwrap();
    let beta = a.least_squares(&DynVector::new(ys.to_vec())).unwrap();
    println!("Dynamisch: y = {:.4} + {:.4}·x", beta[0], beta[1]);
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    eprintln!("Bitte mit --features std oder --features libm ausführen.");
}
