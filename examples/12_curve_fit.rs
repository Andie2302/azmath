// Beispiel 12 – Nichtlineare Kurvenanpassung und Gleichungssysteme
//
// 1. Exponentieller Zerfall y = a·exp(b·t) an verrauschte Messdaten anpassen
//    (Levenberg-Marquardt, mit analytischer und mit Differenzen-Jacobi-Matrix).
// 2. Ein Gauß-Peak mit drei Parametern; die Zahl der Messpunkte ist erst zur Laufzeit bekannt,
//    der Arbeitsspeicher kommt deshalb aus einem Vec.
// 3. Ein nichtlineares Gleichungssystem (Kreis und Gerade) mit Newton lösen.
//
// Funktionen wie exp werden über `Real::exp(x)` aufgerufen, damit das gewählte Backend
// (std, libm) zum Einsatz kommt.
//
//   cargo run --example 12_curve_fit

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::{
    Real,
    optimize::{
        OptimizeOptions, gauss_newton, gauss_newton_work_len, levenberg_marquardt,
        levenberg_marquardt_fd, levenberg_marquardt_work_len, newton_system, newton_work_len,
    },
};

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    let opts = OptimizeOptions::<f64>::default();

    // --- 1. Exponentieller Zerfall mit "Messrauschen" (deterministisch, damit das Ergebnis stabil ist)
    let t = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];
    let rauschen = [0.02, -0.03, 0.01, 0.04, -0.02, -0.01, 0.03, -0.02];
    let mut y = t.map(|t| 2.0 * Real::exp(-0.8 * t));
    for (yi, ri) in y.iter_mut().zip(rauschen) {
        *yi += ri;
    }
    let residual = |p: &[f64], r: &mut [f64]| {
        for i in 0..t.len() {
            r[i] = p[0] * Real::exp(p[1] * t[i]) - y[i];
        }
    };
    let jacobian = |p: &[f64], j: &mut [f64]| {
        for i in 0..t.len() {
            let e = Real::exp(p[1] * t[i]);
            j[2 * i] = e;
            j[2 * i + 1] = p[0] * t[i] * e;
        }
    };

    let mut p = [1.0, -0.2];
    let mut work = [0.0; levenberg_marquardt_work_len(8, 2)];
    let r = levenberg_marquardt(residual, jacobian, &mut p, 8, opts, &mut work).unwrap();
    println!("Zerfall (wahr: a = 2, b = -0.8):");
    println!(
        "  Levenberg-Marquardt: a = {:.5}, b = {:.5}  ({} Iterationen, Rest {:.4}, {:?})",
        p[0], p[1], r.iterations, r.residual_norm, r.termination
    );

    let mut q = [1.0, -0.2];
    levenberg_marquardt_fd(residual, &mut q, 8, opts, &mut work).unwrap();
    println!("  mit Differenzen-J  : a = {:.5}, b = {:.5}", q[0], q[1]);

    let mut g = [1.0, -0.2];
    let mut gwork = [0.0; gauss_newton_work_len(8, 2)];
    gauss_newton(residual, jacobian, &mut g, 8, opts, &mut gwork, &mut [0; 2]).unwrap();
    println!("  Gauß-Newton        : a = {:.5}, b = {:.5}", g[0], g[1]);

    // --- 2. Gauß-Peak, Anzahl der Messpunkte zur Laufzeit
    let n_punkte = 25;
    let ts: Vec<f64> = (0..n_punkte).map(|i| i as f64 * 0.25).collect();
    let modell =
        |p: &[f64], t: f64| p[0] * Real::exp(-(t - p[1]) * (t - p[1]) / (2.0 * p[2] * p[2]));
    let ys: Vec<f64> = ts.iter().map(|&t| modell(&[3.0, 2.5, 0.8], t)).collect();
    let peak_res = |p: &[f64], r: &mut [f64]| {
        for i in 0..ts.len() {
            r[i] = modell(p, ts[i]) - ys[i];
        }
    };
    let mut pp = [2.0, 2.0, 1.5];
    let mut work = vec![0.0; levenberg_marquardt_work_len(n_punkte, 3)];
    let r = levenberg_marquardt_fd(peak_res, &mut pp, n_punkte, opts, &mut work).unwrap();
    println!("Gauß-Peak (wahr: A = 3, c = 2.5, s = 0.8):");
    println!(
        "  A = {:.6}, c = {:.6}, s = {:.6}  ({} Iterationen)",
        pp[0], pp[1], pp[2], r.iterations
    );

    // --- 3. Nichtlineares Gleichungssystem: Kreis x² + y² = 4 und Gerade y = x
    let f = |v: &[f64], out: &mut [f64]| {
        out[0] = v[0] * v[0] + v[1] * v[1] - 4.0;
        out[1] = v[1] - v[0];
    };
    let jac = |v: &[f64], j: &mut [f64]| j.copy_from_slice(&[2.0 * v[0], 2.0 * v[1], -1.0, 1.0]);
    let mut x = [1.0, 0.5];
    let mut nwork = [0.0; newton_work_len(2)];
    let r = newton_system(f, jac, &mut x, opts, &mut nwork).unwrap();
    println!(
        "Kreis ∩ Gerade: ({:.12}, {:.12}), √2 = {:.12}  ({} Iterationen)",
        x[0],
        x[1],
        2f64.sqrt(),
        r.iterations
    );
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    eprintln!("Bitte mit --features std oder --features libm ausführen.");
}
