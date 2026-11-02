// Beispiel 16 – Eigenwerte und Singulärwertzerlegung
//
// 1. Alle Eigenwerte einer symmetrischen Matrix (Jacobi): die Hilbert-Matrix hat sehr kleine, aber
//    genau bestimmte Eigenwerte.
// 2. Potenzmethode matrixfrei: größter Eigenwert eines großen Operators (Poisson), und inverse Iteration
//    für einen Eigenwert nahe einem vorgegebenen Wert.
// 3. SVD: Hauptachsen einer Punktwolke (PCA), Rang und Konditionszahl, Lösung bei Rangmangel.
//
// Funktionen wie cos werden über `Real::cos(x)` aufgerufen, damit das gewählte Backend zum Einsatz kommt.
//
//   cargo run --example 16_eigen_svd

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::{
    Real,
    linalg::{
        FnOperator, Matrix, Vector,
        eigen::{power_iteration, power_iteration_work_len},
        iterative::IterOptions,
    },
};

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    use core::f64::consts::PI;

    // --- 1. Hilbert-Matrix H₅ (schlecht konditioniert, symmetrisch positiv definit)
    let h = Matrix::<f64, 5, 5>::from_fn(|i, j| 1.0 / (i + j + 1) as f64);
    let e = h.symmetric_eigen().unwrap();
    println!("Hilbert-Matrix H₅: {} Jacobi-Sweeps", e.sweeps);
    let werte: Vec<String> = e
        .values
        .as_slice()
        .iter()
        .map(|v| format!("{v:.4e}"))
        .collect();
    println!("  Eigenwerte (aufsteigend): {}", werte.join(", "));
    println!(
        "  Konditionszahl λmax/λmin = {:.3e}",
        e.values[4] / e.values[0]
    );

    // --- 2. Potenzmethode matrixfrei: Poisson-Operator mit n = 50
    const N: usize = 50;
    let poisson = FnOperator::new(N, N, |x: &[f64], y: &mut [f64]| {
        for i in 0..x.len() {
            let links = if i > 0 { x[i - 1] } else { 0.0 };
            let rechts = if i + 1 < x.len() { x[i + 1] } else { 0.0 };
            y[i] = 2.0 * x[i] - links - rechts;
        }
    });
    // Startvektor mit Pseudozufallseinträgen (kleiner Kongruenzgenerator). Ein glatter oder
    // periodischer Startvektor enthält den alternierenden dominanten Eigenvektor kaum, die Potenzmethode
    // würde dann bei einem anderen Eigenwert anhalten (das Residuum ist trotzdem klein).
    let mut zustand: u64 = 0x1234_5678_9ABC_DEF1;
    let mut x: [f64; N] = core::array::from_fn(|_| {
        zustand = zustand
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        0.5 + (zustand >> 11) as f64 / (1u64 << 53) as f64
    });
    let mut work = [0.0; power_iteration_work_len(N)];
    let r = power_iteration(
        &poisson,
        &mut x,
        IterOptions::new(1e-10, 200_000),
        &mut work,
    )
    .unwrap();
    println!(
        "Poisson n = {N}: größter Eigenwert {:.10} nach {} Iterationen (exakt {:.10})",
        r.eigenvalue,
        r.iterations,
        2.0 - 2.0 * Real::cos(N as f64 * PI / (N as f64 + 1.0))
    );
    // inverse Iteration: der Eigenwert von H₅ nahe 0.2
    let (lambda, _, iters) = h.eigenpair_near(0.2, IterOptions::new(1e-12, 100)).unwrap();
    println!("H₅: Eigenwert nahe 0.2 → {lambda:.10} nach {iters} Iterationen");

    // --- 3. SVD einer Punktwolke: Hauptachsen
    let punkte = Matrix::<f64, 6, 2>::new([
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
    ]);
    let mittel = [
        (0..6).map(|i| punkte[(i, 0)]).sum::<f64>() / 6.0,
        (0..6).map(|i| punkte[(i, 1)]).sum::<f64>() / 6.0,
    ];
    let zentriert = Matrix::<f64, 6, 2>::from_fn(|i, j| punkte[(i, j)] - mittel[j]);
    let svd = zentriert.svd().unwrap();
    println!("Punktwolke: Singulärwerte {:.4?}", svd.s.as_array());
    println!(
        "  erste Hauptachse ({:.3}, {:.3}), erklärt {:.1} % der Varianz",
        svd.v[(0, 0)],
        svd.v[(1, 0)],
        100.0 * svd.s[0].powi(2) / (svd.s[0].powi(2) + svd.s[1].powi(2))
    );

    // Rangmangel: dritte Spalte = erste + zweite; QR scheitert, die SVD liefert die Lösung kleinster Norm
    let a = Matrix::new([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 9.0],
        [7.0, 8.0, 15.0],
        [1.0, 0.0, 1.0],
    ]);
    let svd = a.svd().unwrap();
    println!(
        "Rangmangel: Rang {}, Konditionszahl {:.1e}",
        svd.rank(),
        svd.condition_number()
    );
    println!("  QR: {:?}", a.qr().map(|_| ()));
    let b = Vector::new([1.0, 2.0, 3.0, 4.0]);
    let x = svd.solve(&b);
    println!(
        "  SVD-Lösung kleinster Norm: {:.6?}, Residuum ‖Ax − b‖ = {:.4}",
        x.as_array(),
        (a * x - b).norm()
    );
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    eprintln!("Bitte mit --features std oder --features libm ausführen.");
}
