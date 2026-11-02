// Beispiel 10 – BiCGSTAB für nichtsymmetrische Systeme
//
// Ein eindimensionales Konvektions-Diffusions-Problem führt auf eine nichtsymmetrische
// Tridiagonalmatrix (Konvektion verschiebt die Nebendiagonalen gegeneinander). CG ist dafür
// nicht geeignet, BiCGSTAB schon. Der Operator wird wieder matrixfrei als Closure angegeben.
//
// Hinweis: Mit wachsendem Konvektionsanteil `c` und wachsender Größe wird die Matrix schnell
// extrem schlecht konditioniert (exponentiell in n); deshalb hier ein kleines `c`.
//
//   cargo run --example 10_bicgstab

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::linalg::{
    FnOperator, LinalgError, Matrix, Vector,
    iterative::{IterOptions, bicgstab},
};

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    const N: usize = 100;
    let c = 0.05; // Konvektionsanteil
    let operator = FnOperator::new(N, N, move |x: &[f64], y: &mut [f64]| {
        for i in 0..x.len() {
            let left = if i > 0 { x[i - 1] } else { 0.0 };
            let right = if i + 1 < x.len() { x[i + 1] } else { 0.0 };
            y[i] = -(1.0 + c) * left + 2.0 * x[i] - (1.0 - c) * right;
        }
    });

    let b = [1.0f64; N];
    let mut x = [0.0f64; N];
    let mut work = [0.0f64; 8 * N];
    let res = bicgstab(
        &operator,
        &b,
        &mut x,
        IterOptions::new(1e-10, 1000),
        &mut work,
    )
    .unwrap();
    println!(
        "Konvektion-Diffusion, n = {N}: {} Iterationen, Residuum {:e}",
        res.iterations, res.residual_norm
    );
    println!(
        "x[0] = {:.6}, x[Mitte] = {:.6}, x[Ende] = {:.6}",
        x[0],
        x[N / 2],
        x[N - 1]
    );

    // Kleines Beispiel mit expliziter Matrix und der bequemen Methode
    let a = Matrix::new([[4.0f64, 1.0, 0.0], [2.0, 5.0, 1.0], [0.0, 3.0, 6.0]]);
    let rhs = Vector::new([1.0, 2.0, 3.0]);
    let (sol, info) = a.solve_bicgstab(&rhs, IterOptions::default()).unwrap();
    let direkt = a.solve(&rhs).unwrap();
    println!(
        "3x3: BiCGSTAB {:?} ({} Iterationen), LU {:?}",
        sol.as_array(),
        info.iterations,
        direkt.as_array()
    );

    // Verfahren, die zusammenbrechen können, melden das als Fehler
    let schief = Matrix::new([[0.0f64, 1.0], [-1.0, 0.0]]);
    match schief.solve_bicgstab(&Vector::new([1.0, 0.0]), IterOptions::default()) {
        Err(LinalgError::Breakdown) => {
            println!("Schiefsymmetrische Matrix: Breakdown (wie erwartet)")
        }
        other => println!("unerwartet: {other:?}"),
    }
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    eprintln!("Bitte mit --features std oder --features libm ausführen.");
}
