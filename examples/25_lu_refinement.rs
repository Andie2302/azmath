//! LU-Verfeinerung, Rückwärtsfehler und Konditionsschätzung.
#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    use azmath::linalg::{Matrix, RefinementOptions, Vector};
    let a = Matrix::new([[3.0f32, 1.0, 7.0], [2.0, -8.0, 1.0], [5.0, 3.0, -2.0]]);
    let b = Vector::new([1.0, 2.0, -3.0]);
    let lu = a.lu().unwrap();
    let (x, report) = lu
        .solve_refined(
            &b,
            RefinementOptions {
                tolerance: 0.0,
                max_iter: 8,
            },
        )
        .unwrap();
    println!("Lösung: {x:?}");
    println!(
        "Rückwärtsfehler: {} -> {} ({:?})",
        report.initial_backward_error, report.backward_error, report.termination
    );
    println!("Geschätztes 1/κ∞: {}", lu.rcond_inf().unwrap());
    // Ein kleiner Rückwärtsfehler garantiert bei hoher Kondition keinen kleinen Lösungsfehler.
    let h = Matrix::<f64, 6, 6>::from_fn(|i, j| 1.0 / ((i + j + 1) as f64));
    println!(
        "Hilbert-Matrix: geschätztes 1/κ∞ = {}",
        h.lu().unwrap().rcond_inf().unwrap()
    );
}
#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    println!("Dieses Beispiel benötigt std oder libm.");
}
