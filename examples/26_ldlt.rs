//! Symmetrische indefinite Systeme, Blockpivots und Trägheit.
#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    use azmath::linalg::{Matrix, Vector};
    // Kleinstes Sattelpunkt-System: Cholesky scheitert, ein 2×2-Pivot funktioniert.
    let a = Matrix::new([[0.0f64, 2.0], [2.0, 0.0]]);
    let ld = a.ldlt().unwrap();
    println!(
        "Permutation: {:?}, Blöcke: {:?}",
        ld.permutation(),
        ld.block_sizes()
    );
    println!("Trägheit: {:?}", ld.inertia());
    println!("Lösung: {:?}", ld.solve(&Vector::new([6.0, 4.0])).unwrap());
    let l = ld.l();
    let reconstructed = l * ld.d().unwrap() * l.transpose();
    assert_eq!(reconstructed, a);
}
#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    println!("Dieses Beispiel benötigt std oder libm.");
}
