//! Rangmangel erkennen und ein redundantes Ausgleichsproblem lösen.
#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    use azmath::linalg::{Matrix, Vector};
    use azmath::optimize::{OptimizeOptions, gauss_newton, gauss_newton_work_len};

    // Die zweite Spalte ist doppelt so groß wie die erste: nur ein Parameter ist identifizierbar.
    let a = Matrix::new([[1.0f64, 2.0], [2.0, 4.0], [0.0, 0.0]]);
    let b = Vector::new([3.0, 6.0, 2.0]);
    let qr = a.qr_pivoted().unwrap();
    println!(
        "Rang: {}, Spaltenreihenfolge: {:?}",
        qr.rank(),
        qr.permutation()
    );
    println!("Basislösung: {:?}", a.least_squares(&b).unwrap());
    println!(
        "Lösung kleinster Norm (SVD): {:?}",
        a.svd().unwrap().solve(&b)
    );
    println!("Restfehler: {}", qr.residual_norm(&b).unwrap());

    let residual = |x: &[f64], r: &mut [f64]| {
        r[0] = x[0] + 2.0 * x[1] - 3.0;
        r[1] = 2.0 * r[0];
    };
    let jacobian = |_x: &[f64], j: &mut [f64]| j.copy_from_slice(&[1.0, 2.0, 2.0, 4.0]);
    let mut x = [0.0, 0.0];
    let mut work = [0.0; gauss_newton_work_len(2, 2)];
    let mut permutation = [0; 2];
    let result = gauss_newton(
        residual,
        jacobian,
        &mut x,
        2,
        OptimizeOptions::default(),
        &mut work,
        &mut permutation,
    )
    .unwrap();
    println!("Gauß-Newton: {x:?}, Restfehler {}", result.residual_norm);
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    println!("Dieses Beispiel benötigt das Backend std oder libm.");
}
