// Beispiel 6 – Lineare Gleichungssysteme mit dem LU-Löser
//
// Zeigt feste Matrizen (Stack, ohne Allokator), Determinante, Inverse und die
// Wiederverwendung einer LU-Zerlegung für mehrere rechte Seiten. Mit dem
// Feature `alloc` kommt zusätzlich die dynamische Variante dazu:
//
//   cargo run --example 06_linear_system
//   cargo run --example 06_linear_system --features alloc

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
use azmath::linalg::{LinalgError, Matrix, Vector};

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn main() {
    // Gleichungssystem:  2x +  y -  z =   8
    //                   -3x -  y + 2z = -11
    //                   -2x +  y + 2z =  -3      ->  x = 2, y = 3, z = -1
    let a = Matrix::new([[2.0f32, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]);
    let b = Vector::new([8.0f32, -11.0, -3.0]);

    let x = a.solve(&b).expect("Matrix ist regulär");
    println!("Lösung: x = {}, y = {}, z = {}", x[0], x[1], x[2]);

    let residuum = a * x - b;
    println!("Größter Fehler |Ax - b|: {:e}", residuum.max_abs());
    println!("Determinante: {}", a.determinant());

    let inv = a.inverse().expect("Matrix ist regulär");
    println!("Inverse: {:?}", inv.as_array());

    // Eine Zerlegung, mehrere rechte Seiten
    let lu = a.lu().unwrap();
    for rhs in [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
        println!(
            "A⁻¹ · {:?} = {:?}",
            rhs,
            lu.solve(&Vector::new(rhs)).as_array()
        );
    }

    // Singuläre Matrix (zweite Zeile = 2 · erste Zeile)
    let s = Matrix::new([[1.0f32, 2.0], [2.0, 4.0]]);
    match s.solve(&Vector::new([1.0, 2.0])) {
        Err(LinalgError::Singular) => println!("Matrix ist singulär – kein eindeutiges Ergebnis"),
        other => println!("unerwartet: {other:?}"),
    }

    #[cfg(feature = "alloc")]
    dynamisch();
}

#[cfg(all(
    feature = "alloc",
    any(feature = "std", feature = "libm", feature = "micromath")
))]
fn dynamisch() {
    use azmath::linalg::{DynMatrix, DynVector};

    // Größe erst zur Laufzeit bekannt
    let n = 4;
    let mut data = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            data[i * n + j] = if i == j { 4.0 } else { 1.0 };
        }
    }
    let a = DynMatrix::new(n, n, data).unwrap();
    let b = DynVector::new(vec![1.0f32; n]);
    let x = a.solve(&b).unwrap();
    println!("Dynamisch ({n}x{n}): x = {:?}", x.as_slice());
}

#[cfg(not(any(feature = "std", feature = "libm", feature = "micromath")))]
fn main() {
    eprintln!("Bitte mit --features std, libm oder micromath ausführen.");
}
