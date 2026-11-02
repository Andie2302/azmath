// Beispiel 19 – Rotationen mit Quaternionen
//
// 1. Einen Vektor mit einem Quaternion drehen und mit der Drehmatrix vergleichen.
// 2. Interpolation zwischen zwei Lagen (slerp): gleichmäßige Winkelschritte.
// 3. Euler-Winkel nur für Ein-/Ausgabe: Gimbal Lock bei Nickwinkel 90° erkennen.
// 4. Eine durch Rundung verdorbene Drehmatrix mit der Polarzerlegung reparieren.
//
//   cargo run --example 19_rotations

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::{
    geometry::{EulerZyx, Quaternion},
    linalg::{Matrix, Vector},
};

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    use core::f64::consts::FRAC_PI_2;

    // --- 1. 90° um z, dann der Vektor (1, 0, 0)
    let q = Quaternion::from_axis_angle(Vector::new([0.0, 0.0, 1.0]), FRAC_PI_2).unwrap();
    let v = Vector::new([1.0, 0.0, 0.0]);
    let a = q.rotate(v);
    let b = q.to_rotation_matrix() * v;
    println!(
        "Quaternion: ({:.3}, {:.3}, {:.3}), Matrix: ({:.3}, {:.3}, {:.3})",
        a[0], a[1], a[2], b[0], b[1], b[2]
    );

    // --- 2. slerp zwischen 0° und 120° um die x-Achse
    let start = Quaternion::IDENTITY;
    let end =
        Quaternion::from_axis_angle(Vector::new([1.0, 0.0, 0.0]), 120f64.to_radians()).unwrap();
    for k in 0..=4 {
        let t = k as f64 / 4.0;
        let (_, winkel) = start.slerp(end, t).to_axis_angle().unwrap();
        println!("t = {t:.2}: Winkel {:6.2}°", winkel.to_degrees());
    }

    // --- 3. Euler-Winkel und Gimbal Lock
    for nick in [30.0f64, 90.0] {
        let q = Quaternion::from_euler_zyx(EulerZyx::new(
            40f64.to_radians(),
            nick.to_radians(),
            20f64.to_radians(),
        ));
        let e = q.to_euler_zyx().unwrap();
        println!(
            "Nicken {nick:>4}°: Gieren {:6.1}°, Nicken {:5.1}°, Rollen {:5.1}°, Gimbal Lock: {}",
            e.yaw.to_degrees(),
            e.pitch.to_degrees(),
            e.roll.to_degrees(),
            e.gimbal_lock
        );
    }

    // --- 4. Drehmatrix nach vielen Multiplikationen: Drift beheben
    let step = Quaternion::from_axis_angle(Vector::new([1.0, 2.0, 3.0]), 0.01)
        .unwrap()
        .to_rotation_matrix();
    let mut r = Matrix::<f64, 3, 3>::identity();
    for _ in 0..1000 {
        r = r * step + Matrix::from_fn(|i, j| 1e-9 * (i as f64 - j as f64)); // Rundungsdrift (simuliert)
    }
    println!(
        "Orthogonalitätsfehler vorher {:.1e}",
        r.orthogonality_error()
    );
    let fixed = r.nearest_rotation().unwrap();
    println!(
        "                      nachher {:.1e}",
        fixed.orthogonality_error()
    );
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    println!("Dieses Beispiel braucht das Feature `std` oder `libm`.");
}
