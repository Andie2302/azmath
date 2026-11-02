// Beispiel 1 – Generischer Code über das `Real`-Trait
//
// `Real` ist ein Trait, das für `f32` und `f64` implementiert ist. Formeln
// werden einmal generisch geschrieben und funktionieren für beide Typen.
//
// Ausführen:  cargo run --example 01_generic_functions --features std

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::Real;

/// Fläche eines Kreises: A = π · r²
#[cfg(any(feature = "std", feature = "libm"))]
fn kreisflaeche<T: Real>(r: T) -> T {
    T::PI * r * r
}

/// Länge der Hypotenuse (Satz des Pythagoras).
#[cfg(any(feature = "std", feature = "libm"))]
fn hypotenuse<T: Real>(a: T, b: T) -> T {
    (a * a + b * b).sqrt()
}

/// Abstand zweier Punkte in der Ebene.
#[cfg(any(feature = "std", feature = "libm"))]
fn abstand<T: Real>(p: (T, T), q: (T, T)) -> T {
    hypotenuse(q.0 - p.0, q.1 - p.1)
}

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    println!("Kreisfläche r=2 (f32): {}", kreisflaeche(2.0f32));
    println!("Kreisfläche r=2 (f64): {}", kreisflaeche(2.0f64));

    println!("Hypotenuse 3/4 (f32): {}", hypotenuse(3.0f32, 4.0f32));
    println!("Hypotenuse 3/4 (f64): {}", hypotenuse(3.0f64, 4.0f64));

    println!(
        "Abstand (0,0)-(6,8): {}",
        abstand((0.0f64, 0.0f64), (6.0, 8.0))
    );
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    eprintln!("Bitte mit --features std oder --features libm ausführen.");
}
