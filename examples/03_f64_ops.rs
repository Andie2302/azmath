// Beispiel 3 – Operationen auf f64 über das `Real`-Trait
//
// Praktische Anwendung: Lösen einer quadratischen Gleichung ax² + bx + c = 0
// und Berechnen von e^x / ln(x) mit doppelter Genauigkeit.
// (micromath unterstützt kein f64 – hier wird `std` oder `libm` benötigt.)
//
// Ausführen:  cargo run --example 03_f64_ops --features std

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::Real;

/// Reelle Lösungen von ax² + bx + c = 0 (falls vorhanden).
#[cfg(any(feature = "std", feature = "libm"))]
fn quadratisch<T: Real>(a: T, b: T, c: T) -> Option<(T, T)> {
    let diskriminante = b * b - T::from_f64(4.0) * a * c;
    if diskriminante < T::ZERO {
        return None;
    }
    let w = diskriminante.sqrt();
    let zwei_a = T::from_f64(2.0) * a;
    Some(((-b + w) / zwei_a, (-b - w) / zwei_a))
}

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    // x² - 3x + 2 = 0  →  x = 2 und x = 1
    match quadratisch(1.0f64, -3.0, 2.0) {
        Some((x1, x2)) => println!("x² - 3x + 2 = 0  →  x1 = {x1}, x2 = {x2}"),
        None => println!("keine reelle Lösung"),
    }

    // x² + 1 = 0  →  keine reelle Lösung
    match quadratisch(1.0f64, 0.0, 1.0) {
        Some(_) => println!("unerwartet: Lösung gefunden"),
        None => println!("x² + 1 = 0       →  keine reelle Lösung"),
    }

    println!("e^1          = {}", Real::exp(1.0f64));
    println!("ln(e)        = {}", Real::ln(<f64 as Real>::E));
    println!("2^0.5        = {}", Real::powf(2.0f64, 0.5));
    println!("sqrt(2)      = {}", Real::sqrt(2.0f64));
    println!(
        "asin(1)      = {} (π/2 = {})",
        Real::asin(1.0f64),
        <f64 as Real>::PI / 2.0
    );
    println!("tan(π/4)     ≈ {:.12}", Real::tan(<f64 as Real>::PI / 4.0));
    println!("EPSILON      = {:e}", <f64 as Real>::EPSILON);
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    eprintln!("Bitte mit --features std oder --features libm ausführen.");
}
