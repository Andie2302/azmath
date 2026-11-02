// Beispiel 2 – Operationen auf f32 über das `Real`-Trait
//
// Zeigt Arithmetik, Konstanten, Vergleiche sowie Rundung und Trigonometrie.
// Läuft mit jedem Backend (std, libm oder micromath).
//
// Ausführen:  cargo run --example 02_f32_ops --features std

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
use azmath::Real;

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn main() {
    let a = 2.0f32;
    let b = 3.0f32;

    println!("a + b = {}", a + b);
    println!("a - b = {}", a - b);
    println!("a * b = {}", a * b);
    println!("a / b = {:.5}", a / b);
    println!("-a    = {}", -a);

    // Trait-Methoden explizit über `Real` aufrufen
    println!("abs(-a)         = {}", Real::abs(-a));
    println!("min(a, b)       = {}", Real::min(a, b));
    println!("max(a, b)       = {}", Real::max(a, b));
    println!("floor(2.7)      = {}", Real::floor(2.7f32));
    println!("ceil(2.1)       = {}", Real::ceil(2.1f32));
    println!("powi(a, 10)     = {}", Real::powi(a, 10));
    println!("is_nan(0/0)     = {}", Real::is_nan(f32::ZERO / f32::ZERO));
    println!("is_infinite(1/0) = {}", Real::is_infinite(f32::ONE / f32::ZERO));

    // Trigonometrie (Winkel in Bogenmaß)
    let halb_pi = <f32 as Real>::PI / 2.0;
    println!("sin(π/2) ≈ {:.4}", Real::sin(halb_pi));
    println!("cos(0)   = {:.4}", Real::cos(0.0f32));
    println!("atan2(1, 1) ≈ {:.4} (π/4 = {:.4})", Real::atan2(1.0f32, 1.0f32), <f32 as Real>::PI / 4.0);
}

#[cfg(not(any(feature = "std", feature = "libm", feature = "micromath")))]
fn main() {
    eprintln!("Bitte mit --features std, libm oder micromath ausführen.");
}
