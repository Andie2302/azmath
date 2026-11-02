// Beispiel 4 – Wahl des Backends über Feature-Flags
//
// Die Funktionen werden je nach Feature an ein Backend delegiert
// (Priorität: std → libm → micromath):
//
//   cargo run --example 04_backends --features std
//   cargo run --example 04_backends --features libm
//   cargo run --example 04_backends --features micromath   (nur f32)
//
// Die Ergebnisse unterscheiden sich leicht: micromath ist eine schnelle
// Näherung und weniger genau als std/libm.

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
use azmath::Real;

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn backend() -> &'static str {
    if cfg!(feature = "std") {
        "std"
    } else if cfg!(feature = "libm") {
        "libm"
    } else {
        "micromath"
    }
}

#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
fn main() {
    println!("Aktives Backend: {}", backend());

    let x = 0.7f32;
    println!("f32 sqrt({x})  = {:.7}", Real::sqrt(x));
    println!("f32 sin({x})   = {:.7}", Real::sin(x));
    println!("f32 ln({x})    = {:.7}", Real::ln(x));
    println!("f32 exp({x})   = {:.7}", Real::exp(x));

    #[cfg(any(feature = "std", feature = "libm"))]
    {
        let y = 0.7f64;
        println!("f64 sqrt({y})  = {:.15}", Real::sqrt(y));
        println!("f64 sin({y})   = {:.15}", Real::sin(y));
    }
    #[cfg(not(any(feature = "std", feature = "libm")))]
    println!("f64: mit micromath nicht verfügbar (std oder libm nötig)");
}

#[cfg(not(any(feature = "std", feature = "libm", feature = "micromath")))]
fn main() {
    eprintln!("Kein Backend aktiv. Bitte --features std, libm oder micromath angeben.");
}
