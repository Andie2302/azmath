// Beispiel 5 – Konvertierung mit `Real::from_f64`
//
// `from_f64` erlaubt es, Konstanten in generischen Formeln als f64-Literale
// zu schreiben. Für f32 wird (verlustbehaftet) gerundet, für f64 bleibt der
// Wert unverändert.
//
// Ausführen:  cargo run --example 05_convert --features std

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::Real;

/// Umrechnung Celsius → Fahrenheit: F = C · 9/5 + 32
#[cfg(any(feature = "std", feature = "libm"))]
fn celsius_zu_fahrenheit<T: Real>(c: T) -> T {
    c * T::from_f64(1.8) + T::from_f64(32.0)
}

/// Grad → Bogenmaß
#[cfg(any(feature = "std", feature = "libm"))]
fn grad_zu_rad<T: Real>(grad: T) -> T {
    grad * T::PI / T::from_f64(180.0)
}

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    let wert = 0.1f64;
    let als_f32 = <f32 as Real>::from_f64(wert);
    let als_f64 = <f64 as Real>::from_f64(wert);
    println!("0.1 als f32: {als_f32:.10}");
    println!("0.1 als f64: {als_f64:.10}");

    println!("37 °C = {} °F (f32)", celsius_zu_fahrenheit(37.0f32));
    println!("37 °C = {} °F (f64)", celsius_zu_fahrenheit(37.0f64));

    let rad = grad_zu_rad(90.0f64);
    println!("90° = {rad} rad, sin = {:.6}", Real::sin(rad));
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    eprintln!("Bitte mit --features std oder --features libm ausführen.");
}
