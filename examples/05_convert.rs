// Beispiel 5 – Konvertierung f64 -> Real

//
// Die azmath-Bibliothek stellt die Methoden Real::from_f64 auf, um einen f64-Wert nach Real zu wandeln.


use azmath::Real;

fn main() {
    let val: Real = azmath::Real::from_f64(7.5f64);
    println!("Converted 7.5f64 -> Real = {}", val);
}
