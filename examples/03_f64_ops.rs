// Beispiel 3 – Grundlegende Operationen für f64 mittels Real-Trait

//
// Dieses Programm demonstriert dieselben Operationen auf f64.


use azmath::Real;

fn main() {
    let x: Real = 5.0f64;
    let y: Real = 4.0f64;
    println!("5.0f64 + 4.0f64 = {}", x + y);

    let sq: Real = x.sqrt();
    println!("sqrt(5.0f64) = {}", sq);

    let neg: Real = -3.5f64;
    println!("-3.5f64 = {}", neg);
}
