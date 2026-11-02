// Beispiel 4 – Wirkung der Feature Flag(s) auf den gewählten Backend

//
// •default  – keine zusätzlichen Features → die Standardbibliothek wird genutzt.
//
//   --features=libm  – aktiviere die libm-Bibliothek für sqrt, sin, cos, …
//
//   --features=micromath – bevorzuge micromath::F64Ext (für f64) bzw. F32Ext (für f32).

use azmath::Real;

fn main() {
    let a: Real = 1.0f64;
    println!("{} + 2.0f64 = {}", a, 2.0f64);

    // Test für die libm-Bibliothek für fx64
    let b: Real = 1.0;
    println!("sqrt(1.0f64) via libm (feature=libm) = {}", b.sqrt());
}
