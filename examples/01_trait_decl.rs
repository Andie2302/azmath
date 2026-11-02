// Beispiel 1 – Universaliere Trait 'Real' deklariert

//
// Azmath bietet ein einzelnes Trait namens 'Real' für f32 und f64. Durch
// getrennte Implementierungen pro Typ (und je nach Cfg: std, libm, micromath)
// wird der backend-seitig specific routing der Functions erfolgt.

use azmath::Real;

fn main() {
    println!("Real trait declared for f32 and f64");
}
