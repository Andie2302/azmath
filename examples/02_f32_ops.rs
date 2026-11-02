// Beispiel 2 – Grundlegende Operationen für f32 mittels Real-Trait

//
// Diese Programm führt arithmetische Rechenoperationen auf f32 durch, indem sie die
// Operatoren del Trait verwendet:
//
// ├── Add, Sub, Mul, Div, Neg, Abs


use azmath::Real;

fn main() {
    let a: Real = 2.0f32;
    let b: Real = 3.0f32;
    println!("2.0f32 + 3.0f32 = {}", a + b);

    let mul: Real = a * 2.0f32;
    println!("2.0f32 * 2.0 = {}", mul);

    let div: Real = a / 2.0f32;
    println!("2.0f32 / 2.0 = {}", div);

    let neg: Real = -a;
    println!("negative 2.0f32 = {}", neg);

    let abs_val: Real = (-a).abs();
    println!("absolute of -2.0f32 = {}", abs_val);
}
