// Beispiel 9 – Nullstellensuche: Bisektion, Brent, Newton und Sekante im Vergleich
//
// 1. Die Gleichung cos(x) = x wird mit allen vier Verfahren gelöst; die Anzahl der Iterationen
//    zeigt die Unterschiede in der Geschwindigkeit.
// 2. Die Kepler-Gleichung E − e·sin(E) = M wird generisch für f32 und f64 gelöst.
//
// Funktionen wie cos und sin werden über `Real::cos(x)` aufgerufen, damit das gewählte Backend
// (std, libm) zum Einsatz kommt.
//
//   cargo run --example 09_roots

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::{
    Real,
    roots::{RootError, RootOptions, bisection, brent, newton, secant},
};

/// Löst die Kepler-Gleichung E − e·sin(E) = M für die exzentrische Anomalie E.
#[cfg(any(feature = "std", feature = "libm"))]
fn kepler<T: Real>(e: T, m: T) -> Result<T, RootError> {
    let f = |x: T| x - e * x.sin() - m;
    // E liegt immer zwischen M − e und M + e
    let r = brent(f, m - e, m + e, RootOptions::default())?;
    Ok(r.root)
}

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    let f = |x: f64| Real::cos(x) - x;
    let df = |x: f64| -Real::sin(x) - 1.0;
    let opts = RootOptions::default();

    println!("cos(x) = x  (exakt 0.7390851332151607)");
    let b = bisection(f, 0.0, 1.0, opts).unwrap();
    println!(
        "  Bisektion : {:.16}  {:>3} Iterationen",
        b.root, b.iterations
    );
    let br = brent(f, 0.0, 1.0, opts).unwrap();
    println!(
        "  Brent     : {:.16}  {:>3} Iterationen",
        br.root, br.iterations
    );
    let n = newton(f, df, 0.5, opts).unwrap();
    println!(
        "  Newton    : {:.16}  {:>3} Iterationen",
        n.root, n.iterations
    );
    let s = secant(f, 0.0, 1.0, opts).unwrap();
    println!(
        "  Sekante   : {:.16}  {:>3} Iterationen",
        s.root, s.iterations
    );

    // Fehlerfälle werden als Result gemeldet
    match brent(|x: f64| x * x + 1.0, -1.0, 1.0, opts) {
        Err(RootError::NotBracketed) => {
            println!("x² + 1: kein Vorzeichenwechsel, keine Nullstelle")
        }
        other => println!("unerwartet: {other:?}"),
    }

    println!("Kepler (e = 0.6, M = 1.2):");
    println!("  f64: E = {:.15}", kepler(0.6f64, 1.2).unwrap());
    println!("  f32: E = {:.7}", kepler(0.6f32, 1.2).unwrap());
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    eprintln!("Bitte mit --features std oder --features libm ausführen.");
}
