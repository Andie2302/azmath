//! Skalierte Gleichungen mit bekannten Zielwerten (synthetisches Modell, kein IAPWS).
//! cargo run --example 23_scaled_nonlinear_system
#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    use azmath::optimize::{OptimizeOptions, newton_system, newton_work_len};

    // Einfaches Testmodell: p = 1000*T*rho, h = 2000*T + 100*rho.
    // Bekannte Lösung: T = 500 K, rho = 20 kg/m³.
    let target_p = 10_000_000.0;
    let target_h = 1_002_000.0;
    let state = |x: &[f64]| (500.0 * x[0], 20.0 * x[1]);
    let properties = |temperature: f64, density: f64| {
        (
            1000.0 * temperature * density,
            2000.0 * temperature + 100.0 * density,
        )
    };
    // Auch die Unbekannten sind dimensionslos: x = (T/500, rho/20).
    let residual = |x: &[f64], r: &mut [f64]| {
        let (temperature, density) = state(x);
        let (p, h) = properties(temperature, density);
        r[0] = (p - target_p) / target_p;
        r[1] = (h - target_h) / target_h;
    };
    // Die Jacobi-Matrix muss dieselbe Skalierung wie die Residuen verwenden.
    let jacobian = |x: &[f64], j: &mut [f64]| {
        j.copy_from_slice(&[x[1], x[0], 1_000_000.0 / target_h, 2000.0 / target_h]);
    };
    let mut x = [450.0 / 500.0, 25.0 / 20.0];
    let mut work = [0.0; newton_work_len(2)];
    let options = OptimizeOptions::with_tolerance(1e-12);
    let result = newton_system(residual, jacobian, &mut x, options, &mut work).unwrap();

    // Unabhängige Gegenprobe in den ursprünglichen Einheiten.
    let (temperature, density) = state(&x);
    let (p, h) = properties(temperature, density);
    assert!((temperature - 500.0).abs() < 1e-8);
    assert!((density - 20.0).abs() < 1e-8);
    assert!((p - target_p).abs() <= target_p * options.tol_f);
    assert!((h - target_h).abs() <= target_h * options.tol_f);
    println!("T = {temperature:.10} K, rho = {density:.10} kg/m³");
    println!("p = {p:.6} Pa, h = {h:.6} J/kg");
    println!(
        "Skaliertes Residuum: {:e}, {} Iterationen",
        result.residual_norm, result.iterations
    );
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {}
