#![cfg(any(feature = "std", feature = "libm"))]
use azmath::{Real, special::*};
use std::collections::BTreeMap;
#[path = "data/special_refs.rs"]
mod reference;

fn evaluate<T: Real>(name: &str, n: u32, x: T) -> T {
    match name {
        "gamma" => gamma(x),
        "ln_gamma" => ln_gamma(x),
        "erf" => erf(x),
        "erfc" => erfc(x),
        "bessel_j" => bessel_j(n, x),
        "bessel_i" => bessel_i(n, x),
        "bessel_y" => bessel_y(n, x),
        "bessel_k" => bessel_k(n, x),
        _ => unreachable!(),
    }
}
fn ordered(bits: u64, width: u32) -> u64 {
    let sign = 1 << (width - 1);
    let mask = if width == 32 {
        u32::MAX as u64
    } else {
        u64::MAX
    };
    if bits & sign != 0 {
        !bits & mask
    } else {
        bits | sign
    }
}

#[test]
fn high_precision_references() {
    let mut worst = BTreeMap::new();
    let mut failures = Vec::new();
    for &(name, n, width, xb, rb) in reference::CASES {
        let (actual, expected, ab, epsilon) = if width == 32 {
            let a = evaluate(name, n, f32::from_bits(xb as u32));
            (
                a as f64,
                f32::from_bits(rb as u32) as f64,
                a.to_bits() as u64,
                f32::EPSILON as f64,
            )
        } else {
            let a = evaluate(name, n, f64::from_bits(xb));
            (a, f64::from_bits(rb), a.to_bits(), f64::EPSILON)
        };
        let ulps = ordered(ab, width).abs_diff(ordered(rb, width));
        if actual.is_finite()
            && expected.is_finite()
            && !(name == "ln_gamma" && expected.abs() < 1e-6)
        {
            let value = worst.entry((name, width)).or_insert(0u64);
            *value = (*value).max(ulps);
        }
        let budget = match (width, name) {
            (32, "bessel_i") => 8,
            (32, "bessel_j" | "erf" | "gamma") => 32,
            (32, "bessel_y" | "bessel_k" | "erfc") => 64,
            (32, _) => 128,
            (_, "bessel_i" | "erf") => 16,
            (_, "bessel_j" | "bessel_y" | "bessel_k") => 64,
            (_, "erfc") => 256,
            (_, "ln_gamma") => 64,
            _ => 1024,
        };
        // ln|Gamma| vanishes at 1 and 2, making relative/ULP errors ill-conditioned.
        let near_log_zero = name == "ln_gamma"
            && expected.abs() < 1e-6
            && (actual - expected).abs() <= 8.0 * epsilon;
        if actual != expected
            && !(actual.is_finite() && expected.is_finite() && (ulps <= budget || near_log_zero))
        {
            failures.push(format!(
                "{name} n={n} f{width} input={xb:#x}: {actual:e} != {expected:e} ({ulps} ULP)"
            ));
        }
    }
    for ((name, width), ulps) in worst {
        println!("{name:12} f{width}: {ulps} ULP");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
