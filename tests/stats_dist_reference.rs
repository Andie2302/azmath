#![cfg(any(feature = "std", feature = "libm"))]
use azmath::{Real, stats::dist::*};
use std::collections::BTreeMap;
#[path = "data/dist_refs.rs"]
mod reference;
fn evaluate<T: Real>(name: &str, x: T, a: T, b: T) -> T {
    macro_rules! family {($cdf:ident,$sf:ident,$quantile:ident,$isf:ident,$($p:expr),*)=>{
        if name.ends_with("_cdf") {$cdf(x,$($p),*)}else if name.ends_with("_sf") {$sf(x,$($p),*)}
        else if name.ends_with("_quantile") {$quantile(x,$($p),*)}else{$isf(x,$($p),*)}
    };}
    if name.starts_with("normal_") {
        family!(
            normal_cdf_with,
            normal_sf_with,
            normal_quantile_with,
            normal_isf_with,
            a,
            b
        )
    } else if name.starts_with("exponential_") {
        family!(
            exponential_cdf,
            exponential_sf,
            exponential_quantile,
            exponential_isf,
            a
        )
    } else if name.starts_with("uniform_") {
        family!(uniform_cdf, uniform_sf, uniform_quantile, uniform_isf, a, b)
    } else if name.starts_with("gamma_") {
        family!(gamma_cdf, gamma_sf, gamma_quantile, gamma_isf, a, b)
    } else if name.starts_with("chi2_") {
        family!(chi2_cdf, chi2_sf, chi2_quantile, chi2_isf, a)
    } else if name.starts_with("beta_") {
        family!(beta_cdf, beta_sf, beta_quantile, beta_isf, a, b)
    } else if name.starts_with("student_t_") {
        family!(
            student_t_cdf,
            student_t_sf,
            student_t_quantile,
            student_t_isf,
            a
        )
    } else {
        family!(f_cdf, f_sf, f_quantile, f_isf, a, b)
    }
}
fn ordered(b: u64, w: u32) -> u64 {
    let sign = 1 << (w - 1);
    if b & sign == 0 {
        b | sign
    } else {
        !b & if w == 32 { u32::MAX as u64 } else { u64::MAX }
    }
}
#[test]
fn high_precision_distribution_references() {
    let mut worst = BTreeMap::new();
    let mut failures = Vec::new();
    for &(name, width, ab, bb, xb, rb) in reference::CASES {
        let (actual, expected, bits) = if width == 32 {
            let actual = evaluate(
                name,
                f32::from_bits(xb as u32),
                f32::from_bits(ab as u32),
                f32::from_bits(bb as u32),
            );
            (
                actual as f64,
                f32::from_bits(rb as u32) as f64,
                actual.to_bits() as u64,
            )
        } else {
            let actual = evaluate(
                name,
                f64::from_bits(xb),
                f64::from_bits(ab),
                f64::from_bits(bb),
            );
            (actual, f64::from_bits(rb), actual.to_bits())
        };
        let ulps = ordered(bits, width).abs_diff(ordered(rb, width));
        if actual.is_finite() && expected.is_finite() {
            let v = worst.entry((name, width)).or_insert(0u64);
            *v = (*v).max(ulps);
        }
        let budget = if name.starts_with("uniform_") {
            4
        } else if name.starts_with("exponential_")
            || name == "normal_quantile"
            || name == "normal_isf"
        {
            8
        } else if width == 32 {
            128
        } else {
            1024
        };
        if actual != expected && !(actual.is_finite() && expected.is_finite() && ulps <= budget) {
            failures.push(format!("{name} f{width} a={ab:#x} b={bb:#x} x={xb:#x}: {actual:e} != {expected:e} ({ulps} ULP)"));
        }
    }
    for ((name, width), ulps) in worst {
        println!("{name:24} f{width}: {ulps} ULP");
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
