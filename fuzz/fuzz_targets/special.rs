#![no_main]
use azmath::special::{bessel_i, bessel_j, bessel_k, bessel_y, erf, erfc, gamma, ln_gamma};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 9 {
        return;
    }
    let bits = u64::from_le_bytes(data[..8].try_into().unwrap());
    let raw = f64::from_bits(bits);
    // Raw bit patterns include NaN, infinities, poles and subnormal arguments.
    // Large finite gamma arguments are outside this target's accuracy assertions.
    if !raw.is_finite() || raw.abs() < 0.125 {
        let e = erf(raw);
        if raw.is_nan() {
            assert!(e.is_nan());
        } else {
            assert!(e.is_finite() && e.abs() <= 1.0);
        }
        if raw.is_finite() {
            assert!(bessel_j(0, raw).is_finite());
            assert!(bessel_j(1, raw).is_finite());
        }
        let g = gamma(raw);
        if raw.is_nan() || raw == 0.0 || raw == f64::NEG_INFINITY {
            assert!(g.is_nan());
        } else {
            assert!(!g.is_nan());
        }
    }
    // Bound both order and argument: numerical kernels are not constant-time
    // in arbitrary untrusted orders. This keeps fuzz iterations affordable.
    let x = 0.125 + (bits % 1_000_000) as f64 * (7.875 / 1_000_000.0);
    let n = data[8] as u32 % 8;
    let g = gamma(x);
    let log = ln_gamma(x);
    assert!(g.is_finite() && log.is_finite());
    assert!((gamma(x + 1.0) - x * g).abs() <= 1e-11 * x * g);
    assert!((erf(x) + erfc(x) - 1.0).abs() < 1e-13);
    assert!(bessel_j(n, x).is_finite());
    assert!(bessel_i(n, x).is_finite());
    let y = bessel_y(n, x);
    let k = bessel_k(n, x);
    assert!(y.is_finite() && k.is_finite());
    let wronskian = bessel_i(n, x) * bessel_k(n + 1, x) + bessel_i(n + 1, x) * k;
    assert!((wronskian * x - 1.0).abs() < 1e-10);
    let xf = x as f32;
    let gf = gamma(xf);
    assert!(gf.is_finite() && ln_gamma(xf).is_finite());
    assert!((gamma(xf + 1.0) - xf * gf).abs() <= 1e-4 * xf * gf);
    assert!((erf(xf) + erfc(xf) - 1.0).abs() < 1e-5);
    assert!(bessel_j(n, xf).is_finite() && bessel_i(n, xf).is_finite());
    assert!(bessel_y(n, xf).is_finite() && bessel_k(n, xf).is_finite());
});
