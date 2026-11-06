#![cfg(any(feature = "std", feature = "libm"))]
use azmath::{Real, special::*};

fn close<T: Real + std::fmt::Debug>(a: T, b: T, scale: T) {
    assert!(
        a.is_finite() && b.is_finite(),
        "non-finite identity: {a:?}, {b:?}"
    );
    assert!(
        (a - b).abs() <= T::from_f64(512.0) * T::EPSILON * scale,
        "identity: {a:?} != {b:?}"
    );
}
fn identities<T: Real + std::fmt::Debug>() {
    let mut state = 0x7c4a_129b_695e_32d1u64;
    for i in 0..512 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        let x = T::from_f64(0.125 + 7.875 * u);
        let n = (i % 6 + 1) as u32;
        let g = gamma(x);
        close(gamma(x + T::ONE), x * g, (x * g).abs());
        close(erf(-x), -erf(x), T::ONE);
        close(erf(x) + erfc(x), T::ONE, T::ONE);
        let a = T::from_f64(0.25 + 8.0 * (1.0 - u));
        let p = gamma_p(a, x);
        let q = gamma_q(a, x);
        assert!(p >= T::ZERO && p <= T::ONE && q >= T::ZERO && q <= T::ONE);
        close(p + q, T::ONE, T::ONE);
        let j = bessel_j(n, x);
        let jm = bessel_j(n - 1, x);
        let jp = bessel_j(n + 1, x);
        close(
            jm + jp,
            T::from_f64(2.0 * n as f64) / x * j,
            jm.abs() + jp.abs() + j.abs(),
        );
        close(bessel_j(n, -x), if n % 2 == 0 { j } else { -j }, j.abs());
        let iv = bessel_i(n, x);
        let im = bessel_i(n - 1, x);
        let ip = bessel_i(n + 1, x);
        close(
            im - ip,
            T::from_f64(2.0 * n as f64) / x * iv,
            im.abs() + ip.abs() + iv.abs(),
        );
    }
}
#[test]
fn identities_f32() {
    identities::<f32>();
}
#[test]
fn identities_f64() {
    identities::<f64>();
}

#[test]
fn boundary_regressions() {
    assert_eq!(gamma(f64::INFINITY), f64::INFINITY);
    assert_eq!(gamma(f32::INFINITY), f32::INFINITY);
    assert_eq!(ln_gamma(f32::INFINITY), f32::INFINITY);
    assert_eq!(bessel_y(0, f64::INFINITY), 0.0);
    assert_eq!(bessel_y(0, f32::INFINITY), 0.0);
    assert_eq!(bessel_k(0, f64::INFINITY), 0.0);
    assert_eq!(bessel_k(0, f32::INFINITY), 0.0);
    assert_eq!(ln_gamma(f64::INFINITY), f64::INFINITY);
    for x in [f64::NAN, 0.0, -1.0, f64::NEG_INFINITY] {
        assert!(gamma(x).is_nan());
    }
    for x in [f32::NAN, 0.0, -1.0, f32::NEG_INFINITY] {
        assert!(gamma(x).is_nan());
    }
    for x in [1e-200f64, -1e-200, 1e-30, -1e-30] {
        close(erf(x) / x, 2.0 / core::f64::consts::PI.sqrt(), 1.0);
        assert_eq!(bessel_j(0, x), 1.0);
        close(bessel_j(1, x) / x, 0.5, 1.0);
    }
    for x in [1e-30f32, -1e-30] {
        close(erf(x) / x, 2.0 / core::f32::consts::PI.sqrt(), 1.0);
        assert_eq!(bessel_j(0, x), 1.0);
        close(bessel_j(1, x) / x, 0.5, 1.0);
    }
    assert!(gamma(-171.5f64).is_sign_positive());
    assert!(gamma(-171.5f64) > 0.0);
}
