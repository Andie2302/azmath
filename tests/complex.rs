//! Tests für komplexe Zahlen.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::complex::Complex;

type C = Complex<f64>;

fn close(a: C, b: C, tol: f64) {
    assert!((a - b).abs() <= tol, "erwartet {b:?}, erhalten {a:?}");
}

#[test]
fn basic_arithmetic() {
    let a = C::new(3.0, 4.0);
    let b = C::new(1.0, -2.0);
    assert_eq!(a + b, C::new(4.0, 2.0));
    assert_eq!(a - b, C::new(2.0, 6.0));
    assert_eq!(a * b, C::new(11.0, -2.0));
    assert_eq!(-a, C::new(-3.0, -4.0));
    assert_eq!(C::I * C::I, C::new(-1.0, 0.0));
    assert_eq!(a * 2.0, C::new(6.0, 8.0));
    assert_eq!(a / 2.0, C::new(1.5, 2.0));
    assert_eq!(a + 1.0, C::new(4.0, 4.0));
    assert_eq!(C::from(2.0), C::new(2.0, 0.0));
}

#[test]
fn division_is_inverse_of_multiplication() {
    let a = C::new(3.0, 4.0);
    for b in [
        C::new(1.0, -2.0),
        C::new(-0.5, 7.0),
        C::new(1e-3, 1.0),
        C::new(5.0, 0.0),
    ] {
        close((a * b) / b, a, 1e-14 * a.abs());
        close(b * b.recip(), C::ONE, 1e-14);
    }
}

#[test]
fn division_does_not_overflow_in_the_denominator() {
    let big = C::new(1e200, 1e200);
    let q = big / big;
    close(q, C::ONE, 1e-14);
}

#[test]
fn abs_arg_conj_polar() {
    let z = C::new(3.0, 4.0);
    assert_eq!(z.abs(), 5.0);
    assert_eq!(z.norm_sqr(), 25.0);
    assert_eq!(z.conj(), C::new(3.0, -4.0));
    assert!((z * z.conj() - C::from_real(25.0)).abs() < 1e-14);
    let (r, phi) = C::new(-1.0, 0.0).to_polar();
    assert_eq!(r, 1.0);
    assert!((phi - core::f64::consts::PI).abs() < 1e-15);
    close(
        C::from_polar(2.0, 0.7),
        C::new(2.0 * 0.7f64.cos(), 2.0 * 0.7f64.sin()),
        1e-15,
    );
}

#[test]
fn sqrt_principal_branch_and_cancellation() {
    close(C::new(-4.0, 0.0).sqrt(), C::new(0.0, 2.0), 1e-15);
    close(C::new(3.0, 4.0).sqrt(), C::new(2.0, 1.0), 1e-15);
    close(C::new(3.0, -4.0).sqrt(), C::new(2.0, -1.0), 1e-15);
    close(C::new(-3.0, 4.0).sqrt(), C::new(1.0, 2.0), 1e-15);
    close(C::new(-3.0, -4.0).sqrt(), C::new(1.0, -2.0), 1e-15);
    assert_eq!(C::ZERO.sqrt(), C::ZERO);
    // keine Auslöschung bei re ≪ 0 und winzigem im
    let w = C::new(-1.0, 1e-20).sqrt();
    assert!((w.re - 5e-21).abs() < 1e-30 && (w.im - 1.0).abs() < 1e-15);
    for z in [C::new(0.3, -2.1), C::new(-7.0, 0.01), C::new(1e-8, 1e-8)] {
        let s = z.sqrt();
        close(s * s, z, 1e-14 * z.abs().max(1.0));
        assert!(s.re >= 0.0);
    }
}

#[test]
fn exp_ln_roundtrip_and_euler() {
    let pi = core::f64::consts::PI;
    close(C::new(0.0, pi).exp(), C::new(-1.0, 0.0), 1e-15);
    for z in [C::new(0.5, 1.2), C::new(-2.0, -0.4), C::new(3.0, 3.0)] {
        close(z.exp().ln(), z, 1e-14);
        close(z.ln().exp(), z, 1e-14 * z.abs());
    }
    close(C::new(-1.0, 0.0).ln(), C::new(0.0, pi), 1e-15);
}

#[test]
fn powers() {
    let z = C::new(1.0, 1.0);
    close(z.powi(2), C::new(0.0, 2.0), 1e-15);
    close(z.powi(0), C::ONE, 0.0);
    close(z.powi(-2), C::new(0.0, -0.5), 1e-15);
    close(z.powi(8), C::new(16.0, 0.0), 1e-13);
    close(z.powf(2.0), z.powi(2), 1e-14);
    close(z.powf(0.5), z.sqrt(), 1e-14);
    assert_eq!(C::ZERO.powf(2.0), C::ZERO);
    assert!(C::ZERO.powf(-1.0).is_nan());
}

#[test]
fn trig_identities() {
    let z = C::new(0.7, -0.3);
    close(z.sin() * z.sin() + z.cos() * z.cos(), C::ONE, 1e-14);
    close(z.cosh() * z.cosh() - z.sinh() * z.sinh(), C::ONE, 1e-14);
    // sin(iy) = i·sinh(y)
    close(C::new(0.0, 0.9).sin(), C::new(0.0, 0.9f64.sinh()), 1e-15);
}

#[test]
fn nan_and_finite() {
    assert!(C::new(f64::NAN, 0.0).is_nan());
    assert!(!C::new(1.0, 2.0).is_nan());
    assert!(C::new(1.0, 2.0).is_finite());
    assert!(!C::new(f64::INFINITY, 0.0).is_finite());
}

#[test]
fn works_for_f32() {
    let z = Complex::<f32>::new(3.0, 4.0);
    assert_eq!(z.abs(), 5.0);
    let s = z.sqrt();
    assert!((s.re - 2.0).abs() < 1e-6 && (s.im - 1.0).abs() < 1e-6);
}
