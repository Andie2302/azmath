//! Tests für `Scalar` und `ScalarLu` (reell und komplex).
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::complex::Complex;
use azmath::scalar::{Scalar, ScalarLu};

type C = Complex<f64>;

#[test]
fn scalar_basics() {
    let z = C::new(3.0, 4.0);
    assert_eq!(z.modulus(), 5.0);
    assert_eq!(z.modulus_sq(), 25.0);
    assert_eq!(Scalar::conj(z), C::new(3.0, -4.0));
    assert_eq!((-2.5f64).modulus(), 2.5);
    assert_eq!(Scalar::conj(1.5f64), 1.5);
    assert_eq!(Scalar::im(1.5f64), 0.0);
    assert!(!C::new(f64::NAN, 0.0).is_finite());
    assert!(1.0f32.is_finite());
}

#[test]
fn complex_solve_and_det() {
    // A = [[1+i, 2], [3, 4-i]]
    let a = [
        C::new(1.0, 1.0),
        C::new(2.0, 0.0),
        C::new(3.0, 0.0),
        C::new(4.0, -1.0),
    ];
    let lu = ScalarLu::new(2, &a).unwrap();
    let b = [C::new(1.0, 0.0), C::new(0.0, 2.0)];
    let x = lu.solve(&b).unwrap();
    for i in 0..2 {
        let r = a[i * 2] * x[0] + a[i * 2 + 1] * x[1] - b[i];
        assert!(r.abs() < 1e-13);
    }
    // det = (1+i)(4-i) - 6 = 5+3i - 6 = -1+3i
    let d = lu.determinant();
    assert!((d - C::new(-1.0, 3.0)).abs() < 1e-13);
}

#[test]
fn complex_inverse() {
    let a = [
        C::new(2.0, 1.0),
        C::new(0.0, -1.0),
        C::new(1.0, 0.0),
        C::new(0.0, 3.0),
        C::new(1.0, 1.0),
        C::new(2.0, 0.0),
        C::new(-1.0, 0.0),
        C::new(0.5, 0.5),
        C::new(4.0, 0.0),
    ];
    let lu = ScalarLu::new(3, &a).unwrap();
    let inv = lu.inverse();
    for i in 0..3 {
        for j in 0..3 {
            let mut s = C::ZERO;
            for k in 0..3 {
                s += a[i * 3 + k] * inv[k * 3 + j];
            }
            let e = if i == j { C::ONE } else { C::ZERO };
            assert!((s - e).abs() < 1e-12);
        }
    }
}

#[test]
fn real_works_and_errors() {
    let lu = ScalarLu::new(2, &[0.0f64, 1.0, 1.0, 0.0]).unwrap();
    assert_eq!(lu.solve(&[2.0, 3.0]).unwrap(), vec![3.0, 2.0]);
    assert_eq!(lu.determinant(), -1.0);
    assert!(ScalarLu::new(2, &[1.0f64, 2.0, 2.0, 4.0]).is_err());
    assert!(ScalarLu::new(2, &[1.0f64]).is_err());
    assert!(ScalarLu::new(1, &[f64::NAN]).is_err());
    assert!(lu.solve(&[1.0]).is_err());
}
