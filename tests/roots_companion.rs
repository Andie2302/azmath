//! Tests für Polynom-Nullstellen über die Begleitmatrix.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::complex::Complex;
use azmath::roots::{RootError, RootOptions, aberth, companion_roots, companion_work_len};

type C = Complex<f64>;

fn assert_same_roots(found: &[C], expected: &[C], tol: f64) {
    assert_eq!(found.len(), expected.len());
    let mut used = vec![false; found.len()];
    for e in expected {
        let best = (0..found.len())
            .filter(|&i| !used[i])
            .min_by(|&a, &b| (found[a] - *e).abs().total_cmp(&(found[b] - *e).abs()))
            .unwrap();
        assert!(
            (found[best] - *e).abs() <= tol,
            "{e:?} nicht gefunden, alle: {found:?}"
        );
        used[best] = true;
    }
}

fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}

#[test]
fn simple_polynomials() {
    let mut work = [0.0; companion_work_len(4)];
    // x² + 1
    let mut z = [C::ZERO; 2];
    companion_roots(&[1.0, 0.0, 1.0], &mut z, &mut work).unwrap();
    assert_same_roots(&z, &[c(0.0, 1.0), c(0.0, -1.0)], 1e-13);
    // (x − 1)(x − 2)(x − 3)(x + 4)
    let mut z = [C::ZERO; 4];
    companion_roots(&[-24.0, 38.0, -13.0, -2.0, 1.0], &mut z, &mut work).unwrap();
    assert_same_roots(
        &z,
        &[c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(-4.0, 0.0)],
        1e-11,
    );
}

#[test]
fn roots_of_unity() {
    let n = 8;
    let mut coeffs = vec![0.0; n + 1];
    coeffs[0] = -1.0;
    coeffs[n] = 1.0;
    let mut z = vec![C::ZERO; n];
    let mut work = vec![0.0; companion_work_len(n)];
    companion_roots(&coeffs, &mut z, &mut work).unwrap();
    let expected: Vec<C> = (0..n)
        .map(|k| C::from_polar(1.0, 2.0 * core::f64::consts::PI * k as f64 / n as f64))
        .collect();
    assert_same_roots(&z, &expected, 1e-12);
}

#[test]
fn leading_and_trailing_zero_coefficients() {
    let mut work = [0.0; companion_work_len(3)];
    // x³ − x mit zwei führenden Nullen: Wurzeln 0, 1, −1
    let mut z = [C::ZERO; 5];
    let r = companion_roots(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0], &mut z, &mut work).unwrap();
    assert_eq!(r.degree, 3);
    assert_same_roots(&z[..3], &[c(0.0, 0.0), c(1.0, 0.0), c(-1.0, 0.0)], 1e-13);
    // Grad 1 und Grad 0
    let mut one = [C::ZERO; 1];
    let r = companion_roots(&[6.0, 3.0], &mut one, &mut work).unwrap();
    assert_eq!((r.degree, one[0]), (1, c(-2.0, 0.0)));
    let r = companion_roots(&[5.0], &mut one, &mut work).unwrap();
    assert_eq!(r.degree, 0);
}

#[test]
fn agrees_with_aberth_on_a_degree_ten_polynomial() {
    let roots: Vec<f64> = (1..=10).map(f64::from).collect();
    let mut p = vec![1.0];
    for &r in &roots {
        let mut q = vec![0.0; p.len() + 1];
        for (k, &pk) in p.iter().enumerate() {
            q[k + 1] += pk;
            q[k] -= pk * r;
        }
        p = q;
    }
    let mut a = [C::ZERO; 10];
    aberth(&p, &mut a, RootOptions::new(1e-9, 2000)).unwrap();
    let mut b = [C::ZERO; 10];
    let mut work = [0.0; companion_work_len(10)];
    companion_roots(&p, &mut b, &mut work).unwrap();
    let expected: Vec<C> = roots.iter().map(|&r| c(r, 0.0)).collect();
    assert_same_roots(&a, &expected, 1e-5);
    assert_same_roots(&b, &expected, 1e-5);
}

#[test]
fn multiple_root_has_the_expected_reduced_accuracy() {
    let mut z = [C::ZERO; 3];
    let mut work = [0.0; companion_work_len(3)];
    companion_roots(&[-1.0, 3.0, -3.0, 1.0], &mut z, &mut work).unwrap(); // (x − 1)³
    assert_same_roots(&z, &[c(1.0, 0.0); 3], 1e-4);
}

#[test]
fn errors() {
    let mut z = [C::ZERO; 4];
    let mut work = [0.0; companion_work_len(4)];
    assert_eq!(
        companion_roots(&[], &mut z, &mut work),
        Err(RootError::InvalidInput)
    );
    assert_eq!(
        companion_roots(&[0.0, 0.0], &mut z, &mut work),
        Err(RootError::InvalidInput)
    );
    assert_eq!(
        companion_roots(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut z[..2], &mut work),
        Err(RootError::InvalidInput)
    );
    assert_eq!(
        companion_roots(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut z, &mut work[..5]),
        Err(RootError::InvalidInput)
    );
    assert_eq!(
        companion_roots(&[1.0, f64::NAN, 1.0], &mut z, &mut work),
        Err(RootError::NotFinite)
    );
}
