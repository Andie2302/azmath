//! Tests für das Jenkins-Traub-Verfahren.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::complex::Complex;
use azmath::roots::{
    RootError, RootOptions, companion_roots, companion_work_len, jenkins_traub,
    jenkins_traub_complex, jenkins_traub_work_len,
};

type C = Complex<f64>;

fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}

fn opts() -> RootOptions<f64> {
    RootOptions::default()
}

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
            "{e:?} nicht gefunden (nächste: {:?}); alle: {found:?}",
            found[best]
        );
        used[best] = true;
    }
}

/// Koeffizienten (aufsteigend) von `Π (x − r)`.
fn from_roots(roots: &[C]) -> Vec<C> {
    let mut p = vec![C::ONE];
    for &r in roots {
        let mut q = vec![C::ZERO; p.len() + 1];
        for (k, &pk) in p.iter().enumerate() {
            q[k + 1] += pk;
            q[k] -= pk * r;
        }
        p = q;
    }
    p
}

fn real_parts(p: &[C]) -> Vec<f64> {
    p.iter().map(|z| z.re).collect()
}

fn run(coeffs: &[f64]) -> Vec<C> {
    let degree = coeffs.len() - 1;
    let mut z = vec![C::ZERO; degree];
    let mut work = vec![C::ZERO; jenkins_traub_work_len(degree)];
    jenkins_traub(coeffs, &mut z, &mut work, opts()).unwrap();
    z
}

#[test]
fn quadratics_and_cubics() {
    assert_same_roots(&run(&[1.0, 0.0, 1.0]), &[c(0.0, 1.0), c(0.0, -1.0)], 1e-12);
    assert_same_roots(
        &run(&[-6.0, 11.0, -6.0, 1.0]),
        &[c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)],
        1e-11,
    );
    // x³ + 1: −1 und e^{±iπ/3}
    let w = core::f64::consts::PI / 3.0;
    assert_same_roots(
        &run(&[1.0, 0.0, 0.0, 1.0]),
        &[c(-1.0, 0.0), c(w.cos(), w.sin()), c(w.cos(), -w.sin())],
        1e-12,
    );
}

#[test]
fn roots_of_unity_of_degree_twelve() {
    let n = 12;
    let mut coeffs = vec![0.0; n + 1];
    coeffs[0] = -1.0;
    coeffs[n] = 1.0;
    let expected: Vec<C> = (0..n)
        .map(|k| C::from_polar(1.0, 2.0 * core::f64::consts::PI * k as f64 / n as f64))
        .collect();
    assert_same_roots(&run(&coeffs), &expected, 1e-11);
}

#[test]
fn roots_of_very_different_magnitudes() {
    let roots = [
        c(1e-3, 0.0),
        c(1.0, 0.0),
        c(-2.0, 1.0),
        c(-2.0, -1.0),
        c(1e3, 0.0),
    ];
    let coeffs = real_parts(&from_roots(&roots));
    let found = run(&coeffs);
    // relative Genauigkeit je Nullstelle
    for e in &roots {
        let best = found
            .iter()
            .map(|f| (*f - *e).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            best <= 1e-7 * e.abs().max(1.0),
            "{e:?}: Abweichung {best:e}"
        );
    }
}

#[test]
fn degree_ten_with_integer_roots() {
    let roots: Vec<C> = (1..=10).map(|k| c(k as f64, 0.0)).collect();
    let coeffs = real_parts(&from_roots(&roots));
    assert_same_roots(&run(&coeffs), &roots, 1e-5);
}

#[test]
fn agrees_with_the_companion_matrix_method() {
    let coeffs = [3.0, -1.0, 4.0, 1.0, -5.0, 9.0, -2.0, 6.0, 5.0];
    let degree = coeffs.len() - 1;
    let a = run(&coeffs);
    let mut b = vec![C::ZERO; degree];
    let mut work = vec![0.0; companion_work_len(degree)];
    companion_roots(&coeffs, &mut b, &mut work).unwrap();
    assert_same_roots(&a, &b, 1e-8);
}

#[test]
fn complex_coefficients() {
    let roots = [
        c(1.0, 2.0),
        c(-0.5, 0.25),
        c(3.0, -1.0),
        c(0.0, -2.0),
        c(-4.0, 4.0),
    ];
    let coeffs = from_roots(&roots);
    let mut z = vec![C::ZERO; 5];
    let mut work = vec![C::ZERO; jenkins_traub_work_len(5)];
    jenkins_traub_complex(&coeffs, &mut z, &mut work, opts()).unwrap();
    assert_same_roots(&z, &roots, 1e-10);
}

#[test]
fn zero_roots_leading_zeros_and_low_degrees() {
    // x³ − x mit zwei führenden Nullen
    let mut z = [C::ZERO; 5];
    let mut work = vec![C::ZERO; jenkins_traub_work_len(3)];
    let r = jenkins_traub(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0], &mut z, &mut work, opts()).unwrap();
    assert_eq!(r.degree, 3);
    assert_same_roots(&z[..3], &[c(0.0, 0.0), c(1.0, 0.0), c(-1.0, 0.0)], 1e-12);
    // Grad 1 und 0
    let mut one = [C::ZERO; 1];
    let r = jenkins_traub(&[6.0, 3.0], &mut one, &mut work, opts()).unwrap();
    assert_eq!((r.degree, one[0]), (1, c(-2.0, 0.0)));
    assert_eq!(
        jenkins_traub(&[5.0], &mut one, &mut work, opts())
            .unwrap()
            .degree,
        0
    );
}

#[test]
fn multiple_root_with_loose_tolerance() {
    // (x − 1)⁴: Genauigkeit nur etwa ε^(1/4)
    let mut z = [C::ZERO; 4];
    let mut work = vec![C::ZERO; jenkins_traub_work_len(4)];
    jenkins_traub(
        &[1.0, -4.0, 6.0, -4.0, 1.0],
        &mut z,
        &mut work,
        RootOptions::new(1e-7, 200),
    )
    .unwrap();
    assert_same_roots(&z, &[c(1.0, 0.0); 4], 1e-2);
}

#[test]
fn errors() {
    let mut z = [C::ZERO; 4];
    let mut work = vec![C::ZERO; jenkins_traub_work_len(4)];
    assert_eq!(
        jenkins_traub(&[], &mut z, &mut work, opts()),
        Err(RootError::InvalidInput)
    );
    assert_eq!(
        jenkins_traub(&[0.0, 0.0], &mut z, &mut work, opts()),
        Err(RootError::InvalidInput)
    );
    assert_eq!(
        jenkins_traub(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut z[..2], &mut work, opts()),
        Err(RootError::InvalidInput)
    );
    assert_eq!(
        jenkins_traub(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut z, &mut work[..4], opts()),
        Err(RootError::InvalidInput)
    );
    assert_eq!(
        jenkins_traub(&[1.0, f64::NAN, 1.0], &mut z, &mut work, opts()),
        Err(RootError::NotFinite)
    );
}

#[test]
fn works_for_f32() {
    let mut z = [Complex::<f32>::ZERO; 3];
    let mut work = vec![Complex::<f32>::ZERO; jenkins_traub_work_len(3)];
    jenkins_traub(
        &[-6.0f32, 11.0, -6.0, 1.0],
        &mut z,
        &mut work,
        RootOptions::new(1e-5, 200),
    )
    .unwrap();
    let mut re: Vec<f32> = z.iter().map(|r| r.re).collect();
    re.sort_by(|a, b| a.total_cmp(b));
    for (got, want) in re.iter().zip([1.0f32, 2.0, 3.0]) {
        assert!((got - want).abs() < 1e-3, "{re:?}");
    }
}
