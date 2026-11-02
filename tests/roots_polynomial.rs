//! Tests für Polynom-Nullstellen (Aberth-Ehrlich, Durand-Kerner).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::complex::Complex;
use azmath::roots::{
    PolyRootsResult, RootError, RootOptions, aberth, aberth_complex, durand_kerner,
    durand_kerner_complex, polyval, polyval_complex,
};

type C = Complex<f64>;

fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}

fn opts() -> RootOptions<f64> {
    RootOptions::default()
}

/// Jede erwartete Nullstelle muss genau einer gefundenen (innerhalb `tol`) zugeordnet werden können.
fn assert_same_roots(found: &[C], expected: &[C], tol: f64) {
    assert_eq!(found.len(), expected.len());
    let mut used = vec![false; found.len()];
    for e in expected {
        let best = (0..found.len())
            .filter(|&i| !used[i])
            .min_by(|&a, &b| (found[a] - *e).abs().total_cmp(&(found[b] - *e).abs()))
            .expect("zu wenige Nullstellen");
        assert!(
            (found[best] - *e).abs() <= tol,
            "Nullstelle {e:?} nicht gefunden (nächste: {:?}), alle: {found:?}",
            found[best]
        );
        used[best] = true;
    }
}

/// Koeffizienten (aufsteigend) von `Π (x − r)` für reelle oder komplexe Nullstellen.
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

fn both<
    F: Fn(fn(&[f64], &mut [C], RootOptions<f64>) -> Result<PolyRootsResult<f64>, RootError>),
>(
    f: F,
) {
    f(aberth::<f64>);
    f(durand_kerner::<f64>);
}

#[test]
fn polyval_horner() {
    assert_eq!(polyval(&[2.0f64, 3.0, 1.0], 2.0), 12.0);
    assert_eq!(polyval::<f64>(&[], 5.0), 0.0);
    assert_eq!(polyval(&[7.0f64], 100.0), 7.0);
    let z = polyval_complex(&[c(1.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)], c(0.0, 1.0)); // 1 + i² = 0
    assert!(z.abs() < 1e-15);
}

#[test]
fn quadratic_with_complex_roots() {
    both(|solve| {
        let mut z = [C::ZERO; 2];
        let r = solve(&[1.0, 0.0, 1.0], &mut z, opts()).unwrap();
        assert_eq!(r.degree, 2);
        assert_same_roots(&z, &[c(0.0, 1.0), c(0.0, -1.0)], 1e-13);
    });
}

#[test]
fn distinct_real_roots_and_residuals() {
    both(|solve| {
        let coeffs = real_parts(&from_roots(&[
            c(1.0, 0.0),
            c(2.0, 0.0),
            c(3.0, 0.0),
            c(-4.0, 0.0),
        ]));
        let mut z = [C::ZERO; 4];
        solve(&coeffs, &mut z, opts()).unwrap();
        assert_same_roots(
            &z,
            &[c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(-4.0, 0.0)],
            1e-12,
        );
        let cc: Vec<C> = coeffs.iter().map(|&x| C::from_real(x)).collect();
        for r in z {
            assert!(polyval_complex(&cc, r).abs() < 1e-10);
        }
    });
}

#[test]
fn roots_of_unity() {
    both(|solve| {
        let n = 8;
        let mut coeffs = vec![0.0; n + 1];
        coeffs[0] = -1.0;
        coeffs[n] = 1.0;
        let mut z = vec![C::ZERO; n];
        solve(&coeffs, &mut z, opts()).unwrap();
        let expected: Vec<C> = (0..n)
            .map(|k| C::from_polar(1.0, 2.0 * core::f64::consts::PI * k as f64 / n as f64))
            .collect();
        assert_same_roots(&z, &expected, 1e-12);
    });
}

#[test]
fn degree_ten_with_integer_roots() {
    // Wilkinson-artig, aber nur 1..10: gut genug konditioniert für etwa 1e-6 absolut
    let roots: Vec<C> = (1..=10).map(|k| c(k as f64, 0.0)).collect();
    let coeffs = real_parts(&from_roots(&roots));
    both(|solve| {
        let mut z = [C::ZERO; 10];
        solve(&coeffs, &mut z, RootOptions::new(1e-9, 2000)).unwrap();
        assert_same_roots(&z, &roots, 1e-5);
    });
}

#[test]
fn zero_roots_and_leading_zero_coefficients() {
    both(|solve| {
        // x³ − x = x(x − 1)(x + 1), dazu zwei führende Nullkoeffizienten
        let mut z = [C::ZERO; 5];
        let r = solve(&[0.0, -1.0, 0.0, 1.0, 0.0, 0.0], &mut z, opts()).unwrap();
        assert_eq!(r.degree, 3);
        assert_same_roots(&z[..3], &[c(0.0, 0.0), c(1.0, 0.0), c(-1.0, 0.0)], 1e-13);
        assert_eq!(z[0], C::ZERO);
        // x⁴ = 0: vierfache Nullstelle ohne Iteration
        let mut z = [c(9.0, 9.0); 4];
        let r = solve(&[0.0, 0.0, 0.0, 0.0, 1.0], &mut z, opts()).unwrap();
        assert_eq!((r.degree, r.iterations), (4, 0));
        assert!(z.iter().all(|&r| r == C::ZERO));
    });
}

#[test]
fn degree_zero_and_one() {
    both(|solve| {
        let mut z = [C::ZERO; 1];
        let r = solve(&[5.0], &mut z, opts()).unwrap();
        assert_eq!(r.degree, 0);
        let r = solve(&[6.0, 3.0], &mut z, opts()).unwrap();
        assert_eq!((r.degree, z[0]), (1, c(-2.0, 0.0)));
    });
}

#[test]
fn complex_coefficients() {
    let roots = [c(1.0, 2.0), c(-0.5, 0.25), c(3.0, -1.0), c(0.0, -2.0)];
    let coeffs = from_roots(&roots);
    for use_aberth in [true, false] {
        let mut z = [C::ZERO; 4];
        if use_aberth {
            aberth_complex(&coeffs, &mut z, opts()).unwrap();
        } else {
            durand_kerner_complex(&coeffs, &mut z, opts()).unwrap();
        }
        assert_same_roots(&z, &roots, 1e-12);
    }
}

#[test]
fn aberth_needs_fewer_rounds_than_durand_kerner() {
    let roots: Vec<C> = (0..12)
        .map(|k| C::from_polar(1.0 + 0.1 * k as f64, 0.5 * k as f64))
        .collect();
    let coeffs = from_roots(&roots);
    let (mut a, mut d) = ([C::ZERO; 12], [C::ZERO; 12]);
    let o = RootOptions::new(1e-12, 2000);
    let ra = aberth_complex(&coeffs, &mut a, o).unwrap();
    let rd = durand_kerner_complex(&coeffs, &mut d, o).unwrap();
    assert_same_roots(&a, &roots, 1e-8);
    assert_same_roots(&d, &roots, 1e-8);
    assert!(
        ra.iterations <= rd.iterations,
        "Aberth {} gegen Durand-Kerner {}",
        ra.iterations,
        rd.iterations
    );
}

#[test]
fn multiple_roots_need_a_looser_tolerance() {
    // (x − 1)³ = x³ − 3x² + 3x − 1: Genauigkeit nur etwa ε^(1/3)
    let loose = RootOptions::new(1e-6, 2000);
    let mut z = [C::ZERO; 3];
    aberth(&[-1.0, 3.0, -3.0, 1.0], &mut z, loose).unwrap();
    assert_same_roots(&z, &[c(1.0, 0.0); 3], 1e-4);
    let strict = RootOptions::new(1e-16, 50);
    assert_eq!(
        aberth(&[-1.0, 3.0, -3.0, 1.0], &mut z, strict),
        Err(RootError::NoConvergence)
    );
}

#[test]
fn errors() {
    let mut z = [C::ZERO; 4];
    assert_eq!(aberth(&[], &mut z, opts()), Err(RootError::InvalidInput));
    assert_eq!(
        aberth(&[0.0, 0.0], &mut z, opts()),
        Err(RootError::InvalidInput)
    );
    assert_eq!(
        durand_kerner(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut z[..2], opts()),
        Err(RootError::InvalidInput)
    );
    assert_eq!(
        aberth(&[1.0, f64::NAN, 1.0], &mut z, opts()),
        Err(RootError::NotFinite)
    );
    assert_eq!(
        durand_kerner(&[1.0, f64::INFINITY, 1.0], &mut z, opts()),
        Err(RootError::NotFinite)
    );
    let tiny_budget = RootOptions::new(1e-15, 1);
    assert_eq!(
        aberth(&[1.0, 2.0, 3.0, 4.0, 5.0], &mut z, tiny_budget),
        Err(RootError::NoConvergence)
    );
    assert!(RootError::InvalidInput.to_string().contains("ungültig"));
}

#[test]
fn works_for_f32() {
    let mut z = [Complex::<f32>::ZERO; 3];
    aberth(
        &[-6.0f32, 11.0, -6.0, 1.0],
        &mut z,
        RootOptions::new(1e-5, 500),
    )
    .unwrap();
    let mut re: Vec<f32> = z.iter().map(|r| r.re).collect();
    re.sort_by(|a, b| a.total_cmp(b));
    for (got, want) in re.iter().zip([1.0, 2.0, 3.0]) {
        assert!((got - want).abs() < 1e-3, "{re:?}");
    }
}
