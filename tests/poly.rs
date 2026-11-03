//! Tests für `poly`.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::linalg::{DynMatrix, DynVector};
use azmath::poly::*;

fn p(c: &[f64]) -> Polynomial<f64> {
    Polynomial::new(c.to_vec())
}

#[test]
fn arithmetic() {
    let a = p(&[1.0, 2.0, 3.0]); // 1 + 2x + 3x²
    let b = p(&[0.0, 1.0]); // x
    assert_eq!((&a + &b).coefficients(), &[1.0, 3.0, 3.0]);
    assert_eq!((&a - &a).coefficients(), &[] as &[f64]);
    assert_eq!((&a * &b).coefficients(), &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(a.eval(2.0), 17.0);
    assert_eq!(a.derivative().coefficients(), &[2.0, 6.0]);
    assert_eq!(a.integral().coefficients(), &[0.0, 1.0, 1.0, 1.0]);
    assert!((a.integrate(0.0, 1.0) - 3.0).abs() < 1e-14);
    assert_eq!(Polynomial::new(vec![1.0, 0.0, 0.0]).degree(), 0);
    assert!(Polynomial::<f64>::zero().is_zero());
    assert_eq!((-&b).coefficients(), &[0.0, -1.0]);
}

#[test]
fn from_roots_and_compose() {
    let q = Polynomial::from_roots(&[1.0, 2.0, 3.0]);
    assert_eq!(q.coefficients(), &[-6.0, 11.0, -6.0, 1.0]);
    for r in [1.0f64, 2.0, 3.0] {
        assert!(q.eval(r).abs() < 1e-13);
    }
    // (x²)(x + 1) = x²+... : compose(x², x+1) = (x+1)²
    let outer = p(&[0.0, 0.0, 1.0]);
    let inner = p(&[1.0, 1.0]);
    assert_eq!(outer.compose(&inner).coefficients(), &[1.0, 2.0, 1.0]);
}

#[test]
fn division_identity_and_gcd() {
    let a = p(&[5.0, -3.0, 0.0, 2.0, 1.0]);
    let d = p(&[1.0, 0.0, 1.0]);
    let (q, r) = a.div_rem(&d).unwrap();
    let back = &(&q * &d) + &r;
    for (x, y) in back.coefficients().iter().zip(a.coefficients()) {
        assert!((x - y).abs() < 1e-12);
    }
    assert!(r.degree() < d.degree() || r.is_zero());
    assert!(a.div_rem(&Polynomial::zero()).is_err());
    // ggT((x−1)(x−2), (x−1)(x−3)) = x − 1
    let g =
        Polynomial::from_roots(&[1.0f64, 2.0]).gcd(&Polynomial::from_roots(&[1.0f64, 3.0]), 1e-10);
    assert_eq!(g.degree(), 1);
    assert!((g.coefficients()[0] + 1.0).abs() < 1e-10);
}

#[test]
fn polyfit_exact_and_noisy() {
    let x: Vec<f64> = (0..8).map(|i| i as f64 * 0.5 - 1.0).collect();
    let y: Vec<f64> = x.iter().map(|&v| 2.0 - v + 0.5 * v * v * v).collect();
    let fit = polyfit(&x, &y, 3).unwrap();
    for (c, e) in fit.coefficients().iter().zip([2.0, -1.0, 0.0, 0.5]) {
        assert!((c - e).abs() < 1e-10);
    }
    assert!(polyfit(&x, &y[..4], 2).is_err());
    assert!(polyfit(&x[..2], &y[..2], 3).is_err());
}

#[test]
fn chebyshev_approximation() {
    let s = ChebyshevSeries::from_function(|x: f64| x.exp(), -1.0, 2.0, 16);
    for i in 0..=30 {
        let x = -1.0 + 3.0 * i as f64 / 30.0;
        assert!((s.eval(x) - x.exp()).abs() < 1e-10, "x = {x}");
    }
    let d = s.derivative();
    for i in 0..=10 {
        let x = -1.0 + 3.0 * i as f64 / 10.0;
        assert!((d.eval(x) - x.exp()).abs() < 1e-7, "x = {x}");
    }
    let nodes = chebyshev_nodes(5, -1.0f64, 1.0);
    assert!(nodes.windows(2).all(|w| w[0] < w[1]));
    assert!((nodes[2]).abs() < 1e-14);
}

fn mat(rows: usize, cols: usize, d: &[f64]) -> DynMatrix<f64> {
    DynMatrix::new(rows, cols, d.to_vec()).unwrap()
}

#[test]
fn weighted_and_ridge() {
    let a = mat(4, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
    let b = DynVector::new(vec![1.0, 3.0, 5.0, 7.0]);
    let x = weighted_least_squares(&a, &b, &[1.0, 2.0, 0.5, 3.0]).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-10 && (x[1] - 2.0).abs() < 1e-10);
    let r0 = ridge(&a, &b, 0.0).unwrap();
    assert!((r0[0] - 1.0).abs() < 1e-10 && (r0[1] - 2.0).abs() < 1e-10);
    let r = ridge(&a, &b, 100.0).unwrap();
    let (n0, n1) = (r0[0].hypot(r0[1]), r[0].hypot(r[1]));
    assert!(n1 < n0);
    assert!(ridge(&a, &b, -1.0).is_err());
    assert!(weighted_least_squares(&a, &b, &[1.0]).is_err());
}

#[test]
fn nnls_cases() {
    // unbeschränkte Lösung bereits nichtnegativ
    let a = mat(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let b = DynVector::new(vec![1.0, 2.0, 3.0]);
    let r = nnls(&a, &b).unwrap();
    assert!((r.x[0] - 1.0).abs() < 1e-10 && (r.x[1] - 2.0).abs() < 1e-10);
    assert!(r.residual_norm < 1e-10);
    // unbeschränkt negativ -> Schranke aktiv
    let a = mat(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    let b = DynVector::new(vec![1.0, -1.0, 0.5]);
    let r = nnls(&a, &b).unwrap();
    assert!(r.x.iter().all(|&v| v >= 0.0));
    // KKT: w = Aᵀ(b − Ax); w_j ≤ 0 für x_j = 0, w_j ≈ 0 für x_j > 0
    let res: Vec<f64> = (0..3)
        .map(|i| b[i] - (a[(i, 0)] * r.x[0] + a[(i, 1)] * r.x[1]))
        .collect();
    for j in 0..2 {
        let w: f64 = (0..3).map(|i| a[(i, j)] * res[i]).sum();
        if r.x[j] == 0.0 {
            assert!(w <= 1e-9);
        } else {
            assert!(w.abs() < 1e-9);
        }
    }
    assert!(nnls(&a, &DynVector::new(vec![1.0])).is_err());
}
