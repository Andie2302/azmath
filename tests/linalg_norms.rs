//! Tests für Matrixnormen (Frobenius, 1, ∞, 2).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::linalg::{LinalgError, Matrix, Vector};

fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "erwartet {b}, erhalten {a}");
}

const A: [[f64; 3]; 3] = [[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0], [7.0, 8.0, 9.0]];
const B: [[f64; 3]; 3] = [[2.0, 0.0, -1.0], [1.0, 3.0, 0.5], [-2.0, 1.0, 4.0]];

#[test]
fn hand_computed_values() {
    let a = Matrix::new(A);
    close(a.norm_1(), 18.0, 0.0); // Spaltensummen 12, 15, 18
    close(a.norm_inf(), 24.0, 0.0); // Zeilensummen 6, 15, 24
    close(a.norm_frobenius(), 285f64.sqrt(), 1e-14); // 1+4+9+16+25+36+49+64+81
    close(a.max_abs(), 9.0, 0.0);
}

#[test]
fn spectral_norm_of_known_matrices() {
    let d = Matrix::new([[3.0, 0.0], [0.0, -5.0]]);
    close(d.norm_2().unwrap(), 5.0, 1e-14);
    let tall = Matrix::new([[3.0, 0.0], [0.0, -2.0], [0.0, 0.0]]);
    close(tall.norm_2().unwrap(), 3.0, 1e-14);
    // breit (M < N): gleiche Singulärwerte wie die Transponierte
    close(tall.transpose().norm_2().unwrap(), 3.0, 1e-14);
    // Rang 1: ‖u·vᵀ‖₂ = ‖u‖·‖v‖
    let u = Vector::new([1.0, 2.0, 2.0]);
    let v = Vector::new([3.0, 4.0]);
    let r1 = Matrix::<f64, 3, 2>::from_fn(|i, j| u[i] * v[j]);
    close(r1.norm_2().unwrap(), u.norm() * v.norm(), 1e-13);
    close(r1.norm_frobenius(), u.norm() * v.norm(), 1e-13);
}

#[test]
fn norm_inequalities_hold() {
    for m in [Matrix::new(A), Matrix::new(B)] {
        let (n1, ni, nf) = (m.norm_1(), m.norm_inf(), m.norm_frobenius());
        let n2 = m.norm_2().unwrap();
        assert!(n2 <= nf * (1.0 + 1e-14));
        assert!(nf <= 3f64.sqrt() * n2 * (1.0 + 1e-14)); // Rang ≤ 3
        assert!(n2 * n2 <= n1 * ni * (1.0 + 1e-14));
        assert!(m.max_abs() <= n2 * (1.0 + 1e-14));
    }
}

#[test]
fn norms_are_submultiplicative() {
    let (a, b) = (Matrix::new(A), Matrix::new(B));
    let ab = a * b;
    let tol = 1.0 + 1e-14;
    assert!(ab.norm_1() <= a.norm_1() * b.norm_1() * tol);
    assert!(ab.norm_inf() <= a.norm_inf() * b.norm_inf() * tol);
    assert!(ab.norm_frobenius() <= a.norm_frobenius() * b.norm_frobenius() * tol);
    assert!(ab.norm_2().unwrap() <= a.norm_2().unwrap() * b.norm_2().unwrap() * tol);
}

#[test]
fn identity_zero_and_scaling() {
    let i = Matrix::<f64, 4, 4>::identity();
    close(i.norm_1(), 1.0, 0.0);
    close(i.norm_inf(), 1.0, 0.0);
    close(i.norm_2().unwrap(), 1.0, 1e-14);
    close(i.norm_frobenius(), 2.0, 1e-15);
    let z = Matrix::<f64, 3, 2>::zeros();
    assert_eq!(z.norm_frobenius(), 0.0);
    assert_eq!(z.norm_2().unwrap(), 0.0);
    // absolute Homogenität
    let a = Matrix::new(A);
    close((a * -3.0).norm_1(), 3.0 * a.norm_1(), 1e-13);
    close(
        (a * -3.0).norm_2().unwrap(),
        3.0 * a.norm_2().unwrap(),
        1e-12,
    );
}

#[test]
fn frobenius_is_safe_against_overflow_and_underflow() {
    let big = Matrix::new([[1e200f64, 1e200], [1e200, 1e200]]);
    let f = big.norm_frobenius();
    assert!(f.is_finite());
    close(f / 1e200, 2.0, 1e-14);
    let tiny = Matrix::new([[1e-200, 1e-200], [1e-200, 1e-200]]);
    close(tiny.norm_frobenius() / 1e-200, 2.0, 1e-14);
}

#[test]
fn nan_and_infinity() {
    let n = Matrix::new([[1.0, f64::NAN], [0.0, 1.0]]);
    assert!(n.norm_frobenius().is_nan());
    assert!(n.norm_1().is_nan());
    assert!(n.norm_inf().is_nan());
    assert_eq!(n.norm_2(), Err(LinalgError::NotFinite));
    let inf = Matrix::new([[f64::INFINITY, 0.0], [0.0, 1.0]]);
    assert_eq!(inf.norm_frobenius(), f64::INFINITY);
    assert_eq!(inf.norm_1(), f64::INFINITY);
}

#[test]
fn works_for_f32_and_vector_one_norm() {
    let a = Matrix::new([[3.0f32, 0.0], [0.0, 4.0]]);
    assert_eq!(a.norm_frobenius(), 5.0);
    assert!((a.norm_2().unwrap() - 4.0).abs() < 1e-5);
    assert_eq!(Vector::new([1.0, -2.0, 3.0]).norm_1(), 6.0);
}

#[cfg(feature = "alloc")]
mod dynamic {
    use super::*;
    use azmath::linalg::{DynMatrix, DynVector};

    fn dyn_from<const R: usize, const C: usize>(a: [[f64; C]; R]) -> DynMatrix<f64> {
        DynMatrix::new(R, C, a.iter().flatten().copied().collect()).unwrap()
    }

    #[test]
    fn dynamic_norms_match_fixed_size() {
        let (f, d) = (Matrix::new(A), dyn_from(A));
        assert_eq!(d.norm_1(), f.norm_1());
        assert_eq!(d.norm_inf(), f.norm_inf());
        assert_eq!(d.norm_frobenius(), f.norm_frobenius());
        close(d.norm_2().unwrap(), f.norm_2().unwrap(), 1e-14);
        let wide = dyn_from([[3.0, 0.0, 0.0], [0.0, -2.0, 0.0]]);
        close(wide.norm_2().unwrap(), 3.0, 1e-14);
        assert_eq!(DynMatrix::<f64>::zeros(0, 3).norm_2(), Ok(0.0));
        assert_eq!(DynVector::new(vec![1.0, -2.0, 3.0]).norm_1(), 6.0);
    }
}
