//! Tests für Orthonormalisierung (MGS, Householder) und Polarzerlegung.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::linalg::{LinalgError, Matrix, QrDecomposition};

/// Vandermonde-Matrix (fast abhängige Spalten) mit 6 Zeilen und 4 Spalten.
fn vandermonde() -> Matrix<f64, 6, 4> {
    Matrix::from_fn(|i, j| (1.0 + 0.01 * i as f64).powi(j as i32))
}

fn max_diff<const R: usize, const C: usize>(a: &Matrix<f64, R, C>, b: &Matrix<f64, R, C>) -> f64 {
    (*a - *b).max_abs()
}

fn rotation_z(t: f64) -> Matrix<f64, 3, 3> {
    Matrix::new([
        [t.cos(), -t.sin(), 0.0],
        [t.sin(), t.cos(), 0.0],
        [0.0, 0.0, 1.0],
    ])
}

fn rotation_x(t: f64) -> Matrix<f64, 3, 3> {
    Matrix::new([
        [1.0, 0.0, 0.0],
        [0.0, t.cos(), -t.sin()],
        [0.0, t.sin(), t.cos()],
    ])
}

#[test]
fn mgs_reconstructs_and_is_orthonormal() {
    let a = Matrix::new([
        [1.0, 2.0, 0.5],
        [3.0, -1.0, 2.0],
        [0.0, 4.0, 1.0],
        [2.0, 2.0, -3.0],
    ]);
    let o = a.orthonormalize_mgs(false).unwrap();
    assert!(max_diff(&(o.q * o.r), &a) < 1e-14);
    assert!(o.q.orthogonality_error() < 1e-14);
    for i in 0..3 {
        assert!(o.r[(i, i)] > 0.0);
        for j in 0..i {
            assert_eq!(o.r[(i, j)], 0.0);
        }
    }
}

#[test]
fn reorthogonalization_keeps_nearly_dependent_columns_orthogonal() {
    let a = vandermonde();
    let plain = a.orthonormalize_mgs(false).unwrap();
    let twice = a.orthonormalize_mgs(true).unwrap();
    let (e1, e2) = (plain.q.orthogonality_error(), twice.q.orthogonality_error());
    assert!(e2 < 1e-14, "mit Reorthogonalisierung: {e2:e}");
    assert!(e1 > 10.0 * e2, "ohne: {e1:e}, mit: {e2:e}");
    assert!(max_diff(&(twice.q * twice.r), &a) < 1e-13);
}

#[test]
fn mgs_errors() {
    let dup = Matrix::new([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]);
    assert_eq!(dup.orthonormalize_mgs(true), Err(LinalgError::Singular));
    let zero_col = Matrix::new([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]);
    assert_eq!(
        zero_col.orthonormalize_mgs(false),
        Err(LinalgError::Singular)
    );
    let wide = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(
        wide.orthonormalize_mgs(false),
        Err(LinalgError::DimensionMismatch)
    );
    let nan = Matrix::new([[1.0, f64::NAN], [0.0, 1.0]]);
    assert_eq!(nan.orthonormalize_mgs(false), Err(LinalgError::NotFinite));
}

#[test]
fn householder_thin_q_spans_the_same_space_as_mgs() {
    let a = vandermonde();
    let qr = QrDecomposition::new(&a).unwrap();
    let q = qr.thin_q();
    assert!(q.orthogonality_error() < 1e-14);
    assert!(max_diff(&(q * qr.r()), &a) < 1e-13);
    let m = a.orthonormalize_mgs(true).unwrap().q;
    for j in 0..4 {
        let dot: f64 = (0..6).map(|i| q[(i, j)] * m[(i, j)]).sum();
        assert!((dot.abs() - 1.0).abs() < 1e-9, "Spalte {j}: {dot}");
    }
}

#[test]
fn polar_factors() {
    let a = Matrix::new([[2.0, 1.0], [0.5, 3.0], [1.0, -1.0]]);
    let p = a.polar().unwrap();
    assert!(max_diff(&(p.u * p.p), &a) < 1e-13);
    assert!(p.u.orthogonality_error() < 1e-13);
    assert!(max_diff(&p.p, &p.p.transpose()) < 1e-14);
    let eig = p.p.symmetric_eigen().unwrap();
    assert!(eig.values.as_slice().iter().all(|&l| l > 0.0));
}

#[test]
fn polar_errors() {
    let rank1 = Matrix::new([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]);
    assert_eq!(rank1.polar(), Err(LinalgError::Singular));
    let wide = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(wide.polar(), Err(LinalgError::DimensionMismatch));
}

#[test]
fn nearest_orthogonal_repairs_a_perturbed_rotation() {
    let r = rotation_z(0.7) * rotation_x(-0.4);
    let noisy = r + Matrix::new([
        [1e-3, -2e-3, 5e-4],
        [-1e-3, 1e-3, 2e-3],
        [3e-4, -1e-3, 1e-3],
    ]);
    assert!(noisy.orthogonality_error() > 1e-4);
    let fixed = noisy.nearest_orthogonal().unwrap();
    assert!(fixed.orthogonality_error() < 1e-14);
    assert!((fixed - r).norm_frobenius() < (noisy - r).norm_frobenius());
    assert!((fixed.determinant() - 1.0).abs() < 1e-13);
    let rot = noisy.nearest_rotation().unwrap();
    assert!(max_diff(&rot, &fixed) < 1e-13);
}

#[test]
fn rotation_versus_reflection() {
    let mirror =
        Matrix::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]) * rotation_z(0.3);
    let o = mirror.nearest_orthogonal().unwrap();
    assert!((o.determinant() + 1.0).abs() < 1e-13);
    assert!(max_diff(&o, &mirror) < 1e-13);
    let r = mirror.nearest_rotation().unwrap();
    assert!((r.determinant() - 1.0).abs() < 1e-13);
    assert!(r.orthogonality_error() < 1e-14);
}

#[test]
fn orthogonality_error_basics() {
    assert_eq!(Matrix::<f64, 3, 3>::identity().orthogonality_error(), 0.0);
    let scaled = Matrix::<f64, 2, 2>::identity() * 2.0;
    assert!((scaled.orthogonality_error() - (2.0f64 * 9.0).sqrt()).abs() < 1e-14);
    assert!(
        Matrix::new([[f64::NAN, 0.0], [0.0, 1.0]])
            .orthogonality_error()
            .is_nan()
    );
}

#[test]
fn works_for_f32() {
    let a = Matrix::new([[3.0f32, 1.0], [1.0, 2.0], [0.0, 1.0]]);
    let o = a.orthonormalize_mgs(true).unwrap();
    assert!(o.q.orthogonality_error() < 1e-5);
    let r = rotation_z(0.5);
    let rf = Matrix::<f32, 3, 3>::from_fn(|i, j| r[(i, j)] as f32);
    assert!(rf.nearest_rotation().unwrap().orthogonality_error() < 1e-5);
}

#[cfg(feature = "alloc")]
mod dynamic {
    use super::*;
    use azmath::linalg::DynMatrix;

    fn dyn_of<const R: usize, const C: usize>(a: &Matrix<f64, R, C>) -> DynMatrix<f64> {
        DynMatrix::new(R, C, a.as_array().iter().flatten().copied().collect()).unwrap()
    }

    #[test]
    fn dynamic_matches_fixed_size() {
        let a = vandermonde();
        let d = dyn_of(&a);
        let (f, g) = (
            a.orthonormalize_mgs(true).unwrap(),
            d.orthonormalize_mgs(true).unwrap(),
        );
        assert_eq!(g.q.as_slice(), f.q.as_array().as_flattened());
        assert_eq!(g.r.as_slice(), f.r.as_array().as_flattened());
        assert_eq!(d.orthogonality_error(), a.orthogonality_error());

        let qr = QrDecomposition::new(&a).unwrap();
        let dq = d.qr().unwrap().thin_q();
        for (x, y) in dq
            .as_slice()
            .iter()
            .zip(qr.thin_q().as_array().as_flattened())
        {
            assert!((x - y).abs() < 1e-14);
        }

        let m = Matrix::new([[2.0, 1.0], [0.5, 3.0], [1.0, -1.0]]);
        let (pf, pd) = (m.polar().unwrap(), dyn_of(&m).polar().unwrap());
        for (x, y) in pd.u.as_slice().iter().zip(pf.u.as_array().as_flattened()) {
            assert!((x - y).abs() < 1e-13);
        }
    }

    #[test]
    fn dynamic_rotation_and_errors() {
        let mirror =
            Matrix::new([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]) * rotation_z(0.3);
        let r = dyn_of(&mirror).nearest_rotation().unwrap();
        assert!((r.determinant().unwrap() - 1.0).abs() < 1e-13);
        let o = dyn_of(&mirror).nearest_orthogonal().unwrap();
        assert!((o.determinant().unwrap() + 1.0).abs() < 1e-13);
        let tall = dyn_of(&Matrix::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]));
        assert_eq!(tall.nearest_rotation(), Err(LinalgError::DimensionMismatch));
        let dup = dyn_of(&Matrix::new([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]));
        assert_eq!(dup.orthonormalize_mgs(true), Err(LinalgError::Singular));
    }
}
