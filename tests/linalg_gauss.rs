//! Tests für die Gauß-Elimination (`gauss_solve`, `rank`, `row_echelon`).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::linalg::{LinalgError, Matrix, Vector};

/// Deterministische Pseudozufallszahlen in [-1, 1).
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn hilbert<const N: usize>() -> Matrix<f64, N, N> {
    Matrix::from_fn(|i, j| 1.0 / (i + j + 1) as f64)
}

#[test]
fn solves_a_small_system() {
    let a = Matrix::new([[2.0f64, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]);
    let x = a.gauss_solve(&Vector::new([8.0, -11.0, -3.0])).unwrap();
    for (got, want) in x.as_slice().iter().zip([2.0, 3.0, -1.0]) {
        assert!((got - want).abs() < 1e-13);
    }
}

#[test]
fn requires_pivoting() {
    let a = Matrix::new([[0.0, 1.0], [1.0, 0.0]]);
    let x = a.gauss_solve(&Vector::new([3.0, 4.0])).unwrap();
    assert_eq!(x.as_slice(), &[4.0, 3.0]);
    // winziger Pivot: ohne Pivotsuche würde die Lösung zerstört
    let t = Matrix::new([[1e-20f64, 1.0], [1.0, 1.0]]);
    let x = t.gauss_solve(&Vector::new([1.0, 2.0])).unwrap();
    assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12);
}

#[test]
fn agrees_with_lu_solve() {
    let mut s = 42u64;
    for _ in 0..20 {
        let a = Matrix::<f64, 6, 6>::from_fn(|_, _| lcg(&mut s));
        let b = Vector::<f64, 6>::new([(); 6].map(|_| lcg(&mut s)));
        let (g, l) = (a.gauss_solve(&b).unwrap(), a.solve(&b).unwrap());
        for k in 0..6 {
            assert!(
                (g[k] - l[k]).abs() <= 1e-9 * (1.0 + l[k].abs()),
                "{} gegen {}",
                g[k],
                l[k]
            );
        }
    }
    // Hilbert-Matrix: schlecht konditioniert, beide Wege liefern dasselbe
    let h = hilbert::<6>();
    let b = h * Vector::new([1.0; 6]);
    let (g, l) = (h.gauss_solve(&b).unwrap(), h.solve(&b).unwrap());
    for k in 0..6 {
        assert!((g[k] - l[k]).abs() < 1e-8, "{k}: {} gegen {}", g[k], l[k]);
        assert!((g[k] - 1.0).abs() < 1e-6);
    }
}

#[test]
fn singular_and_non_finite() {
    let sing = Matrix::new([[1.0, 2.0], [2.0, 4.0]]);
    assert_eq!(
        sing.gauss_solve(&Vector::new([1.0, 2.0])),
        Err(LinalgError::Singular)
    );
    assert_eq!(
        Matrix::<f64, 2, 2>::zeros().gauss_solve(&Vector::new([0.0, 0.0])),
        Err(LinalgError::Singular)
    );
    let nan = Matrix::new([[1.0, f64::NAN], [0.0, 1.0]]);
    assert_eq!(
        nan.gauss_solve(&Vector::new([1.0, 1.0])),
        Err(LinalgError::NotFinite)
    );
    let ok = Matrix::<f64, 2, 2>::identity();
    assert_eq!(
        ok.gauss_solve(&Vector::new([f64::INFINITY, 1.0])),
        Err(LinalgError::NotFinite)
    );
}

#[test]
fn rank_of_various_matrices() {
    assert_eq!(Matrix::<f64, 3, 3>::zeros().rank().unwrap(), 0);
    assert_eq!(Matrix::<f64, 3, 3>::identity().rank().unwrap(), 3);
    assert_eq!(Matrix::new([[1.0, 2.0], [2.0, 4.0]]).rank().unwrap(), 1);
    assert_eq!(
        Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
            .rank()
            .unwrap(),
        2
    );
    // rechteckig
    assert_eq!(
        Matrix::new([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0]])
            .rank()
            .unwrap(),
        2
    );
    assert_eq!(
        Matrix::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            .rank()
            .unwrap(),
        2
    );
    assert_eq!(
        Matrix::new([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
            .rank()
            .unwrap(),
        1
    );
    // Rang gleich dem der SVD bei gut konditionierten Matrizen
    let a = Matrix::new([
        [2.0, 1.0, 0.0],
        [1.0, 2.0, 1.0],
        [0.0, 1.0, 2.0],
        [1.0, 1.0, 1.0],
    ]);
    assert_eq!(a.rank().unwrap(), a.svd().unwrap().rank());
    assert_eq!(
        Matrix::new([[1.0, f64::NAN]]).rank(),
        Err(LinalgError::NotFinite)
    );
}

#[test]
fn row_echelon_form() {
    let a = Matrix::new([[1.0, 2.0, 3.0], [2.0, 4.0, 7.0], [3.0, 6.0, 10.0]]);
    let e = a.row_echelon().unwrap();
    assert_eq!(e.rank, 2);
    assert_eq!(e.pivots(), &[0, 2]);
    // unterhalb der Pivots stehen Nullen, die Nullzeile ganz unten
    assert_eq!(e.matrix[(1, 0)], 0.0);
    assert_eq!(e.matrix[(2, 0)], 0.0);
    assert_eq!(e.matrix[(2, 1)], 0.0);
    assert_eq!(e.matrix[(2, 2)], 0.0);
    // Stufenform bleibt zeilenäquivalent: gleicher Rang
    assert_eq!(e.matrix.rank().unwrap(), 2);
}

#[test]
fn works_for_f32() {
    let a = Matrix::new([[4.0f32, 1.0], [1.0, 3.0]]);
    let x = a.gauss_solve(&Vector::new([1.0, 2.0])).unwrap();
    assert!((x[0] - 1.0 / 11.0).abs() < 1e-6 && (x[1] - 7.0 / 11.0).abs() < 1e-6);
}

#[cfg(feature = "alloc")]
mod dynamic {
    use super::*;
    use azmath::linalg::{DynMatrix, DynVector};

    fn dyn_of<const R: usize, const C: usize>(a: &Matrix<f64, R, C>) -> DynMatrix<f64> {
        DynMatrix::new(R, C, a.as_array().iter().flatten().copied().collect()).unwrap()
    }

    #[test]
    fn dynamic_matches_fixed_size() {
        let h = hilbert::<5>();
        let b = h * Vector::new([1.0; 5]);
        let xf = h.gauss_solve(&b).unwrap();
        let xd = dyn_of(&h)
            .gauss_solve(&DynVector::new(b.as_slice().to_vec()))
            .unwrap();
        assert_eq!(xd.as_slice(), xf.as_slice());

        let a = Matrix::new([[1.0, 2.0, 3.0], [2.0, 4.0, 7.0], [3.0, 6.0, 10.0]]);
        let (ef, ed) = (a.row_echelon().unwrap(), dyn_of(&a).row_echelon().unwrap());
        assert_eq!(ed.rank, ef.rank);
        assert_eq!(ed.pivots.as_slice(), ef.pivots());
        assert_eq!(ed.matrix.as_slice(), ef.matrix.as_array().as_flattened());
        assert_eq!(dyn_of(&a).rank().unwrap(), 2);
    }

    #[test]
    fn dynamic_errors() {
        let tall = DynMatrix::new(3, 2, vec![1.0; 6]).unwrap();
        assert_eq!(
            tall.gauss_solve(&DynVector::new(vec![1.0; 3])),
            Err(LinalgError::DimensionMismatch)
        );
        let sq = DynMatrix::<f64>::identity(2);
        assert_eq!(
            sq.gauss_solve(&DynVector::new(vec![1.0; 3])),
            Err(LinalgError::DimensionMismatch)
        );
        let sing = DynMatrix::new(2, 2, vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        assert_eq!(
            sing.gauss_solve(&DynVector::new(vec![1.0, 2.0])),
            Err(LinalgError::Singular)
        );
    }
}
