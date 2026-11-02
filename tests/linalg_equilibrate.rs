//! Tests für die Äquilibrierung (Ruiz, Sinkhorn).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::linalg::{EquilibrateOptions, LinalgError, Matrix, Vector};

type Opts = EquilibrateOptions<f64>;

/// Schlecht skalierte, aber gut konditionierte Matrix: `D1·B·D2` mit Faktoren um 1e±8.
fn badly_scaled() -> Matrix<f64, 3, 3> {
    let b = Matrix::new([[2.0, 1.0, 0.5], [1.0, 3.0, 1.0], [0.5, 1.0, 4.0]]);
    let d1 = [1e8, 1.0, 1e-8];
    let d2 = [1e-6, 1.0, 1e6];
    Matrix::from_fn(|i, j| d1[i] * b[(i, j)] * d2[j])
}

fn cond(a: &Matrix<f64, 3, 3>) -> f64 {
    a.svd().unwrap().condition_number()
}

#[test]
fn ruiz_reduces_the_condition_number_drastically() {
    let a = badly_scaled();
    let e = a.equilibrate_ruiz(&Opts::default()).unwrap();
    assert!(cond(&a) > 1e12, "Ausgangslage: {}", cond(&a));
    assert!(cond(&e.matrix) < 100.0, "danach: {}", cond(&e.matrix));
}

#[test]
fn ruiz_without_pow2_reaches_unit_maxima() {
    let a = badly_scaled();
    let opts = Opts {
        pow2: false,
        ..Opts::default()
    };
    let e = a.equilibrate_ruiz(&opts).unwrap();
    for i in 0..3 {
        let row = (0..3).map(|j| e.matrix[(i, j)].abs()).fold(0.0, f64::max);
        let col = (0..3).map(|j| e.matrix[(j, i)].abs()).fold(0.0, f64::max);
        assert!((row - 1.0).abs() < 1e-7 && (col - 1.0).abs() < 1e-7);
    }
}

#[test]
fn pow2_scaling_is_exact_and_reversible() {
    let a = badly_scaled();
    let e = a.equilibrate_ruiz(&Opts::default()).unwrap();
    for &s in e.row_scale.as_slice().iter().chain(e.col_scale.as_slice()) {
        assert_eq!(s.log2().fract(), 0.0, "{s} ist keine Zweierpotenz");
    }
    // R⁻¹·Â·C⁻¹ == A, bitgleich
    for i in 0..3 {
        for j in 0..3 {
            let back = e.matrix[(i, j)] / e.row_scale[i] / e.col_scale[j];
            assert_eq!(back, a[(i, j)]);
        }
    }
}

#[test]
fn sinkhorn_gives_unit_row_and_column_sums() {
    let a = Matrix::new([[1e4, 2.0, 0.0], [3.0, 1e-3, 5.0], [0.5, 7.0, 1e2]]);
    let opts = Opts {
        pow2: false,
        ..Opts::default()
    };
    let e = a.equilibrate_sinkhorn(&opts).unwrap();
    for k in 0..3 {
        let row: f64 = (0..3).map(|j| e.matrix[(k, j)].abs()).sum();
        let col: f64 = (0..3).map(|i| e.matrix[(i, k)].abs()).sum();
        assert!((row - 1.0).abs() < 1e-6, "Zeile {k}: {row}");
        assert!((col - 1.0).abs() < 1e-6, "Spalte {k}: {col}");
    }
}

#[test]
fn rectangular_matrices() {
    let tall = Matrix::new([[1e5, 1.0], [2.0, 3e-5], [4.0, 5.0]]);
    let opts = Opts {
        pow2: false,
        ..Opts::default()
    };
    let e = tall.equilibrate_sinkhorn(&opts).unwrap();
    for i in 0..3 {
        let row: f64 = (0..2).map(|j| e.matrix[(i, j)].abs()).sum();
        assert!((row - 1.0).abs() < 1e-6);
    }
    for j in 0..2 {
        let col: f64 = (0..3).map(|i| e.matrix[(i, j)].abs()).sum();
        assert!(
            (col - 1.5).abs() < 1e-6,
            "Spaltensumme {col} statt m/n = 1.5"
        );
    }
    let wide = tall.transpose();
    let e = wide.equilibrate_ruiz(&Opts::default()).unwrap();
    assert!(e.matrix.max_abs() <= 2.0 + 1e-9);
}

#[test]
fn solving_through_the_scaled_system_matches() {
    let a = badly_scaled();
    let x_true = Vector::new([1.0, -2.0, 3.0]);
    let b = a * x_true;
    let e = a.equilibrate_ruiz(&Opts::default()).unwrap();
    let y = e.matrix.solve(&e.scale_rhs(&b)).unwrap();
    let x = e.recover_solution(&y);
    for k in 0..3 {
        assert!(
            (x[k] - x_true[k]).abs() <= 1e-10 * x_true[k].abs(),
            "x[{k}] = {}",
            x[k]
        );
    }
}

#[test]
fn zero_rows_and_columns_keep_scale_one() {
    let a = Matrix::new([[0.0, 0.0, 0.0], [1e4, 0.0, 2.0], [3.0, 0.0, 1e-4]]);
    let e = a.equilibrate_ruiz(&Opts::default()).unwrap();
    assert_eq!(e.row_scale[0], 1.0);
    assert_eq!(e.col_scale[1], 1.0);
    assert!(e.matrix.max_abs() <= 2.0 + 1e-9);
    let z = Matrix::<f64, 2, 2>::zeros();
    let e = z.equilibrate_sinkhorn(&Opts::default()).unwrap();
    assert_eq!(e.iterations, 0);
    assert_eq!(e.matrix, z);
}

#[test]
fn errors() {
    let nan = Matrix::new([[1.0, f64::NAN], [0.0, 1.0]]);
    assert_eq!(
        nan.equilibrate_ruiz(&Opts::default()),
        Err(LinalgError::NotFinite)
    );
    let inf = Matrix::new([[f64::INFINITY, 1.0], [1.0, 1.0]]);
    assert_eq!(
        inf.equilibrate_sinkhorn(&Opts::default()),
        Err(LinalgError::NotFinite)
    );
    // max_iter = 0 und schlecht skaliert: keine Konvergenz
    let none = Opts {
        max_iter: 0,
        ..Opts::default()
    };
    assert_eq!(
        badly_scaled().equilibrate_ruiz(&none),
        Err(LinalgError::NoConvergence)
    );
    // keine Gesamtstütze (obere Dreiecksmatrix): Sinkhorn konvergiert nicht
    let tri = Matrix::new([[1.0, 1.0], [0.0, 1.0]]);
    let few = Opts {
        max_iter: 200,
        ..Opts::default()
    };
    assert_eq!(
        tri.equilibrate_sinkhorn(&few),
        Err(LinalgError::NoConvergence)
    );
}

#[test]
fn works_for_f32() {
    let a = Matrix::new([[1.0e4f32, 2.0], [3.0, 4.0e-4]]);
    let e = a.equilibrate_ruiz(&EquilibrateOptions::default()).unwrap();
    assert!(e.matrix.max_abs() <= 2.0);
}

#[cfg(feature = "alloc")]
mod dynamic {
    use super::*;
    use azmath::linalg::{DynMatrix, DynVector};

    #[test]
    fn dynamic_matches_fixed_size() {
        let a = badly_scaled();
        let d = DynMatrix::new(3, 3, a.as_array().iter().flatten().copied().collect()).unwrap();
        for ruiz in [true, false] {
            let (f, g) = if ruiz {
                (
                    a.equilibrate_ruiz(&Opts::default()),
                    d.equilibrate_ruiz(&Opts::default()),
                )
            } else {
                (
                    a.equilibrate_sinkhorn(&Opts::default()),
                    d.equilibrate_sinkhorn(&Opts::default()),
                )
            };
            let (f, g) = (f.unwrap(), g.unwrap());
            assert_eq!(g.row_scale.as_slice(), f.row_scale.as_slice());
            assert_eq!(g.col_scale.as_slice(), f.col_scale.as_slice());
            assert_eq!(g.matrix.as_slice(), f.matrix.as_array().as_flattened());
            assert_eq!(g.iterations, f.iterations);
        }
        let e = d.equilibrate_ruiz(&Opts::default()).unwrap();
        assert_eq!(
            e.scale_rhs(&DynVector::new(vec![1.0, 1.0])),
            Err(LinalgError::DimensionMismatch)
        );
        let y = e
            .recover_solution(&DynVector::new(vec![1.0, 1.0, 1.0]))
            .unwrap();
        assert_eq!(y.as_slice(), e.col_scale.as_slice());
    }
}
