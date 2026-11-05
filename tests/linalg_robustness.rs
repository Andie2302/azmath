//! Wertebereich, NaN-Weitergabe und unabhängige Residuenprüfung für Normen/LU.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::linalg::{LinalgError, Matrix, Vector};

macro_rules! robustness_tests {
    ($module:ident, $t:ty, $big:expr, $tiny:expr, $scale:expr) => {
        mod $module {
            use super::*;

            #[test]
            fn vector_norm_extreme_scales() {
                for s in [$big, $tiny, 1.0] {
                    let v = Vector::new([3.0 * s, -4.0 * s]);
                    let norm = v.norm();
                    assert!(norm.is_finite() && norm > 0.0);
                    assert!((norm / s - 5.0).abs() < 16.0 * <$t>::EPSILON);
                    let matrix = Matrix::new([[3.0 * s, -4.0 * s]]);
                    assert_eq!(norm, matrix.norm_frobenius());
                }
                let smallest = <$t>::from_bits(1);
                assert_eq!(Vector::new([smallest, 0.0]).norm(), smallest);
                assert_eq!(Vector::<$t, 0>::zeros().norm(), 0.0);
                assert_eq!(Vector::<$t, 2>::zeros().norm(), 0.0);
                assert_eq!(Vector::new([<$t>::INFINITY, 0.0]).norm(), <$t>::INFINITY);
                assert!(Vector::new([<$t>::INFINITY, <$t>::NAN]).norm().is_nan());
            }

            #[test]
            fn max_norm_propagates_nan_in_every_position() {
                for i in 0..4 {
                    let mut values = [1.0 as $t, -2.0, 3.0, -4.0];
                    values[i] = <$t>::NAN;
                    assert!(Vector::new(values).max_abs().is_nan());
                    let matrix = Matrix::new([[values[0], values[1]], [values[2], values[3]]]);
                    assert!(matrix.max_abs().is_nan());
                    assert!(matrix.norm_frobenius().is_nan());
                    assert!(matrix.norm_1().is_nan());
                    assert!(matrix.norm_inf().is_nan());
                    #[cfg(feature = "alloc")]
                    {
                        use azmath::linalg::{DynMatrix, DynVector};
                        assert!(DynVector::new(values.to_vec()).max_abs().is_nan());
                        assert!(
                            DynMatrix::new(2, 2, values.to_vec())
                                .unwrap()
                                .max_abs()
                                .is_nan()
                        );
                    }
                }
            }

            #[test]
            fn lu_rejects_overflow_during_elimination() {
                let v = <$t>::MAX * 0.75;
                let a = Matrix::new([[v, v], [-v, v]]);
                assert_eq!(a.lu().unwrap_err(), LinalgError::NotFinite);
                #[cfg(feature = "alloc")]
                {
                    use azmath::linalg::DynMatrix;
                    let a = DynMatrix::new(2, 2, vec![v, v, -v, v]).unwrap();
                    assert_eq!(a.lu().unwrap_err(), LinalgError::NotFinite);
                }
            }

            #[test]
            fn lu_known_solutions_and_scaled_backward_error() {
                let expected = [1.0 as $t, -2.0, 0.5, 3.0];
                for seed in 0..24 {
                    for scale in [1.0, $scale, 1.0 / $scale] {
                        // Streng diagonaldominante Ausgangsmatrix, danach zyklisch
                        // vertauschte Zeilen: regulär, aber Pivotisierung erforderlich.
                        let data: [[$t; 4]; 4] = core::array::from_fn(|i| {
                            let row = (i + seed % 4) % 4;
                            core::array::from_fn(|j| {
                                let value = if row == j {
                                    16.0
                                } else {
                                    ((row * 7 + j * 3 + seed) % 7) as $t - 3.0
                                };
                                value * scale
                            })
                        });
                        // Rechte Seite separat von der Matrix-Implementierung bilden.
                        let rhs: [$t; 4] = core::array::from_fn(|i| {
                            (0..4).map(|j| data[i][j] * expected[j]).sum()
                        });
                        let a = Matrix::new(data);
                        let lu = a.lu().unwrap();
                        let x = lu.solve(&Vector::new(rhs));
                        let tolerance = 128.0 * <$t>::EPSILON;
                        for i in 0..4 {
                            assert!((x[i] - expected[i]).abs() <= tolerance);
                            // Komponentenweise rückwärtsgerichteter Fehler:
                            // |Ax-b| / (|A||x|+|b|), unabhängig berechnet.
                            let residual =
                                ((0..4).map(|j| data[i][j] * x[j]).sum::<$t>() - rhs[i]).abs();
                            let denominator = rhs[i].abs()
                                + (0..4).map(|j| data[i][j].abs() * x[j].abs()).sum::<$t>();
                            assert!(residual / denominator <= tolerance);
                        }
                        #[cfg(feature = "alloc")]
                        {
                            use azmath::linalg::{DynMatrix, DynVector};
                            let d =
                                DynMatrix::new(4, 4, data.into_iter().flatten().collect()).unwrap();
                            let dx = d.solve(&DynVector::new(rhs.to_vec())).unwrap();
                            for i in 0..4 {
                                assert!((dx[i] - expected[i]).abs() <= tolerance);
                            }
                        }
                    }
                }
            }

            #[test]
            fn lu_hilbert_backward_error() {
                // Hilbert(5) ist schlecht konditioniert (kappa_2 etwa 4.8e5).
                // Ein kleines Residuum allein garantiert daher keine genaue Lösung.
                let a = Matrix::<$t, 5, 5>::from_fn(|i, j| 1.0 / (i + j + 1) as $t);
                let b = Vector::new(core::array::from_fn(|i| {
                    (0..5).map(|j| 1.0 / (i + j + 1) as $t).sum()
                }));
                let x = a.solve(&b).unwrap();
                for i in 0..5 {
                    assert!(x[i].is_finite());
                    let ax = (0..5).map(|j| a[(i, j)] * x[j]).sum::<$t>();
                    let denominator =
                        b[i].abs() + (0..5).map(|j| a[(i, j)].abs() * x[j].abs()).sum::<$t>();
                    assert!((ax - b[i]).abs() / denominator <= 32.0 * <$t>::EPSILON);
                    // Die gerundete rechte Seite und die Kondition begrenzen den Vorwärtsfehler.
                    assert!((x[i] - 1.0).abs() <= 500_000.0 * <$t>::EPSILON);
                }
            }

            #[test]
            fn lu_singular_and_nonfinite_inputs() {
                let singular = Matrix::new([[1.0 as $t, 2.0], [2.0, 4.0]]);
                assert_eq!(singular.lu().unwrap_err(), LinalgError::Singular);
                for invalid in [<$t>::NAN, <$t>::INFINITY, <$t>::NEG_INFINITY] {
                    let a = Matrix::new([[1.0, invalid], [0.0, 1.0]]);
                    assert_eq!(a.lu().unwrap_err(), LinalgError::NotFinite);
                }
            }
        }
    };
}

robustness_tests!(double, f64, 1e200_f64, 1e-200_f64, 1e150_f64);
robustness_tests!(single, f32, 1e20_f32, 1e-30_f32, 1e15_f32);
