#![cfg(any(feature = "std", feature = "libm"))]
use azmath::linalg::{Inertia, LinalgError, Matrix, Vector};
macro_rules! checks {
    ($name:ident,$t:ty,$tiny:expr,$large:expr) => {
        mod $name {
            use super::*;
            #[test]
            fn zero_diagonal_requires_two_by_two_pivot() {
                let a = Matrix::new([[0.0 as $t, 2.0], [2.0, 0.0]]);
                let ld = a.ldlt().unwrap();
                assert_eq!(ld.block_sizes(), &[2, 0]);
                assert_eq!(
                    ld.inertia(),
                    Inertia {
                        positive: 1,
                        negative: 1,
                        zero: 0
                    }
                );
                let x = ld.solve(&Vector::new([6.0, 4.0])).unwrap();
                assert_eq!(x, Vector::new([2.0, 3.0]));
                assert_eq!(ld.l(), Matrix::identity());
                assert_eq!(ld.d().unwrap(), a);
            }
            #[test]
            fn reconstruction_solve_and_eigenvalue_inertia_on_varied_symmetric_matrices() {
                let mut seed = 7919u64;
                for _ in 0..40 {
                    let mut a = Matrix::<$t, 6, 6>::zeros();
                    for i in 0..6 {
                        for j in 0..=i {
                            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                            let v = ((seed >> 32) % 19) as i32 - 9;
                            a[(i, j)] = v as $t;
                            a[(j, i)] = v as $t;
                        }
                    }
                    let ld = a.ldlt().unwrap();
                    let l = ld.l();
                    let d = ld.d().unwrap();
                    let rec = l * d * l.transpose();
                    let p = ld.permutation();
                    let tol = 1000.0 * <$t>::EPSILON;
                    for i in 0..6 {
                        for j in 0..6 {
                            assert!(
                                (rec[(i, j)] - a[(p[i], p[j])]).abs() < tol * 20.0,
                                "reconstruction {:?}",
                                p
                            );
                        }
                    }
                    let eig = a.symmetric_eigen().unwrap();
                    let mut reference = Inertia {
                        positive: 0,
                        negative: 0,
                        zero: 0,
                    };
                    for i in 0..6 {
                        let v = eig.values[i];
                        if v.abs() < tol {
                            reference.zero += 1;
                        } else if v > 0.0 {
                            reference.positive += 1;
                        } else {
                            reference.negative += 1;
                        }
                    }
                    if reference.zero == 0 {
                        assert_eq!(ld.inertia(), reference);
                    }
                    if ld.inertia().zero == 0 {
                        let target = Vector::new([1.0 as $t, -2.0, 3.0, -4.0, 0.5, 1.5]);
                        let b = a * target;
                        let x = ld.solve(&b).unwrap();
                        let ax = a * x;
                        for i in 0..6 {
                            assert!((ax[i] - b[i]).abs() < tol * 100.0);
                        }
                    }
                }
            }
            #[test]
            fn global_scale_and_pivot_permutation() {
                for s in [$tiny as $t, 1.0, $large] {
                    let a =
                        Matrix::new([[0.0, 0.0, 2.0 * s], [0.0, 3.0 * s, s], [2.0 * s, s, 0.0]]);
                    let ld = a.ldlt().unwrap();
                    assert_eq!(ld.permutation(), &[0, 2, 1]);
                    assert_eq!(
                        ld.inertia(),
                        Inertia {
                            positive: 2,
                            negative: 1,
                            zero: 0
                        }
                    );
                    let x = ld.solve(&Vector::new([6.0 * s, 9.0 * s, 4.0 * s])).unwrap();
                    for (a, b) in x.as_array().iter().zip([1.0, 2.0, 3.0]) {
                        assert!((*a - b).abs() < 100.0 * <$t>::EPSILON);
                    }
                }
                for value in [<$t>::MAX, <$t>::MIN_POSITIVE, <$t>::from_bits(1)] {
                    let ld = Matrix::new([[value]]).ldlt().unwrap();
                    assert_eq!(ld.solve(&Vector::new([value])).unwrap()[0], 1.0);
                }
            }
            #[test]
            fn singular_blocks_and_tolerance_are_explicit() {
                let ld = Matrix::new([[-2.0 as $t, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
                    .ldlt()
                    .unwrap();
                assert_eq!(
                    ld.inertia(),
                    Inertia {
                        positive: 1,
                        negative: 1,
                        zero: 1
                    }
                );
                assert_eq!(ld.solve(&Vector::zeros()), Err(LinalgError::Singular));
                let eps = <$t>::EPSILON;
                let ld = Matrix::new([[eps / 4.0, 0.0], [0.0, 1.0]]).ldlt().unwrap();
                assert_eq!(ld.inertia().zero, 1);
                assert_eq!(ld.inertia_with_tolerance(0.0).unwrap().positive, 2);
                for tol in [-1.0, <$t>::NAN, <$t>::INFINITY] {
                    assert_eq!(
                        ld.inertia_with_tolerance(tol),
                        Err(LinalgError::InvalidInput)
                    );
                }
                let empty = Matrix::<$t, 0, 0>::zeros().ldlt().unwrap();
                assert_eq!(
                    empty.inertia(),
                    Inertia {
                        positive: 0,
                        negative: 0,
                        zero: 0
                    }
                );
                assert_eq!(empty.solve(&Vector::zeros()).unwrap(), Vector::zeros());
            }
            #[test]
            fn rejects_asymmetry_nonfinite_data_and_unrepresentable_solutions() {
                assert_eq!(
                    Matrix::new([[1.0 as $t, 2.0], [0.0, 1.0]]).ldlt(),
                    Err(LinalgError::NotSymmetric)
                );
                for v in [<$t>::NAN, <$t>::INFINITY] {
                    assert_eq!(Matrix::new([[v]]).ldlt(), Err(LinalgError::NotFinite));
                    let ld = Matrix::new([[1.0 as $t]]).ldlt().unwrap();
                    assert_eq!(ld.solve(&Vector::new([v])), Err(LinalgError::NotFinite));
                }
                let ld = Matrix::new([[<$t>::MIN_POSITIVE]]).ldlt().unwrap();
                assert_eq!(
                    ld.solve(&Vector::new([<$t>::MAX])),
                    Err(LinalgError::NotFinite)
                );
            }
        }
    };
}
checks!(single, f32, 1e-20, 1e20);
checks!(double, f64, 1e-200, 1e200);
#[cfg(feature = "alloc")]
#[test]
fn dynamic_fixed_parity_and_dimensions() {
    use azmath::linalg::{DynMatrix, DynVector};
    let a = Matrix::new([[0.0f64, 0.0, 2.0], [0.0, 3.0, 1.0], [2.0, 1.0, 0.0]]);
    let fixed = a.ldlt().unwrap();
    let d = DynMatrix::new(3, 3, a.as_array().as_flattened().to_vec())
        .unwrap()
        .ldlt()
        .unwrap();
    assert_eq!(d.permutation(), fixed.permutation());
    assert_eq!(d.block_sizes(), fixed.block_sizes());
    assert_eq!(d.inertia(), fixed.inertia());
    assert_eq!(d.l().as_slice(), fixed.l().as_array().as_flattened());
    assert_eq!(
        d.d().unwrap().as_slice(),
        fixed.d().unwrap().as_array().as_flattened()
    );
    let b = Vector::new([6.0, 9.0, 4.0]);
    assert_eq!(
        d.solve(&DynVector::new(b.as_array().to_vec()))
            .unwrap()
            .as_slice(),
        fixed.solve(&b).unwrap().as_array()
    );
    assert_eq!(
        d.solve(&DynVector::zeros(1)),
        Err(LinalgError::DimensionMismatch)
    );
    assert_eq!(
        DynMatrix::<f64>::zeros(2, 3).ldlt(),
        Err(LinalgError::DimensionMismatch)
    );
}

#[test]
fn scalar_pivot_swap_and_saddle_point_system_match_lu() {
    let a = Matrix::new([[0.0f64, 1.0, 2.0], [1.0, 10.0, 0.0], [2.0, 0.0, 20.0]]);
    let ld = a.ldlt().unwrap();
    assert_eq!(ld.permutation()[0], 2);
    assert_eq!(ld.block_sizes()[0], 1);
    let kkt = Matrix::new([
        [2.0f64, 0.5, 1.0, 0.0],
        [0.5, 3.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]);
    let b = Vector::new([2.0, 1.0, -1.0, 3.0]);
    let ld = kkt.ldlt().unwrap();
    assert_eq!(
        ld.inertia(),
        Inertia {
            positive: 2,
            negative: 2,
            zero: 0
        }
    );
    let x = ld.solve(&b).unwrap();
    let reference = kkt.solve(&b).unwrap();
    for i in 0..4 {
        assert!((x[i] - reference[i]).abs() < 1e-13);
    }
}

#[test]
fn scaled_factors_can_solve_even_when_unscaled_d_is_not_representable() {
    let m = f64::MAX;
    let ld = Matrix::new([[m, m], [m, -m]]).ldlt().unwrap();
    assert_eq!(ld.d(), Err(LinalgError::NotFinite));
    assert_eq!(
        ld.solve(&Vector::new([m, m])).unwrap(),
        Vector::new([1.0, 0.0])
    );
}
