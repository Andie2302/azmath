//! Tests für die Sparse-Formate (CSR, COO) und ihren Einsatz mit den iterativen Lösern.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::sparse::{CsrView, SparseError};

#[test]
fn view_basics_and_validation() {
    // [[2, 0, 1], [0, 3, 0]]
    let (rp, ci, v) = ([0usize, 2, 3], [0usize, 2, 1], [2.0f64, 1.0, 3.0]);
    let a = CsrView::new(2, 3, &rp, &ci, &v).unwrap();
    assert_eq!((a.rows(), a.cols(), a.nnz()), (2, 3, 3));
    assert_eq!(a.get(0, 2), Some(1.0));
    assert_eq!(a.get(0, 1), Some(0.0));
    assert_eq!(a.get(2, 0), None);
    assert_eq!(a.get(0, 3), None);
    assert_eq!(a.row(0).collect::<Vec<_>>(), vec![(0, 2.0), (2, 1.0)]);
    let mut y = [0.0; 2];
    a.matvec(&[1.0, 2.0, 3.0], &mut y).unwrap();
    assert_eq!(y, [5.0, 6.0]);
    let mut yt = [0.0; 3];
    a.matvec_transposed(&[1.0, 1.0], &mut yt).unwrap();
    assert_eq!(yt, [2.0, 3.0, 1.0]);
    assert_eq!(
        a.matvec(&[1.0], &mut y),
        Err(SparseError::DimensionMismatch)
    );
    let mut d = [0.0; 2];
    a.diagonal_into(&mut d).unwrap();
    assert_eq!(d, [2.0, 3.0]);
    assert!(!a.is_symmetric(1e-12));
}

#[test]
fn view_rejects_invalid_structures() {
    let v = [1.0f64, 2.0, 3.0];
    // row_ptr zu kurz / falscher Anfang / Ende passt nicht / nicht monoton
    assert_eq!(
        CsrView::new(2, 3, &[0, 3], &[0, 1, 2], &v),
        Err(SparseError::InvalidStructure)
    );
    assert_eq!(
        CsrView::new(2, 3, &[1, 2, 3], &[0, 1, 2], &v),
        Err(SparseError::InvalidStructure)
    );
    assert_eq!(
        CsrView::new(2, 3, &[0, 1, 2], &[0, 1, 2], &v),
        Err(SparseError::InvalidStructure)
    );
    assert_eq!(
        CsrView::new(2, 3, &[0, 3, 2], &[0, 1, 2], &v),
        Err(SparseError::InvalidStructure)
    );
    // Spalte außerhalb, unsortiert, doppelt
    assert_eq!(
        CsrView::new(1, 2, &[0, 3], &[0, 1, 2], &v),
        Err(SparseError::IndexOutOfBounds)
    );
    assert_eq!(
        CsrView::new(1, 3, &[0, 3], &[0, 2, 1], &v),
        Err(SparseError::InvalidStructure)
    );
    assert_eq!(
        CsrView::new(1, 3, &[0, 3], &[0, 1, 1], &v),
        Err(SparseError::InvalidStructure)
    );
    // Wertezahl passt nicht, NaN
    assert_eq!(
        CsrView::new(1, 3, &[0, 3], &[0, 1, 2], &v[..2]),
        Err(SparseError::InvalidStructure)
    );
    assert_eq!(
        CsrView::new(1, 1, &[0, 1], &[0], &[f64::NAN]),
        Err(SparseError::NotFinite)
    );
    // leere Matrix ist gültig
    assert!(CsrView::<f64>::new(0, 0, &[0], &[], &[]).is_ok());
}

#[cfg(feature = "alloc")]
mod owned {
    use super::*;
    use azmath::linalg::iterative::{IterOptions, bicgstab, cg, gmres};
    use azmath::linalg::{DynMatrix, DynVector, LinearOperator, Matrix};
    use azmath::multigrid::Poisson2d;
    use azmath::sparse::{CooMatrix, CsrMatrix};

    /// Deterministische Pseudozufallszahlen in [-1, 1).
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    fn random_dense(rows: usize, cols: usize, density: f64, seed: u64) -> DynMatrix<f64> {
        let mut s = seed;
        let data = (0..rows * cols)
            .map(|_| {
                let keep = (lcg(&mut s) + 1.0) / 2.0 < density;
                let v = lcg(&mut s);
                if keep { v } else { 0.0 }
            })
            .collect();
        DynMatrix::new(rows, cols, data).unwrap()
    }

    fn poisson_1d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 2.0).unwrap();
            if i + 1 < n {
                coo.push(i, i + 1, -1.0).unwrap();
                coo.push(i + 1, i, -1.0).unwrap();
            }
        }
        CsrMatrix::from_coo(&coo)
    }

    /// 5-Punkt-Stern des 2D-Laplace auf einem `n × n`-Gitter (ohne Faktor 1/h²).
    pub fn poisson_2d(n: usize) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(n * n, n * n);
        for i in 0..n {
            for j in 0..n {
                let k = i * n + j;
                coo.push(k, k, 4.0).unwrap();
                if i > 0 {
                    coo.push(k, k - n, -1.0).unwrap();
                }
                if i + 1 < n {
                    coo.push(k, k + n, -1.0).unwrap();
                }
                if j > 0 {
                    coo.push(k, k - 1, -1.0).unwrap();
                }
                if j + 1 < n {
                    coo.push(k, k + 1, -1.0).unwrap();
                }
            }
        }
        CsrMatrix::from_coo(&coo)
    }

    #[test]
    fn coo_sums_duplicates_and_sorts() {
        let mut coo = CooMatrix::with_capacity(3, 3, 8);
        for &(i, j, v) in &[
            (2, 1, 1.0),
            (0, 2, 5.0),
            (0, 0, 1.0),
            (2, 1, 4.0),
            (0, 0, -1.0),
            (1, 1, 7.0),
        ] {
            coo.push(i, j, v).unwrap();
        }
        assert_eq!(coo.len(), 6);
        let a = CsrMatrix::from_coo(&coo);
        a.validate().unwrap();
        assert_eq!(a.nnz(), 4); // (0,0) bleibt als explizite 0 gespeichert
        assert_eq!(a.get(0, 0), Some(0.0));
        assert_eq!(a.get(2, 1), Some(5.0));
        assert_eq!(a.row(0).map(|(j, _)| j).collect::<Vec<_>>(), vec![0, 2]);
        assert_eq!(coo.push(3, 0, 1.0), Err(SparseError::IndexOutOfBounds));
        assert_eq!(coo.push(0, 0, f64::NAN), Err(SparseError::NotFinite));
        let empty = CsrMatrix::<f64>::from_coo(&CooMatrix::new(4, 2));
        assert_eq!((empty.rows(), empty.cols(), empty.nnz()), (4, 2, 0));
    }

    #[test]
    fn matvec_matches_the_dense_product() {
        for (rows, cols, density) in [(7, 5, 0.3), (12, 12, 0.15), (3, 9, 0.6), (20, 20, 0.05)] {
            let dense = random_dense(rows, cols, density, (rows * 31 + cols) as u64);
            let a = CsrMatrix::from_dense(&dense, 0.0).unwrap();
            a.validate().unwrap();
            let mut s = 99;
            let x = DynVector::new((0..cols).map(|_| lcg(&mut s)).collect());
            let want = dense.mul_vec(&x).unwrap();
            let mut got = vec![0.0; rows];
            a.matvec(x.as_slice(), &mut got).unwrap();
            for i in 0..rows {
                assert!((got[i] - want[i]).abs() < 1e-13);
            }
            // über den Operator
            let mut got2 = vec![0.0; rows];
            LinearOperator::apply(&a, x.as_slice(), &mut got2);
            assert_eq!(got, got2);
            // Transponierte: Aᵀ·u über matvec_transposed und über transpose().matvec
            let u: Vec<f64> = (0..rows).map(|_| lcg(&mut s)).collect();
            let mut t1 = vec![0.0; cols];
            let mut t2 = vec![0.0; cols];
            a.matvec_transposed(&u, &mut t1).unwrap();
            a.transpose().matvec(&u, &mut t2).unwrap();
            for j in 0..cols {
                assert!((t1[j] - t2[j]).abs() < 1e-13);
            }
        }
    }

    #[test]
    fn dense_roundtrip_and_conversions() {
        let dense = random_dense(6, 8, 0.4, 5);
        let a = CsrMatrix::from_dense(&dense, 0.0).unwrap();
        assert_eq!(a.to_dense(), dense);
        assert_eq!(a.transpose().transpose(), a);
        assert_eq!(a.transpose().to_dense(), dense.transpose());
        // Schwelle: kleine Einträge fallen weg
        let thin = CsrMatrix::from_dense(&dense, 0.5).unwrap();
        assert!(thin.nnz() < a.nnz());
        assert!(thin.values().iter().all(|v| v.abs() > 0.5));
        // feste Größe
        let m = Matrix::new([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]);
        let f = CsrMatrix::from_matrix(&m, 0.0).unwrap();
        assert_eq!(f.nnz(), 3);
        assert_eq!(f.diagonal(), vec![1.0, 2.0]);
        assert_eq!(
            CsrMatrix::<f64>::from_dense_slice(2, 2, &[1.0], 0.0),
            Err(SparseError::DimensionMismatch)
        );
        assert_eq!(
            CsrMatrix::from_dense_slice(1, 1, &[f64::INFINITY], 0.0),
            Err(SparseError::NotFinite)
        );
    }

    #[test]
    fn add_scale_identity_and_symmetry() {
        let a = random_dense(5, 5, 0.4, 1);
        let b = random_dense(5, 5, 0.4, 2);
        let (sa, sb) = (
            CsrMatrix::from_dense(&a, 0.0).unwrap(),
            CsrMatrix::from_dense(&b, 0.0).unwrap(),
        );
        let sum = sa.add(&sb).unwrap();
        sum.validate().unwrap();
        for i in 0..5 {
            for j in 0..5 {
                assert!((sum.get(i, j).unwrap() - (a[(i, j)] + b[(i, j)])).abs() < 1e-15);
            }
        }
        let other = CsrMatrix::<f64>::identity(4);
        assert_eq!(sa.add(&other), Err(SparseError::DimensionMismatch));

        let mut m = sa.clone();
        let rs: Vec<f64> = (1..=5).map(|k| k as f64).collect();
        m.scale_rows(&rs).unwrap();
        m.scale_cols(&rs).unwrap();
        for i in 0..5 {
            for j in 0..5 {
                assert!((m.get(i, j).unwrap() - a[(i, j)] * rs[i] * rs[j]).abs() < 1e-13);
            }
        }
        assert_eq!(m.scale_rows(&rs[..2]), Err(SparseError::DimensionMismatch));

        assert!(poisson_1d(6).is_symmetric(1e-14));
        assert!(!sa.is_symmetric(1e-14));
        let id = CsrMatrix::<f64>::identity(3);
        assert_eq!(id.to_dense(), DynMatrix::identity(3));
        // Wertezugriff bei festem Muster
        let mut p = poisson_1d(4);
        p.values_mut().iter_mut().for_each(|v| *v *= 2.0);
        assert_eq!(p.get(0, 0), Some(4.0));
        assert!(CsrMatrix::from_parts(1, 2, vec![0, 1], vec![5], vec![1.0]).is_err());
    }

    #[test]
    fn poisson_2d_operator_matches_the_multigrid_stencil() {
        let n = 9;
        let a = poisson_2d(n);
        let h = 1.0 / (n + 1) as f64;
        let mut s = 7;
        let x: Vec<f64> = (0..n * n).map(|_| lcg(&mut s)).collect();
        let mut y1 = vec![0.0; n * n];
        let mut y2 = vec![0.0; n * n];
        a.matvec(&x, &mut y1).unwrap();
        LinearOperator::apply(&Poisson2d::new(n), &x[..], &mut y2);
        for k in 0..n * n {
            assert!((y1[k] / (h * h) - y2[k]).abs() < 1e-8 * (1.0 + y2[k].abs()));
        }
    }

    #[test]
    fn iterative_solvers_work_with_csr() {
        let n = 12;
        let a = poisson_2d(n);
        let size = n * n;
        let x_true: Vec<f64> = (0..size).map(|k| ((k % 7) as f64) - 3.0).collect();
        let mut b = vec![0.0; size];
        a.matvec(&x_true, &mut b).unwrap();
        let opts = IterOptions::new(1e-11, 2000);
        for solver in 0..3 {
            let mut x = vec![0.0; size];
            match solver {
                0 => {
                    let mut work = vec![0.0; 4 * size];
                    cg(&a, &b, &mut x, opts, &mut work).unwrap();
                }
                1 => {
                    let mut work = vec![0.0; 8 * size];
                    bicgstab(&a, &b, &mut x, opts, &mut work).unwrap();
                }
                _ => {
                    let restart = 30;
                    let mut work =
                        vec![0.0; azmath::linalg::iterative::gmres_work_len(size, restart)];
                    gmres(&a, &b, &mut x, restart, opts, &mut work).unwrap();
                }
            }
            for k in 0..size {
                assert!(
                    (x[k] - x_true[k]).abs() < 1e-7,
                    "Löser {solver}: x[{k}] = {}",
                    x[k]
                );
            }
        }
    }

    #[test]
    fn large_systems_are_cheap() {
        // 1D-Laplace mit 3000 Unbekannten (CG braucht dafür etwa n Schritte)
        let n = 3_000;
        let a = poisson_1d(n);
        assert_eq!(a.nnz(), 3 * n - 2);
        let b = vec![1.0; n];
        let mut x = vec![0.0; n];
        let mut work = vec![0.0; 4 * n];
        let r = cg(&a, &b, &mut x, IterOptions::new(1e-8, 100_000), &mut work).unwrap();
        assert!(r.iterations > 0);
        let mut res = vec![0.0; n];
        a.matvec(&x, &mut res).unwrap();
        let err = res
            .iter()
            .zip(&b)
            .map(|(r, b)| (r - b).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-5, "Residuum {err}");
    }

    #[test]
    fn works_for_f32() {
        let mut coo = CooMatrix::new(2, 2);
        coo.push(0, 0, 4.0f32).unwrap();
        coo.push(0, 1, 1.0).unwrap();
        coo.push(1, 0, 1.0).unwrap();
        coo.push(1, 1, 3.0).unwrap();
        let a = CsrMatrix::from_coo(&coo);
        let mut x = [0.0f32; 2];
        let mut work = [0.0f32; 8];
        cg(
            &a,
            &[1.0f32, 2.0],
            &mut x,
            IterOptions::new(1e-6, 50),
            &mut work,
        )
        .unwrap();
        assert!((x[0] - 1.0 / 11.0).abs() < 1e-5 && (x[1] - 7.0 / 11.0).abs() < 1e-5);
    }
}
