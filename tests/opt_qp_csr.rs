#![cfg(all(feature = "alloc", any(feature = "std", feature = "libm")))]
use azmath::{
    opt::{
        OptError, QpKktBackend, QpOptions, QpStatus, interior_point_lp_sparse, interior_point_qp,
        interior_point_qp_sparse,
    },
    sparse::{CooMatrix, CsrMatrix},
};
fn csr(rows: usize, cols: usize, v: &[f64]) -> CsrMatrix<f64> {
    CsrMatrix::from_dense_slice(rows, cols, v, 0.0).unwrap()
}
#[test]
fn csr_and_dense_share_solutions_and_original_kkt_residuals() {
    let q = [3.0, -0.2, 0.0, -0.2, 2.0, -0.1, 0.0, -0.1, 4.0];
    let c = [-3.0, -1.0, -2.0];
    let a = [1.0, 1.0, 1.0];
    let b = [1.5];
    let g = [
        -1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let h = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let (qs, as_, gs) = (csr(3, 3, &q), csr(1, 3, &a), csr(6, 3, &g));
    for backend in [
        QpKktBackend::Dense,
        QpKktBackend::Sparse,
        QpKktBackend::Auto,
    ] {
        let opts = QpOptions {
            kkt_backend: backend,
            ..QpOptions::default()
        };
        let d = interior_point_qp(&q, &c, &a, &b, &g, &h, opts).unwrap();
        let s = interior_point_qp_sparse(&qs, &c, &as_, &b, &gs, &h, opts).unwrap();
        assert_eq!(d.status, QpStatus::Optimal);
        assert_eq!(s.status, QpStatus::Optimal);
        assert!((d.objective - s.objective).abs() < 1e-8);
        assert!((s.x.iter().sum::<f64>() - 1.5).abs() < 1e-7);
        for i in 0..3 {
            assert!((d.x[i] - s.x[i]).abs() < 1e-8);
            let dual = (0..3).map(|j| q[3 * i + j] * s.x[j]).sum::<f64>() + c[i] + s.y[0] - s.z[i]
                + s.z[3 + i];
            assert!(dual.abs() < 1e-7);
            assert!(s.x[i] >= -1e-7 && s.x[i] <= 1.0 + 1e-7);
            assert!((s.x[i] * s.z[i]).abs() < 1e-7);
            assert!(((1.0 - s.x[i]) * s.z[3 + i]).abs() < 1e-7);
        }
    }
    assert_eq!(qs.values(), &[3.0, -0.2, -0.2, 2.0, -0.1, -0.1, 4.0]);
}
#[test]
fn sparse_presolve_removes_redundancy_and_detects_contradictions() {
    let q = CsrMatrix::<f64>::identity(3);
    let g = csr(0, 3, &[]);
    let a = csr(3, 3, &[1.0, 1.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0]);
    let s = interior_point_qp_sparse(
        &q,
        &[0.0; 3],
        &a,
        &[2.0, 4.0, 3.0],
        &g,
        &[],
        QpOptions::default(),
    )
    .unwrap();
    assert_eq!(s.status, QpStatus::Optimal);
    for (v, t) in s.x.iter().zip([1.0, 1.0, 3.0]) {
        assert!((*v - t).abs() < 1e-7);
    }
    assert_eq!(s.y.len(), 3);
    let s = interior_point_qp_sparse(
        &q,
        &[0.0; 3],
        &a,
        &[2.0, 5.0, 3.0],
        &g,
        &[],
        QpOptions::default(),
    )
    .unwrap();
    assert_eq!(s.status, QpStatus::NumericalFailure);
}
#[test]
fn sparse_lp_with_zero_hessian_and_dense_fallback() {
    let a = csr(1, 1, &[1.0]);
    let s = interior_point_lp_sparse(&[1.0], &a, &[2.0], QpOptions::default()).unwrap();
    assert_eq!(s.status, QpStatus::Optimal);
    assert!((s.x[0] - 2.0).abs() < 1e-7);
    let zero = csr(1, 1, &[0.0]);
    let empty = csr(0, 1, &[]);
    let s = interior_point_qp_sparse(
        &zero,
        &[1.0],
        &a,
        &[2.0],
        &empty,
        &[],
        QpOptions {
            regularization: 0.0,
            kkt_backend: QpKktBackend::Sparse,
            ..QpOptions::default()
        },
    )
    .unwrap();
    assert_eq!(s.status, QpStatus::Optimal);
    assert_eq!(s.kkt.dense_fallbacks, 1);
}
#[test]
fn validates_shape_symmetry_and_values_without_panics() {
    let mut q = CsrMatrix::<f64>::identity(2);
    let empty = csr(0, 2, &[]);
    assert_eq!(
        interior_point_qp_sparse(
            &q,
            &[0.0; 2],
            &csr(0, 0, &[]),
            &[],
            &empty,
            &[],
            QpOptions::default()
        ),
        Err(OptError::InvalidInput)
    );
    let asym = csr(2, 2, &[1.0, 1.0, 0.0, 1.0]);
    assert_eq!(
        interior_point_qp_sparse(
            &asym,
            &[0.0; 2],
            &empty,
            &[],
            &empty,
            &[],
            QpOptions::default()
        ),
        Err(OptError::InvalidInput)
    );
    q.values_mut()[0] = f64::NAN;
    assert_eq!(
        interior_point_qp_sparse(
            &q,
            &[0.0; 2],
            &empty,
            &[],
            &empty,
            &[],
            QpOptions::default()
        ),
        Err(OptError::NotFinite)
    );
}
#[test]
fn larger_sparse_equalities_and_single_precision() {
    let n = 100;
    let q = CsrMatrix::<f64>::identity(n);
    let empty = CsrMatrix::from_parts(0, n, vec![0], vec![], vec![]).unwrap();
    let s = interior_point_qp_sparse(
        &q,
        &vec![-1.0; n],
        &q,
        &vec![1.0; n],
        &empty,
        &[],
        QpOptions::default(),
    )
    .unwrap();
    assert_eq!(s.status, QpStatus::Optimal);
    for x in s.x {
        assert!((x - 1.0).abs() < 1e-7);
    }
    let q = CsrMatrix::<f32>::identity(2);
    let empty = CsrMatrix::from_parts(0, 2, vec![0], vec![], vec![]).unwrap();
    let mut coo = CooMatrix::new(1, 2);
    coo.push(0, 0, 1.0).unwrap();
    coo.push(0, 1, 1.0).unwrap();
    let s = interior_point_qp_sparse(
        &q,
        &[-2.0, -1.0],
        &empty,
        &[],
        &CsrMatrix::from_coo(&coo),
        &[2.0],
        QpOptions {
            tol: 1e-4,
            ..QpOptions::default()
        },
    )
    .unwrap();
    assert_eq!(s.status, QpStatus::Optimal);
    assert!((s.x[0] - 1.5).abs() < 1e-3);
}
