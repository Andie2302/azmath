#![cfg(all(feature = "alloc", any(feature = "std", feature = "libm")))]
use azmath::{
    opt::{QpKktBackend, QpOptions, QpStatus, QpWorkspace, interior_point_qp_sparse},
    sparse::CsrMatrix,
};
#[test]
fn related_qps_reuse_analysis_and_warm_solution() {
    let n = 16;
    let q = CsrMatrix::<f64>::identity(n);
    let a = CsrMatrix::from_parts(0, n, vec![0], vec![], vec![]).unwrap();
    let g = CsrMatrix::identity(n);
    let opts = QpOptions {
        kkt_backend: QpKktBackend::Sparse,
        ..QpOptions::default()
    };
    let mut workspace = QpWorkspace::new(q.clone(), a.clone(), g.clone(), opts);
    let mut warm_iters = 0;
    let mut cold_iters = 0;
    for k in 0..8 {
        let c = vec![-1.0 - k as f64 * 0.0001; n];
        let h = vec![0.4; n];
        let w = workspace.solve(&c, &[], &h).unwrap();
        let cold = interior_point_qp_sparse(&q, &c, &a, &[], &g, &h, opts).unwrap();
        assert_eq!(w.status, QpStatus::Optimal);
        assert_eq!(w.kkt.symbolic_analyses, if k == 0 { 1 } else { 0 });
        for (x, y) in w.x.iter().zip(&cold.x) {
            assert!((*x - *y).abs() < 1e-7);
        }
        warm_iters += w.iterations;
        cold_iters += cold.iterations;
    }
    assert!(warm_iters < cold_iters, "{warm_iters} vs {cold_iters}");
    workspace.clear_warm_start();
    let result = workspace.solve(&vec![-1.0; n], &[], &vec![0.4; n]).unwrap();
    assert_eq!(result.kkt.symbolic_analyses, 0);
    assert_eq!(result.status, QpStatus::Optimal);
    assert!(workspace.solve(&[1.0], &[], &vec![0.4; n]).is_err());
    assert_eq!(
        workspace
            .solve(&vec![-1.0; n], &[], &vec![0.4; n])
            .unwrap()
            .status,
        QpStatus::Optimal
    );
}
#[test]
fn changing_rhs_bounds_and_redundant_equalities_are_safe() {
    let q = CsrMatrix::<f64>::identity(2);
    let a = CsrMatrix::from_dense_slice(2, 2, &[1.0, 1.0, 2.0, 2.0], 0.0).unwrap();
    let g = CsrMatrix::identity(2);
    let mut w = QpWorkspace::new(q.clone(), a.clone(), g.clone(), QpOptions::default());
    for (b, h) in [
        (vec![1.0, 2.0], vec![2.0; 2]),
        (vec![0.8, 1.6], vec![1.5; 2]),
        (vec![1.2, 2.4], vec![2.0; 2]),
    ] {
        let s = w.solve(&[-1.0, -1.0], &b, &h).unwrap();
        assert_eq!(s.status, QpStatus::Optimal);
        assert!((s.x.iter().sum::<f64>() - b[0]).abs() < 1e-7);
    }
    assert_eq!(
        w.solve(&[-1.0, -1.0], &[1.0, 3.0], &[2.0; 2])
            .unwrap()
            .status,
        QpStatus::NumericalFailure
    );
    assert_eq!(
        w.solve(&[-1.0, -1.0], &[1.0, 2.0], &[2.0; 2])
            .unwrap()
            .status,
        QpStatus::Optimal
    );
}
