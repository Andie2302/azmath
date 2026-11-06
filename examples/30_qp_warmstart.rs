//! Folge verwandter QPs: Kaltstart gegenüber persistentem Workspace/Warmstart.
#[cfg(all(feature = "std", feature = "alloc"))]
fn main() {
    use azmath::{
        opt::{QpKktBackend, QpOptions, QpStatus, QpWorkspace, interior_point_qp_sparse},
        sparse::CsrMatrix,
    };
    let n = 32;
    let q = CsrMatrix::<f64>::identity(n);
    let a = CsrMatrix::from_parts(0, n, vec![0], vec![], vec![]).unwrap();
    let g = CsrMatrix::identity(n);
    let opts = QpOptions {
        kkt_backend: QpKktBackend::Sparse,
        ..QpOptions::default()
    };
    let h = vec![0.4; n];
    let mut w = QpWorkspace::new(q.clone(), a.clone(), g.clone(), opts);
    let (mut cold, mut warm, mut analyses) = (0, 0, 0);
    let start = std::time::Instant::now();
    for i in 0..20 {
        let c = vec![-1.0 - i as f64 * 0.0001; n];
        let s = w.solve(&c, &[], &h).unwrap();
        assert_eq!(s.status, QpStatus::Optimal);
        warm += s.iterations;
        analyses += s.kkt.symbolic_analyses;
    }
    let warm_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = std::time::Instant::now();
    for i in 0..20 {
        let c = vec![-1.0 - i as f64 * 0.0001; n];
        let s = interior_point_qp_sparse(&q, &c, &a, &[], &g, &h, opts).unwrap();
        assert_eq!(s.status, QpStatus::Optimal);
        cold += s.iterations;
    }
    println!(
        "20 QPs: cold iterations {cold}, warm iterations {warm}, cached analyses {analyses}, cold ms {:.3}, warm ms {warm_ms:.3}",
        start.elapsed().as_secs_f64() * 1000.0
    );
}
#[cfg(not(all(feature = "std", feature = "alloc")))]
fn main() {
    println!("Bitte mit std und alloc bauen.");
}
