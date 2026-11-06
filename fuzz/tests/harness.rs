#[allow(dead_code)]
#[path = "../fuzz_targets/common.rs"]
mod common;
#[path = "../fuzz_targets/qp_kkt.rs"]
mod qp_kkt;
use azmath::{
    opt::{QpKktStats, QpSolution, QpStatus},
    sparse::CsrMatrix,
};
#[test]
fn fft_boundary_lengths_are_reachable() {
    for expected in [0usize, 1, 255, 256, 257] {
        let bytes = (expected as u16).to_le_bytes();
        assert_eq!(common::Bytes(&bytes).range(0, 257), expected);
    }
    assert_eq!(common::Bytes(&[255]).range(0, 255), 255);
    assert_eq!(common::Bytes(&[]).range(0, 257), 0);
}
#[test]
fn optimality_oracle_accepts_optimum_and_rejects_feasible_wrong_answers() {
    let q = CsrMatrix::<f64>::identity(1);
    let a = CsrMatrix::from_dense_slice(0, 1, &[], 0.).unwrap();
    let g = CsrMatrix::identity(1);
    let solution = |x, z| QpSolution {
        status: QpStatus::Optimal,
        x: vec![x],
        y: vec![],
        z: vec![z],
        objective: 0.,
        iterations: 0,
        kkt: QpKktStats::default(),
    };
    assert!(qp_kkt::check(&q, &[-2.], &a, &g, &[1.], &solution(1., 1.), 1e-8).is_ok());
    for (x, z) in [(0., 0.), (1., -1.), (0., 2.), (1., f64::NAN)] {
        assert!(qp_kkt::check(&q, &[-2.], &a, &g, &[1.], &solution(x, z), 1e-8).is_err());
    }
    // Equality-constrained optimum: x=1 and y=-1.
    let eq = CsrMatrix::identity(1);
    let none = CsrMatrix::from_dense_slice(0, 1, &[], 0.).unwrap();
    let mut s = solution(1., 0.);
    s.z.clear();
    s.y = vec![-1.];
    assert!(qp_kkt::check(&q, &[0.], &eq, &none, &[], &s, 1e-8).is_ok());
}
