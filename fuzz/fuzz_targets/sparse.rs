#![no_main]
#[path = "common.rs"]
mod common;

use azmath::sparse::{
    CooMatrix, CsrMatrix, Ordering, SparseCholesky, SparseLdl, SparseLu, minimum_degree_ordering,
    permute_symmetric, rcm_ordering,
};
use common::Bytes;
use libfuzzer_sys::fuzz_target;

const ORDERINGS: [Ordering; 3] = [
    Ordering::Natural,
    Ordering::ReverseCuthillMcKee,
    Ordering::MinimumDegree,
];

fn is_permutation(p: &[usize], n: usize) -> bool {
    let mut seen = vec![false; n];
    p.len() == n
        && p.iter()
            .all(|&i| i < n && !std::mem::replace(&mut seen[i], true))
}

fn residual_inf(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    (0..a.rows())
        .map(|i| (a.row(i).map(|(j, v)| v * x[j]).sum::<f64>() - b[i]).abs())
        .fold(0.0, f64::max)
}

fuzz_target!(|data: &[u8]| {
    let mut r = Bytes(data);
    let n = r.range(1, 14);
    let special = r.u8() % 4 == 0;
    let count = r.range(0, 60);
    let mut coo = CooMatrix::new(n, n);
    let mut dense = vec![0.0f64; n * n];
    for _ in 0..count {
        // Indizes dürfen auch außerhalb liegen: dann muss `push` einen Fehler liefern.
        let (i, j) = (r.range(0, n), r.range(0, n));
        let v = r.value(special);
        match coo.push(i, j, v) {
            Ok(()) => dense[i * n + j] += v,
            Err(_) => assert!(i >= n || j >= n || !v.is_finite()),
        }
    }
    let a = CsrMatrix::from_coo(&coo);
    a.validate().unwrap();
    assert_eq!((a.rows(), a.cols()), (n, n));

    // Umordnungen und Transposition auf beliebiger (auch nichtendlicher) Struktur
    let t = a.transpose();
    t.validate().unwrap();
    for perm in [rcm_ordering(&a), minimum_degree_ordering(&a)] {
        assert!(is_permutation(&perm, n));
        permute_symmetric(&a, &perm).validate().unwrap();
    }

    // Direkte Löser dürfen Fehler melden, aber nie paniken
    let b: Vec<f64> = (0..n).map(|_| r.value(special)).collect();
    for ordering in ORDERINGS {
        let _ = SparseLdl::new(&a, ordering).map(|f| f.solve(&b));
        let _ = SparseCholesky::new(&a, ordering).map(|f| f.solve(&b));
        let _ = SparseLu::new(&a, ordering).map(|f| (f.solve(&b), f.solve_transposed(&b)));
    }

    // Streng diagonaldominante, endliche Matrix: Löser müssen gelingen und genau sein.
    if !special {
        let mut sym = CooMatrix::new(n, n);
        let mut rowsum = vec![0.0f64; n];
        for i in 0..n {
            for (j, v) in a.row(i) {
                if i != j && v.is_finite() {
                    sym.push(i, j, v).unwrap();
                    sym.push(j, i, v).unwrap();
                    rowsum[i] += v.abs();
                    rowsum[j] += v.abs();
                }
            }
        }
        for (i, s) in rowsum.iter().enumerate() {
            sym.push(i, i, 2.0 * s + 1.0).unwrap();
        }
        let m = CsrMatrix::from_coo(&sym);
        let rhs: Vec<f64> = (0..n).map(|_| r.moderate()).collect();
        let scale = 1.0 + rhs.iter().fold(0.0f64, |p, v| p.max(v.abs()));
        for ordering in ORDERINGS {
            let x = SparseCholesky::new(&m, ordering)
                .unwrap()
                .solve(&rhs)
                .unwrap();
            assert!(
                residual_inf(&m, &x, &rhs)
                    <= 1e-8 * scale * (1.0 + m.values().iter().fold(0.0f64, |p, v| p.max(v.abs())))
            );
            let x = SparseLdl::new(&m, ordering).unwrap().solve(&rhs).unwrap();
            assert!(
                residual_inf(&m, &x, &rhs)
                    <= 1e-8 * scale * (1.0 + m.values().iter().fold(0.0f64, |p, v| p.max(v.abs())))
            );
            let x = SparseLu::new(&m, ordering).unwrap().solve(&rhs).unwrap();
            assert!(
                residual_inf(&m, &x, &rhs)
                    <= 1e-8 * scale * (1.0 + m.values().iter().fold(0.0f64, |p, v| p.max(v.abs())))
            );
        }
    }

    // matvec gegen die dichte Referenz (nur für endliche Werte vergleichbar)
    if !special {
        let x: Vec<f64> = (0..n).map(|_| r.moderate()).collect();
        let mut y = vec![0.0; n];
        a.matvec(&x, &mut y).unwrap();
        for i in 0..n {
            let e: f64 = (0..n).map(|j| dense[i * n + j] * x[j]).sum();
            assert!((y[i] - e).abs() <= 1e-9 * (1.0 + e.abs()), "matvec");
        }
    }
});
