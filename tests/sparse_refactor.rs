#![cfg(all(feature = "alloc", any(feature = "std", feature = "libm")))]
use azmath::linalg::LinalgError;
use azmath::sparse::{
    CooMatrix, CsrMatrix, Ordering, SparseLdl, SparseLdlSymbolic, SparseLu, SparseLuSymbolic,
};
fn tridiagonal(n: usize, diag: f64) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n, n);
    for i in 0..n {
        coo.push(i, i, diag).unwrap();
        if i > 0 {
            coo.push(i, i - 1, -1.0).unwrap();
            coo.push(i - 1, i, -1.0).unwrap();
        }
    }
    CsrMatrix::from_coo(&coo)
}
fn residual(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) {
    for (i, &bi) in b.iter().enumerate().take(a.rows()) {
        let r = a.row(i).map(|(j, v)| v * x[j]).sum::<f64>() - bi;
        assert!(r.abs() < 1e-10, "row {i} residual {r}");
    }
}
#[test]
fn repeated_ldl_cholesky_match_fresh_factors_for_every_ordering() {
    for ordering in [
        Ordering::Natural,
        Ordering::ReverseCuthillMcKee,
        Ordering::MinimumDegree,
    ] {
        let mut a = tridiagonal(15, 4.0);
        let symbolic = SparseLdlSymbolic::analyze(&a, ordering).unwrap();
        let mut ldl = symbolic.factorize(&a).unwrap();
        let mut chol = symbolic.factorize_cholesky(&a).unwrap();
        assert_eq!(symbolic.nnz_l(), ldl.nnz_l());
        assert_eq!(symbolic.n(), 15);
        for k in 0..20 {
            for i in 0..15 {
                for p in a.row_ptr()[i]..a.row_ptr()[i + 1] {
                    if a.col_idx()[p] == i {
                        a.values_mut()[p] = 3.0 + k as f64 / 10.0;
                    }
                }
            }
            ldl.refactor(&a).unwrap();
            chol.refactor(&a).unwrap();
            let b = vec![1.0; 15];
            let x = ldl.solve(&b).unwrap();
            residual(&a, &x, &b);
            assert_eq!(x, SparseLdl::new(&a, ordering).unwrap().solve(&b).unwrap());
            assert_eq!(x, chol.solve(&b).unwrap());
        }
    }
}
#[test]
fn lower_triangle_is_authoritative_after_reordering() {
    let mut c = CooMatrix::new(4, 4);
    for i in 0..4 {
        c.push(i, i, 4.0).unwrap();
        if i > 0 {
            c.push(i, i - 1, -1.0).unwrap();
        }
    }
    let lower = CsrMatrix::from_coo(&c);
    let full = tridiagonal(4, 4.0);
    for order in [
        Ordering::Natural,
        Ordering::ReverseCuthillMcKee,
        Ordering::MinimumDegree,
    ] {
        let l = SparseLdlSymbolic::analyze(&lower, order)
            .unwrap()
            .factorize(&lower)
            .unwrap();
        residual(&full, &l.solve(&[1.0; 4]).unwrap(), &[1.0; 4]);
    }
}
#[test]
fn lu_recomputes_pivots_and_supports_transposed_solves() {
    for order in [
        Ordering::Natural,
        Ordering::ReverseCuthillMcKee,
        Ordering::MinimumDegree,
    ] {
        let mut a = CsrMatrix::from_parts(
            3,
            3,
            vec![0, 3, 6, 9],
            vec![0, 1, 2, 0, 1, 2, 0, 1, 2],
            vec![1.0, 2.0, 0.0, 4.0, 1.0, 2.0, 0.0, 3.0, 5.0],
        )
        .unwrap();
        let sym = SparseLuSymbolic::analyze(&a, order).unwrap();
        let mut lu = sym.factorize(&a).unwrap();
        for values in [
            [1.0, 2.0, 0.0, 4.0, 1.0, 2.0, 0.0, 3.0, 5.0],
            [9.0, 1.0, 2.0, 0.0, 5.0, 3.0, 1.0, 0.0, 4.0],
            [0.0, 2.0, 1.0, 3.0, 0.0, 2.0, 1.0, 4.0, 0.0],
        ] {
            a.values_mut().copy_from_slice(&values);
            lu.refactor(&a).unwrap();
            let fresh = SparseLu::new(&a, order).unwrap();
            let b = [1.0, -2.0, 3.0];
            let x = lu.solve(&b).unwrap();
            residual(&a, &x, &b);
            assert_eq!(x, fresh.solve(&b).unwrap());
            let x = lu.solve_transposed(&b).unwrap();
            residual(&a.transpose(), &x, &b);
        }
    }
}
#[test]
fn structural_input_errors_preserve_old_factors_numeric_errors_invalidate_and_recover() {
    let mut a = tridiagonal(3, 4.0);
    let mut ldl = SparseLdl::new(&a, Ordering::Natural).unwrap();
    let mut lu = SparseLu::new(&a, Ordering::Natural).unwrap();
    let different = CsrMatrix::identity(3);
    assert_eq!(ldl.refactor(&different), Err(LinalgError::InvalidInput));
    assert_eq!(lu.refactor(&different), Err(LinalgError::InvalidInput));
    assert!(ldl.is_valid() && lu.is_valid());
    a.values_mut()[0] = f64::NAN;
    assert_eq!(ldl.refactor(&a), Err(LinalgError::NotFinite));
    assert_eq!(lu.refactor(&a), Err(LinalgError::NotFinite));
    assert!(ldl.is_valid() && lu.is_valid());
    a.values_mut().fill(0.0);
    assert_eq!(ldl.refactor(&a), Err(LinalgError::Singular));
    assert_eq!(lu.refactor(&a), Err(LinalgError::Singular));
    assert!(!ldl.is_valid() && !lu.is_valid());
    assert_eq!(ldl.solve(&[1.0; 3]), Err(LinalgError::InvalidInput));
    assert_eq!(lu.solve(&[1.0; 3]), Err(LinalgError::InvalidInput));
    assert_eq!(
        lu.solve_transposed(&[1.0; 3]),
        Err(LinalgError::InvalidInput)
    );
    a = tridiagonal(3, 4.0);
    ldl.refactor(&a).unwrap();
    lu.refactor(&a).unwrap();
    residual(&a, &ldl.solve(&[1.0; 3]).unwrap(), &[1.0; 3]);
    residual(&a, &lu.solve(&[1.0; 3]).unwrap(), &[1.0; 3]);
    let sym = SparseLdlSymbolic::analyze(&a, Ordering::Natural).unwrap();
    let mut chol = sym.factorize_cholesky(&a).unwrap();
    for v in a.values_mut() {
        *v = -*v;
    }
    assert_eq!(chol.refactor(&a), Err(LinalgError::NotPositiveDefinite));
    assert!(!chol.is_valid());
    assert_eq!(chol.solve(&[1.0; 3]), Err(LinalgError::InvalidInput));
    a = tridiagonal(3, 4.0);
    chol.refactor(&a).unwrap();
    assert!(chol.is_valid());
}
#[test]
fn empty_dimensions_and_nonfinite_solves() {
    let a = CsrMatrix::<f64>::from_parts(0, 0, vec![0], vec![], vec![]).unwrap();
    let s = SparseLdlSymbolic::analyze(&a, Ordering::MinimumDegree).unwrap();
    assert_eq!(
        s.factorize(&a).unwrap().solve(&[]).unwrap(),
        Vec::<f64>::new()
    );
    assert_eq!(
        SparseLuSymbolic::analyze(&a, Ordering::Natural)
            .unwrap()
            .factorize(&a)
            .unwrap()
            .solve(&[])
            .unwrap(),
        Vec::<f64>::new()
    );
    let a = tridiagonal(2, 4.0);
    let l = SparseLdl::new(&a, Ordering::Natural).unwrap();
    let u = SparseLu::new(&a, Ordering::Natural).unwrap();
    assert_eq!(l.solve(&[f64::NAN, 0.0]), Err(LinalgError::NotFinite));
    assert_eq!(u.solve(&[f64::NAN, 0.0]), Err(LinalgError::NotFinite));
}

#[test]
fn partial_numeric_failure_overflow_and_single_precision_recovery() {
    let mut a = CsrMatrix::from_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![2.0f64, 1.0, 1.0, 3.0],
    )
    .unwrap();
    let mut l = SparseLdl::new(&a, Ordering::Natural).unwrap();
    let mut u = SparseLu::new(&a, Ordering::Natural).unwrap();
    for values in [
        [1.0, 0.0, 0.0, 0.0],
        [f64::MAX, f64::MAX, f64::MAX, -f64::MAX],
    ] {
        a.values_mut().copy_from_slice(&values);
        assert!(l.refactor(&a).is_err());
        assert!(u.refactor(&a).is_err());
        assert!(!l.is_valid() && !u.is_valid());
        a.values_mut().copy_from_slice(&[2.0, 1.0, 1.0, 3.0]);
        l.refactor(&a).unwrap();
        u.refactor(&a).unwrap();
        residual(&a, &l.solve(&[1.0, 2.0]).unwrap(), &[1.0, 2.0]);
        residual(&a, &u.solve(&[1.0, 2.0]).unwrap(), &[1.0, 2.0]);
    }
    let different_size = CsrMatrix::identity(3);
    assert_eq!(
        l.refactor(&different_size),
        Err(LinalgError::DimensionMismatch)
    );
    assert_eq!(
        u.refactor(&different_size),
        Err(LinalgError::DimensionMismatch)
    );
    assert!(l.is_valid() && u.is_valid());
    let mut a = CsrMatrix::from_parts(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![2.0f32, 1.0, 1.0, 3.0],
    )
    .unwrap();
    let mut l = SparseLdl::new(&a, Ordering::MinimumDegree).unwrap();
    let mut u = SparseLu::new(&a, Ordering::MinimumDegree).unwrap();
    for d in [4.0, 2.0, 8.0] {
        a.values_mut()[0] = d;
        l.refactor(&a).unwrap();
        u.refactor(&a).unwrap();
        for x in [
            l.solve(&[d + 2.0, 7.0]).unwrap(),
            u.solve(&[d + 2.0, 7.0]).unwrap(),
        ] {
            assert!((x[0] - 1.0).abs() < 1e-5 && (x[1] - 2.0).abs() < 1e-5);
        }
    }
}
