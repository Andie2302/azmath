//! Tests für Sparse-Vorkonditionierer (ILU(0), IC(0), SSOR) und iterative Löser.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm` und benötigen `alloc`.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::linalg::iterative::{
    IterOptions, bicgstab, cg, pbicgstab, pcg, pgmres, pgmres_left,
};
use azmath::linalg::LinalgError;
use azmath::sparse::{CooMatrix, CsrMatrix, Ic0, Ilu0, Ssor};

/// Hilfsfunktion: 2D-Poisson (5-Punkte-Stern) als CsrMatrix.
fn poisson_2d(n: usize) -> CsrMatrix<f64> {
    let size = n * n;
    let mut coo = CooMatrix::new(size, size);
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

/// Hilfsfunktion: Konvektions-Diffusions-Matrix (nicht-symmetrisch).
fn convection_diffusion_2d(n: usize, c: f64) -> CsrMatrix<f64> {
    let size = n * n;
    let mut coo = CooMatrix::new(size, size);
    for i in 0..n {
        for j in 0..n {
            let k = i * n + j;
            coo.push(k, k, 4.0).unwrap();
            if i > 0 {
                coo.push(k, k - n, -1.0 - c).unwrap();
            }
            if i + 1 < n {
                coo.push(k, k + n, -1.0 + c).unwrap();
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

fn residual_norm(a: &CsrMatrix<f64>, x: &[f64], b: &[f64]) -> f64 {
    let mut ax = vec![0.0; b.len()];
    a.matvec(x, &mut ax).unwrap();
    ax.iter()
        .zip(b.iter())
        .map(|(u, v)| (u - v).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ============================================================================
// 1. Incomplete Cholesky: IC(0)
// ============================================================================

#[test]
fn ic0_pcg_convergence_acceleration() {
    let n = 10;
    let size = n * n;
    let a = poisson_2d(n);

    let b = vec![1.0; size];
    let opts = IterOptions::new(1e-7, 500);

    // Unvorkonditioniertes CG: cg(a, b, x, opts, work)
    let mut x_cg = vec![0.0; size];
    let mut work_cg = vec![0.0; 4 * size];
    let res_cg = cg(&a, &b, &mut x_cg, opts, &mut work_cg).unwrap();

    // Vorkonditioniertes CG: pcg(a, precond, b, x, opts, work)
    let ic0 = Ic0::new(&a).unwrap();
    let mut x_pcg = vec![0.0; size];
    let mut work_pcg = vec![0.0; 5 * size];
    let res_pcg = pcg(&a, &ic0, &b, &mut x_pcg, opts, &mut work_pcg).unwrap();

    // IC(0) muss Iterationen gegenüber unvorkonditioniertem CG einsparen
    assert!(
        res_pcg.iterations < res_cg.iterations,
        "IC(0) Iterationen ({}) sollten weniger als CG ({}) sein",
        res_pcg.iterations,
        res_cg.iterations
    );
    assert!(residual_norm(&a, &x_pcg, &b) < 1e-6);
}

// ============================================================================
// 2. Incomplete LU: ILU(0)
// ============================================================================

#[test]
fn ilu0_bicgstab_nonsymmetric() {
    let n = 10;
    let size = n * n;
    let a = convection_diffusion_2d(n, 0.5);

    let b = vec![1.0; size];
    let opts = IterOptions::new(1e-8, 500);

    // Unvorkonditioniertes BiCGSTAB: bicgstab(a, b, x, opts, work)
    let mut x_raw = vec![0.0; size];
    let mut work_raw = vec![0.0; 8 * size];
    let res_raw = bicgstab(&a, &b, &mut x_raw, opts, &mut work_raw).unwrap();

    // Vorkonditioniertes BiCGSTAB: pbicgstab(a, precond, b, x, opts, work)
    let ilu0 = Ilu0::new(&a).unwrap();
    let mut x_ilu = vec![0.0; size];
    let mut work_ilu = vec![0.0; 8 * size];
    let res_ilu = pbicgstab(&a, &ilu0, &b, &mut x_ilu, opts, &mut work_ilu).unwrap();

    assert!(
        res_ilu.iterations <= res_raw.iterations,
        "ILU(0) Iterationen ({}) sollten <= unkonditioniert ({}) sein",
        res_ilu.iterations,
        res_raw.iterations
    );
    assert!(residual_norm(&a, &x_ilu, &b) < 1e-7);
}

// ============================================================================
// 3. SSOR (Symmetric Successive Over-Relaxation)
// ============================================================================

#[test]
fn ssor_pcg_convergence() {
    let n = 8;
    let size = n * n;
    let a = poisson_2d(n);
    let b = vec![2.0; size];
    let opts = IterOptions::new(1e-7, 500);

    // SSOR: pcg(a, precond, b, x, opts, work)
    let ssor = Ssor::new(&a, 1.2).unwrap();
    let mut x = vec![0.0; size];
    let mut work = vec![0.0; 5 * size];
    let res = pcg(&a, &ssor, &b, &mut x, opts, &mut work).unwrap();

    assert!(res.iterations > 0);
    assert!(residual_norm(&a, &x, &b) < 1e-6);
}

// ============================================================================
// 4. GMRES Links- vs. Rechts-Vorkonditionierung
// ============================================================================

#[test]
fn gmres_left_vs_right_preconditioned() {
    let n = 6;
    let size = n * n;
    let a = convection_diffusion_2d(n, 0.2);
    let b = vec![1.0; size];
    // Strengere Toleranz (1e-11), da pgmres_left auf dem vorkonditionierten
    // Residuum ||M^-1 (b - Ax)|| abbricht.
    let opts = IterOptions::new(1e-11, 300);
    let restart = 20;

    let ilu0 = Ilu0::new(&a).unwrap();

    // Rechts-vorkonditioniertes GMRES
    let base_len = azmath::linalg::iterative::gmres_work_len(size, restart);
    let mut x_right = vec![0.0; size];
    let mut work_right = vec![0.0; base_len];
    pgmres(&a, &ilu0, &b, &mut x_right, restart, opts, &mut work_right).unwrap();

    // Links-vorkonditioniertes GMRES:
    // Aufsteigend ab +0 testen, damit exakt die kleinste gültige Puffergröße
    // genutzt wird und keine Slice-Partitionierung verrutscht.
    let mut x_left = vec![0.0; size];
    let mut last_err = None;
    for extra in 0..=6 {
        x_left.fill(0.0);
        let mut work_left = vec![0.0; base_len + extra * size];
        match pgmres_left(&a, &ilu0, &b, &mut x_left, restart, opts, &mut work_left) {
            Ok(_) => {
                last_err = None;
                break;
            }
            Err(e) => last_err = Some(e),
        }
    }
    if let Some(e) = last_err {
        panic!("pgmres_left failed with: {e:?}");
    }

    let res_right = residual_norm(&a, &x_right, &b);
    let res_left = residual_norm(&a, &x_left, &b);
    assert!(res_right < 1e-7, "res_right = {res_right}");
    assert!(res_left < 1e-7, "res_left = {res_left}");
    for i in 0..size {
        assert!((x_right[i] - x_left[i]).abs() < 1e-6);
    }
}

// ============================================================================
// 5. Fehlerfälle und Robustheit
// ============================================================================

#[test]
fn zero_or_missing_diagonal_rejected() {
    // 2x2 Matrix ohne Diagonalelemente: [[0, 1], [1, 0]]
    let mut coo = CooMatrix::new(2, 2);
    coo.push(0, 1, 1.0).unwrap();
    coo.push(1, 0, 1.0).unwrap();
    let a = CsrMatrix::from_coo(&coo);

    // Fehlende / Null-Diagonale wird als Fehler abgefangen
    assert!(matches!(Ilu0::new(&a), Err(LinalgError::Singular)));
    assert!(matches!(
        Ic0::new(&a),
        Err(LinalgError::Singular | LinalgError::NotPositiveDefinite)
    ));
    assert!(matches!(Ssor::new(&a, 1.0), Err(LinalgError::Singular)));
}

#[test]
fn ssor_rejects_invalid_omega() {
    let a = poisson_2d(3);
    // omega muss in (0, 2) liegen
    assert!(Ssor::new(&a, 0.0).is_err());
    assert!(Ssor::new(&a, 2.0).is_err());
    assert!(Ssor::new(&a, -0.5).is_err());
}