#![cfg(all(feature = "alloc", any(feature = "std", feature = "libm")))]
use azmath::opt::{OptError, QpKktBackend, QpOptions, QpStatus, interior_point_qp};

#[test]
fn cached_sparse_backend_matches_dense_and_satisfies_original_kkt_conditions() {
    for seed in 0..15 {
        let n = 4;
        let mut q = vec![0.0f64; n * n];
        for i in 0..n {
            q[i * n + i] = 2.0 + seed as f64 * 0.1;
            if i > 0 {
                q[i * n + i - 1] = -0.2;
                q[(i - 1) * n + i] = -0.2;
            }
        }
        let c = [-2.0, -1.0, -3.0, -0.5];
        let a = [1.0; 4];
        let b = [1.5];
        let mut g = vec![0.0; 8 * n];
        let mut h = vec![0.0; 8];
        for i in 0..n {
            g[i * n + i] = -1.0;
            g[(n + i) * n + i] = 1.0;
            h[n + i] = 1.0;
        }
        let solve = |backend| {
            interior_point_qp(
                &q,
                &c,
                &a,
                &b,
                &g,
                &h,
                QpOptions {
                    kkt_backend: backend,
                    ..QpOptions::default()
                },
            )
            .unwrap()
        };
        let sparse = solve(QpKktBackend::Sparse);
        let dense = solve(QpKktBackend::Dense);
        assert_eq!(sparse.status, QpStatus::Optimal);
        assert_eq!(dense.status, QpStatus::Optimal);
        assert_eq!(sparse.kkt.symbolic_analyses, 1);
        assert!(sparse.kkt.sparse_refactorizations > 0);
        assert_eq!(sparse.kkt.dense_fallbacks, 0);
        assert_eq!(dense.kkt.symbolic_analyses, 0);
        assert!(dense.kkt.dense_factorizations > 0);
        assert!((sparse.objective - dense.objective).abs() < 1e-7);
        assert!(sparse.kkt.max_linear_backward_error < 1e-10);
        for sol in [&sparse, &dense] {
            assert!((sol.x.iter().sum::<f64>() - 1.5).abs() < 1e-7);
            for i in 0..n {
                let stationarity =
                    (0..n).map(|j| q[i * n + j] * sol.x[j]).sum::<f64>() + c[i] + sol.y[0]
                        - sol.z[i]
                        + sol.z[n + i];
                assert!(stationarity.abs() < 1e-7);
                assert!(sol.x[i] >= -1e-7 && sol.x[i] <= 1.0 + 1e-7);
                assert!(sol.z[i] >= 0.0 && sol.z[n + i] >= 0.0);
                assert!((sol.z[i] * sol.x[i]).abs() < 1e-7);
                assert!((sol.z[n + i] * (1.0 - sol.x[i])).abs() < 1e-7);
                assert!((sparse.x[i] - dense.x[i]).abs() < 1e-7);
            }
        }
    }
}
#[test]
fn zero_sparse_pivot_falls_back_to_dense_two_by_two_pivot() {
    // Fully constrained convex LP: K=[[0,1],[1,0]], nonsingular but no scalar LDL pivot.
    let sol = interior_point_qp(
        &[0.0f64],
        &[1.0],
        &[1.0],
        &[2.0],
        &[],
        &[],
        QpOptions {
            regularization: 0.0,
            kkt_backend: QpKktBackend::Sparse,
            ..QpOptions::default()
        },
    )
    .unwrap();
    assert_eq!(sol.status, QpStatus::Optimal);
    assert!((sol.x[0] - 2.0).abs() < 1e-10);
    assert!((sol.y[0] + 1.0).abs() < 1e-10);
    assert_eq!(sol.kkt.symbolic_analyses, 1);
    assert_eq!(sol.kkt.dense_fallbacks, 1);
    assert_eq!(sol.kkt.dense_factorizations, 1);
}
#[test]
fn sparse_and_dense_single_precision_results_agree() {
    let solve = |backend| {
        interior_point_qp(
            &[1.0f32, 0.0, 0.0, 1.0],
            &[-2.0, -1.0],
            &[],
            &[],
            &[1.0, 1.0],
            &[2.0],
            QpOptions {
                tol: 1e-4,
                kkt_backend: backend,
                ..QpOptions::default()
            },
        )
        .unwrap()
    };
    let s = solve(QpKktBackend::Sparse);
    let d = solve(QpKktBackend::Dense);
    assert_eq!(s.status, QpStatus::Optimal);
    assert_eq!(d.status, QpStatus::Optimal);
    assert!((s.x[0] - d.x[0]).abs() < 1e-3);
    assert!((s.x[0] - 1.5).abs() < 1e-3);
}
#[test]
fn option_validation_and_redundant_constraint_diagnostics() {
    for tol in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            interior_point_qp(
                &[1.0],
                &[0.0],
                &[],
                &[],
                &[],
                &[],
                QpOptions {
                    tol,
                    ..QpOptions::default()
                }
            ),
            Err(OptError::InvalidInput)
        );
    }
    for regularization in [-1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(
            interior_point_qp(
                &[1.0],
                &[0.0],
                &[],
                &[],
                &[],
                &[],
                QpOptions {
                    regularization,
                    ..QpOptions::default()
                }
            ),
            Err(OptError::InvalidInput)
        );
    }
    let sol = interior_point_qp(
        &[1.0f64, 0.0, 0.0, 1.0],
        &[0.0, 0.0],
        &[1.0, 1.0, 2.0, 2.0],
        &[2.0, 4.0],
        &[],
        &[],
        QpOptions::default(),
    )
    .unwrap();
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_eq!(sol.y.len(), 2);
    assert_eq!(sol.kkt.selected_backend, Some(QpKktBackend::Dense));
    assert!((sol.x[0] - 1.0).abs() < 1e-7);
}

#[test]
fn auto_selects_by_structural_density_without_changing_solutions() {
    for n in [2, 16] {
        let mut q = vec![0.0f64; n * n];
        for i in 0..n {
            q[i * n + i] = 2.0;
        }
        let sol = interior_point_qp(&q, &vec![-1.0; n], &[], &[], &[], &[], QpOptions::default())
            .unwrap();
        assert_eq!(sol.status, QpStatus::Optimal);
        assert_eq!(
            sol.kkt.selected_backend,
            Some(if n == 2 {
                QpKktBackend::Dense
            } else {
                QpKktBackend::Sparse
            })
        );
        for x in sol.x {
            assert!((x - 0.5).abs() < 1e-7);
        }
    }
}

#[test]
fn changing_kkt_values_keep_structural_zeros_in_the_cached_pattern() {
    // Initially Q01 + (GᵀWG)01 = -1 + 1 = 0; later weights change.
    let sol = interior_point_qp(
        &[2.0f64, -1.0, -1.0, 2.0],
        &[-3.0, -1.0],
        &[],
        &[],
        &[1.0, 1.0],
        &[1.5],
        QpOptions {
            kkt_backend: QpKktBackend::Sparse,
            ..QpOptions::default()
        },
    )
    .unwrap();
    assert_eq!(sol.status, QpStatus::Optimal);
    assert_eq!(sol.kkt.symbolic_analyses, 1);
    assert!(sol.kkt.sparse_refactorizations > 0);
    assert!((sol.x[0] - 13.0 / 12.0).abs() < 1e-7);
    assert!((sol.x[1] - 5.0 / 12.0).abs() < 1e-7);
}
