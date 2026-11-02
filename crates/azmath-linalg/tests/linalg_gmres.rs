//! Tests für GMRES(m) und vorkonditioniertes GMRES (fest, matrixfrei und dynamisch).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath_linalg::iterative::{
    Identity, IterOptions, Jacobi, bicgstab, cg, gmres, gmres_work_len, pgmres,
};
use azmath_linalg::{FnOperator, LinalgError, LinearOperator, Matrix, Vector};

fn opts() -> IterOptions<f64> {
    IterOptions::new(1e-12, 2000)
}

/// 1D-Poisson-Operator (SPD) matrixfrei.
fn poisson(x: &[f64], y: &mut [f64]) {
    let n = x.len();
    for i in 0..n {
        let l = if i > 0 { x[i - 1] } else { 0.0 };
        let r = if i + 1 < n { x[i + 1] } else { 0.0 };
        y[i] = 2.0 * x[i] - l - r;
    }
}

/// 1D-Konvektions-Diffusion (nichtsymmetrisch), kleines `c` wegen der Kondition.
fn conv_diff(c: f64) -> impl Fn(&[f64], &mut [f64]) {
    move |x: &[f64], y: &mut [f64]| {
        let n = x.len();
        for i in 0..n {
            let l = if i > 0 { x[i - 1] } else { 0.0 };
            let r = if i + 1 < n { x[i + 1] } else { 0.0 };
            y[i] = -(1.0 + c) * l + 2.0 * x[i] - (1.0 - c) * r;
        }
    }
}

fn nonsym3() -> Matrix<f64, 3, 3> {
    Matrix::new([[4.0, 1.0, 0.0], [2.0, 5.0, 1.0], [0.0, 3.0, 6.0]])
}

#[test]
fn work_len_formula() {
    // m = 3, n = 5: V 20 + H 12 + Givens/g/y 4m+1 = 13 + w,z 10
    assert_eq!(gmres_work_len(5, 3), 55);
    // restart > n wird auf n begrenzt
    assert_eq!(gmres_work_len(3, 100), gmres_work_len(3, 3));
}

#[test]
fn nonsymmetric_system_matches_lu() {
    let a = nonsym3();
    let b = Vector::new([1.0, 2.0, 3.0]);
    let mut x = [0.0; 3];
    let mut work = [0.0; gmres_work_len(3, 3)];
    let r = gmres(&a, b.as_slice(), &mut x, 3, opts(), &mut work).unwrap();
    let x_ref = a.solve(&b).unwrap();
    for i in 0..3 {
        assert!((x[i] - x_ref[i]).abs() < 1e-11, "{} vs {}", x[i], x_ref[i]);
    }
    assert!(r.iterations <= 3, "{} Iterationen", r.iterations);
    assert!(r.residual_norm < 1e-10);
}

#[test]
fn full_gmres_needs_at_most_n_iterations() {
    const N: usize = 10;
    // diagonaldominante, nichtsymmetrische Matrix
    let a = Matrix::<f64, N, N>::from_fn(|i, j| {
        if i == j {
            10.0
        } else {
            ((i * 7 + j * 3) % 5) as f64 - 2.0
        }
    });
    let x_true = Vector::<f64, N>::new(core::array::from_fn(|i| i as f64 - 4.0));
    let b = a * x_true;
    let mut x = [0.0; N];
    let mut work = [0.0; gmres_work_len(N, N)];
    let r = gmres(
        &a,
        b.as_slice(),
        &mut x,
        N,
        IterOptions::new(1e-13, 100),
        &mut work,
    )
    .unwrap();
    assert!(r.iterations <= N, "{} Iterationen", r.iterations);
    for i in 0..N {
        assert!((x[i] - x_true[i]).abs() < 1e-10, "x[{i}]");
    }
}

#[test]
fn solves_skew_symmetric_system_where_cg_and_bicgstab_fail() {
    let a = Matrix::new([[0.0f64, 1.0], [-1.0, 0.0]]);
    let b = [1.0, 0.0];

    let mut x = [0.0; 2];
    let mut work = [0.0; gmres_work_len(2, 2)];
    gmres(&a, &b, &mut x, 2, opts(), &mut work).unwrap();
    // A·x = (x₁, −x₀) = (1, 0)  ->  x = (0, 1)
    assert!(x[0].abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12, "{x:?}");

    // zum Vergleich: die beiden anderen Verfahren versagen auf derselben Aufgabe
    let mut y = [0.0; 2];
    let mut w1 = [0.0; 8];
    assert_eq!(
        cg(&a, &b, &mut y, opts(), &mut w1),
        Err(LinalgError::NotPositiveDefinite)
    );
    let mut z = [0.0; 2];
    let mut w2 = [0.0; 16];
    assert_eq!(
        bicgstab(&a, &b, &mut z, opts(), &mut w2),
        Err(LinalgError::Breakdown)
    );
}

#[test]
fn solves_symmetric_indefinite_system_where_cg_fails() {
    // Eigenwerte 3 und −1
    let a = Matrix::new([[1.0f64, 2.0], [2.0, 1.0]]);
    let b = [1.0, -1.0];
    let mut x = [0.0; 2];
    let mut work = [0.0; gmres_work_len(2, 2)];
    gmres(&a, &b, &mut x, 2, opts(), &mut work).unwrap();
    // A·x = b  ->  x = (−1, 1)
    assert!(
        (x[0] + 1.0).abs() < 1e-12 && (x[1] - 1.0).abs() < 1e-12,
        "{x:?}"
    );
    let mut y = [0.0; 2];
    let mut w1 = [0.0; 8];
    assert_eq!(
        cg(&a, &b, &mut y, opts(), &mut w1),
        Err(LinalgError::NotPositiveDefinite)
    );
}

#[test]
fn restarted_gmres_converges_but_needs_more_iterations_than_full() {
    const N: usize = 50;
    let op = FnOperator::new(N, N, poisson);
    let x_true: [f64; N] = core::array::from_fn(|i| {
        let t = (i + 1) as f64 / (N + 1) as f64;
        t * (1.0 - t)
    });
    let mut b = [0.0; N];
    poisson(&x_true, &mut b);

    let mut full = [0.0; N];
    let mut w_full = [0.0; gmres_work_len(N, N)];
    let r_full = gmres(&op, &b, &mut full, N, opts(), &mut w_full).unwrap();

    let mut restarted = [0.0; N];
    let mut w_small = [0.0; gmres_work_len(N, 10)];
    let r_small = gmres(&op, &b, &mut restarted, 10, opts(), &mut w_small).unwrap();

    for i in 0..N {
        assert!((full[i] - x_true[i]).abs() < 1e-9, "voll x[{i}]");
        assert!((restarted[i] - x_true[i]).abs() < 1e-9, "Neustart x[{i}]");
    }
    assert!(r_full.iterations <= N);
    assert!(
        r_small.iterations > r_full.iterations,
        "Neustart {} vs. voll {}",
        r_small.iterations,
        r_full.iterations
    );
    // Der Speicherbedarf von GMRES(10) ist deutlich kleiner
    assert!(gmres_work_len(N, 10) * 3 < gmres_work_len(N, N));
}

#[test]
fn matrix_free_convection_diffusion_and_the_cost_of_restarting() {
    const N: usize = 100;
    let op = FnOperator::new(N, N, conv_diff(0.05));
    let x_true: [f64; N] = core::array::from_fn(|i| {
        let t = (i + 1) as f64 / (N + 1) as f64;
        t * (1.0 - t)
    });
    let mut b = [0.0; N];
    conv_diff(0.05)(&x_true, &mut b);
    let o = IterOptions::new(1e-10, 100_000);

    let solve = |restart: usize| {
        let mut x = [0.0; N];
        let mut work = vec![0.0; gmres_work_len(N, restart)];
        let r = gmres(&op, &b, &mut x, restart, o, &mut work).unwrap();
        let mut ax = [0.0; N];
        op.apply(&x, &mut ax);
        let res = ax
            .iter()
            .zip(&b)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(res < 1e-8, "Neustart {restart}: Residuum {res:e}");
        for i in 0..N {
            assert!(
                (x[i] - x_true[i]).abs() < 1e-6,
                "Neustart {restart}: x[{i}]"
            );
        }
        r.iterations
    };
    let full = solve(N);
    let restarted = solve(20);
    assert!(full <= N, "volles GMRES: {full} Iterationen");
    assert!(
        restarted > 3 * full,
        "GMRES(20): {restarted} vs. voll {full}"
    );

    // BiCGSTAB braucht zwei Anwendungen von A pro Iteration
    let mut x = [0.0; N];
    let mut work = [0.0; 8 * N];
    let rb = bicgstab(&op, &b, &mut x, o, &mut work).unwrap();
    assert!(
        full < 2 * rb.iterations,
        "GMRES {full} vs. BiCGSTAB {} Anwendungen",
        2 * rb.iterations
    );
}

#[test]
fn jacobi_preconditioner_helps_on_column_scaled_matrix() {
    // A = B·S mit ungleichmäßiger *Spalten*skalierung S. Der Rechts-Vorkonditionierer M = diag(A)
    // wirkt auf die Spalten (A·M⁻¹ = B·diag(B)⁻¹) und beseitigt genau diese Skalierung.
    let s = [1.0f64, 100.0, 1.0e4, 10.0, 1.0e3, 1.0];
    let a = Matrix::<f64, 6, 6>::from_fn(|i, j| {
        let basis = if i == j {
            5.0
        } else if j + 1 == i {
            2.0
        } else if i + 1 == j {
            -1.0
        } else {
            0.0
        };
        basis * s[j]
    });
    let b = Vector::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let o = IterOptions::new(1e-12, 500);
    let x_ref = a.solve(&b).unwrap();

    let mut plain = [0.0; 6];
    let mut w1 = [0.0; gmres_work_len(6, 3)];
    let r_plain = gmres(&a, b.as_slice(), &mut plain, 3, o, &mut w1);

    let jac = Jacobi::new(&a).unwrap();
    let mut pre = [0.0; 6];
    let mut w2 = [0.0; gmres_work_len(6, 3)];
    let r_pre = pgmres(&a, &jac, b.as_slice(), &mut pre, 3, o, &mut w2).unwrap();

    for i in 0..6 {
        assert!(
            (pre[i] - x_ref[i]).abs() <= 1e-9 * x_ref[i].abs().max(1e-6),
            "x[{i}]"
        );
    }
    assert!(r_pre.iterations <= 30, "{} Iterationen", r_pre.iterations);
    match r_plain {
        Ok(rp) => assert!(
            r_pre.iterations < rp.iterations,
            "{} vs. {}",
            r_pre.iterations,
            rp.iterations
        ),
        Err(e) => assert_eq!(e, LinalgError::NoConvergence),
    }
}

#[test]
fn right_preconditioner_does_not_fix_row_scaling() {
    // A = S·B (Zeilenskalierung): Rechts-Jacobi bringt hier nichts, Zeilen bräuchten einen
    // Links-Vorkonditionierer. Die Lösung ist trotzdem korrekt, nur nicht schneller.
    let s = [1.0f64, 10.0, 100.0, 10.0, 1.0, 1.0];
    let a = Matrix::<f64, 6, 6>::from_fn(|i, j| {
        let basis = if i == j {
            5.0
        } else if j + 1 == i {
            2.0
        } else if i + 1 == j {
            -1.0
        } else {
            0.0
        };
        s[i] * basis
    });
    let b = Vector::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let jac = Jacobi::new(&a).unwrap();
    let mut x = [0.0; 6];
    let mut work = [0.0; gmres_work_len(6, 6)];
    pgmres(
        &a,
        &jac,
        b.as_slice(),
        &mut x,
        6,
        IterOptions::new(1e-12, 100),
        &mut work,
    )
    .unwrap();
    let x_ref = a.solve(&b).unwrap();
    for i in 0..6 {
        assert!((x[i] - x_ref[i]).abs() <= 1e-8 * x_ref[i].abs().max(1e-6));
    }
}

#[test]
fn identity_preconditioner_equals_plain() {
    let a = nonsym3();
    let b = [1.0, 2.0, 3.0];
    let (mut x1, mut x2) = ([0.0; 3], [0.0; 3]);
    let (mut w1, mut w2) = ([0.0; gmres_work_len(3, 3)], [0.0; gmres_work_len(3, 3)]);
    let r1 = gmres(&a, &b, &mut x1, 3, opts(), &mut w1).unwrap();
    let r2 = pgmres(&a, &Identity, &b, &mut x2, 3, opts(), &mut w2).unwrap();
    assert_eq!(x1, x2);
    assert_eq!(r1, r2);
}

#[test]
fn start_guess_and_zero_rhs() {
    let a = nonsym3();
    let b = Vector::new([1.0, 2.0, 3.0]);
    let mut work = [0.0; gmres_work_len(3, 3)];
    // exakte Startnäherung
    let mut x = a.solve(&b).unwrap().into_array();
    let r = gmres(&a, b.as_slice(), &mut x, 3, opts(), &mut work).unwrap();
    assert_eq!(r.iterations, 0);
    // rechte Seite 0
    let mut z = [5.0, 5.0, 5.0];
    let r = gmres(&a, &[0.0; 3], &mut z, 3, opts(), &mut work).unwrap();
    assert_eq!((z, r.iterations, r.residual_norm), ([0.0; 3], 0, 0.0));
}

#[test]
fn restart_larger_than_n_is_clamped() {
    let a = nonsym3();
    let b = [1.0, 2.0, 3.0];
    let mut x = [0.0; 3];
    let mut work = [0.0; gmres_work_len(3, 100)];
    gmres(&a, &b, &mut x, 100, opts(), &mut work).unwrap();
    let r = (a * Vector::new(x) - Vector::new(b)).max_abs();
    assert!(r < 1e-10);
}

#[test]
fn no_convergence_with_a_tiny_budget() {
    const N: usize = 100;
    let op = FnOperator::new(N, N, poisson);
    let b = [1.0; N];
    let mut x = [0.0; N];
    let mut work = [0.0; gmres_work_len(N, 2)];
    let r = gmres(&op, &b, &mut x, 2, IterOptions::new(1e-12, 5), &mut work);
    assert_eq!(r, Err(LinalgError::NoConvergence));
    assert!(x.iter().any(|&v| v != 0.0), "x enthält die letzte Näherung");
}

#[test]
fn singular_system() {
    let a = Matrix::new([[1.0f64, 1.0], [1.0, 1.0]]);
    let mut work = [0.0; gmres_work_len(2, 2)];
    // konsistent: (1, 1) ist erreichbar, GMRES liefert eine Lösung
    let mut x = [0.0; 2];
    gmres(&a, &[1.0, 1.0], &mut x, 2, opts(), &mut work).unwrap();
    assert!((x[0] + x[1] - 1.0).abs() < 1e-10);
    // inkonsistent: keine Lösung
    let mut y = [0.0; 2];
    let r = gmres(
        &a,
        &[1.0, 0.0],
        &mut y,
        2,
        IterOptions::new(1e-12, 50),
        &mut work,
    );
    assert!(
        matches!(r, Err(LinalgError::Singular | LinalgError::NoConvergence)),
        "{r:?}"
    );
}

#[test]
fn dimension_errors() {
    let a = nonsym3();
    let b = [1.0, 2.0, 3.0];
    let mut x = [0.0; 3];
    let mut work = [0.0; 200];
    assert_eq!(
        gmres(&a, &b, &mut x, 0, opts(), &mut work),
        Err(LinalgError::DimensionMismatch)
    );
    let mut short = [0.0; gmres_work_len(3, 3) - 1];
    assert_eq!(
        gmres(&a, &b, &mut x, 3, opts(), &mut short),
        Err(LinalgError::DimensionMismatch)
    );
    let mut x2 = [0.0; 2];
    assert_eq!(
        gmres(&a, &b, &mut x2, 3, opts(), &mut work),
        Err(LinalgError::DimensionMismatch)
    );
    let rect = Matrix::new([[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(
        gmres(&rect, &b, &mut x, 3, opts(), &mut work),
        Err(LinalgError::DimensionMismatch)
    );
}

#[test]
fn non_finite_input_is_rejected() {
    let a = nonsym3();
    let mut x = [0.0; 3];
    let mut work = [0.0; gmres_work_len(3, 3)];
    assert_eq!(
        gmres(&a, &[1.0, f64::NAN, 0.0], &mut x, 3, opts(), &mut work),
        Err(LinalgError::NotFinite)
    );
    let bad = Matrix::new([[f64::NAN, 0.0], [0.0, 1.0]]);
    let mut x2 = [0.0; 2];
    let mut w2 = [0.0; gmres_work_len(2, 2)];
    assert_eq!(
        gmres(&bad, &[1.0, 1.0], &mut x2, 2, opts(), &mut w2),
        Err(LinalgError::NotFinite)
    );
}

#[test]
fn f32_converges() {
    let a = Matrix::new([[4.0f32, 1.0, 0.0], [2.0, 5.0, 1.0], [0.0, 3.0, 6.0]]);
    let b = [1.0f32, 2.0, 3.0];
    let mut x = [0.0f32; 3];
    let mut work = [0.0f32; gmres_work_len(3, 3)];
    let r = gmres(&a, &b, &mut x, 3, IterOptions::new(1e-5, 50), &mut work).unwrap();
    assert!((a * Vector::new(x) - Vector::new(b)).max_abs() < 1e-4);
    assert!(r.iterations <= 4);
}

#[cfg(feature = "alloc")]
mod dynamic {
    extern crate alloc;
    use alloc::vec;
    use azmath_linalg::iterative::{DynJacobi, IterOptions};
    use azmath_linalg::{DynMatrix, DynVector, LinalgError};

    fn conv_diff_matrix(n: usize, c: f64) -> DynMatrix<f64> {
        let mut a = DynMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 2.0;
            if i > 0 {
                a[(i, i - 1)] = -(1.0 + c);
            }
            if i + 1 < n {
                a[(i, i + 1)] = -(1.0 - c);
            }
        }
        a
    }

    #[test]
    fn solve_gmres_matches_lu() {
        let a = conv_diff_matrix(40, 0.2);
        let b = DynVector::new((0..40).map(|i| (i % 7) as f64 - 2.0).collect());
        let (x, res) = a
            .solve_gmres(&b, 40, IterOptions::new(1e-12, 2000))
            .unwrap();
        let x_ref = a.solve(&b).unwrap();
        for i in 0..40 {
            assert!((x[i] - x_ref[i]).abs() < 1e-8, "x[{i}]");
        }
        assert!(res.iterations <= 40);
        // mit Neustart ebenfalls
        let (y, _) = a
            .solve_gmres(&b, 15, IterOptions::new(1e-12, 5000))
            .unwrap();
        for i in 0..40 {
            assert!((y[i] - x_ref[i]).abs() < 1e-7, "Neustart x[{i}]");
        }
    }

    #[test]
    fn dyn_jacobi_preconditioner() {
        let n = 8;
        let s: alloc::vec::Vec<f64> = (0..n).map(|i| 10f64.powi(i as i32 / 2)).collect();
        let mut a = DynMatrix::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = 5.0 * s[i];
            if i > 0 {
                a[(i, i - 1)] = 2.0 * s[i];
            }
            if i + 1 < n {
                a[(i, i + 1)] = -s[i];
            }
        }
        let b = DynVector::new(vec![1.0; n]);
        let o = IterOptions::new(1e-12, 5000);
        let (x, pre) = a
            .solve_pgmres(&DynJacobi::new(&a).unwrap(), &b, n, o)
            .unwrap();
        let x_ref = a.solve(&b).unwrap();
        for i in 0..n {
            assert!((x[i] - x_ref[i]).abs() <= 1e-9 * x_ref[i].abs().max(1e-6));
        }
        assert!(pre.iterations <= 12);
    }

    #[test]
    fn skew_symmetric_and_errors() {
        let skew = DynMatrix::new(2, 2, vec![0.0f64, 1.0, -1.0, 0.0]).unwrap();
        let (x, _) = skew
            .solve_gmres(&DynVector::new(vec![1.0, 0.0]), 2, IterOptions::default())
            .unwrap();
        // A·x = (x₁, −x₀) = (1, 0)  ->  x = (0, 1)
        assert!(x[0].abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6);

        let a = conv_diff_matrix(4, 0.1);
        assert_eq!(
            a.solve_gmres(&DynVector::new(vec![1.0; 3]), 4, IterOptions::default())
                .unwrap_err(),
            LinalgError::DimensionMismatch
        );
        assert_eq!(
            a.solve_gmres(&DynVector::new(vec![1.0; 4]), 0, IterOptions::default())
                .unwrap_err(),
            LinalgError::DimensionMismatch
        );
        let big = conv_diff_matrix(80, 0.3);
        assert_eq!(
            big.solve_gmres(
                &DynVector::new(vec![1.0; 80]),
                3,
                IterOptions::new(1e-12, 4)
            )
            .unwrap_err(),
            LinalgError::NoConvergence
        );
    }

    #[test]
    fn large_matrix_free_with_vec_workspace() {
        use azmath_linalg::FnOperator;
        use azmath_linalg::iterative::{gmres, gmres_work_len};
        let n = 400;
        let op = FnOperator::new(n, n, |x: &[f64], y: &mut [f64]| {
            for i in 0..x.len() {
                let l = if i > 0 { x[i - 1] } else { 0.0 };
                let r = if i + 1 < x.len() { x[i + 1] } else { 0.0 };
                y[i] = -1.02 * l + 2.0 * x[i] - 0.98 * r;
            }
        });
        let b = vec![1.0; n];
        let mut x = vec![0.0; n];
        let restart = 50;
        let mut work = vec![0.0; gmres_work_len(n, restart)];
        let r = gmres(
            &op,
            &b,
            &mut x,
            restart,
            IterOptions::new(1e-8, 20_000),
            &mut work,
        )
        .unwrap();
        assert!(r.residual_norm <= 1e-8 * (n as f64).sqrt());
        assert!(x.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn fixed_and_dynamic_agree() {
        use azmath_linalg::Matrix;
        use azmath_linalg::iterative::{gmres, gmres_work_len};
        let fixed = Matrix::new([[4.0f64, 1.0, 0.0], [2.0, 5.0, 1.0], [0.0, 3.0, 6.0]]);
        let dynamic =
            DynMatrix::new(3, 3, vec![4.0, 1.0, 0.0, 2.0, 5.0, 1.0, 0.0, 3.0, 6.0]).unwrap();
        let o = IterOptions::new(1e-12, 100);
        let mut xf = [0.0; 3];
        let mut work = [0.0; gmres_work_len(3, 3)];
        gmres(&fixed, &[1.0, 2.0, 3.0], &mut xf, 3, o, &mut work).unwrap();
        let (xd, _) = dynamic
            .solve_gmres(&DynVector::new(vec![1.0, 2.0, 3.0]), 3, o)
            .unwrap();
        for i in 0..3 {
            assert_eq!(xf[i], xd[i]);
        }
    }
}
