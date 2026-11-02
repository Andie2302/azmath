//! Tests für Eigenwerte (Jacobi, Potenzmethode, inverse Iteration) und die SVD.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::Real;
use azmath::linalg::eigen::{inverse_iteration, power_iteration, power_iteration_work_len};
use azmath::linalg::iterative::IterOptions;
use azmath::linalg::{FnOperator, LinalgError, Matrix, Vector};
use core::f64::consts::PI;

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "erwartet {b}, erhalten {a} (Δ = {:e})",
        (a - b).abs()
    );
}

fn rel_close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol * b.abs(),
        "erwartet {b}, erhalten {a} (rel. Δ = {:e})",
        ((a - b) / b).abs()
    );
}

fn opts() -> IterOptions<f64> {
    IterOptions::new(1e-12, 100_000)
}

/// Symmetrische Beispielmatrix der Größe N (deterministisch, nicht trivial).
fn sym<const N: usize>() -> Matrix<f64, N, N> {
    Matrix::from_fn(|i, j| {
        let (lo, hi) = (i.min(j), i.max(j));
        ((lo * 7 + hi * 3) % 11) as f64 * 0.3 - 1.0 + if i == j { 2.0 } else { 0.0 }
    })
}

/// Prüft A·V = V·Λ und VᵀV = I.
fn check_eigen<const N: usize>(
    a: &Matrix<f64, N, N>,
    values: &Vector<f64, N>,
    vectors: &Matrix<f64, N, N>,
    tol: f64,
) {
    let av = *a * *vectors;
    for j in 0..N {
        for i in 0..N {
            close(av[(i, j)], vectors[(i, j)] * values[j], tol);
        }
    }
    let vtv = vectors.transpose() * *vectors;
    for i in 0..N {
        for j in 0..N {
            close(vtv[(i, j)], if i == j { 1.0 } else { 0.0 }, tol);
        }
    }
}

// ---------------------------------------------------------------------------
// Symmetrische Eigenwerte (Jacobi)
// ---------------------------------------------------------------------------

#[test]
fn two_by_two_known_eigenpairs() {
    let a = Matrix::new([[2.0, 1.0], [1.0, 2.0]]);
    let e = a.symmetric_eigen().unwrap();
    close(e.values[0], 1.0, 1e-14);
    close(e.values[1], 3.0, 1e-14);
    check_eigen(&a, &e.values, &e.vectors, 1e-13);
    // Eigenvektor zu 3 ist ±(1, 1)/√2
    close(e.vectors[(0, 1)].abs(), 1.0 / 2f64.sqrt(), 1e-13);
    close(e.vectors[(0, 1)] * e.vectors[(1, 1)], 0.5, 1e-13);
}

#[test]
fn spectral_decomposition_of_a_symmetric_matrix() {
    let a = sym::<6>();
    let e = a.symmetric_eigen().unwrap();
    check_eigen(&a, &e.values, &e.vectors, 1e-12);
    // aufsteigend sortiert
    assert!(e.values.as_slice().windows(2).all(|w| w[0] <= w[1]));
    // Rekonstruktion, Spur und Determinante
    let rec = e.reconstruct();
    for i in 0..6 {
        for j in 0..6 {
            close(rec[(i, j)], a[(i, j)], 1e-12);
        }
    }
    close(e.values.as_slice().iter().sum::<f64>(), a.trace(), 1e-12);
    let det: f64 = e.values.as_slice().iter().product();
    close(det, a.determinant(), 1e-9 * a.determinant().abs().max(1.0));
    assert!(e.sweeps >= 1 && e.sweeps < 20, "{} Sweeps", e.sweeps);
}

#[test]
fn larger_matrix() {
    let a = sym::<25>();
    let e = a.symmetric_eigen().unwrap();
    check_eigen(&a, &e.values, &e.vectors, 1e-11);
}

#[test]
fn diagonal_matrix_needs_no_rotation_and_is_sorted() {
    let a = Matrix::new([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]);
    let e = a.symmetric_eigen().unwrap();
    assert_eq!(e.sweeps, 0);
    assert_eq!(e.values.as_slice(), &[1.0, 2.0, 3.0]);
    check_eigen(&a, &e.values, &e.vectors, 1e-15);
}

#[test]
fn repeated_eigenvalues() {
    // Eigenwerte 1, 1, 4 (Identität plus Rang-1-Anteil)
    let a = Matrix::new([[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]);
    let e = a.symmetric_eigen().unwrap();
    close(e.values[0], 1.0, 1e-13);
    close(e.values[1], 1.0, 1e-13);
    close(e.values[2], 4.0, 1e-13);
    check_eigen(&a, &e.values, &e.vectors, 1e-12);
    let id = Matrix::<f64, 4, 4>::identity();
    let ei = id.symmetric_eigen().unwrap();
    assert!(ei.values.as_slice().iter().all(|&v| v == 1.0));
}

#[test]
fn hilbert_matrix_has_accurate_small_eigenvalues() {
    // H₅: λ_max ≈ 1.5670506910982, λ_min ≈ 3.2879287721719e-6 (Konditionszahl ≈ 4.8e5)
    let h = Matrix::<f64, 5, 5>::from_fn(|i, j| 1.0 / (i + j + 1) as f64);
    let e = h.symmetric_eigen().unwrap();
    rel_close(e.values[4], 1.567_050_691_098_231, 1e-12);
    rel_close(e.values[0], 3.287_928_772_171_892e-6, 1e-8);
    assert!(e.values.as_slice().iter().all(|&v| v > 0.0));
    check_eigen(&h, &e.values, &e.vectors, 1e-12);
    // symmetrisch positiv definit: Cholesky gelingt ebenfalls
    assert!(h.cholesky().is_ok());
}

#[test]
fn poisson_spectrum_matches_the_analytic_formula() {
    // Tridiagonal(−1, 2, −1): λ_k = 2 − 2·cos(kπ/(n+1)), k = 1 … n
    const N: usize = 12;
    let a = Matrix::<f64, N, N>::from_fn(|i, j| {
        if i == j {
            2.0
        } else if i.abs_diff(j) == 1 {
            -1.0
        } else {
            0.0
        }
    });
    let e = a.symmetric_eigen().unwrap();
    for k in 1..=N {
        let exact = 2.0 - 2.0 * Real::cos(k as f64 * PI / (N as f64 + 1.0));
        close(e.values[k - 1], exact, 1e-13);
    }
}

#[test]
fn indefinite_and_negative_eigenvalues() {
    let a = Matrix::new([[1.0, 2.0], [2.0, 1.0]]); // −1 und 3
    let e = a.symmetric_eigen().unwrap();
    close(e.values[0], -1.0, 1e-14);
    close(e.values[1], 3.0, 1e-14);
}

#[test]
fn jacobi_error_cases() {
    let nonsym = Matrix::new([[1.0, 2.0], [0.0, 1.0]]);
    assert_eq!(
        nonsym.symmetric_eigen().unwrap_err(),
        LinalgError::NotSymmetric
    );
    let nan = Matrix::new([[1.0, f64::NAN], [f64::NAN, 1.0]]);
    assert_eq!(nan.symmetric_eigen().unwrap_err(), LinalgError::NotFinite);
    // Sonderfälle
    let one = Matrix::new([[5.0]]);
    let e = one.symmetric_eigen().unwrap();
    assert_eq!((e.values[0], e.vectors[(0, 0)]), (5.0, 1.0));
    assert!(Matrix::<f64, 0, 0>::zeros().symmetric_eigen().is_ok());
}

// ---------------------------------------------------------------------------
// Potenzmethode und inverse Iteration
// ---------------------------------------------------------------------------

#[test]
fn power_iteration_finds_the_dominant_eigenpair() {
    let a = sym::<6>();
    let e = a.symmetric_eigen().unwrap();
    let (lambda, v, iters) = a.dominant_eigenpair(opts()).unwrap();
    // betragsgrößter Eigenwert (Jacobi: kleinster oder größter)
    let reference = if e.values[0].abs() > e.values[5].abs() {
        e.values[0]
    } else {
        e.values[5]
    };
    close(lambda, reference, 1e-9);
    // A·v = λ·v
    let r = a * v - v.scale(lambda);
    assert!(r.max_abs() < 1e-9, "Residuum {:e}", r.max_abs());
    assert!(iters > 0);
}

#[test]
fn power_iteration_handles_a_negative_dominant_eigenvalue() {
    // Eigenwerte −5 und 2: der Vektor wechselt in jedem Schritt das Vorzeichen
    let a = Matrix::new([[-4.0, 1.0], [1.0, 1.0 - 0.0]]);
    let e = a.symmetric_eigen().unwrap();
    let (lambda, v, _) = a.dominant_eigenpair(opts()).unwrap();
    close(lambda, e.values[0], 1e-10);
    assert!(lambda < -3.0);
    assert!((a * v - v.scale(lambda)).max_abs() < 1e-9);
}

#[test]
fn power_iteration_for_a_nonsymmetric_matrix_with_a_real_dominant_eigenvalue() {
    // Dreiecksmatrix: Eigenwerte 3 und 1
    let a = Matrix::new([[3.0, 1.0], [0.0, 1.0]]);
    let (lambda, v, _) = a.dominant_eigenpair(opts()).unwrap();
    close(lambda, 3.0, 1e-10);
    assert!((a * v - v.scale(lambda)).max_abs() < 1e-9);
}

fn poisson_operator<const N: usize>() -> FnOperator<f64, impl Fn(&[f64], &mut [f64])> {
    FnOperator::new(N, N, |x: &[f64], y: &mut [f64]| {
        for i in 0..x.len() {
            let l = if i > 0 { x[i - 1] } else { 0.0 };
            let r = if i + 1 < x.len() { x[i + 1] } else { 0.0 };
            y[i] = 2.0 * x[i] - l - r;
        }
    })
}

/// λ_k des Tridiagonal(−1, 2, −1)-Operators der Größe n.
fn poisson_eigenvalue(k: usize, n: usize) -> f64 {
    2.0 - 2.0 * Real::cos(k as f64 * PI / (n as f64 + 1.0))
}

/// Deterministischer Pseudozufalls-Startvektor (breites Frequenzspektrum).
fn mixed_start<const N: usize>() -> [f64; N] {
    let mut state: u64 = 0x1234_5678_9ABC_DEF1;
    core::array::from_fn(|_| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        0.5 + (state >> 11) as f64 / (1u64 << 53) as f64
    })
}

#[test]
fn matrix_free_power_iteration_on_the_poisson_operator() {
    const N: usize = 50;
    let op = poisson_operator::<N>();
    let mut x = mixed_start::<N>();
    let mut work = [0.0; power_iteration_work_len(N)];
    let r = power_iteration(&op, &mut x, IterOptions::new(1e-10, 200_000), &mut work).unwrap();
    close(r.eigenvalue, poisson_eigenvalue(N, N), 1e-9);
    let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
    close(norm, 1.0, 1e-12);
    // A·x = λ·x
    let mut ax = [0.0; N];
    use azmath::linalg::LinearOperator;
    op.apply(&x, &mut ax);
    for i in 0..N {
        close(ax[i], r.eigenvalue * x[i], 1e-8);
    }
}

#[test]
fn tolerance_controls_accuracy_when_the_top_eigenvalues_are_clustered() {
    // n = 200: die beiden größten Eigenwerte unterscheiden sich nur um 7·10⁻⁴, die Potenzmethode
    // konvergiert entsprechend langsam (Faktor 0.99982 pro Schritt). Das Residuumskriterium garantiert,
    // dass der gemeldete Wert innerhalb von tol·|λ| an einem Eigenwert liegt; strengere Toleranz kostet
    // viele Iterationen.
    const N: usize = 200;
    let op = poisson_operator::<N>();
    let top = poisson_eigenvalue(N, N);
    let mut work = [0.0; power_iteration_work_len(N)];

    let mut x = mixed_start::<N>();
    let loose = power_iteration(&op, &mut x, IterOptions::new(1e-4, 5_000_000), &mut work).unwrap();
    assert!(
        (loose.eigenvalue - top).abs() <= 1e-4 * top,
        "λ = {}",
        loose.eigenvalue
    );

    let mut y = mixed_start::<N>();
    let strict =
        power_iteration(&op, &mut y, IterOptions::new(1e-10, 5_000_000), &mut work).unwrap();
    close(strict.eigenvalue, top, 1e-9);
    assert!(
        strict.iterations > 2 * loose.iterations,
        "{} vs. {}",
        strict.iterations,
        loose.iterations
    );
}

#[test]
fn a_start_vector_orthogonal_to_the_dominant_mode_never_finds_it() {
    // Der dominante Eigenvektor des Poisson-Operators alterniert im Vorzeichen; das periodische Muster
    // 1 + (i mod 3) ist rechnerisch senkrecht dazu (Überlapp ≈ 6·10⁻¹⁶). Die Potenzmethode meldet dann
    // ein anderes, gültiges Eigenpaar, nie den dominanten Eigenwert.
    const N: usize = 50;
    let op = poisson_operator::<N>();
    let mut x: [f64; N] = core::array::from_fn(|i| 1.0 + (i % 3) as f64);
    let mut work = [0.0; power_iteration_work_len(N)];
    let r = power_iteration(&op, &mut x, IterOptions::new(1e-10, 200_000), &mut work).unwrap();
    assert!(
        poisson_eigenvalue(N, N) - r.eigenvalue > 1e-3,
        "λ = {}",
        r.eigenvalue
    );
}

#[test]
fn inverse_iteration_finds_the_eigenvalue_nearest_the_shift() {
    let a = sym::<6>();
    let e = a.symmetric_eigen().unwrap();
    for &target in &[e.values[0], e.values[2], e.values[4]] {
        let (lambda, v, iters) = a.eigenpair_near(target + 1e-3, opts()).unwrap();
        close(lambda, target, 1e-9);
        assert!((a * v - v.scale(lambda)).max_abs() < 1e-8);
        assert!(iters <= 20, "{iters} Iterationen");
    }
}

#[test]
fn inverse_iteration_with_a_custom_solver() {
    // (A − σI)⁻¹ für eine Diagonalmatrix: trivial, hier als beliebige Closure
    let diag = [1.0f64, 4.0, 9.0];
    let sigma = 4.2;
    let solve = |b: &[f64], y: &mut [f64]| {
        for i in 0..3 {
            y[i] = b[i] / (diag[i] - sigma);
        }
    };
    let mut x = [1.0, 1.0, 1.0];
    let mut work = [0.0; 3];
    let r = inverse_iteration(solve, sigma, &mut x, opts(), &mut work).unwrap();
    close(r.eigenvalue, 4.0, 1e-10);
    close(x[1].abs(), 1.0, 1e-9);
}

#[test]
fn eigen_iteration_error_cases() {
    let a = Matrix::new([[2.0, 1.0], [1.0, 2.0]]);
    let mut work = [0.0; 2];
    // Nullvektor
    assert_eq!(
        power_iteration(&a, &mut [0.0, 0.0], opts(), &mut work),
        Err(LinalgError::Breakdown)
    );
    // Längen
    assert_eq!(
        power_iteration(&a, &mut [1.0, 1.0, 1.0], opts(), &mut [0.0; 3]),
        Err(LinalgError::DimensionMismatch)
    );
    assert_eq!(
        power_iteration(&a, &mut [1.0, 1.0], opts(), &mut [0.0; 1]),
        Err(LinalgError::DimensionMismatch)
    );
    // NaN
    assert_eq!(
        power_iteration(&a, &mut [1.0, f64::NAN], opts(), &mut work),
        Err(LinalgError::NotFinite)
    );
    // Drehung: Eigenwerte ±i, es gibt keinen reellen dominanten Eigenwert
    let rot = Matrix::new([[0.0, -1.0], [1.0, 0.0]]);
    assert_eq!(
        power_iteration(
            &rot,
            &mut [1.0, 0.5],
            IterOptions::new(1e-10, 500),
            &mut work
        ),
        Err(LinalgError::NoConvergence)
    );
    // Verschiebung genau auf einem Eigenwert: A − σI singulär
    assert_eq!(
        a.eigenpair_near(3.0, opts()).unwrap_err(),
        LinalgError::Singular
    );
}

// ---------------------------------------------------------------------------
// SVD
// ---------------------------------------------------------------------------

fn tall() -> Matrix<f64, 5, 3> {
    Matrix::new([
        [1.0, 2.0, 0.5],
        [3.0, -1.0, 2.0],
        [0.0, 4.0, 1.0],
        [2.0, 2.0, -3.0],
        [-1.0, 0.5, 1.5],
    ])
}

#[test]
fn svd_reconstructs_and_is_orthogonal() {
    let a = tall();
    let svd = a.svd().unwrap();
    let rec = svd.reconstruct();
    for i in 0..5 {
        for j in 0..3 {
            close(rec[(i, j)], a[(i, j)], 1e-13);
        }
    }
    let utu = svd.u.transpose() * svd.u;
    let vtv = svd.v.transpose() * svd.v;
    for i in 0..3 {
        for j in 0..3 {
            let id = if i == j { 1.0 } else { 0.0 };
            close(utu[(i, j)], id, 1e-13);
            close(vtv[(i, j)], id, 1e-13);
        }
    }
    // absteigend und nicht negativ
    assert!(svd.s.as_slice().windows(2).all(|w| w[0] >= w[1]));
    assert!(svd.s.as_slice().iter().all(|&s| s >= 0.0));
    assert_eq!(svd.rank(), 3);
}

#[test]
fn singular_values_are_the_roots_of_the_eigenvalues_of_ata() {
    let a = tall();
    let svd = a.svd().unwrap();
    let ata = a.transpose() * a;
    let e = ata.symmetric_eigen().unwrap(); // aufsteigend
    for k in 0..3 {
        close(svd.s[k], Real::sqrt(e.values[2 - k]), 1e-12);
    }
    // Frobeniusnorm² = Σσ²
    let fro2: f64 = (0..5)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| a[(i, j)].powi(2))
        .sum();
    close(
        svd.s.as_slice().iter().map(|s| s * s).sum::<f64>(),
        fro2,
        1e-11,
    );
}

#[test]
fn square_matrix_determinant_and_condition_number() {
    let a = Matrix::new([[4.0, 1.0, 0.0], [2.0, 5.0, 1.0], [0.0, 3.0, 6.0]]);
    let svd = a.svd().unwrap();
    let prod: f64 = svd.s.as_slice().iter().product();
    close(prod, a.determinant().abs(), 1e-10);
    close(svd.condition_number(), svd.s[0] / svd.s[2], 1e-14);
    let d = Matrix::new([[1.0, 0.0], [0.0, 1.0e-6]]);
    close(d.svd().unwrap().condition_number(), 1.0e6, 1e-6);
}

#[test]
fn rank_deficient_matrix() {
    // dritte Spalte = erste + zweite
    let a = Matrix::new([
        [1.0f64, 2.0, 3.0],
        [4.0, 5.0, 9.0],
        [7.0, 8.0, 15.0],
        [1.0, 0.0, 1.0],
    ]);
    let svd = a.svd().unwrap();
    assert_eq!(svd.rank(), 2);
    assert!(svd.s[2] < 1e-13 * svd.s[0], "σ₃ = {:e}", svd.s[2]);
    assert!(svd.condition_number() > 1e12);
    // QR scheitert hier, SVD löst
    assert_eq!(a.qr().unwrap_err(), LinalgError::Singular);
    let b = Vector::new([1.0, 2.0, 3.0, 4.0]);
    let x = svd.solve(&b);
    assert!(x.as_slice().iter().all(|v| v.is_finite()));
    // x minimiert ‖Ax − b‖: Normalengleichungen Aᵀ(Ax − b) = 0
    let grad = a.transpose() * (a * x - b);
    assert!(grad.max_abs() < 1e-10, "Gradient {:e}", grad.max_abs());
}

#[test]
fn minimum_norm_solution() {
    // A = [[1,1],[1,1]], b = (2, 2): alle (t, 2−t) lösen das System, minimale Norm bei (1, 1)
    let a = Matrix::new([[1.0, 1.0], [1.0, 1.0]]);
    let svd = a.svd().unwrap();
    assert_eq!(svd.rank(), 1);
    let x = svd.solve(&Vector::new([2.0, 2.0]));
    close(x[0], 1.0, 1e-13);
    close(x[1], 1.0, 1e-13);
}

#[test]
fn svd_solve_matches_qr_for_full_rank_least_squares() {
    let a = tall();
    let b = Vector::new([1.0, 2.0, 0.5, -1.0, 3.0]);
    let x_svd = a.svd().unwrap().solve(&b);
    let x_qr = a.least_squares(&b).unwrap();
    for i in 0..3 {
        close(x_svd[i], x_qr[i], 1e-11);
    }
}

#[test]
fn pseudo_inverse_satisfies_the_moore_penrose_conditions() {
    let a = tall();
    let svd = a.svd().unwrap();
    let p = svd.pseudo_inverse();
    let apa = a * p * a;
    for i in 0..5 {
        for j in 0..3 {
            close(apa[(i, j)], a[(i, j)], 1e-12);
        }
    }
    let pap = p * a * p;
    for i in 0..3 {
        for j in 0..5 {
            close(pap[(i, j)], p[(i, j)], 1e-12);
        }
    }
    // auch bei Rangmangel
    let r = Matrix::new([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]);
    let rp = r.svd().unwrap().pseudo_inverse();
    let rpr = r * rp * r;
    for i in 0..3 {
        for j in 0..2 {
            close(rpr[(i, j)], r[(i, j)], 1e-12);
        }
    }
}

#[test]
fn zero_and_special_matrices() {
    let z = Matrix::<f64, 3, 2>::zeros();
    let svd = z.svd().unwrap();
    assert_eq!(svd.s.as_slice(), &[0.0, 0.0]);
    assert_eq!(svd.rank(), 0);
    assert_eq!(svd.condition_number(), f64::INFINITY);
    assert_eq!(
        svd.solve(&Vector::new([1.0, 2.0, 3.0])).as_slice(),
        &[0.0, 0.0]
    );

    let diag = Matrix::new([[3.0, 0.0], [0.0, -2.0], [0.0, 0.0]]);
    let s = diag.svd().unwrap();
    close(s.s[0], 3.0, 1e-14);
    close(s.s[1], 2.0, 1e-14);

    let single = Matrix::new([[3.0], [4.0]]);
    close(single.svd().unwrap().s[0], 5.0, 1e-14);
}

#[test]
fn svd_error_cases() {
    let wide = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    assert_eq!(wide.svd().unwrap_err(), LinalgError::DimensionMismatch);
    // die Transponierte ist zerlegbar und hat dieselben Singulärwerte
    assert!(wide.transpose().svd().is_ok());
    let nan = Matrix::new([[1.0, f64::NAN], [0.0, 1.0]]);
    assert_eq!(nan.svd().unwrap_err(), LinalgError::NotFinite);
}

#[test]
fn f32_works() {
    let a = Matrix::new([[2.0f32, 1.0], [1.0, 2.0]]);
    let e = a.symmetric_eigen().unwrap();
    assert!((e.values[0] - 1.0).abs() < 1e-5 && (e.values[1] - 3.0).abs() < 1e-5);
    let svd = Matrix::new([[3.0f32, 0.0], [0.0, 4.0]]).svd().unwrap();
    assert!((svd.s[0] - 4.0).abs() < 1e-5 && (svd.s[1] - 3.0).abs() < 1e-5);
}

// ---------------------------------------------------------------------------
// Dynamische Varianten
// ---------------------------------------------------------------------------

#[cfg(feature = "alloc")]
mod dynamic {
    use super::*;
    use azmath::linalg::{DynMatrix, DynVector};

    fn dyn_sym(n: usize) -> DynMatrix<f64> {
        DynMatrix::new(
            n,
            n,
            (0..n * n)
                .map(|k| {
                    let (i, j) = (k / n, k % n);
                    let (lo, hi) = (i.min(j), i.max(j));
                    ((lo * 7 + hi * 3) % 11) as f64 * 0.3 - 1.0 + if i == j { 2.0 } else { 0.0 }
                })
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn symmetric_eigen_matches_the_fixed_size_version() {
        let fixed = sym::<6>();
        let dynamic = dyn_sym(6);
        let ef = fixed.symmetric_eigen().unwrap();
        let ed = dynamic.symmetric_eigen().unwrap();
        for i in 0..6 {
            close(ef.values[i], ed.values[i], 1e-13);
        }
        // A·V = V·Λ
        let av = dynamic.matmul(&ed.vectors).unwrap();
        for j in 0..6 {
            for i in 0..6 {
                close(av[(i, j)], ed.vectors[(i, j)] * ed.values[j], 1e-12);
            }
        }
    }

    #[test]
    fn large_runtime_sized_matrix() {
        let n = 40;
        let a = dyn_sym(n);
        let e = a.symmetric_eigen().unwrap();
        let tr: f64 = (0..n).map(|i| a[(i, i)]).sum();
        close(e.values.as_slice().iter().sum::<f64>(), tr, 1e-10);
        assert!(e.values.as_slice().windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn power_and_inverse_iteration() {
        let a = dyn_sym(6);
        let e = a.symmetric_eigen().unwrap();
        let (lambda, v, _) = a.dominant_eigenpair(opts()).unwrap();
        let reference = if e.values[0].abs() > e.values[5].abs() {
            e.values[0]
        } else {
            e.values[5]
        };
        close(lambda, reference, 1e-9);
        let av = a.mul_vec(&v).unwrap();
        for i in 0..6 {
            close(av[i], lambda * v[i], 1e-8);
        }
        let (mu, _, _) = a.eigenpair_near(e.values[2] + 1e-3, opts()).unwrap();
        close(mu, e.values[2], 1e-9);
    }

    #[test]
    fn svd_matches_the_fixed_size_version_and_solves_rank_deficient_systems() {
        let t = tall();
        let dt = DynMatrix::new(5, 3, (0..15).map(|k| t[(k / 3, k % 3)]).collect()).unwrap();
        let sf = t.svd().unwrap();
        let sd = dt.svd().unwrap();
        for k in 0..3 {
            close(sf.s[k], sd.s[k], 1e-13);
        }
        let rec = sd.reconstruct();
        for i in 0..5 {
            for j in 0..3 {
                close(rec[(i, j)], t[(i, j)], 1e-13);
            }
        }
        assert_eq!(sd.rank(), 3);

        let r = DynMatrix::new(2, 2, vec![1.0, 1.0, 1.0, 1.0]).unwrap();
        let svd = r.svd().unwrap();
        assert_eq!(svd.rank(), 1);
        let x = svd.solve(&DynVector::new(vec![2.0, 2.0])).unwrap();
        close(x[0], 1.0, 1e-13);
        close(x[1], 1.0, 1e-13);
        assert_eq!(svd.condition_number(), f64::INFINITY);
        let p = svd.pseudo_inverse();
        assert_eq!((p.rows(), p.cols()), (2, 2));
    }

    #[test]
    fn dynamic_errors() {
        let rect = DynMatrix::new(2, 3, vec![1.0; 6]).unwrap();
        assert_eq!(
            rect.symmetric_eigen().unwrap_err(),
            LinalgError::DimensionMismatch
        );
        assert_eq!(rect.svd().unwrap_err(), LinalgError::DimensionMismatch);
        assert_eq!(
            rect.eigenpair_near(1.0, opts()).unwrap_err(),
            LinalgError::DimensionMismatch
        );
        let nonsym = DynMatrix::new(2, 2, vec![1.0, 2.0, 0.0, 1.0]).unwrap();
        assert_eq!(
            nonsym.symmetric_eigen().unwrap_err(),
            LinalgError::NotSymmetric
        );
        let svd = DynMatrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0])
            .unwrap()
            .svd()
            .unwrap();
        assert_eq!(
            svd.solve(&DynVector::new(vec![1.0])).unwrap_err(),
            LinalgError::DimensionMismatch
        );
    }
}
