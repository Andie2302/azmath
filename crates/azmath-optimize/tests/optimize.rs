//! Tests für Newton (Gleichungssysteme), Gauß-Newton und Levenberg-Marquardt.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath_core::Real;
use azmath_linalg::{Matrix, Vector};
use azmath_optimize::{
    OptimizeError, OptimizeOptions, Termination, gauss_newton, gauss_newton_fd,
    gauss_newton_work_len, levenberg_marquardt, levenberg_marquardt_fd,
    levenberg_marquardt_work_len, newton_system, newton_system_fd, newton_work_len,
    numerical_jacobian, numerical_jacobian_work_len,
};

fn opts() -> OptimizeOptions<f64> {
    OptimizeOptions::default()
}

fn close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() <= tol,
        "erwartet {b}, erhalten {a} (Δ = {:e})",
        (a - b).abs()
    );
}

// ---------------------------------------------------------------------------
// Testprobleme
// ---------------------------------------------------------------------------

/// Rosenbrock als Residuen: r = (10(y − x²), 1 − x); Minimum 0 bei (1, 1).
fn rosenbrock(v: &[f64], r: &mut [f64]) {
    r[0] = 10.0 * (v[1] - v[0] * v[0]);
    r[1] = 1.0 - v[0];
}
fn rosenbrock_jac(v: &[f64], j: &mut [f64]) {
    j.copy_from_slice(&[-20.0 * v[0], 10.0, -1.0, 0.0]);
}

/// Kreis x² + y² = 4 und Gerade y = x.
fn circle_line(v: &[f64], out: &mut [f64]) {
    out[0] = v[0] * v[0] + v[1] * v[1] - 4.0;
    out[1] = v[1] - v[0];
}
fn circle_line_jac(v: &[f64], j: &mut [f64]) {
    j.copy_from_slice(&[2.0 * v[0], 2.0 * v[1], -1.0, 1.0]);
}

/// Modell y = a·exp(b·t); Parameter p = (a, b).
const T: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0];

fn exp_data(a: f64, b: f64) -> [f64; 8] {
    T.map(|t| a * Real::exp(b * t))
}
fn exp_residual(y: [f64; 8]) -> impl Fn(&[f64], &mut [f64]) {
    move |p, r| {
        for i in 0..8 {
            r[i] = p[0] * Real::exp(p[1] * T[i]) - y[i];
        }
    }
}
fn exp_jac(p: &[f64], j: &mut [f64]) {
    for i in 0..8 {
        let e = Real::exp(p[1] * T[i]);
        j[2 * i] = e;
        j[2 * i + 1] = p[0] * T[i] * e;
    }
}

// ---------------------------------------------------------------------------
// numerical_jacobian
// ---------------------------------------------------------------------------

#[test]
fn numerical_jacobian_matches_analytic() {
    let x = [0.7, 1.9];
    let mut r0 = [0.0; 2];
    rosenbrock(&x, &mut r0);
    let mut jn = [0.0; 4];
    let mut ja = [0.0; 4];
    let mut work = [0.0; numerical_jacobian_work_len(2, 2)];
    numerical_jacobian(rosenbrock, &x, &r0, &mut jn, &mut work);
    rosenbrock_jac(&x, &mut ja);
    for i in 0..4 {
        close(jn[i], ja[i], 1e-6);
    }
    // nichtquadratisch (8 × 2)
    let y = exp_data(2.0, -1.0);
    let res = exp_residual(y);
    let p = [1.5, -0.7];
    let mut r = [0.0; 8];
    res(&p, &mut r);
    let mut jn = [0.0; 16];
    let mut ja = [0.0; 16];
    let mut work = [0.0; numerical_jacobian_work_len(8, 2)];
    numerical_jacobian(&res, &p, &r, &mut jn, &mut work);
    exp_jac(&p, &mut ja);
    for i in 0..16 {
        close(jn[i], ja[i], 1e-6 * ja[i].abs().max(1.0));
    }
}

// ---------------------------------------------------------------------------
// newton_system
// ---------------------------------------------------------------------------

#[test]
fn newton_circle_line_quadratic_convergence() {
    let mut x = [1.0, 0.5];
    let mut work = [0.0; newton_work_len(2)];
    let r = newton_system(circle_line, circle_line_jac, &mut x, opts(), &mut work).unwrap();
    close(x[0], 2f64.sqrt(), 1e-12);
    close(x[1], 2f64.sqrt(), 1e-12);
    assert!(r.iterations <= 8, "{} Iterationen", r.iterations);
    assert!(r.residual_norm < 1e-12);
}

#[test]
fn newton_finite_difference_jacobian_agrees() {
    let mut a = [1.0, 0.5];
    let mut b = a;
    let mut work = [0.0; newton_work_len(2)];
    newton_system(circle_line, circle_line_jac, &mut a, opts(), &mut work).unwrap();
    newton_system_fd(circle_line, &mut b, opts(), &mut work).unwrap();
    close(a[0], b[0], 1e-7);
    close(a[1], b[1], 1e-7);
}

#[test]
fn newton_rosenbrock_system() {
    let mut x = [-1.2, 1.0];
    let mut work = [0.0; newton_work_len(2)];
    let r = newton_system(rosenbrock, rosenbrock_jac, &mut x, opts(), &mut work).unwrap();
    close(x[0], 1.0, 1e-10);
    close(x[1], 1.0, 1e-10);
    assert!(r.iterations < 30);
}

#[test]
fn newton_three_dimensional_system() {
    // x² + y² + z² = 3, x·y = 1, y − z = 0  ->  (1, 1, 1)
    let f = |v: &[f64], o: &mut [f64]| {
        o[0] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - 3.0;
        o[1] = v[0] * v[1] - 1.0;
        o[2] = v[1] - v[2];
    };
    let jac = |v: &[f64], j: &mut [f64]| {
        j.copy_from_slice(&[
            2.0 * v[0],
            2.0 * v[1],
            2.0 * v[2],
            v[1],
            v[0],
            0.0,
            0.0,
            1.0,
            -1.0,
        ]);
    };
    let mut x = [1.5, 0.8, 1.2];
    let mut work = [0.0; newton_work_len(3)];
    newton_system(f, jac, &mut x, opts(), &mut work).unwrap();
    // Das System hat mehrere Lösungen, (1, 1, 1) und (√2, 1/√2, 1/√2); Newton findet hier die
    // zweite. Maßgeblich ist daher das Residuum.
    let mut out = [0.0; 3];
    f(&x, &mut out);
    for v in out {
        close(v, 0.0, 1e-10);
    }
    close(x[1], x[2], 1e-10);
    close(x[0] * x[1], 1.0, 1e-10);
    // Von einem Startwert nahe (1, 1, 1) findet es diese Lösung
    let mut y = [1.1, 0.9, 1.05];
    newton_system(f, jac, &mut y, opts(), &mut work).unwrap();
    for v in y {
        close(v, 1.0, 1e-10);
    }
}

#[test]
fn newton_on_linear_system_needs_one_step() {
    // F(x) = A·x − b ist linear: ein Newton-Schritt genügt (der zweite bestätigt)
    let a = Matrix::new([[3.0, 1.0], [1.0, 2.0]]);
    let b = Vector::new([9.0, 8.0]);
    let f = |x: &[f64], o: &mut [f64]| {
        let r = a * Vector::new([x[0], x[1]]) - b;
        o.copy_from_slice(r.as_slice());
    };
    let jac = |_x: &[f64], j: &mut [f64]| {
        for i in 0..2 {
            for k in 0..2 {
                j[2 * i + k] = a[(i, k)];
            }
        }
    };
    let mut x = [0.0, 0.0];
    let mut work = [0.0; newton_work_len(2)];
    let r = newton_system(f, jac, &mut x, opts(), &mut work).unwrap();
    let exact = a.solve(&b).unwrap();
    close(x[0], exact[0], 1e-12);
    close(x[1], exact[1], 1e-12);
    assert!(r.iterations <= 2);
}

#[test]
fn newton_start_at_solution_stops_immediately() {
    let s = 2f64.sqrt();
    let mut x = [s, s];
    let mut work = [0.0; newton_work_len(2)];
    let r = newton_system(circle_line, circle_line_jac, &mut x, opts(), &mut work).unwrap();
    assert_eq!(r.iterations, 0);
    assert!(matches!(
        r.termination,
        Termination::Residual | Termination::Gradient
    ));
}

#[test]
fn newton_singular_jacobian() {
    // F = (x², y²) bei (0, 1): J = [[0,0],[0,2]] hat Rangmangel
    let f = |v: &[f64], o: &mut [f64]| {
        o[0] = v[0] * v[0];
        o[1] = v[1] * v[1];
    };
    let jac = |v: &[f64], j: &mut [f64]| j.copy_from_slice(&[2.0 * v[0], 0.0, 0.0, 2.0 * v[1]]);
    let mut x = [0.0, 1.0];
    let mut work = [0.0; newton_work_len(2)];
    // Gradient Jᵀr ist hier 0 · … + 2·1 ≠ 0, Rang aber 1
    assert_eq!(
        newton_system(f, jac, &mut x, opts(), &mut work),
        Err(OptimizeError::SingularJacobian)
    );
}

// ---------------------------------------------------------------------------
// gauss_newton
// ---------------------------------------------------------------------------

#[test]
fn gauss_newton_exponential_fit_exact_data() {
    let y = exp_data(2.5, -1.3);
    let mut p = [2.0, -1.0];
    let mut work = [0.0; gauss_newton_work_len(8, 2)];
    let r = gauss_newton(exp_residual(y), exp_jac, &mut p, 8, opts(), &mut work).unwrap();
    close(p[0], 2.5, 1e-9);
    close(p[1], -1.3, 1e-9);
    assert!(r.residual_norm < 1e-8);
}

#[test]
fn gauss_newton_linear_problem_equals_least_squares() {
    // Lineares Modell y = c0 + c1·t: ein Schritt liefert die Kleinste-Quadrate-Lösung (QR)
    let ts = [1.0, 2.0, 3.0, 4.0];
    let ys = [1.0, 2.0, 2.0, 5.0];
    let res = |p: &[f64], r: &mut [f64]| {
        for i in 0..4 {
            r[i] = p[0] + p[1] * ts[i] - ys[i];
        }
    };
    let jac = |_p: &[f64], j: &mut [f64]| {
        for i in 0..4 {
            j[2 * i] = 1.0;
            j[2 * i + 1] = ts[i];
        }
    };
    let mut p = [0.0, 0.0];
    let mut work = [0.0; gauss_newton_work_len(4, 2)];
    let r = gauss_newton(res, jac, &mut p, 4, opts(), &mut work).unwrap();
    close(p[0], -0.5, 1e-12);
    close(p[1], 1.2, 1e-12);
    // Residuum ist nicht null: Abbruch über Gradient oder Schrittweite
    assert!((r.residual_norm - 1.8f64.sqrt()).abs() < 1e-12);
    assert!(matches!(
        r.termination,
        Termination::Gradient | Termination::Step
    ));

    let a = Matrix::new([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]]);
    let ls = a.least_squares(&Vector::new(ys)).unwrap();
    close(p[0], ls[0], 1e-12);
    close(p[1], ls[1], 1e-12);
}

#[test]
fn gauss_newton_fd_matches_analytic() {
    let y = exp_data(2.5, -1.3);
    let (mut p1, mut p2) = ([2.0, -1.0], [2.0, -1.0]);
    let mut work = [0.0; gauss_newton_work_len(8, 2)];
    gauss_newton(exp_residual(y), exp_jac, &mut p1, 8, opts(), &mut work).unwrap();
    gauss_newton_fd(exp_residual(y), &mut p2, 8, opts(), &mut work).unwrap();
    close(p1[0], p2[0], 1e-6);
    close(p1[1], p2[1], 1e-6);
}

#[test]
fn gauss_newton_rank_deficient_jacobian() {
    // Zwei identische Parameter: r = x0 + x1 − 3; J = [1, 1] hat Rang 1 bei n = 2
    let res = |p: &[f64], r: &mut [f64]| {
        r[0] = p[0] + p[1] - 3.0;
        r[1] = p[0] + p[1] - 3.0;
    };
    let jac = |_p: &[f64], j: &mut [f64]| j.copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
    let mut p = [0.0, 0.0];
    let mut work = [0.0; gauss_newton_work_len(2, 2)];
    assert_eq!(
        gauss_newton(res, jac, &mut p, 2, opts(), &mut work),
        Err(OptimizeError::SingularJacobian)
    );
}

// ---------------------------------------------------------------------------
// levenberg_marquardt
// ---------------------------------------------------------------------------

#[test]
fn lm_rosenbrock() {
    let mut x = [-1.2, 1.0];
    let mut work = [0.0; levenberg_marquardt_work_len(2, 2)];
    let r = levenberg_marquardt(rosenbrock, rosenbrock_jac, &mut x, 2, opts(), &mut work).unwrap();
    close(x[0], 1.0, 1e-8);
    close(x[1], 1.0, 1e-8);
    assert!(r.residual_norm < 1e-8);
}

#[test]
fn lm_converges_from_far_away_starts() {
    let y = exp_data(2.5, -1.3);
    let mut work = [0.0; levenberg_marquardt_work_len(8, 2)];
    for start in [
        [0.1, 1.0],
        [8.0, 1.5],
        [20.0, 0.5],
        [1.0, -4.0],
        [0.5, 0.0],
        [3.0, -2.5],
    ] {
        let mut p = start;
        let r = levenberg_marquardt(exp_residual(y), exp_jac, &mut p, 8, opts(), &mut work)
            .unwrap_or_else(|e| panic!("Start {start:?}: {e:?}"));
        close(p[0], 2.5, 1e-6);
        close(p[1], -1.3, 1e-6);
        assert!(
            r.residual_norm < 1e-6,
            "Start {start:?}: {}",
            r.residual_norm
        );
    }
}

#[test]
fn lm_is_a_local_method() {
    // Vom Startwert (0.1, 2) läuft der Exponent in das flache Plateau b ≈ −40, in dem das Modell
    // fast nur noch bei t = 0 etwas liefert. Dort bleibt ein Restfehler; das globale Optimum
    // (Rest 0) wird nicht gefunden. Lokale Verfahren garantieren das nicht.
    let y = exp_data(2.5, -1.3);
    let mut p = [0.1, 2.0];
    let mut work = [0.0; levenberg_marquardt_work_len(8, 2)];
    let r = levenberg_marquardt(exp_residual(y), exp_jac, &mut p, 8, opts(), &mut work);
    let mut res = [0.0; 8];
    exp_residual(y)(&p, &mut res);
    let rest: f64 = res.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(rest > 1.0, "Rest {rest} (Ergebnis {r:?})");
    assert!(p.iter().all(|v| v.is_finite()));
}

#[test]
fn lm_fd_matches_analytic() {
    let y = exp_data(2.5, -1.3);
    let (mut p1, mut p2) = ([1.0, -0.5], [1.0, -0.5]);
    let mut work = [0.0; levenberg_marquardt_work_len(8, 2)];
    levenberg_marquardt(exp_residual(y), exp_jac, &mut p1, 8, opts(), &mut work).unwrap();
    levenberg_marquardt_fd(exp_residual(y), &mut p2, 8, opts(), &mut work).unwrap();
    close(p1[0], p2[0], 1e-5);
    close(p1[1], p2[1], 1e-5);
}

#[test]
fn lm_noisy_data_agrees_with_gauss_newton() {
    // leicht verrauschte Daten (deterministisch): die beste Anpassung hat Rest > 0
    let noise = [0.02, -0.03, 0.01, 0.04, -0.02, -0.01, 0.03, -0.02];
    let mut y = exp_data(2.0, -0.8);
    for i in 0..8 {
        y[i] += noise[i];
    }
    let (mut pl, mut pg) = ([1.0, -0.2], [1.0, -0.2]);
    let mut work = [0.0; levenberg_marquardt_work_len(8, 2)];
    let rl = levenberg_marquardt(exp_residual(y), exp_jac, &mut pl, 8, opts(), &mut work).unwrap();
    let rg = gauss_newton(exp_residual(y), exp_jac, &mut pg, 8, opts(), &mut work).unwrap();
    close(pl[0], pg[0], 1e-6);
    close(pl[1], pg[1], 1e-6);
    close(rl.residual_norm, rg.residual_norm, 1e-9);
    assert!(rl.residual_norm > 1e-3, "Rest muss > 0 sein");
    // nahe an den Ausgangsparametern
    close(pl[0], 2.0, 0.1);
    close(pl[1], -0.8, 0.1);
    // Gradient am Optimum ≈ 0
    let mut r = [0.0; 8];
    exp_residual(y)(&pl, &mut r);
    let mut j = [0.0; 16];
    exp_jac(&pl, &mut j);
    for k in 0..2 {
        let g: f64 = (0..8).map(|i| j[2 * i + k] * r[i]).sum();
        assert!(g.abs() < 1e-6, "Gradient {g:e}");
    }
}

#[test]
fn lm_gaussian_peak_with_three_parameters() {
    // y = A·exp(−(t − c)²/(2 s²)); wahre Parameter (3, 2.5, 0.8)
    let ts: Vec<f64> = (0..25).map(|i| i as f64 * 0.25).collect();
    let model =
        |p: &[f64], t: f64| p[0] * Real::exp(-(t - p[1]) * (t - p[1]) / (2.0 * p[2] * p[2]));
    let ys: Vec<f64> = ts.iter().map(|&t| model(&[3.0, 2.5, 0.8], t)).collect();
    let m = ts.len();
    let res = |p: &[f64], r: &mut [f64]| {
        for i in 0..m {
            r[i] = model(p, ts[i]) - ys[i];
        }
    };
    let mut p = [2.0, 2.0, 1.5];
    // zur Laufzeit festgelegte Größe: Arbeitsspeicher als Vec
    let mut work = vec![0.0; levenberg_marquardt_work_len(m, 3)];
    let r = levenberg_marquardt_fd(res, &mut p, m, opts(), &mut work).unwrap();
    close(p[0], 3.0, 1e-6);
    close(p[1], 2.5, 1e-6);
    close(p[2], 0.8, 1e-6);
    assert!(r.residual_norm < 1e-6);
}

#[test]
fn lm_copes_with_rank_deficient_jacobian() {
    // r = x0 + x1 − 3 (doppelt): unendlich viele Lösungen; LM findet eine davon, Gauß-Newton nicht
    let res = |p: &[f64], r: &mut [f64]| {
        r[0] = p[0] + p[1] - 3.0;
        r[1] = p[0] + p[1] - 3.0;
    };
    let jac = |_p: &[f64], j: &mut [f64]| j.copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
    let mut p = [0.0, 0.0];
    let mut work = [0.0; levenberg_marquardt_work_len(2, 2)];
    let r = levenberg_marquardt(res, jac, &mut p, 2, opts(), &mut work).unwrap();
    close(p[0] + p[1], 3.0, 1e-9);
    assert!(r.residual_norm < 1e-8);
}

#[test]
fn lm_linear_problem_matches_least_squares() {
    let ts = [1.0, 2.0, 3.0, 4.0];
    let ys = [1.0, 2.0, 2.0, 5.0];
    let res = |p: &[f64], r: &mut [f64]| {
        for i in 0..4 {
            r[i] = p[0] + p[1] * ts[i] - ys[i];
        }
    };
    let mut p = [10.0, -10.0];
    let mut work = [0.0; levenberg_marquardt_work_len(4, 2)];
    levenberg_marquardt_fd(res, &mut p, 4, opts(), &mut work).unwrap();
    close(p[0], -0.5, 1e-7);
    close(p[1], 1.2, 1e-7);
}

// ---------------------------------------------------------------------------
// Fehlerfälle
// ---------------------------------------------------------------------------

#[test]
fn invalid_input_is_rejected() {
    let mut x = [1.0, 1.0];
    let mut work = [0.0; 200];
    // m < n
    assert_eq!(
        levenberg_marquardt(rosenbrock, rosenbrock_jac, &mut x, 1, opts(), &mut work),
        Err(OptimizeError::InvalidInput)
    );
    assert_eq!(
        gauss_newton(rosenbrock, rosenbrock_jac, &mut x, 1, opts(), &mut work),
        Err(OptimizeError::InvalidInput)
    );
    // leeres x
    let mut empty: [f64; 0] = [];
    assert_eq!(
        newton_system(rosenbrock, rosenbrock_jac, &mut empty, opts(), &mut work),
        Err(OptimizeError::InvalidInput)
    );
    // zu wenig Arbeitsspeicher
    let mut short = [0.0; 5];
    assert_eq!(
        levenberg_marquardt(rosenbrock, rosenbrock_jac, &mut x, 2, opts(), &mut short),
        Err(OptimizeError::InvalidInput)
    );
    assert_eq!(
        newton_system(rosenbrock, rosenbrock_jac, &mut x, opts(), &mut short),
        Err(OptimizeError::InvalidInput)
    );
}

#[test]
fn non_finite_values_are_reported() {
    let mut work = [0.0; 200];
    let mut x = [f64::NAN, 1.0];
    assert_eq!(
        levenberg_marquardt(rosenbrock, rosenbrock_jac, &mut x, 2, opts(), &mut work),
        Err(OptimizeError::NotFinite)
    );
    let nan_res = |_x: &[f64], r: &mut [f64]| r.fill(f64::NAN);
    let mut x = [1.0, 1.0];
    assert_eq!(
        gauss_newton_fd(nan_res, &mut x, 2, opts(), &mut work),
        Err(OptimizeError::NotFinite)
    );
    let nan_jac = |_x: &[f64], j: &mut [f64]| j.fill(f64::NAN);
    let mut x = [-1.2, 1.0];
    assert_eq!(
        levenberg_marquardt(rosenbrock, nan_jac, &mut x, 2, opts(), &mut work),
        Err(OptimizeError::NotFinite)
    );
}

#[test]
fn iteration_limit_gives_no_convergence() {
    let mut x = [-1.2, 1.0];
    let mut work = [0.0; levenberg_marquardt_work_len(2, 2)];
    let few = OptimizeOptions {
        max_iter: 2,
        ..opts()
    };
    assert_eq!(
        levenberg_marquardt(rosenbrock, rosenbrock_jac, &mut x, 2, few, &mut work),
        Err(OptimizeError::NoConvergence)
    );
    // x enthält trotzdem den letzten (verbesserten) Stand
    let mut r0 = [0.0; 2];
    rosenbrock(&[-1.2, 1.0], &mut r0);
    let mut r1 = [0.0; 2];
    rosenbrock(&x, &mut r1);
    assert!(r1[0].hypot(r1[1]) < r0[0].hypot(r0[1]));
}

#[test]
fn newton_without_root_does_not_converge() {
    // x² + 1 = 0 hat keine reelle Lösung
    let f = |v: &[f64], o: &mut [f64]| o[0] = v[0] * v[0] + 1.0;
    let jac = |v: &[f64], j: &mut [f64]| j[0] = 2.0 * v[0];
    let mut x = [3.0];
    let mut work = [0.0; newton_work_len(1)];
    let r = newton_system(
        f,
        jac,
        &mut x,
        OptimizeOptions {
            max_iter: 50,
            ..opts()
        },
        &mut work,
    );
    assert!(
        matches!(
            r,
            Err(OptimizeError::NoConvergence | OptimizeError::SingularJacobian)
        ) || matches!(r, Ok(res) if res.residual_norm >= 1.0),
        "{r:?}"
    );
    assert!(x[0].is_finite());
}

// ---------------------------------------------------------------------------
// Sonstiges
// ---------------------------------------------------------------------------

#[test]
fn f32_works() {
    let res = |p: &[f32], r: &mut [f32]| {
        r[0] = 10.0 * (p[1] - p[0] * p[0]);
        r[1] = 1.0 - p[0];
    };
    let jac = |p: &[f32], j: &mut [f32]| j.copy_from_slice(&[-20.0 * p[0], 10.0, -1.0, 0.0]);
    let mut x = [-1.2f32, 1.0];
    let mut work = [0.0f32; levenberg_marquardt_work_len(2, 2)];
    levenberg_marquardt(res, jac, &mut x, 2, OptimizeOptions::default(), &mut work).unwrap();
    assert!(
        (x[0] - 1.0).abs() < 1e-3 && (x[1] - 1.0).abs() < 1e-3,
        "{x:?}"
    );
}

#[test]
fn options_and_error_display() {
    let o = OptimizeOptions::<f64>::default();
    assert!(o.tol_x < 1e-10 && o.tol_g < 1e-10 && o.tol_f > 0.0);
    assert_eq!(o.max_iter, 200);
    let t = OptimizeOptions::with_tolerance(1e-6);
    assert_eq!((t.tol_x, t.tol_f, t.tol_g), (1e-6, 1e-6, 1e-6));
    extern crate std;
    use std::string::ToString;
    for e in [
        OptimizeError::InvalidInput,
        OptimizeError::NotFinite,
        OptimizeError::SingularJacobian,
        OptimizeError::NoConvergence,
    ] {
        assert!(!e.to_string().is_empty());
    }
}
