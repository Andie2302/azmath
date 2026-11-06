//! Fehlerpfade und Zustandsinvarianten von `QpWorkspace` (Warmstarts, ungültige Eingaben,
//! Fehlschläge) gegen Kaltstarts mit `interior_point_qp_sparse`.
#![cfg(all(feature = "alloc", any(feature = "std", feature = "libm")))]

use azmath::opt::{
    OptError, QpKktBackend, QpOptions, QpSolution, QpStatus, QpWorkspace, interior_point_qp_sparse,
};
use azmath::sparse::CsrMatrix;

const BACKENDS: [QpKktBackend; 3] = [
    QpKktBackend::Auto,
    QpKktBackend::Sparse,
    QpKktBackend::Dense,
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

/// Q (SPD, diagonaldominant), eine Gleichung Σx = b, Schranken −h₂ ≤ x ≤ h₁.
struct Problem {
    q: CsrMatrix<f64>,
    a: CsrMatrix<f64>,
    g: CsrMatrix<f64>,
}

fn problem(rng: &mut Rng, n: usize) -> Problem {
    let mut qd = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..i {
            if rng.next() > 0.3 {
                let v = 0.5 * rng.next();
                qd[i * n + j] = v;
                qd[j * n + i] = v;
            }
        }
        qd[i * n + i] = n as f64 + 1.0 + rng.next().abs();
    }
    let q = CsrMatrix::from_dense_slice(n, n, &qd, 0.0).unwrap();
    let a = CsrMatrix::from_dense_slice(1, n, &vec![1.0; n], 0.0).unwrap();
    let mut gd = vec![0.0; 2 * n * n];
    for i in 0..n {
        gd[i * n + i] = 1.0;
        gd[(n + i) * n + i] = -1.0;
    }
    let g = CsrMatrix::from_dense_slice(2 * n, n, &gd, 0.0).unwrap();
    Problem { q, a, g }
}

struct Data {
    c: Vec<f64>,
    b: Vec<f64>,
    h: Vec<f64>,
}

fn data(rng: &mut Rng, n: usize) -> Data {
    Data {
        c: (0..n).map(|_| 2.0 * rng.next()).collect(),
        b: vec![0.3 * rng.next()],
        h: (0..2 * n).map(|_| 1.0 + rng.next().abs()).collect(),
    }
}

fn cold(p: &Problem, d: &Data, opts: QpOptions<f64>) -> QpSolution<f64> {
    interior_point_qp_sparse(&p.q, &d.c, &p.a, &d.b, &p.g, &d.h, opts).unwrap()
}

fn assert_same(a: &QpSolution<f64>, b: &QpSolution<f64>, tol: f64, tag: &str) {
    assert_eq!(a.status, b.status, "{tag}");
    assert_eq!(a.x.len(), b.x.len(), "{tag}");
    for (u, v) in a.x.iter().zip(&b.x) {
        assert!((u - v).abs() <= tol, "{tag}: {u} vs {v}");
    }
}

fn opts(backend: QpKktBackend) -> QpOptions<f64> {
    QpOptions {
        kkt_backend: backend,
        ..QpOptions::default()
    }
}

#[test]
fn random_sequences_match_cold_starts_for_every_backend() {
    let mut rng = Rng(21);
    for backend in BACKENDS {
        for case in 0..8 {
            let n = 3 + case % 4;
            let p = problem(&mut rng, n);
            let mut w = QpWorkspace::new(p.q.clone(), p.a.clone(), p.g.clone(), opts(backend));
            for round in 0..12 {
                let d = data(&mut rng, n);
                if rng.next() > 0.6 {
                    w.clear_warm_start();
                }
                // Zwischen gültigen Aufrufen: ungültige Eingaben dürfen nichts verändern.
                if rng.next() > 0.3 {
                    assert!(w.solve(&d.c[..n - 1], &d.b, &d.h).is_err());
                    let mut bad = d.c.clone();
                    bad[0] = f64::NAN;
                    assert_eq!(w.solve(&bad, &d.b, &d.h).err(), Some(OptError::NotFinite));
                }
                let warm = w.solve(&d.c, &d.b, &d.h).unwrap();
                let reference = cold(&p, &d, opts(backend));
                let tag = format!("{backend:?} case {case} round {round}");
                assert_eq!(warm.status, QpStatus::Optimal, "{tag}");
                assert_same(&warm, &reference, 1e-5, &tag);
                assert!(
                    (warm.objective - reference.objective).abs() <= 1e-7,
                    "{tag}"
                );
                // zulässig: Gleichung und Schranken
                assert!((warm.x.iter().sum::<f64>() - d.b[0]).abs() <= 1e-6, "{tag}");
                for i in 0..n {
                    assert!(
                        warm.x[i] <= d.h[i] + 1e-6 && -warm.x[i] <= d.h[n + i] + 1e-6,
                        "{tag}"
                    );
                }
                assert!(warm.z.iter().all(|&z| z >= -1e-9), "{tag}: z < 0");
            }
        }
    }
}

#[test]
fn invalid_inputs_do_not_change_the_state_seen_by_the_next_solve() {
    let mut rng = Rng(22);
    for backend in BACKENDS {
        let n = 5;
        let p = problem(&mut rng, n);
        let (d1, d2) = (data(&mut rng, n), data(&mut rng, n));
        let mut with_errors =
            QpWorkspace::new(p.q.clone(), p.a.clone(), p.g.clone(), opts(backend));
        let mut without = QpWorkspace::new(p.q.clone(), p.a.clone(), p.g.clone(), opts(backend));
        with_errors.solve(&d1.c, &d1.b, &d1.h).unwrap();
        without.solve(&d1.c, &d1.b, &d1.h).unwrap();

        let nan = |v: &Vec<f64>| {
            let mut v = v.clone();
            v[0] = f64::NAN;
            v
        };
        let inf = |v: &Vec<f64>| {
            let mut v = v.clone();
            v[0] = f64::INFINITY;
            v
        };
        assert_eq!(
            with_errors.solve(&nan(&d2.c), &d2.b, &d2.h).err(),
            Some(OptError::NotFinite)
        );
        assert_eq!(
            with_errors.solve(&d2.c, &inf(&d2.b), &d2.h).err(),
            Some(OptError::NotFinite)
        );
        assert_eq!(
            with_errors.solve(&d2.c, &d2.b, &nan(&d2.h)).err(),
            Some(OptError::NotFinite)
        );
        assert_eq!(
            with_errors.solve(&d2.c, &d2.b, &inf(&d2.h)).err(),
            Some(OptError::NotFinite)
        );
        assert_eq!(
            with_errors.solve(&[], &d2.b, &d2.h).err(),
            Some(OptError::InvalidInput)
        );
        assert_eq!(
            with_errors.solve(&d2.c, &[], &d2.h).err(),
            Some(OptError::InvalidInput)
        );
        assert_eq!(
            with_errors.solve(&d2.c, &d2.b, &d2.h[..3]).err(),
            Some(OptError::InvalidInput)
        );

        let a = with_errors.solve(&d2.c, &d2.b, &d2.h).unwrap();
        let b = without.solve(&d2.c, &d2.b, &d2.h).unwrap();
        // Identisches Verhalten, auch bitweise bei den Iterationen: der Warmstart blieb erhalten.
        assert_eq!(a.iterations, b.iterations, "{backend:?}");
        assert_same(&a, &b, 0.0, &format!("{backend:?}"));
    }
}

#[test]
fn invalid_coefficients_and_options_are_reported_at_solve() {
    let mut rng = Rng(23);
    let n = 4;
    let p = problem(&mut rng, n);
    let d = data(&mut rng, n);

    let mut nan_q = p.q.clone();
    nan_q.values_mut()[0] = f64::NAN;
    let mut w = QpWorkspace::new(nan_q, p.a.clone(), p.g.clone(), QpOptions::default());
    assert_eq!(w.solve(&d.c, &d.b, &d.h).err(), Some(OptError::NotFinite));

    // nicht symmetrisches Q: [[2, 1], [0, 2]]
    let q = CsrMatrix::from_dense_slice(2, 2, &[2.0, 1.0, 0.0, 2.0], 0.0).unwrap();
    let a = CsrMatrix::from_dense_slice(1, 2, &[1.0, 1.0], 0.0).unwrap();
    let g = CsrMatrix::identity(2);
    let mut w = QpWorkspace::new(q, a, g, QpOptions::default());
    assert_eq!(
        w.solve(&[1.0, 1.0], &[1.0], &[2.0, 2.0]).err(),
        Some(OptError::InvalidInput)
    );

    // falsche Spaltenzahl von A
    let bad_a = CsrMatrix::from_dense_slice(1, n + 1, &vec![1.0; n + 1], 0.0).unwrap();
    let mut w = QpWorkspace::new(p.q.clone(), bad_a, p.g.clone(), QpOptions::default());
    assert_eq!(
        w.solve(&d.c, &d.b, &d.h).err(),
        Some(OptError::InvalidInput)
    );

    // ungültige Optionen
    for bad in [0.0, -1.0, f64::NAN] {
        let o = QpOptions {
            tol: bad,
            ..QpOptions::default()
        };
        let mut w = QpWorkspace::new(p.q.clone(), p.a.clone(), p.g.clone(), o);
        assert_eq!(
            w.solve(&d.c, &d.b, &d.h).err(),
            Some(OptError::InvalidInput),
            "tol={bad}"
        );
    }
    let o = QpOptions {
        regularization: -1.0,
        ..QpOptions::default()
    };
    let mut w = QpWorkspace::new(p.q.clone(), p.a.clone(), p.g.clone(), o);
    assert_eq!(
        w.solve(&d.c, &d.b, &d.h).err(),
        Some(OptError::InvalidInput)
    );
}

#[test]
fn infeasible_problem_fails_without_poisoning_the_next_solve() {
    let mut rng = Rng(24);
    for backend in BACKENDS {
        let n = 4;
        let p = problem(&mut rng, n);
        let good = data(&mut rng, n);
        let mut w = QpWorkspace::new(p.q.clone(), p.a.clone(), p.g.clone(), opts(backend));
        w.solve(&good.c, &good.b, &good.h).unwrap();

        // Schranken x ≤ −1 und −x ≤ −1 widersprechen sich.
        let infeasible = Data {
            c: good.c.clone(),
            b: good.b.clone(),
            h: vec![-1.0; 2 * n],
        };
        let r = w
            .solve(&infeasible.c, &infeasible.b, &infeasible.h)
            .unwrap();
        assert_ne!(r.status, QpStatus::Optimal, "{backend:?}");
        assert!(
            r.x.iter().all(|v| v.is_finite()),
            "{backend:?}: x nicht endlich"
        );

        // Nach dem Fehlschlag wird kalt gestartet: gleiche Iterationszahl und Lösung wie ein Kaltstart.
        let next = data(&mut rng, n);
        let after = w.solve(&next.c, &next.b, &next.h).unwrap();
        let reference = cold(&p, &next, opts(backend));
        assert_eq!(after.status, QpStatus::Optimal, "{backend:?}");
        assert_eq!(
            after.iterations, reference.iterations,
            "{backend:?}: Warmstart nicht verworfen"
        );
        assert_same(&after, &reference, 1e-7, &format!("{backend:?}"));
    }
}

#[test]
fn inconsistent_redundant_equalities_report_failure_and_workspace_recovers() {
    let q = CsrMatrix::<f64>::identity(2);
    let a = CsrMatrix::from_dense_slice(2, 2, &[1.0, 1.0, 2.0, 2.0], 0.0).unwrap();
    let g = CsrMatrix::identity(2);
    let mut w = QpWorkspace::new(q, a, g, QpOptions::default());
    for _ in 0..3 {
        let bad = w.solve(&[-1.0, -1.0], &[1.0, 3.0], &[2.0, 2.0]).unwrap();
        assert_eq!(bad.status, QpStatus::NumericalFailure);
        assert_eq!(bad.y.len(), 2);
        let ok = w.solve(&[-1.0, -1.0], &[1.0, 2.0], &[2.0, 2.0]).unwrap();
        assert_eq!(ok.status, QpStatus::Optimal);
        assert!((ok.x[0] + ok.x[1] - 1.0).abs() < 1e-7);
    }
}

#[test]
fn iteration_limit_is_reported_and_never_stored_as_warm_start() {
    let mut rng = Rng(25);
    let n = 5;
    let p = problem(&mut rng, n);
    let d = data(&mut rng, n);
    let o = QpOptions {
        max_iter: 1,
        ..QpOptions::default()
    };
    let mut w = QpWorkspace::new(p.q.clone(), p.a.clone(), p.g.clone(), o);
    for _ in 0..3 {
        // ohne gespeicherten Startwert bleibt jeder Aufruf identisch (Zustand wächst nicht)
        let r = w.solve(&d.c, &d.b, &d.h).unwrap();
        assert_eq!(r.status, QpStatus::MaxIterations);
        assert_eq!(r.iterations, 1);
        assert!(r.x.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn solving_the_same_problem_twice_is_consistent_and_not_slower() {
    let mut rng = Rng(26);
    for backend in BACKENDS {
        let n = 6;
        let p = problem(&mut rng, n);
        let d = data(&mut rng, n);
        let mut w = QpWorkspace::new(p.q.clone(), p.a.clone(), p.g.clone(), opts(backend));
        let first = w.solve(&d.c, &d.b, &d.h).unwrap();
        let second = w.solve(&d.c, &d.b, &d.h).unwrap();
        assert_eq!(second.status, QpStatus::Optimal);
        assert!(second.iterations <= first.iterations, "{backend:?}");
        assert_same(&first, &second, 1e-6, &format!("{backend:?}"));
    }
}
