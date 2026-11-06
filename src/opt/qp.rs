#![allow(clippy::needless_range_loop)]

use super::OptError;
use super::qp_kkt::KktSystem;
use super::qp_matrix::{QpMatrix, independent_rows};
use crate::Real;
use crate::sparse::CsrMatrix;
use alloc::vec;
use alloc::vec::Vec;

/// Ausgang des Innere-Punkte-Verfahrens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpStatus {
    /// Optimale Lösung innerhalb der Toleranz gefunden.
    Optimal,
    /// Iterationslimit erreicht (`x` ist die letzte Näherung).
    MaxIterations,
    /// Das KKT-System war nicht lösbar oder die Iterierten laufen davon (typisch bei **unzulässigen** oder
    /// **unbeschränkten** Problemen); das Verfahren liefert dafür kein Zertifikat.
    NumericalFailure,
}

/// Faktorisierung der QP-KKT-Systeme; beide Modi verwenden dieselbe musterfeste Assemblierung.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QpKktBackend {
    /// Wählt Sparse bei höchstens 25 % struktureller KKT-Dichte, sonst Dense. Heuristik.
    Auto,
    /// Wiederverwendetes Sparse-LDLᵀ, mit dichtem pivotiertem LDLᵀ als Rückfall.
    Sparse,
    /// Dichtes pivotiertes LDLᵀ für alle Iterationen.
    Dense,
}
/// Diagnose der tatsächlich verwendeten KKT-Faktorisierungen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QpKktStats<T> {
    /// Vorausgewähltes Backend (Sparse oder Dense), vor eventuellen Rückfällen.
    pub selected_backend: Option<QpKktBackend>,
    /// Anzahl symbolischer Sparse-Analysen (höchstens eine pro QP-Aufruf).
    pub symbolic_analyses: usize,
    /// Versuche numerischer Sparse-Faktorisierung inklusive Refaktorisierungen.
    pub sparse_factorizations: usize,
    /// Davon Refaktorisierungen eines vorhandenen Faktors.
    pub sparse_refactorizations: usize,
    /// Versuche dichter Faktorisierung.
    pub dense_factorizations: usize,
    /// Davon Rückfälle aus dem Sparse-Modus.
    pub dense_fallbacks: usize,
    /// Größter normweiser Rückwärtsfehler akzeptierter linearer KKT-Lösungen.
    pub max_linear_backward_error: T,
}
impl<T: Real> Default for QpKktStats<T> {
    fn default() -> Self {
        Self {
            selected_backend: None,
            symbolic_analyses: 0,
            sparse_factorizations: 0,
            sparse_refactorizations: 0,
            dense_factorizations: 0,
            dense_fallbacks: 0,
            max_linear_backward_error: T::ZERO,
        }
    }
}

/// Einstellungen für dichte und CSR-Varianten von QP und LP.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QpOptions<T> {
    /// Toleranz für Residuen und Komplementarität (relativ zur Größe der Daten).
    pub tol: T,
    /// Höchstzahl der Innere-Punkte-Iterationen.
    pub max_iter: usize,
    /// Regularisierung `δ` der KKT-Matrix (macht singuläre Fälle wie reine LPs lösbar).
    pub regularization: T,
    /// Standard: Auto; Sparse verwendet Wiederverwendung und dichten Rückfall.
    pub kkt_backend: QpKktBackend,
}

impl<T: Real> Default for QpOptions<T> {
    /// `tol = √ε`, 100 Iterationen, `δ = √ε·10⁻³`.
    fn default() -> Self {
        let r = T::EPSILON.sqrt();
        Self {
            tol: r,
            max_iter: 100,
            regularization: r * T::from_f64(1e-3),
            kkt_backend: QpKktBackend::Auto,
        }
    }
}

/// Ergebnis des Innere-Punkte-Verfahrens.
#[derive(Debug, Clone, PartialEq)]
pub struct QpSolution<T> {
    /// Ausgang.
    pub status: QpStatus,
    /// Lösung (`n` Einträge).
    pub x: Vec<T>,
    /// Multiplikatoren der Gleichungen (`p` Einträge).
    pub y: Vec<T>,
    /// Multiplikatoren der Ungleichungen (`q` Einträge, `≥ 0`).
    pub z: Vec<T>,
    /// Zielfunktionswert `½·xᵀQx + cᵀx`.
    pub objective: T,
    /// Anzahl der Iterationen.
    pub iterations: usize,
    /// KKT-Backend- und Genauigkeitsdiagnostik.
    pub kkt: QpKktStats<T>,
}

fn inf_norm<T: Real>(v: &[T]) -> T {
    v.iter().fold(T::ZERO, |m, x| m.max(x.abs()))
}

fn finite_all<T: Real>(v: &[T]) -> bool {
    v.iter().all(|x| !x.is_nan() && !x.is_infinite())
}

/// Größte Schrittweite `α` mit `v + α·dv ≥ 0` (unendlich, wenn keine Komponente fällt).
fn max_step<T: Real>(v: &[T], dv: &[T]) -> T {
    let mut a = T::INFINITY;
    for (x, d) in v.iter().zip(dv) {
        if *d < T::ZERO {
            a = a.min(-*x / *d);
        }
    }
    a
}

/// Löst das konvexe quadratische Programm
///
/// ```text
/// min  ½·xᵀQx + cᵀx    unter   A·x = b,   G·x ≤ h
/// ```
///
/// mit einem **primal-dualen Innere-Punkte-Verfahren** (Mehrotra-Prädiktor-Korrektor, unzulässiger Startpunkt). Pro
/// Iteration wird dasselbe strukturelle KKT-Muster neu assembliert. `QpOptions::kkt_backend`
/// wählt Auto (Standard), wiederverwendetes Sparse-LDLᵀ oder dichtes pivotiertes LDLᵀ.
/// Auto verwendet Sparse bei höchstens 25 % struktureller Dichte, sonst Dense (Heuristik).
/// Sparse analysiert das Muster einmal. Bei Faktorisierungs-, Trägheits- oder Residuenproblemen
/// wird für die betreffende Iteration auf dichte pivotierte LDLᵀ zurückgefallen.
/// Jede lineare Lösung wird auf normweisen Rückwärtsfehler `≤ 64·(n+p)·ε` geprüft.
/// Diagnose siehe `QpSolution::kkt`. Die Koeffizienten q/a_eq/g werden weiterhin dicht übergeben.
///
/// - `q` (`n × n`, zeilenweise, symmetrisch positiv semidefinit), `c` (`n`); `a_eq` (`p × n`), `b_eq` (`p`) und `g` (`q' × n`),
///   `h` (`q'`) dürfen leer sein (`p = 0` bzw. `q' = 0`).
/// - **Redundante Gleichungen** (linear abhängige Zeilen von `A`) werden vor dem Lösen erkannt und gestrichen; ihre Multiplikatoren
///   sind 0. Widersprüchliche Gleichungen enden sofort als [`QpStatus::NumericalFailure`].
/// - **Kein Zertifikat** für Unzulässigkeit oder Unbeschränktheit: solche Probleme enden als
///   [`QpStatus::NumericalFailure`] oder [`QpStatus::MaxIterations`]. Dafür liefert der Simplex ([`simplex`](super::simplex))
///   einen eindeutigen Status. Nicht konvexe `Q` werden nicht erkannt (das Ergebnis ist dann nicht verlässlich).
///
/// Fehler: [`OptError::InvalidInput`] bei ungültigen Dimensionen, unsymmetrischem Q,
/// nichtpositiver/nichtendlicher Toleranz oder negativer/nichtendlicher Regularisierung;
/// [`OptError::NotFinite`] bei nichtendlichen Daten oder initialer Assemblierung.
///
/// ```
/// use azmath::opt::{interior_point_qp, QpOptions, QpStatus};
///
/// // Projektion von (2, 1) auf den Halbraum x + y ≤ 2:  min ½‖x − p‖²  =  ½xᵀx − pᵀx + const
/// let sol = interior_point_qp(
///     &[1.0f64, 0.0, 0.0, 1.0], &[-2.0, -1.0],
///     &[], &[],
///     &[1.0, 1.0], &[2.0],
///     QpOptions::default(),
/// ).unwrap();
/// assert_eq!(sol.status, QpStatus::Optimal);
/// assert!((sol.x[0] - 1.5).abs() < 1e-6 && (sol.x[1] - 0.5).abs() < 1e-6);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn interior_point_qp<T: Real>(
    q: &[T],
    c: &[T],
    a_eq: &[T],
    b_eq: &[T],
    g: &[T],
    h: &[T],
    opts: QpOptions<T>,
) -> Result<QpSolution<T>, OptError> {
    let n = c.len();
    let p = b_eq.len();
    let m = h.len();
    if n.checked_mul(n) != Some(q.len())
        || p.checked_mul(n) != Some(a_eq.len())
        || m.checked_mul(n) != Some(g.len())
    {
        return Err(OptError::InvalidInput);
    }
    solve_qp(
        QpMatrix::Dense {
            data: q,
            rows: n,
            cols: n,
        },
        c,
        QpMatrix::Dense {
            data: a_eq,
            rows: p,
            cols: n,
        },
        b_eq,
        QpMatrix::Dense {
            data: g,
            rows: m,
            cols: n,
        },
        h,
        opts,
    )
}

/// Konvexes QP mit geliehenen CSR-Koeffizienten Q (n×n), A (p×n), G (m×n).
/// Q muss vollständig symmetrisch gespeichert sein (fehlende Einträge gelten als 0).
/// Leere A/G haben 0 Zeilen und n Spalten. Keine pauschale dichte Umwandlung:
/// Produkte, Residuen, Zielfunktion, Presolve und KKT-Aufbau arbeiten auf Sparse-Zeilen.
/// Fill-in im Presolve/Faktoren kann Speicherbedarf erhöhen; Dense/Auto und numerische
/// Rückfälle dürfen weiterhin eine dichte KKT-Matrix erzeugen. Übrige Verträge wie
/// bei [`interior_point_qp`], einschließlich fehlender Unzulässigkeitszertifikate.
///
/// ```
/// use azmath::{sparse::CsrMatrix, opt::{interior_point_qp_sparse, QpOptions, QpStatus}};
/// let q = CsrMatrix::identity(2);
/// let a = CsrMatrix::from_parts(0, 2, vec![0], vec![], vec![]).unwrap();
/// let g = CsrMatrix::identity(2);
/// let result = interior_point_qp_sparse(&q, &[-1.0f64, -1.0], &a, &[], &g,
///     &[0.4, 0.4], QpOptions::default()).unwrap();
/// assert_eq!(result.status, QpStatus::Optimal);
/// assert!((result.x[0] - 0.4).abs() < 1e-7);
/// ```
pub fn interior_point_qp_sparse<T: Real>(
    q: &CsrMatrix<T>,
    c: &[T],
    a_eq: &CsrMatrix<T>,
    b_eq: &[T],
    g: &CsrMatrix<T>,
    h: &[T],
    opts: QpOptions<T>,
) -> Result<QpSolution<T>, OptError> {
    solve_qp(
        QpMatrix::Csr(q),
        c,
        QpMatrix::Csr(a_eq),
        b_eq,
        QpMatrix::Csr(g),
        h,
        opts,
    )
}

fn solve_qp<T: Real>(
    q: QpMatrix<'_, T>,
    c: &[T],
    a_eq: QpMatrix<'_, T>,
    b_eq: &[T],
    g: QpMatrix<'_, T>,
    h: &[T],
    opts: QpOptions<T>,
) -> Result<QpSolution<T>, OptError> {
    if !opts.tol.is_finite()
        || opts.tol <= T::ZERO
        || !opts.regularization.is_finite()
        || opts.regularization < T::ZERO
    {
        return Err(OptError::InvalidInput);
    }
    let mut stats = QpKktStats::default();
    let n = c.len();
    let p = b_eq.len();
    let m = h.len();
    if n == 0
        || q.rows() != n
        || q.cols() != n
        || a_eq.rows() != p
        || a_eq.cols() != n
        || g.rows() != m
        || g.cols() != n
    {
        return Err(OptError::InvalidInput);
    }
    if !(finite_all(q.values())
        && finite_all(c)
        && finite_all(a_eq.values())
        && finite_all(b_eq)
        && finite_all(g.values())
        && finite_all(h))
    {
        return Err(OptError::NotFinite);
    }
    let qscale = inf_norm(q.values());
    let qscale = if qscale == T::ZERO { T::ONE } else { qscale };
    let symtol = T::from_f64(8.0) * T::from_usize(n) * T::EPSILON;
    for i in 0..n {
        let mut symmetric = true;
        q.row(i, |j, v| {
            if (v / qscale - q.get(j, i) / qscale).abs() > symtol {
                symmetric = false;
            }
        });
        if !symmetric {
            return Err(OptError::InvalidInput);
        }
    }
    // Presolve: linear abhängige Gleichungen entfernen (sonst wird die KKT-Matrix singulär)
    let p_orig = p;
    let Ok(keep) = independent_rows(a_eq, b_eq) else {
        return Ok(QpSolution {
            status: QpStatus::NumericalFailure,
            x: vec![T::ZERO; n],
            y: vec![T::ZERO; p_orig],
            z: vec![T::ZERO; m],
            objective: T::ZERO,
            iterations: 0,
            kkt: stats,
        });
    };
    let a_red = a_eq.selected_rows(&keep);
    let b_red: Vec<T> = keep.iter().map(|&i| b_eq[i]).collect();
    let (a_eq, b_eq) = (QpMatrix::Csr(&a_red), &b_red[..]);
    let p = keep.len();
    let tol = opts.tol;
    let delta = opts.regularization;
    let dim = n + p;
    let (c_norm, b_norm, h_norm) = (inf_norm(c), inf_norm(b_eq), inf_norm(h));

    let mut x = vec![T::ZERO; n];
    let mut y = vec![T::ZERO; p];
    let mut s = vec![T::ONE; m];
    let mut z = vec![T::ONE; m];
    let objective = |x: &[T]| {
        let mut qx = vec![T::ZERO; n];
        q.mul(x, &mut qx);
        let mut quad = T::ZERO;
        let mut lin = T::ZERO;
        for i in 0..n {
            quad += x[i] * qx[i];
            lin += c[i] * x[i];
        }
        T::from_f64(0.5) * quad + lin
    };
    let result = |status, x: Vec<T>, y: Vec<T>, z: Vec<T>, iterations, kkt| {
        // Multiplikatoren auf die ursprünglichen Gleichungen zurückschreiben (gestrichene Zeilen: 0)
        let mut y_full = vec![T::ZERO; p_orig];
        for (k, &row) in keep.iter().enumerate() {
            y_full[row] = y[k];
        }
        QpSolution {
            status,
            objective: objective(&x),
            x,
            y: y_full,
            z,
            iterations,
            kkt,
        }
    };

    let mut kkt = KktSystem::new(q, a_eq, g, n, p, m, delta, opts.kkt_backend, &mut stats)?;
    // Arbeitsvektoren
    let (mut rd, mut re, mut ri) = (vec![T::ZERO; n], vec![T::ZERO; p], vec![T::ZERO; m]);
    let (mut tmp_n, mut tmp_m) = (vec![T::ZERO; n], vec![T::ZERO; m]);
    let mut tmp_p = vec![T::ZERO; p];
    for iter in 0..opts.max_iter {
        // Residuen
        q.mul(&x, &mut rd);
        a_eq.mul_t(&y, &mut tmp_n);
        for i in 0..n {
            rd[i] += c[i] + tmp_n[i];
        }
        g.mul_t(&z, &mut tmp_n);
        for i in 0..n {
            rd[i] += tmp_n[i];
        }
        a_eq.mul(&x, &mut tmp_p);
        for i in 0..p {
            re[i] = tmp_p[i] - b_eq[i];
        }
        g.mul(&x, &mut tmp_m);
        for i in 0..m {
            ri[i] = tmp_m[i] + s[i] - h[i];
        }
        let mu = if m > 0 {
            s.iter().zip(&z).fold(T::ZERO, |a, (si, zi)| a + *si * *zi) / T::from_f64(m as f64)
        } else {
            T::ZERO
        };
        if !(finite_all(&rd) && finite_all(&re) && finite_all(&ri) && finite_all(&[mu])) {
            return Ok(result(QpStatus::NumericalFailure, x, y, z, iter, stats));
        }
        if inf_norm(&rd) <= tol * (T::ONE + c_norm)
            && inf_norm(&re) <= tol * (T::ONE + b_norm)
            && inf_norm(&ri) <= tol * (T::ONE + h_norm)
            && mu <= tol
        {
            return Ok(result(QpStatus::Optimal, x, y, z, iter, stats));
        }
        if inf_norm(&x) > T::from_f64(1e12) {
            return Ok(result(QpStatus::NumericalFailure, x, y, z, iter, stats));
        }

        if !kkt.factor(&s, &z, &mut stats) {
            return Ok(result(QpStatus::NumericalFailure, x, y, z, iter, stats));
        }

        // Löst das Newton-System für gegebene Komplementaritätsresiduen rc; Rückgabe (dx, dy, ds, dz)
        let mut direction = |rc: &[T]| -> Option<(Vec<T>, Vec<T>, Vec<T>, Vec<T>)> {
            // rechte Seite: −rd + Gᵀ(S⁻¹·rc) − Gᵀ(W·ri) und −re
            let mut rhs = vec![T::ZERO; dim];
            let mut t = vec![T::ZERO; m];
            for r in 0..m {
                t[r] = rc[r] / s[r] - (z[r] / s[r]) * ri[r];
            }
            let mut gt = vec![T::ZERO; n];
            g.mul_t(&t, &mut gt);
            for i in 0..n {
                rhs[i] = -rd[i] + gt[i];
            }
            for i in 0..p {
                rhs[n + i] = -re[i];
            }
            let sol = kkt.solve(&rhs, &mut stats)?;
            let dx: Vec<T> = sol[..n].to_vec();
            let dy: Vec<T> = sol[n..].to_vec();
            let mut gdx = vec![T::ZERO; m];
            g.mul(&dx, &mut gdx);
            let ds: Vec<T> = (0..m).map(|r| -ri[r] - gdx[r]).collect();
            let dz: Vec<T> = (0..m).map(|r| -(rc[r] + z[r] * ds[r]) / s[r]).collect();
            if finite_all(&dx) && finite_all(&dy) && finite_all(&ds) && finite_all(&dz) {
                Some((dx, dy, ds, dz))
            } else {
                None
            }
        };

        // Prädiktor (σ = 0)
        let rc_aff: Vec<T> = (0..m).map(|r| s[r] * z[r]).collect();
        let Some((_, _, ds_a, dz_a)) = direction(&rc_aff) else {
            return Ok(result(QpStatus::NumericalFailure, x, y, z, iter, stats));
        };
        let sigma = if m > 0 {
            let alpha = max_step(&s, &ds_a).min(max_step(&z, &dz_a)).min(T::ONE);
            let mu_aff = (0..m).fold(T::ZERO, |a, r| {
                a + (s[r] + alpha * ds_a[r]) * (z[r] + alpha * dz_a[r])
            }) / T::from_f64(m as f64);
            let ratio = (mu_aff / mu).max(T::ZERO).min(T::ONE);
            ratio * ratio * ratio
        } else {
            T::ZERO
        };
        // Korrektor
        let rc: Vec<T> = (0..m)
            .map(|r| s[r] * z[r] + ds_a[r] * dz_a[r] - sigma * mu)
            .collect();
        let Some((dx, dy, ds, dz)) = direction(&rc) else {
            return Ok(result(QpStatus::NumericalFailure, x, y, z, iter, stats));
        };
        let alpha =
            (T::from_f64(0.99) * max_step(&s, &ds).min(max_step(&z, &dz)) / T::ONE).min(T::ONE);
        let alpha = if m > 0 { alpha } else { T::ONE };
        for i in 0..n {
            x[i] += alpha * dx[i];
        }
        for i in 0..p {
            y[i] += alpha * dy[i];
        }
        for r in 0..m {
            s[r] += alpha * ds[r];
            z[r] += alpha * dz[r];
        }
    }
    Ok(result(
        QpStatus::MaxIterations,
        x,
        y,
        z,
        opts.max_iter,
        stats,
    ))
}

/// Löst das lineare Programm `min cᵀx` unter `A·x = b`, `x ≥ 0` mit dem Innere-Punkte-Verfahren
/// ([`interior_point_qp`] mit `Q = 0`, `G = −I`, `h = 0`).
///
/// Anders als der Simplex ([`simplex`](super::simplex)) liefert es keine Ecke, sondern einen Punkt im Inneren der
/// optimalen Seite und kein Zertifikat für Unzulässigkeit/Unbeschränktheit; dafür wächst die Iterationszahl kaum mit der
/// Problemgröße. `a` ist `p × n` zeilenweise.
///
/// Fehler: [`OptError::InvalidInput`], [`OptError::NotFinite`].
pub fn interior_point_lp<T: Real>(
    c: &[T],
    a: &[T],
    b: &[T],
    opts: QpOptions<T>,
) -> Result<QpSolution<T>, OptError> {
    let n = c.len();
    if a.len() != b.len().checked_mul(n).ok_or(OptError::InvalidInput)? {
        return Err(OptError::InvalidInput);
    }
    let (q, g) = lp_matrices(n);
    solve_qp(
        QpMatrix::Csr(&q),
        c,
        QpMatrix::Dense {
            data: a,
            rows: b.len(),
            cols: n,
        },
        b,
        QpMatrix::Csr(&g),
        &vec![T::ZERO; n],
        opts,
    )
}

/// Sparse-LP `min cᵀx`, A x=b, x≥0; A ist CSR. Q=0 und G=−I werden in O(n)
/// Speicher aufgebaut. Solver-Verträge wie bei [`interior_point_lp`].
pub fn interior_point_lp_sparse<T: Real>(
    c: &[T],
    a: &CsrMatrix<T>,
    b: &[T],
    opts: QpOptions<T>,
) -> Result<QpSolution<T>, OptError> {
    let n = c.len();
    let (q, g) = lp_matrices(n);
    solve_qp(
        QpMatrix::Csr(&q),
        c,
        QpMatrix::Csr(a),
        b,
        QpMatrix::Csr(&g),
        &vec![T::ZERO; n],
        opts,
    )
}
fn lp_matrices<T: Real>(n: usize) -> (CsrMatrix<T>, CsrMatrix<T>) {
    (
        CsrMatrix::from_parts(n, n, vec![0; n + 1], vec![], vec![]).expect("empty CSR"),
        CsrMatrix::from_parts(n, n, (0..=n).collect(), (0..n).collect(), vec![-T::ONE; n])
            .expect("diagonal CSR"),
    )
}
