#![allow(clippy::needless_range_loop)]

use super::OptError;
use crate::Real;
use crate::linalg::{DynMatrix, DynVector};
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

/// Einstellungen für [`interior_point_qp`] und [`interior_point_lp`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QpOptions<T> {
    /// Toleranz für Residuen und Komplementarität (relativ zur Größe der Daten).
    pub tol: T,
    /// Höchstzahl der Innere-Punkte-Iterationen.
    pub max_iter: usize,
    /// Regularisierung `δ` der KKT-Matrix (macht singuläre Fälle wie reine LPs lösbar).
    pub regularization: T,
}

impl<T: Real> Default for QpOptions<T> {
    /// `tol = √ε`, 100 Iterationen, `δ = √ε·10⁻³`.
    fn default() -> Self {
        let r = T::EPSILON.sqrt();
        Self {
            tol: r,
            max_iter: 100,
            regularization: r * T::from_f64(1e-3),
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
}

fn inf_norm<T: Real>(v: &[T]) -> T {
    v.iter().fold(T::ZERO, |m, x| m.max(x.abs()))
}

fn finite_all<T: Real>(v: &[T]) -> bool {
    v.iter().all(|x| !x.is_nan() && !x.is_infinite())
}

/// `y = M·x` für eine zeilenweise `rows × cols`-Matrix.
fn mat_vec<T: Real>(m: &[T], rows: usize, cols: usize, x: &[T], y: &mut [T]) {
    for i in 0..rows {
        let mut s = T::ZERO;
        for j in 0..cols {
            s += m[i * cols + j] * x[j];
        }
        y[i] = s;
    }
}

/// `y = Mᵀ·x` für eine zeilenweise `rows × cols`-Matrix (`x` hat `rows`, `y` hat `cols` Einträge).
fn mat_t_vec<T: Real>(m: &[T], rows: usize, cols: usize, x: &[T], y: &mut [T]) {
    y.fill(T::ZERO);
    for i in 0..rows {
        for j in 0..cols {
            y[j] += m[i * cols + j] * x[i];
        }
    }
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
/// Iteration wird ein dichtes KKT-System gelöst (LU); der Aufwand ist `O((n+p)³)` pro Schritt, bei 10 bis 30 Schritten.
///
/// - `q` (`n × n`, zeilenweise, symmetrisch positiv semidefinit), `c` (`n`); `a_eq` (`p × n`), `b_eq` (`p`) und `g` (`q' × n`),
///   `h` (`q'`) dürfen leer sein (`p = 0` bzw. `q' = 0`).
/// - **Kein Zertifikat** für Unzulässigkeit oder Unbeschränktheit: solche Probleme enden als
///   [`QpStatus::NumericalFailure`] oder [`QpStatus::MaxIterations`]. Dafür liefert der Simplex ([`simplex`](super::simplex))
///   einen eindeutigen Status. Nicht konvexe `Q` werden nicht erkannt (das Ergebnis ist dann nicht verlässlich).
///
/// Fehler: [`OptError::InvalidInput`], [`OptError::NotFinite`].
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
    if n == 0 || q.len() != n * n || a_eq.len() != p * n || g.len() != m * n {
        return Err(OptError::InvalidInput);
    }
    if !(finite_all(q)
        && finite_all(c)
        && finite_all(a_eq)
        && finite_all(b_eq)
        && finite_all(g)
        && finite_all(h))
    {
        return Err(OptError::NotFinite);
    }
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
        mat_vec(q, n, n, x, &mut qx);
        let mut quad = T::ZERO;
        let mut lin = T::ZERO;
        for i in 0..n {
            quad += x[i] * qx[i];
            lin += c[i] * x[i];
        }
        T::from_f64(0.5) * quad + lin
    };
    let result = |status, x: Vec<T>, y: Vec<T>, z: Vec<T>, iterations| QpSolution {
        status,
        objective: objective(&x),
        x,
        y,
        z,
        iterations,
    };

    // Arbeitsvektoren
    let (mut rd, mut re, mut ri) = (vec![T::ZERO; n], vec![T::ZERO; p], vec![T::ZERO; m]);
    let (mut tmp_n, mut tmp_m) = (vec![T::ZERO; n], vec![T::ZERO; m]);
    let mut tmp_p = vec![T::ZERO; p];
    for iter in 0..opts.max_iter {
        // Residuen
        mat_vec(q, n, n, &x, &mut rd);
        mat_t_vec(a_eq, p, n, &y, &mut tmp_n);
        for i in 0..n {
            rd[i] += c[i] + tmp_n[i];
        }
        mat_t_vec(g, m, n, &z, &mut tmp_n);
        for i in 0..n {
            rd[i] += tmp_n[i];
        }
        mat_vec(a_eq, p, n, &x, &mut tmp_p);
        for i in 0..p {
            re[i] = tmp_p[i] - b_eq[i];
        }
        mat_vec(g, m, n, &x, &mut tmp_m);
        for i in 0..m {
            ri[i] = tmp_m[i] + s[i] - h[i];
        }
        let mu = if m > 0 {
            s.iter().zip(&z).fold(T::ZERO, |a, (si, zi)| a + *si * *zi) / T::from_f64(m as f64)
        } else {
            T::ZERO
        };
        if !(finite_all(&rd) && finite_all(&re) && finite_all(&ri) && finite_all(&[mu])) {
            return Ok(result(QpStatus::NumericalFailure, x, y, z, iter));
        }
        if inf_norm(&rd) <= tol * (T::ONE + c_norm)
            && inf_norm(&re) <= tol * (T::ONE + b_norm)
            && inf_norm(&ri) <= tol * (T::ONE + h_norm)
            && mu <= tol
        {
            return Ok(result(QpStatus::Optimal, x, y, z, iter));
        }
        if inf_norm(&x) > T::from_f64(1e12) {
            return Ok(result(QpStatus::NumericalFailure, x, y, z, iter));
        }

        // KKT-Matrix [[Q + GᵀWG + δI, Aᵀ], [A, −δI]] mit W = diag(z/s)
        let mut k = vec![T::ZERO; dim * dim];
        for i in 0..n {
            for j in 0..n {
                k[i * dim + j] = q[i * n + j];
            }
            k[i * dim + i] += delta;
        }
        for r in 0..m {
            let w = z[r] / s[r];
            for i in 0..n {
                let gi = g[r * n + i] * w;
                if gi != T::ZERO {
                    for j in 0..n {
                        k[i * dim + j] += gi * g[r * n + j];
                    }
                }
            }
        }
        for r in 0..p {
            for j in 0..n {
                k[(n + r) * dim + j] = a_eq[r * n + j];
                k[j * dim + n + r] = a_eq[r * n + j];
            }
            k[(n + r) * dim + n + r] = -delta;
        }
        let Ok(kkt) = DynMatrix::new(dim, dim, k) else {
            return Ok(result(QpStatus::NumericalFailure, x, y, z, iter));
        };
        let Ok(lu) = kkt.lu() else {
            return Ok(result(QpStatus::NumericalFailure, x, y, z, iter));
        };

        // Löst das Newton-System für gegebene Komplementaritätsresiduen rc; Rückgabe (dx, dy, ds, dz)
        let direction = |rc: &[T]| -> Option<(Vec<T>, Vec<T>, Vec<T>, Vec<T>)> {
            // rechte Seite: −rd + Gᵀ(S⁻¹·rc) − Gᵀ(W·ri) und −re
            let mut rhs = vec![T::ZERO; dim];
            let mut t = vec![T::ZERO; m];
            for r in 0..m {
                t[r] = rc[r] / s[r] - (z[r] / s[r]) * ri[r];
            }
            let mut gt = vec![T::ZERO; n];
            mat_t_vec(g, m, n, &t, &mut gt);
            for i in 0..n {
                rhs[i] = -rd[i] + gt[i];
            }
            for i in 0..p {
                rhs[n + i] = -re[i];
            }
            let sol = lu.solve(&DynVector::new(rhs)).ok()?;
            let dx: Vec<T> = sol.as_slice()[..n].to_vec();
            let dy: Vec<T> = sol.as_slice()[n..].to_vec();
            let mut gdx = vec![T::ZERO; m];
            mat_vec(g, m, n, &dx, &mut gdx);
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
            return Ok(result(QpStatus::NumericalFailure, x, y, z, iter));
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
            return Ok(result(QpStatus::NumericalFailure, x, y, z, iter));
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
    Ok(result(QpStatus::MaxIterations, x, y, z, opts.max_iter))
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
    let q = vec![T::ZERO; n * n];
    let mut g = vec![T::ZERO; n * n];
    for i in 0..n {
        g[i * n + i] = -T::ONE;
    }
    let h = vec![T::ZERO; n];
    interior_point_qp(&q, c, a, b, &g, &h, opts)
}
