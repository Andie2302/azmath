#![allow(clippy::needless_range_loop)]

use super::OptError;
use crate::Real;
use alloc::vec;
use alloc::vec::Vec;

/// Art einer Nebenbedingung `aᵢ·x ? bᵢ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Relation {
    /// `aᵢ·x ≤ bᵢ`
    Le,
    /// `aᵢ·x = bᵢ`
    Eq,
    /// `aᵢ·x ≥ bᵢ`
    Ge,
}

/// Ausgang der Optimierung.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpStatus {
    /// Eine optimale Ecke wurde gefunden.
    Optimal,
    /// Es gibt keinen zulässigen Punkt.
    Infeasible,
    /// Das Ziel ist auf dem zulässigen Bereich nach unten unbeschränkt.
    Unbounded,
    /// Das Iterationslimit wurde erreicht (`x` ist dann nicht verlässlich).
    IterationLimit,
}

/// Einstellungen für [`simplex`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimplexOptions<T> {
    /// Toleranz für „ist null“ bei Reduzierten Kosten, Pivotelementen und der Zulässigkeit.
    pub tol: T,
    /// Höchstzahl der Pivotschritte (beider Phasen zusammen).
    pub max_iter: usize,
}

impl<T: Real> Default for SimplexOptions<T> {
    /// `tol = √ε`, höchstens 10 000 Pivotschritte.
    fn default() -> Self {
        Self {
            tol: T::EPSILON.sqrt(),
            max_iter: 10_000,
        }
    }
}

/// Ergebnis von [`simplex`].
#[derive(Debug, Clone, PartialEq)]
pub struct LpSolution<T> {
    /// Ausgang; `x` und `objective` sind nur bei [`LpStatus::Optimal`] aussagekräftig.
    pub status: LpStatus,
    /// Die Lösung (`n` Einträge).
    pub x: Vec<T>,
    /// Zielfunktionswert `cᵀx`.
    pub objective: T,
    /// Anzahl der Pivotschritte.
    pub iterations: usize,
}

/// Ein Tableau mit Basis und Reduzierte-Kosten-Zeile.
struct Tableau<T> {
    rows: usize,
    cols: usize, // ohne rechte Seite
    t: Vec<T>,   // rows × (cols + 1), letzte Spalte = rechte Seite
    basis: Vec<usize>,
    d: Vec<T>, // reduzierte Kosten (cols), danach d[cols] = −Zielwert
}

impl<T: Real> Tableau<T> {
    fn at(&self, i: usize, j: usize) -> T {
        self.t[i * (self.cols + 1) + j]
    }

    fn set_costs(&mut self, cost: &[T]) {
        let w = self.cols + 1;
        self.d = vec![T::ZERO; w];
        self.d[..self.cols].copy_from_slice(cost);
        for i in 0..self.rows {
            let cb = cost[self.basis[i]];
            if cb != T::ZERO {
                for j in 0..w {
                    self.d[j] -= cb * self.t[i * w + j];
                }
            }
        }
    }

    fn pivot(&mut self, row: usize, col: usize) {
        let w = self.cols + 1;
        let p = self.t[row * w + col];
        for j in 0..w {
            self.t[row * w + j] /= p;
        }
        for i in 0..self.rows {
            if i != row {
                let f = self.t[i * w + col];
                if f != T::ZERO {
                    for j in 0..w {
                        let v = self.t[row * w + j];
                        self.t[i * w + j] -= f * v;
                    }
                    self.t[i * w + col] = T::ZERO;
                }
            }
        }
        let f = self.d[col];
        if f != T::ZERO {
            for j in 0..w {
                let v = self.t[row * w + j];
                self.d[j] -= f * v;
            }
            self.d[col] = T::ZERO;
        }
        self.basis[row] = col;
    }
}

enum Outcome {
    Optimal,
    Unbounded,
    IterationLimit,
}

/// Simplex-Iteration auf dem aktuellen Tableau; erlaubt sind nur Spalten `j < allowed`.
fn iterate<T: Real>(
    tab: &mut Tableau<T>,
    allowed: usize,
    tol: T,
    max_iter: usize,
    iters: &mut usize,
) -> Outcome {
    let w = tab.cols + 1;
    let mut stalled = 0usize;
    loop {
        // Eintretende Variable: Dantzig (kleinste reduzierte Kosten), bei Stillstand Bland (kleinster Index)
        let bland = stalled > 20;
        let mut enter = None;
        let mut best = -tol;
        for j in 0..allowed {
            if tab.d[j] < best {
                enter = Some(j);
                if bland {
                    break;
                }
                best = tab.d[j];
            }
        }
        let Some(col) = enter else {
            return Outcome::Optimal;
        };
        if *iters >= max_iter {
            return Outcome::IterationLimit;
        }
        // Quotiententest, bei Gleichstand kleinster Basisindex (Bland)
        let mut leave: Option<usize> = None;
        let mut ratio = T::ZERO;
        for i in 0..tab.rows {
            let a = tab.at(i, col);
            if a > tol {
                let r = tab.t[i * w + tab.cols] / a;
                let better = match leave {
                    None => true,
                    Some(l) => {
                        r < ratio - tol * T::ONE.max(ratio.abs())
                            || (r <= ratio + tol * T::ONE.max(ratio.abs())
                                && tab.basis[i] < tab.basis[l])
                    }
                };
                if better {
                    leave = Some(i);
                    ratio = r;
                }
            }
        }
        let Some(row) = leave else {
            return Outcome::Unbounded;
        };
        if ratio.abs() <= tol {
            stalled += 1;
        } else {
            stalled = 0;
        }
        tab.pivot(row, col);
        *iters += 1;
    }
}

/// Löst das lineare Programm `min cᵀx` unter `A·x ≤ / = / ≥ b` und `x ≥ 0` mit dem **Zwei-Phasen-Simplex-Verfahren** auf
/// einem dichten Tableau.
///
/// - `c`: Zielfunktion (`n` Einträge). `a`: Restriktionsmatrix **zeilenweise** (`m × n`), `b`: rechte Seiten (`m`),
///   `relations`: Art jeder Zeile (`m`). Maximieren: `c` negieren.
/// - Phase 1 sucht mit künstlichen Variablen einen zulässigen Punkt (sonst [`LpStatus::Infeasible`]), Phase 2
///   optimiert. Gegen Zyklen bei entarteten Ecken wechselt die Spaltenwahl nach längerem Stillstand auf die
///   Bland-Regel. Redundante Gleichungen sind erlaubt.
/// - Aufwand: `O(Pivotschritte · m · (n + m))` Rechenzeit, `O(m · (n + m))` Speicher – für kleine und mittlere
///   Probleme; sehr große, dünn besetzte Probleme brauchen ein anderes Verfahren.
///
/// Fehler: [`OptError::InvalidInput`] (Längen, `n = 0`), [`OptError::NotFinite`]. Unzulässigkeit und Unbeschränktheit
/// sind **keine** Fehler, sondern der [`LpStatus`] im Ergebnis.
///
/// ```
/// use azmath::opt::{simplex, LpStatus, Relation, SimplexOptions};
///
/// // maximiere 3x + 2y  unter  x + y ≤ 4,  x + 3y ≤ 6,  x ≤ 3   (also minimiere −3x − 2y)
/// let c = [-3.0f64, -2.0];
/// let a = [1.0, 1.0, 1.0, 3.0, 1.0, 0.0];
/// let b = [4.0, 6.0, 3.0];
/// let sol = simplex(&c, &a, &b, &[Relation::Le; 3], SimplexOptions::default()).unwrap();
/// assert_eq!(sol.status, LpStatus::Optimal);
/// assert!((sol.x[0] - 3.0).abs() < 1e-9 && (sol.x[1] - 1.0).abs() < 1e-9);
/// assert!((sol.objective + 11.0).abs() < 1e-9);
/// ```
pub fn simplex<T: Real>(
    c: &[T],
    a: &[T],
    b: &[T],
    relations: &[Relation],
    opts: SimplexOptions<T>,
) -> Result<LpSolution<T>, OptError> {
    let n = c.len();
    let m = b.len();
    if n == 0 || a.len() != m * n || relations.len() != m {
        return Err(OptError::InvalidInput);
    }
    let finite = |v: &T| !v.is_nan() && !v.is_infinite();
    if !(c.iter().all(finite) && a.iter().all(finite) && b.iter().all(finite)) {
        return Err(OptError::NotFinite);
    }
    let tol = opts.tol;

    // Zeilen mit negativer rechter Seite umkehren (≤ ↔ ≥)
    let mut rel = relations.to_vec();
    let mut sign = vec![T::ONE; m];
    for i in 0..m {
        if b[i] < T::ZERO {
            sign[i] = -T::ONE;
            rel[i] = match rel[i] {
                Relation::Le => Relation::Ge,
                Relation::Ge => Relation::Le,
                Relation::Eq => Relation::Eq,
            };
        }
    }
    let n_slack = rel.iter().filter(|r| **r != Relation::Eq).count();
    let n_art = rel.iter().filter(|r| **r != Relation::Le).count();
    let cols = n + n_slack + n_art;
    let w = cols + 1;
    let mut t = vec![T::ZERO; m * w];
    let mut basis = vec![0usize; m];
    let (mut s_idx, mut a_idx) = (n, n + n_slack);
    for i in 0..m {
        for j in 0..n {
            t[i * w + j] = sign[i] * a[i * n + j];
        }
        t[i * w + cols] = sign[i] * b[i];
        match rel[i] {
            Relation::Le => {
                t[i * w + s_idx] = T::ONE;
                basis[i] = s_idx;
                s_idx += 1;
            }
            Relation::Ge => {
                t[i * w + s_idx] = -T::ONE;
                s_idx += 1;
                t[i * w + a_idx] = T::ONE;
                basis[i] = a_idx;
                a_idx += 1;
            }
            Relation::Eq => {
                t[i * w + a_idx] = T::ONE;
                basis[i] = a_idx;
                a_idx += 1;
            }
        }
    }
    let mut tab = Tableau {
        rows: m,
        cols,
        t,
        basis,
        d: Vec::new(),
    };
    let mut iters = 0usize;
    let fail = |status, iters| LpSolution {
        status,
        x: vec![T::ZERO; n],
        objective: T::ZERO,
        iterations: iters,
    };

    // Phase 1: Summe der künstlichen Variablen minimieren
    if n_art > 0 {
        let mut cost1 = vec![T::ZERO; cols];
        for j in (n + n_slack)..cols {
            cost1[j] = T::ONE;
        }
        tab.set_costs(&cost1);
        match iterate(&mut tab, cols, tol, opts.max_iter, &mut iters) {
            Outcome::IterationLimit => return Ok(fail(LpStatus::IterationLimit, iters)),
            Outcome::Unbounded | Outcome::Optimal => {}
        }
        let scale = T::ONE + b.iter().fold(T::ZERO, |s, v| s + v.abs());
        if -tab.d[cols] > tol * scale {
            return Ok(fail(LpStatus::Infeasible, iters));
        }
        // künstliche Variablen aus der Basis entfernen, soweit möglich (sonst ist die Zeile redundant)
        for i in 0..m {
            if tab.basis[i] >= n + n_slack {
                if let Some(j) = (0..n + n_slack).find(|&j| tab.at(i, j).abs() > tol) {
                    tab.pivot(i, j);
                }
            }
        }
    }

    // Phase 2: ursprüngliches Ziel, künstliche Spalten gesperrt
    let mut cost2 = vec![T::ZERO; cols];
    cost2[..n].copy_from_slice(c);
    tab.set_costs(&cost2);
    match iterate(&mut tab, n + n_slack, tol, opts.max_iter, &mut iters) {
        Outcome::Unbounded => return Ok(fail(LpStatus::Unbounded, iters)),
        Outcome::IterationLimit => return Ok(fail(LpStatus::IterationLimit, iters)),
        Outcome::Optimal => {}
    }
    let mut x = vec![T::ZERO; n];
    for i in 0..m {
        if tab.basis[i] < n {
            x[tab.basis[i]] = tab.t[i * w + cols];
        }
    }
    let objective = c.iter().zip(&x).fold(T::ZERO, |s, (ci, xi)| s + *ci * *xi);
    Ok(LpSolution {
        status: LpStatus::Optimal,
        x,
        objective,
        iterations: iters,
    })
}
