#![allow(clippy::needless_range_loop)]

use super::simplex::{Tableau, simplex_tableau};
use super::{LpSolution, LpStatus, OptError, Relation, SimplexOptions};
use crate::Real;
use alloc::vec;
use alloc::vec::Vec;

/// Gelöstes lineares Programm `min cᵀx`, `A·x ≤ / = / ≥ b`, `x ≥ 0` mit **Warmstart**: Nach dem Lösen lassen sich weitere
/// Nebenbedingungen hinzufügen; das optimale Tableau bleibt dual zulässig und wird mit dem **Dual-Simplex-Verfahren** in wenigen
/// Schritten neu optimiert (Grundlage für Branch-and-Bound und Schnittebenen).
///
/// ```
/// use azmath::opt::{LpStatus, LpWarmStart, Relation, SimplexOptions};
///
/// // min −x − y  unter  x + y ≤ 4,  x ≤ 3
/// let mut lp = LpWarmStart::new(
///     &[-1.0f64, -1.0], &[1.0, 1.0, 1.0, 0.0], &[4.0, 3.0], &[Relation::Le; 2], SimplexOptions::default(),
/// ).unwrap();
/// assert!((lp.solution().objective + 4.0).abs() < 1e-9);
/// // zusätzlich y ≤ 0,5: Dual-Simplex statt Neustart
/// assert_eq!(lp.add_constraint(&[0.0, 1.0], Relation::Le, 0.5).unwrap(), LpStatus::Optimal);
/// assert!((lp.solution().objective + 4.0).abs() < 1e-9);
/// assert_eq!(lp.add_constraint(&[0.0, 1.0], Relation::Ge, 2.0).unwrap(), LpStatus::Infeasible);
/// ```
#[derive(Clone)]
pub struct LpWarmStart<T> {
    pub(super) tab: Tableau<T>,
    pub(super) allowed: Vec<bool>,
    pub(super) n: usize,
    c: Vec<T>,
    opts: SimplexOptions<T>,
    status: LpStatus,
    iterations: usize,
}

impl<T: Real> LpWarmStart<T> {
    /// Löst das LP (Zwei-Phasen-Simplex) und merkt sich das optimale Tableau. Fehler wie bei [`simplex`](super::simplex).
    pub fn new(
        c: &[T],
        a: &[T],
        b: &[T],
        relations: &[Relation],
        opts: SimplexOptions<T>,
    ) -> Result<Self, OptError> {
        let (sol, tab) = simplex_tableau(c, a, b, relations, opts)?;
        let n = c.len();
        match tab {
            Some((tab, allowed_cols)) => {
                let allowed = (0..tab.cols).map(|j| j < allowed_cols).collect();
                Ok(Self {
                    tab,
                    allowed,
                    n,
                    c: c.to_vec(),
                    opts,
                    status: LpStatus::Optimal,
                    iterations: sol.iterations,
                })
            }
            None => Ok(Self {
                tab: Tableau {
                    rows: 0,
                    cols: 0,
                    t: vec![T::ZERO; 1],
                    basis: Vec::new(),
                    d: vec![T::ZERO; 1],
                },
                allowed: Vec::new(),
                n,
                c: c.to_vec(),
                opts,
                status: sol.status,
                iterations: sol.iterations,
            }),
        }
    }

    /// Status der letzten Optimierung.
    pub fn status(&self) -> LpStatus {
        self.status
    }

    /// Summe der bisherigen Pivotschritte.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// Aktuelle Lösung (nur bei [`LpStatus::Optimal`] aussagekräftig).
    pub fn solution(&self) -> LpSolution<T> {
        let mut x = vec![T::ZERO; self.n];
        let mut obj = T::ZERO;
        if self.status == LpStatus::Optimal {
            let w = self.tab.cols + 1;
            for i in 0..self.tab.rows {
                let k = self.tab.basis[i];
                if k < self.n {
                    x[k] = self.tab.t[i * w + self.tab.cols];
                }
            }
            obj = self
                .c
                .iter()
                .zip(&x)
                .fold(T::ZERO, |s, (ci, xi)| s + *ci * *xi);
        }
        LpSolution {
            status: self.status,
            x,
            objective: obj,
            iterations: self.iterations,
        }
    }

    /// Fügt `coeffs·x ? rhs` hinzu und optimiert mit dem Dual-Simplex-Verfahren neu. Eine Gleichung wird als `≤` und `≥` eingefügt.
    /// Fehler: [`OptError::InvalidInput`] (Länge), [`OptError::NotFinite`]. Ist das LP schon unzulässig, bleibt der Status.
    pub fn add_constraint(
        &mut self,
        coeffs: &[T],
        relation: Relation,
        rhs: T,
    ) -> Result<LpStatus, OptError> {
        if coeffs.len() != self.n {
            return Err(OptError::InvalidInput);
        }
        if coeffs.iter().any(|v| !v.is_finite()) || !rhs.is_finite() {
            return Err(OptError::NotFinite);
        }
        match relation {
            Relation::Le => self.add_le(coeffs, rhs),
            Relation::Ge => {
                let neg: Vec<T> = coeffs.iter().map(|&v| -v).collect();
                self.add_le(&neg, -rhs)
            }
            Relation::Eq => {
                self.add_le(coeffs, rhs)?;
                let neg: Vec<T> = coeffs.iter().map(|&v| -v).collect();
                self.add_le(&neg, -rhs)
            }
        }
    }

    fn add_le(&mut self, coeffs: &[T], rhs: T) -> Result<LpStatus, OptError> {
        if self.status != LpStatus::Optimal {
            return Ok(self.status);
        }
        let cols = self.tab.cols;
        let w = cols + 1;
        let mut row = vec![T::ZERO; cols];
        row[..self.n].copy_from_slice(coeffs);
        let mut r = rhs;
        for i in 0..self.tab.rows {
            let k = self.tab.basis[i];
            if k < self.n && coeffs[k] != T::ZERO {
                let f = coeffs[k];
                for j in 0..cols {
                    row[j] -= f * self.tab.t[i * w + j];
                }
                r -= f * self.tab.t[i * w + cols];
            }
        }
        self.append_row(row, r);
        self.reoptimize();
        Ok(self.status)
    }

    /// Hängt die Zeile `row·(Spalten) + s = rhs` mit neuer Schlupfvariable `s` (Basisvariable) an.
    pub(super) fn append_row(&mut self, row: Vec<T>, rhs: T) {
        let tab = &mut self.tab;
        let (cols, rows) = (tab.cols, tab.rows);
        let w_old = cols + 1;
        let w_new = cols + 2;
        let mut t = vec![T::ZERO; (rows + 1) * w_new];
        for i in 0..rows {
            t[i * w_new..i * w_new + cols].copy_from_slice(&tab.t[i * w_old..i * w_old + cols]);
            t[i * w_new + cols + 1] = tab.t[i * w_old + cols];
        }
        t[rows * w_new..rows * w_new + cols].copy_from_slice(&row);
        t[rows * w_new + cols] = T::ONE;
        t[rows * w_new + cols + 1] = rhs;
        tab.t = t;
        tab.basis.push(cols);
        let obj = tab.d[cols];
        tab.d[cols] = T::ZERO;
        tab.d.push(obj);
        tab.rows += 1;
        tab.cols += 1;
        self.allowed.push(true);
    }

    /// Dual-Simplex-Iterationen, bis alle rechten Seiten nichtnegativ sind (optimal), keine Pivotspalte existiert (unzulässig)
    /// oder das Iterationslimit erreicht ist.
    pub(super) fn reoptimize(&mut self) {
        let tol = self.opts.tol;
        let tab = &mut self.tab;
        let mut local = 0usize;
        loop {
            let w = tab.cols + 1;
            let mut leave: Option<usize> = None;
            let mut worst = -tol;
            for i in 0..tab.rows {
                let v = tab.t[i * w + tab.cols];
                if v < worst {
                    worst = v;
                    leave = Some(i);
                }
            }
            let Some(r) = leave else {
                self.status = LpStatus::Optimal;
                return;
            };
            let mut enter: Option<usize> = None;
            let mut best = T::INFINITY;
            for j in 0..tab.cols {
                if !self.allowed[j] {
                    continue;
                }
                let a = tab.t[r * w + j];
                if a < -tol {
                    let ratio = tab.d[j].max(T::ZERO) / (-a);
                    if ratio < best - tol * T::ONE.max(best.abs().min(T::from_f64(1.0e12))) {
                        best = ratio;
                        enter = Some(j);
                    }
                }
            }
            let Some(j) = enter else {
                self.status = LpStatus::Infeasible;
                return;
            };
            if local >= self.opts.max_iter {
                self.status = LpStatus::IterationLimit;
                return;
            }
            tab.pivot(r, j);
            local += 1;
            self.iterations += 1;
        }
    }
}
