//! Gewichtete, regularisierte und nichtnegative kleinste Quadrate.

use crate::Real;
use crate::linalg::{DynMatrix, DynVector, LinalgError};
use alloc::vec::Vec;

/// Gewichtete kleinste Quadrate: minimiert `Σ wᵢ·(Ax − b)ᵢ²` (Gewichte `≥ 0`).
pub fn weighted_least_squares<T: Real>(
    a: &DynMatrix<T>,
    b: &DynVector<T>,
    w: &[T],
) -> Result<DynVector<T>, LinalgError> {
    let (m, n) = (a.rows(), a.cols());
    if b.len() != m || w.len() != m {
        return Err(LinalgError::DimensionMismatch);
    }
    if w.iter().any(|&v| !(v >= T::ZERO)) {
        return Err(LinalgError::NotFinite);
    }
    let mut aw = DynMatrix::zeros(m, n);
    let mut bw = DynVector::zeros(m);
    for i in 0..m {
        let s = w[i].sqrt();
        for j in 0..n {
            aw[(i, j)] = a[(i, j)] * s;
        }
        bw[i] = b[i] * s;
    }
    aw.least_squares(&bw)
}

/// Tikhonov-Regularisierung (Ridge): minimiert `‖Ax − b‖² + λ·‖x‖²` mit `λ ≥ 0`, gelöst über das erweiterte System
/// `[A; √λ·I]`. Funktioniert auch bei Rangdefizit von `A`, solange `λ > 0`.
pub fn ridge<T: Real>(
    a: &DynMatrix<T>,
    b: &DynVector<T>,
    lambda: T,
) -> Result<DynVector<T>, LinalgError> {
    let (m, n) = (a.rows(), a.cols());
    if b.len() != m {
        return Err(LinalgError::DimensionMismatch);
    }
    if !(lambda >= T::ZERO) {
        return Err(LinalgError::NotFinite);
    }
    let s = lambda.sqrt();
    let mut aug = DynMatrix::zeros(m + n, n);
    let mut rhs = DynVector::zeros(m + n);
    for i in 0..m {
        for j in 0..n {
            aug[(i, j)] = a[(i, j)];
        }
        rhs[i] = b[i];
    }
    for j in 0..n {
        aug[(m + j, j)] = s;
    }
    aug.least_squares(&rhs)
}

/// Ergebnis von [`nnls`].
#[derive(Debug, Clone, PartialEq)]
pub struct NnlsResult<T> {
    /// Lösung `x ≥ 0`.
    pub x: Vec<T>,
    /// Norm des Residuums `‖Ax − b‖₂`.
    pub residual_norm: T,
    /// Anzahl der äußeren Iterationen.
    pub iterations: usize,
}

/// Nichtnegative kleinste Quadrate `min ‖Ax − b‖₂` mit `x ≥ 0` (Lawson-Hanson, aktive Mengen).
///
/// Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NotFinite`], [`LinalgError::NoConvergence`] (nach `3·n + 10`
/// Iterationen), sowie [`LinalgError::Singular`], wenn die gewählten Spalten linear abhängig sind.
pub fn nnls<T: Real>(a: &DynMatrix<T>, b: &DynVector<T>) -> Result<NnlsResult<T>, LinalgError> {
    let (m, n) = (a.rows(), a.cols());
    if b.len() != m {
        return Err(LinalgError::DimensionMismatch);
    }
    if a.as_slice()
        .iter()
        .chain(b.as_slice())
        .any(|v| !v.is_finite())
    {
        return Err(LinalgError::NotFinite);
    }
    let tol = T::EPSILON * T::from_f64(10.0) * (T::ONE + a.max_abs()) * T::from_usize(m.max(n));
    let mut x = alloc::vec![T::ZERO; n];
    let mut in_p = alloc::vec![false; n];
    let residual = |x: &[T]| -> Vec<T> {
        (0..m)
            .map(|i| {
                let mut s = b[i];
                for j in 0..n {
                    s -= a[(i, j)] * x[j];
                }
                s
            })
            .collect()
    };
    let max_iter = 3 * n + 10;
    let mut iterations = 0;
    loop {
        let r = residual(&x);
        // w = Aᵀ r
        let mut best: Option<(usize, T)> = None;
        for j in 0..n {
            if in_p[j] {
                continue;
            }
            let mut wj = T::ZERO;
            for i in 0..m {
                wj += a[(i, j)] * r[i];
            }
            if wj > tol && best.is_none_or(|(_, bw)| wj > bw) {
                best = Some((j, wj));
            }
        }
        let Some((jmax, _)) = best else { break };
        iterations += 1;
        if iterations > max_iter {
            return Err(LinalgError::NoConvergence);
        }
        in_p[jmax] = true;
        loop {
            let cols: Vec<usize> = (0..n).filter(|&j| in_p[j]).collect();
            if cols.len() > m {
                return Err(LinalgError::NoConvergence);
            }
            let mut sub = DynMatrix::zeros(m, cols.len());
            for (k, &j) in cols.iter().enumerate() {
                for i in 0..m {
                    sub[(i, k)] = a[(i, j)];
                }
            }
            let s = sub.least_squares(b)?;
            if cols.iter().enumerate().all(|(k, _)| s[k] > T::ZERO) {
                x.iter_mut().for_each(|v| *v = T::ZERO);
                for (k, &j) in cols.iter().enumerate() {
                    x[j] = s[k];
                }
                break;
            }
            let mut alpha = T::ONE;
            for (k, &j) in cols.iter().enumerate() {
                if s[k] <= T::ZERO {
                    let q = x[j] / (x[j] - s[k]);
                    if q < alpha {
                        alpha = q;
                    }
                }
            }
            for (k, &j) in cols.iter().enumerate() {
                x[j] = x[j] + alpha * (s[k] - x[j]);
            }
            for &j in &cols {
                if x[j] <= tol {
                    x[j] = T::ZERO;
                    in_p[j] = false;
                }
            }
        }
    }
    let r = residual(&x);
    let mut s = T::ZERO;
    for v in &r {
        s += *v * *v;
    }
    Ok(NnlsResult {
        x,
        residual_norm: s.sqrt(),
        iterations,
    })
}
