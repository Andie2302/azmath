//! Minimale dünn besetzte Matrizen über [`Scalar`] (CSR) und GMRES(m) für komplexe lineare Systeme.

use super::Scalar;
use super::qr::norm2;
use crate::Real;
use crate::linalg::LinalgError;
use alloc::vec;
use alloc::vec::Vec;

/// CSR-Matrix über einem beliebigen [`Scalar`].
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarCsr<T: Scalar> {
    rows: usize,
    cols: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<T>,
}

impl<T: Scalar> ScalarCsr<T> {
    /// Aus Tripletten `(Zeile, Spalte, Wert)`; Duplikate werden summiert. Fehler [`LinalgError::DimensionMismatch`] bei
    /// Indizes außerhalb, [`LinalgError::NotFinite`].
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: &[(usize, usize, T)],
    ) -> Result<Self, LinalgError> {
        if triplets.iter().any(|&(i, j, _)| i >= rows || j >= cols) {
            return Err(LinalgError::DimensionMismatch);
        }
        if triplets.iter().any(|&(_, _, v)| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        let mut t = triplets.to_vec();
        t.sort_by_key(|&(i, j, _)| (i, j));
        let mut row_ptr = vec![0usize; rows + 1];
        let mut col_idx = Vec::with_capacity(t.len());
        let mut values: Vec<T> = Vec::with_capacity(t.len());
        let mut last: Option<(usize, usize)> = None;
        for (i, j, v) in t {
            if last == Some((i, j)) {
                if let Some(slot) = values.last_mut() {
                    *slot += v;
                }
            } else {
                col_idx.push(j);
                values.push(v);
                row_ptr[i + 1] += 1;
                last = Some((i, j));
            }
        }
        for i in 0..rows {
            row_ptr[i + 1] += row_ptr[i];
        }
        Ok(Self {
            rows,
            cols,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Anzahl der Zeilen.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Anzahl der Spalten.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Alle gespeicherten Einträge als `(Zeile, Spalte, Wert)`.
    pub fn entries(&self) -> Vec<(usize, usize, T)> {
        let mut v = Vec::with_capacity(self.values.len());
        for i in 0..self.rows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                v.push((i, self.col_idx[k], self.values[k]));
            }
        }
        v
    }

    /// Anzahl der gespeicherten Einträge.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// `y = A·x`.
    pub fn matvec(&self, x: &[T], y: &mut [T]) -> Result<(), LinalgError> {
        if x.len() != self.cols || y.len() != self.rows {
            return Err(LinalgError::DimensionMismatch);
        }
        for i in 0..self.rows {
            let mut s = T::ZERO;
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                s += self.values[k] * x[self.col_idx[k]];
            }
            y[i] = s;
        }
        Ok(())
    }

    /// `y = Aᴴ·x`.
    pub fn adjoint_matvec(&self, x: &[T], y: &mut [T]) -> Result<(), LinalgError> {
        if x.len() != self.rows || y.len() != self.cols {
            return Err(LinalgError::DimensionMismatch);
        }
        y.iter_mut().for_each(|v| *v = T::ZERO);
        for i in 0..self.rows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                y[self.col_idx[k]] += self.values[k].conj() * x[i];
            }
        }
        Ok(())
    }

    /// Dichte Darstellung (zeilenweise).
    pub fn to_dense(&self) -> Vec<T> {
        let mut d = vec![T::ZERO; self.rows * self.cols];
        for i in 0..self.rows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                d[i * self.cols + self.col_idx[k]] = self.values[k];
            }
        }
        d
    }
}

/// Ergebnis von [`gmres_scalar`].
#[derive(Debug, Clone, PartialEq)]
pub struct GmresScalarResult<T: Scalar> {
    /// Näherungslösung.
    pub x: Vec<T>,
    /// Gesamtzahl der inneren Iterationen.
    pub iterations: usize,
    /// Norm des Residuums `‖b − A·x‖₂` zum Schluss.
    pub residual_norm: T::Real,
}

/// GMRES(`restart`) für `A·x = b` mit quadratischer [`ScalarCsr`]-Matrix, Start `x = 0`. Abbruch bei
/// `‖b − A·x‖ ≤ tol·‖b‖` (relative Toleranz).
///
/// Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NotFinite`], [`LinalgError::NoConvergence`] nach `max_iter`
/// inneren Iterationen.
pub fn gmres_scalar<T: Scalar>(
    a: &ScalarCsr<T>,
    b: &[T],
    restart: usize,
    tol: T::Real,
    max_iter: usize,
) -> Result<GmresScalarResult<T>, LinalgError> {
    let n = a.rows();
    if a.cols() != n || b.len() != n || restart == 0 {
        return Err(LinalgError::DimensionMismatch);
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    let rz = <T::Real as Real>::ZERO;
    let bnorm = norm2(b);
    let mut x = vec![T::ZERO; n];
    if bnorm == rz {
        return Ok(GmresScalarResult {
            x,
            iterations: 0,
            residual_norm: rz,
        });
    }
    let m = restart.min(n.max(1));
    let mut total = 0usize;
    let mut ax = vec![T::ZERO; n];
    loop {
        a.matvec(&x, &mut ax)?;
        let r: Vec<T> = (0..n).map(|i| b[i] - ax[i]).collect();
        let beta = norm2(&r);
        if beta <= tol * bnorm {
            return Ok(GmresScalarResult {
                x,
                iterations: total,
                residual_norm: beta,
            });
        }
        if total >= max_iter {
            return Err(LinalgError::NoConvergence);
        }
        let mut v: Vec<Vec<T>> = Vec::with_capacity(m + 1);
        v.push(r.iter().map(|&z| z / T::from_real(beta)).collect());
        let mut hmat = vec![vec![T::ZERO; m]; m + 1];
        let mut cs = vec![T::ZERO; m];
        let mut sn = vec![T::ZERO; m];
        let mut g = vec![T::ZERO; m + 1];
        g[0] = T::from_real(beta);
        let mut k = 0usize;
        let mut w = vec![T::ZERO; n];
        for j in 0..m {
            a.matvec(&v[j], &mut w)?;
            for i in 0..=j {
                let mut h = T::ZERO;
                for t in 0..n {
                    h += v[i][t].conj() * w[t];
                }
                hmat[i][j] = h;
                for t in 0..n {
                    w[t] -= h * v[i][t];
                }
            }
            let hn = norm2(&w);
            hmat[j + 1][j] = T::from_real(hn);
            let breakdown = !(hn > rz);
            if !breakdown {
                v.push(w.iter().map(|&z| z / T::from_real(hn)).collect());
            }
            for i in 0..j {
                let (u, t) = (hmat[i][j], hmat[i + 1][j]);
                hmat[i][j] = cs[i].conj() * u + sn[i].conj() * t;
                hmat[i + 1][j] = -(sn[i] * u) + cs[i] * t;
            }
            let (p, q) = (hmat[j][j], hmat[j + 1][j]);
            let r = p.modulus().hypot(q.modulus());
            if r == rz {
                cs[j] = T::ONE;
                sn[j] = T::ZERO;
            } else {
                cs[j] = p / T::from_real(r);
                sn[j] = q / T::from_real(r);
            }
            hmat[j][j] = T::from_real(r);
            hmat[j + 1][j] = T::ZERO;
            g[j + 1] = -(sn[j] * g[j]);
            g[j] = cs[j].conj() * g[j];
            total += 1;
            k = j + 1;
            if g[j + 1].modulus() <= tol * bnorm || breakdown || total >= max_iter {
                break;
            }
        }
        // R·y = g
        let mut y = vec![T::ZERO; k];
        for i in (0..k).rev() {
            let mut s = g[i];
            for j in i + 1..k {
                s -= hmat[i][j] * y[j];
            }
            if hmat[i][i] == T::ZERO {
                return Err(LinalgError::Singular);
            }
            y[i] = s / hmat[i][i];
        }
        for (i, yi) in y.iter().enumerate() {
            for t in 0..n {
                x[t] += *yi * v[i][t];
            }
        }
    }
}
