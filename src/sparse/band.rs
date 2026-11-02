#![allow(clippy::needless_range_loop)] // Bandindizes: Indexschleifen sind hier klarer

use super::SparseError;
use crate::Real;
use crate::linalg::{DynMatrix, DynVector, LinalgError, LinearOperator};
use alloc::vec;
use alloc::vec::Vec;

fn finite<T: Real>(x: T) -> bool {
    !x.is_nan() && !x.is_infinite()
}

/// Quadratische **Bandmatrix** `n × n` mit `kl` unteren und `ku` oberen Nebendiagonalen: `aᵢⱼ = 0` für
/// `j < i − kl` und `j > i + ku`. Gespeichert werden nur die `n·(kl + ku + 1)` Bandeinträge (zeilenweise).
///
/// Mit [`lu`](Self::lu) (allgemein, Teilpivotisierung) und [`cholesky`](Self::cholesky) (symmetrisch positiv definit)
/// gibt es direkte Löser in `O(n·kl·(kl + ku))` bzw. `O(n·kd²)` – für tridiagonale und ähnliche Systeme
/// (Differenzenverfahren, Splines) schnell und ohne Rundungsfehler eines iterativen Verfahrens.
///
/// ```
/// use azmath::sparse::BandMatrix;
///
/// // Tridiagonale Matrix: 2 auf der Diagonale, −1 daneben
/// let n = 5;
/// let mut a = BandMatrix::new(n, 1, 1);
/// for i in 0..n {
///     a.set(i, i, 2.0f64).unwrap();
///     if i + 1 < n {
///         a.set(i, i + 1, -1.0).unwrap();
///         a.set(i + 1, i, -1.0).unwrap();
///     }
/// }
/// let x = a.lu().unwrap().solve(&[1.0; 5]).unwrap();
/// assert!((x[2] - 4.5).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BandMatrix<T> {
    n: usize,
    kl: usize,
    ku: usize,
    /// Zeile `i`, Spalte `j` (im Band) liegt bei `i·(kl+ku+1) + (j + kl − i)`.
    data: Vec<T>,
}

impl<T: Real> BandMatrix<T> {
    /// Nullmatrix mit der Bandstruktur `(kl, ku)`.
    pub fn new(n: usize, kl: usize, ku: usize) -> Self {
        Self {
            n,
            kl,
            ku,
            data: vec![T::ZERO; n * (kl + ku + 1)],
        }
    }

    /// Tridiagonale Matrix aus Unter-, Haupt- und Oberdiagonale (`lower` und `upper` haben `n − 1`, `diag` hat `n` Einträge).
    pub fn from_tridiagonal(lower: &[T], diag: &[T], upper: &[T]) -> Result<Self, SparseError> {
        let n = diag.len();
        if n > 0 && (lower.len() != n - 1 || upper.len() != n - 1) {
            return Err(SparseError::DimensionMismatch);
        }
        let mut a = Self::new(n, 1, 1);
        for i in 0..n {
            a.set(i, i, diag[i])?;
            if i + 1 < n {
                a.set(i, i + 1, upper[i])?;
                a.set(i + 1, i, lower[i])?;
            }
        }
        Ok(a)
    }

    /// Aus einer dichten quadratischen Matrix. Fehler, wenn außerhalb des Bands Einträge ungleich 0 stehen
    /// ([`SparseError::IndexOutOfBounds`]) oder die Matrix nicht quadratisch ist ([`SparseError::DimensionMismatch`]).
    pub fn from_dense(a: &DynMatrix<T>, kl: usize, ku: usize) -> Result<Self, SparseError> {
        let n = a.rows();
        if a.cols() != n {
            return Err(SparseError::DimensionMismatch);
        }
        let mut b = Self::new(n, kl, ku);
        for i in 0..n {
            for j in 0..n {
                let v = a[(i, j)];
                if v != T::ZERO {
                    b.set(i, j, v)?;
                }
            }
        }
        Ok(b)
    }

    /// Größe `n`.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Zahl der unteren Nebendiagonalen.
    pub fn kl(&self) -> usize {
        self.kl
    }

    /// Zahl der oberen Nebendiagonalen.
    pub fn ku(&self) -> usize {
        self.ku
    }

    fn width(&self) -> usize {
        self.kl + self.ku + 1
    }

    fn in_band(&self, i: usize, j: usize) -> bool {
        i < self.n && j < self.n && j + self.kl >= i && j <= i + self.ku
    }

    fn idx(&self, i: usize, j: usize) -> usize {
        i * self.width() + (j + self.kl - i)
    }

    /// Eintrag `(i, j)` (außerhalb des Bands 0); `None` außerhalb der Matrix.
    pub fn get(&self, i: usize, j: usize) -> Option<T> {
        if i >= self.n || j >= self.n {
            return None;
        }
        Some(if self.in_band(i, j) {
            self.data[self.idx(i, j)]
        } else {
            T::ZERO
        })
    }

    /// Setzt einen Eintrag im Band. Fehler außerhalb des Bands ([`SparseError::IndexOutOfBounds`]) oder bei NaN/∞.
    pub fn set(&mut self, i: usize, j: usize, v: T) -> Result<(), SparseError> {
        if !self.in_band(i, j) {
            return Err(SparseError::IndexOutOfBounds);
        }
        if !finite(v) {
            return Err(SparseError::NotFinite);
        }
        let k = self.idx(i, j);
        self.data[k] = v;
        Ok(())
    }

    /// Als dichte Matrix (Speicher `n²`).
    pub fn to_dense(&self) -> DynMatrix<T> {
        let mut d = DynMatrix::zeros(self.n, self.n);
        for i in 0..self.n {
            for j in i.saturating_sub(self.kl)..=(i + self.ku).min(self.n.saturating_sub(1)) {
                d[(i, j)] = self.data[self.idx(i, j)];
            }
        }
        d
    }

    /// `y = A·x`. Fehler bei falschen Längen.
    pub fn matvec(&self, x: &[T], y: &mut [T]) -> Result<(), SparseError> {
        if x.len() != self.n || y.len() != self.n {
            return Err(SparseError::DimensionMismatch);
        }
        self.apply_unchecked(x, y);
        Ok(())
    }

    fn apply_unchecked(&self, x: &[T], y: &mut [T]) {
        for (i, yi) in y.iter_mut().enumerate() {
            let mut s = T::ZERO;
            for j in i.saturating_sub(self.kl)..=(i + self.ku).min(self.n.saturating_sub(1)) {
                s += self.data[self.idx(i, j)] * x[j];
            }
            *yi = s;
        }
    }

    /// LU-Zerlegung mit Teilpivotisierung. Fehler: [`LinalgError::Singular`], [`LinalgError::NotFinite`].
    pub fn lu(&self) -> Result<BandLu<T>, LinalgError> {
        BandLu::new(self)
    }

    /// Cholesky-Zerlegung `A = L·Lᵀ` einer symmetrisch positiv definiten Bandmatrix (`kl == ku`).
    ///
    /// Fehler: [`LinalgError::DimensionMismatch`] (`kl ≠ ku`), [`LinalgError::NotSymmetric`],
    /// [`LinalgError::NotPositiveDefinite`], [`LinalgError::NotFinite`].
    pub fn cholesky(&self) -> Result<BandCholesky<T>, LinalgError> {
        BandCholesky::new(self)
    }

    /// Löst `A·x = b` (LU mit Teilpivotisierung).
    pub fn solve(&self, b: &DynVector<T>) -> Result<DynVector<T>, LinalgError> {
        Ok(DynVector::new(self.lu()?.solve(b.as_slice())?))
    }
}

impl<T: Real> LinearOperator<T> for BandMatrix<T> {
    fn rows(&self) -> usize {
        self.n
    }
    fn cols(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        self.apply_unchecked(x, y);
    }
}

/// LU-Zerlegung einer [`BandMatrix`] mit Zeilenvertauschungen (`P·A = L·U`). `U` hat bis zu `kl + ku` obere
/// Nebendiagonalen (Fill-in durch das Pivotisieren), `L` höchstens `kl` untere.
#[derive(Debug, Clone, PartialEq)]
pub struct BandLu<T> {
    n: usize,
    kl: usize,
    ku: usize,
    /// `U` zeilenweise, Breite `2·kl + ku + 1`, Spalte `j` der Zeile `i` bei `j + kl − i`.
    u: Vec<T>,
    /// Eliminationsfaktoren: `l[k·kl + (i − k − 1)]` für Zeile `i` in `k+1..=k+kl`.
    l: Vec<T>,
    piv: Vec<usize>,
    odd: bool,
}

impl<T: Real> BandLu<T> {
    fn new(a: &BandMatrix<T>) -> Result<Self, LinalgError> {
        let (n, kl, ku) = (a.n, a.kl, a.ku);
        if a.data.iter().any(|&v| !finite(v)) {
            return Err(LinalgError::NotFinite);
        }
        let w = 2 * kl + ku + 1;
        let mut u = vec![T::ZERO; n * w];
        let mut scale = T::ZERO;
        for i in 0..n {
            for j in i.saturating_sub(kl)..=(i + ku).min(n.saturating_sub(1)) {
                let v = a.data[a.idx(i, j)];
                u[i * w + (j + kl - i)] = v;
                scale = scale.max(v.abs());
            }
        }
        if n > 0 && scale == T::ZERO {
            return Err(LinalgError::Singular);
        }
        let tol = T::from_f64(n as f64) * T::EPSILON * scale;
        let mut l = vec![T::ZERO; n * kl];
        let mut piv = vec![0usize; n];
        let mut odd = false;
        let at = |i: usize, j: usize| i * w + (j + kl - i);
        for k in 0..n {
            let last_row = (k + kl).min(n - 1);
            let mut p = k;
            let mut best = u[at(k, k)].abs();
            for i in (k + 1)..=last_row {
                let v = u[at(i, k)].abs();
                if v > best {
                    best = v;
                    p = i;
                }
            }
            if best.is_nan() || best <= tol {
                return Err(LinalgError::Singular);
            }
            piv[k] = p;
            let last_col = (k + kl + ku).min(n - 1);
            if p != k {
                for j in k..=last_col {
                    u.swap(at(k, j), at(p, j));
                }
                odd = !odd;
            }
            let pivot = u[at(k, k)];
            for i in (k + 1)..=last_row {
                let m = u[at(i, k)] / pivot;
                l[k * kl + (i - k - 1)] = m;
                u[at(i, k)] = T::ZERO;
                if m != T::ZERO {
                    for j in (k + 1)..=last_col {
                        let ukj = u[at(k, j)];
                        u[at(i, j)] -= m * ukj;
                    }
                }
            }
        }
        Ok(Self {
            n,
            kl,
            ku,
            u,
            l,
            piv,
            odd,
        })
    }

    /// Löst `A·x = b`. Fehler bei falscher Länge von `b`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, LinalgError> {
        if b.len() != self.n {
            return Err(LinalgError::DimensionMismatch);
        }
        let (n, kl, ku) = (self.n, self.kl, self.ku);
        let w = 2 * kl + ku + 1;
        let mut x = b.to_vec();
        for k in 0..n {
            x.swap(k, self.piv[k]);
            let xk = x[k];
            for i in (k + 1)..=(k + kl).min(n - 1) {
                x[i] -= self.l[k * kl + (i - k - 1)] * xk;
            }
        }
        for i in (0..n).rev() {
            let mut s = x[i];
            for j in (i + 1)..=(i + kl + ku).min(n - 1) {
                s -= self.u[i * w + (j + kl - i)] * x[j];
            }
            x[i] = s / self.u[i * w + kl];
        }
        Ok(x)
    }

    /// Determinante: Produkt der Diagonale von `U` mit dem Vorzeichen der Vertauschungen.
    pub fn determinant(&self) -> T {
        let w = 2 * self.kl + self.ku + 1;
        let mut det = if self.odd { -T::ONE } else { T::ONE };
        for i in 0..self.n {
            det *= self.u[i * w + self.kl];
        }
        det
    }
}

/// Cholesky-Zerlegung `A = L·Lᵀ` einer symmetrisch positiv definiten [`BandMatrix`]; `L` hat dieselbe Bandbreite `kd`.
#[derive(Debug, Clone, PartialEq)]
pub struct BandCholesky<T> {
    n: usize,
    kd: usize,
    /// `L` zeilenweise, Breite `kd + 1`, Spalte `j ≤ i` bei `j + kd − i`.
    l: Vec<T>,
}

impl<T: Real> BandCholesky<T> {
    fn new(a: &BandMatrix<T>) -> Result<Self, LinalgError> {
        if a.kl != a.ku {
            return Err(LinalgError::DimensionMismatch);
        }
        let (n, kd) = (a.n, a.kl);
        if a.data.iter().any(|&v| !finite(v)) {
            return Err(LinalgError::NotFinite);
        }
        let mut scale = T::ZERO;
        let mut max_diag = T::ZERO;
        for i in 0..n {
            for j in i.saturating_sub(kd)..=i {
                let (lo, up) = (a.data[a.idx(i, j)], a.data[a.idx(j, i)]);
                if (lo - up).abs() > T::EPSILON.sqrt() * lo.abs().max(up.abs()) {
                    return Err(LinalgError::NotSymmetric);
                }
                scale = scale.max(lo.abs());
            }
            max_diag = max_diag.max(a.data[a.idx(i, i)]);
        }
        let tol = T::from_f64(n as f64) * T::EPSILON * max_diag;
        let w = kd + 1;
        let mut l = vec![T::ZERO; n * w];
        let at = |i: usize, j: usize| i * w + (j + kd - i);
        for i in 0..n {
            for j in i.saturating_sub(kd)..=i {
                let mut s = a.data[a.idx(i, j)];
                for k in i.saturating_sub(kd).max(j.saturating_sub(kd))..j {
                    s -= l[at(i, k)] * l[at(j, k)];
                }
                if i == j {
                    if s.is_nan() || s <= tol {
                        return Err(LinalgError::NotPositiveDefinite);
                    }
                    l[at(i, i)] = s.sqrt();
                } else {
                    l[at(i, j)] = s / l[at(j, j)];
                }
            }
        }
        Ok(Self { n, kd, l })
    }

    /// Löst `A·x = b`. Fehler bei falscher Länge von `b`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, LinalgError> {
        if b.len() != self.n {
            return Err(LinalgError::DimensionMismatch);
        }
        let (n, kd) = (self.n, self.kd);
        let w = kd + 1;
        let mut x = b.to_vec();
        for i in 0..n {
            let mut s = x[i];
            for k in i.saturating_sub(kd)..i {
                s -= self.l[i * w + (k + kd - i)] * x[k];
            }
            x[i] = s / self.l[i * w + kd];
        }
        for i in (0..n).rev() {
            let mut s = x[i];
            for k in (i + 1)..=(i + kd).min(n - 1) {
                s -= self.l[k * w + (i + kd - k)] * x[k];
            }
            x[i] = s / self.l[i * w + kd];
        }
        Ok(x)
    }

    /// Determinante `(Π lᵢᵢ)²`.
    pub fn determinant(&self) -> T {
        let mut p = T::ONE;
        for i in 0..self.n {
            p *= self.l[i * (self.kd + 1) + self.kd];
        }
        p * p
    }
}
