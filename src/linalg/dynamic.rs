//! Matrizen und Vektoren mit zur Laufzeit festgelegter Größe (Feature `alloc`).

use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Index, IndexMut};

use super::LinalgError;
use super::kernel::{lu_decompose, lu_determinant, lu_solve};
use crate::Real;

/// Vektor mit dynamischer Länge.
#[derive(Debug, Clone, PartialEq)]
pub struct DynVector<T> {
    data: Vec<T>,
}

impl<T: Real> DynVector<T> {
    /// Erzeugt einen Vektor aus einem `Vec`.
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Der Nullvektor der Länge `n`.
    pub fn zeros(n: usize) -> Self {
        Self {
            data: vec![T::ZERO; n],
        }
    }

    /// Die Elemente als Slice.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Anzahl der Elemente.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// `true`, wenn der Vektor leer ist.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Skalarprodukt; Fehler bei unterschiedlicher Länge.
    pub fn dot(&self, other: &Self) -> Result<T, LinalgError> {
        if self.len() != other.len() {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut s = T::ZERO;
        for (a, b) in self.data.iter().zip(&other.data) {
            s += *a * *b;
        }
        Ok(s)
    }

    /// Größter Betrag aller Elemente (Maximumnorm).
    pub fn max_abs(&self) -> T {
        self.data.iter().fold(T::ZERO, |m, v| m.max(v.abs()))
    }
}

impl<T> Index<usize> for DynVector<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T> IndexMut<usize> for DynVector<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

/// Matrix mit dynamischer Größe, zeilenweise gespeichert.
#[derive(Debug, Clone, PartialEq)]
pub struct DynMatrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Real> DynMatrix<T> {
    /// Erzeugt eine Matrix aus zeilenweise angeordneten Daten.
    ///
    /// Fehler, wenn `data.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, LinalgError> {
        if data.len() != rows * cols {
            return Err(LinalgError::DimensionMismatch);
        }
        Ok(Self { rows, cols, data })
    }

    /// Die Nullmatrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::ZERO; rows * cols],
        }
    }

    /// Die `n x n`-Einheitsmatrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = T::ONE;
        }
        m
    }

    /// Anzahl der Zeilen.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Anzahl der Spalten.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Die Daten zeilenweise als Slice.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Die transponierte Matrix.
    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.data[j * self.rows + i] = self.data[i * self.cols + j];
            }
        }
        out
    }

    /// Matrix-Vektor-Produkt.
    pub fn mul_vec(&self, v: &DynVector<T>) -> Result<DynVector<T>, LinalgError> {
        if self.cols != v.len() {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut out = DynVector::zeros(self.rows);
        for i in 0..self.rows {
            let mut s = T::ZERO;
            for j in 0..self.cols {
                s += self.data[i * self.cols + j] * v.data[j];
            }
            out.data[i] = s;
        }
        Ok(out)
    }

    /// Matrix-Matrix-Produkt.
    pub fn matmul(&self, rhs: &Self) -> Result<Self, LinalgError> {
        if self.cols != rhs.rows {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut out = Self::zeros(self.rows, rhs.cols);
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut s = T::ZERO;
                for k in 0..self.cols {
                    s += self.data[i * self.cols + k] * rhs.data[k * rhs.cols + j];
                }
                out.data[i * rhs.cols + j] = s;
            }
        }
        Ok(out)
    }

    /// Größter Betrag aller Elemente (Maximumnorm).
    pub fn max_abs(&self) -> T {
        self.data.iter().fold(T::ZERO, |m, v| m.max(v.abs()))
    }

    /// LU-Zerlegung; die Matrix muss quadratisch sein.
    pub fn lu(&self) -> Result<DynLu<T>, LinalgError> {
        DynLu::new(self)
    }

    /// Löst `self · x = b`.
    pub fn solve(&self, b: &DynVector<T>) -> Result<DynVector<T>, LinalgError> {
        self.lu()?.solve(b)
    }

    /// Determinante; Fehler bei nicht quadratischen Matrizen.
    /// Für singuläre Matrizen `0`, bei NaN/Unendlich NaN.
    pub fn determinant(&self) -> Result<T, LinalgError> {
        match self.lu() {
            Ok(lu) => Ok(lu.determinant()),
            Err(LinalgError::NotFinite) => Ok(T::NAN),
            Err(LinalgError::Singular) => Ok(T::ZERO),
            Err(e) => Err(e),
        }
    }

    /// Inverse Matrix.
    pub fn inverse(&self) -> Result<Self, LinalgError> {
        Ok(self.lu()?.inverse())
    }
}

impl<T> Index<(usize, usize)> for DynMatrix<T> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &T {
        assert!(i < self.rows && j < self.cols, "Index außerhalb der Matrix");
        &self.data[i * self.cols + j]
    }
}

impl<T> IndexMut<(usize, usize)> for DynMatrix<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        assert!(i < self.rows && j < self.cols, "Index außerhalb der Matrix");
        &mut self.data[i * self.cols + j]
    }
}

/// LU-Zerlegung `P·A = L·U` einer dynamischen quadratischen Matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct DynLu<T> {
    n: usize,
    lu: Vec<T>,
    perm: Vec<usize>,
    odd: bool,
}

impl<T: Real> DynLu<T> {
    /// Berechnet die Zerlegung von `a`.
    ///
    /// Fehler: [`LinalgError::DimensionMismatch`] bei nicht quadratischer Matrix,
    /// [`LinalgError::Singular`] bei (numerisch) singulärer Matrix,
    /// [`LinalgError::NotFinite`] bei NaN oder unendlichen Einträgen.
    pub fn new(a: &DynMatrix<T>) -> Result<Self, LinalgError> {
        if a.rows != a.cols {
            return Err(LinalgError::DimensionMismatch);
        }
        let n = a.rows;
        let mut lu = a.data.clone();
        let mut perm = vec![0usize; n];
        let odd = lu_decompose(&mut lu, n, &mut perm)?;
        Ok(Self { n, lu, perm, odd })
    }

    /// Löst `A·x = b`; Fehler, wenn die Länge von `b` nicht passt.
    pub fn solve(&self, b: &DynVector<T>) -> Result<DynVector<T>, LinalgError> {
        if b.len() != self.n {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut x = DynVector::zeros(self.n);
        lu_solve(&self.lu, self.n, &self.perm, &b.data, &mut x.data);
        Ok(x)
    }

    /// Determinante von `A`.
    pub fn determinant(&self) -> T {
        lu_determinant(&self.lu, self.n, self.odd)
    }

    /// Inverse von `A`.
    pub fn inverse(&self) -> DynMatrix<T> {
        let n = self.n;
        let mut inv = DynMatrix::zeros(n, n);
        let mut e = DynVector::zeros(n);
        let mut col = DynVector::zeros(n);
        for j in 0..n {
            e.data[j] = T::ONE;
            lu_solve(&self.lu, n, &self.perm, &e.data, &mut col.data);
            e.data[j] = T::ZERO;
            for i in 0..n {
                inv.data[i * n + j] = col.data[i];
            }
        }
        inv
    }

    /// Die Zeilenpermutation: Zeile `i` von `P·A` ist Zeile `permutation()[i]` von `A`.
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }
}
