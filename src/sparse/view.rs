use super::SparseError;
use crate::Real;
use crate::linalg::LinearOperator;

/// Geliehene Sicht auf eine Matrix im **CSR-Format** (Compressed Sparse Row) – ohne Allokator nutzbar.
///
/// Zeile `i` besteht aus den Einträgen `row_ptr[i]..row_ptr[i + 1]` der Arrays `col_idx` und `values`. Die Spaltenindizes
/// jeder Zeile sind **streng aufsteigend** (keine Duplikate) und kleiner als `cols`.
///
/// ```
/// use azmath::sparse::CsrView;
/// use azmath::linalg::LinearOperator;
///
/// // [[2, 0, 1],
/// //  [0, 3, 0]]
/// let a = CsrView::new(2, 3, &[0, 2, 3], &[0, 2, 1], &[2.0f64, 1.0, 3.0]).unwrap();
/// let mut y = [0.0; 2];
/// a.apply(&[1.0, 1.0, 1.0], &mut y);
/// assert_eq!(y, [3.0, 3.0]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CsrView<'a, T> {
    rows: usize,
    cols: usize,
    row_ptr: &'a [usize],
    col_idx: &'a [usize],
    values: &'a [T],
}

/// Prüft die CSR-Invarianten.
pub(crate) fn validate(
    rows: usize,
    cols: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
) -> Result<(), SparseError> {
    if row_ptr.len() != rows + 1 || row_ptr[0] != 0 || row_ptr[rows] != col_idx.len() {
        return Err(SparseError::InvalidStructure);
    }
    for i in 0..rows {
        let (a, b) = (row_ptr[i], row_ptr[i + 1]);
        if a > b {
            return Err(SparseError::InvalidStructure);
        }
        for k in a..b {
            if col_idx[k] >= cols {
                return Err(SparseError::IndexOutOfBounds);
            }
            if k > a && col_idx[k] <= col_idx[k - 1] {
                return Err(SparseError::InvalidStructure);
            }
        }
    }
    Ok(())
}

impl<'a, T: Real> CsrView<'a, T> {
    /// Baut eine Sicht und prüft alle Invarianten.
    ///
    /// Fehler: [`SparseError::InvalidStructure`], [`SparseError::IndexOutOfBounds`], [`SparseError::NotFinite`].
    pub fn new(
        rows: usize,
        cols: usize,
        row_ptr: &'a [usize],
        col_idx: &'a [usize],
        values: &'a [T],
    ) -> Result<Self, SparseError> {
        if values.len() != col_idx.len() {
            return Err(SparseError::InvalidStructure);
        }
        validate(rows, cols, row_ptr, col_idx)?;
        if values.iter().any(|v| v.is_nan() || v.is_infinite()) {
            return Err(SparseError::NotFinite);
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

    /// Anzahl der gespeicherten Einträge.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Zeilenzeiger (`rows + 1` Einträge).
    pub fn row_ptr(&self) -> &'a [usize] {
        self.row_ptr
    }

    /// Spaltenindizes aller Einträge.
    pub fn col_idx(&self) -> &'a [usize] {
        self.col_idx
    }

    /// Werte aller Einträge.
    pub fn values(&self) -> &'a [T] {
        self.values
    }

    /// Die Einträge `(Spalte, Wert)` der Zeile `i` (aufsteigend nach Spalte). Panik bei `i >= rows`.
    pub fn row(&self, i: usize) -> impl Iterator<Item = (usize, T)> + 'a {
        let (a, b) = (self.row_ptr[i], self.row_ptr[i + 1]);
        self.col_idx[a..b]
            .iter()
            .copied()
            .zip(self.values[a..b].iter().copied())
    }

    /// Eintrag `(i, j)`; nicht gespeicherte Einträge sind 0. `None` außerhalb der Matrix. Binärsuche in der Zeile.
    pub fn get(&self, i: usize, j: usize) -> Option<T> {
        if i >= self.rows || j >= self.cols {
            return None;
        }
        let (a, b) = (self.row_ptr[i], self.row_ptr[i + 1]);
        Some(match self.col_idx[a..b].binary_search(&j) {
            Ok(k) => self.values[a + k],
            Err(_) => T::ZERO,
        })
    }

    /// Schreibt die Diagonale nach `out` (Länge `min(rows, cols)`).
    pub fn diagonal_into(&self, out: &mut [T]) -> Result<(), SparseError> {
        if out.len() != self.rows.min(self.cols) {
            return Err(SparseError::DimensionMismatch);
        }
        for (i, o) in out.iter_mut().enumerate() {
            *o = self.get(i, i).unwrap_or(T::ZERO);
        }
        Ok(())
    }

    /// `y = A·x`. Fehler bei falschen Längen.
    pub fn matvec(&self, x: &[T], y: &mut [T]) -> Result<(), SparseError> {
        if x.len() != self.cols || y.len() != self.rows {
            return Err(SparseError::DimensionMismatch);
        }
        self.apply_unchecked(x, y);
        Ok(())
    }

    /// `y = Aᵀ·x` (`x` hat `rows`, `y` hat `cols` Elemente).
    pub fn matvec_transposed(&self, x: &[T], y: &mut [T]) -> Result<(), SparseError> {
        if x.len() != self.rows || y.len() != self.cols {
            return Err(SparseError::DimensionMismatch);
        }
        y.fill(T::ZERO);
        for (i, &xi) in x.iter().enumerate() {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                y[self.col_idx[k]] += self.values[k] * xi;
            }
        }
        Ok(())
    }

    fn apply_unchecked(&self, x: &[T], y: &mut [T]) {
        for (i, yi) in y.iter_mut().enumerate() {
            let mut s = T::ZERO;
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                s += self.values[k] * x[self.col_idx[k]];
            }
            *yi = s;
        }
    }

    /// `true`, wenn die Matrix quadratisch ist und `|aᵢⱼ − aⱼᵢ| ≤ tol·max(|aᵢⱼ|, |aⱼᵢ|)` für alle Einträge gilt
    /// (nicht gespeicherte Einträge zählen als 0).
    pub fn is_symmetric(&self, tol: T) -> bool {
        if self.rows != self.cols {
            return false;
        }
        for i in 0..self.rows {
            for (j, v) in self.row(i) {
                let w = self.get(j, i).unwrap_or(T::ZERO);
                if (v - w).abs() > tol * v.abs().max(w.abs()) {
                    return false;
                }
            }
        }
        true
    }
}

impl<T: Real> LinearOperator<T> for CsrView<'_, T> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        self.apply_unchecked(x, y);
    }
}
