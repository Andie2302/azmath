use super::view::validate;
use super::{CooMatrix, CsrView, SparseError};
use crate::Real;
use crate::linalg::{DynMatrix, LinearOperator, Matrix};
use alloc::vec;
use alloc::vec::Vec;

/// Matrix im **CSR-Format** (Compressed Sparse Row) mit eigenem Speicher; die Lesezugriffe entsprechen
/// [`CsrView`] (siehe dort für das Format). Implementiert [`LinearOperator`] und lässt sich damit direkt an `cg`, `gmres`
/// und `bicgstab` übergeben.
///
/// ```
/// use azmath::linalg::iterative::{cg, IterOptions};
/// use azmath::sparse::{CooMatrix, CsrMatrix};
///
/// // 1D-Laplace mit n = 5
/// let n = 5;
/// let mut coo = CooMatrix::new(n, n);
/// for i in 0..n {
///     coo.push(i, i, 2.0f64).unwrap();
///     if i + 1 < n {
///         coo.push(i, i + 1, -1.0).unwrap();
///         coo.push(i + 1, i, -1.0).unwrap();
///     }
/// }
/// let a = CsrMatrix::from_coo(&coo);
/// let b = vec![1.0; n];
/// let mut x = vec![0.0; n];
/// let mut work = vec![0.0; 4 * n];
/// cg(&a, &b, &mut x, IterOptions::new(1e-12, 100), &mut work).unwrap();
/// assert!((x[2] - 4.5).abs() < 1e-9); // x_i = i·(n+1−i)/2 mit i = 3
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix<T> {
    rows: usize,
    cols: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<T>,
}

impl<T: Real> CsrMatrix<T> {
    /// Aus den drei CSR-Arrays; prüft alle Invarianten (siehe [`CsrView::new`]).
    pub fn from_parts(
        rows: usize,
        cols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<T>,
    ) -> Result<Self, SparseError> {
        CsrView::new(rows, cols, &row_ptr, &col_idx, &values)?;
        Ok(Self {
            rows,
            cols,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Aus einer Tripletliste: sortiert nach Zeile und Spalte und **summiert Duplikate**.
    pub fn from_coo(coo: &CooMatrix<T>) -> Self {
        let mut t = coo.triplets.clone();
        t.sort_unstable_by_key(|&(i, j, _)| (i, j));
        let mut row_ptr = vec![0usize; coo.rows + 1];
        let mut col_idx: Vec<usize> = Vec::with_capacity(t.len());
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
        for i in 0..coo.rows {
            row_ptr[i + 1] += row_ptr[i];
        }
        Self {
            rows: coo.rows,
            cols: coo.cols,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Aus einer zeilenweise gespeicherten dichten Matrix; Einträge mit `|a| ≤ drop_tol` werden weggelassen
    /// (`drop_tol = 0` lässt nur exakte Nullen weg). Fehler bei falscher Länge oder nicht endlichen Werten.
    pub fn from_dense_slice(
        rows: usize,
        cols: usize,
        data: &[T],
        drop_tol: T,
    ) -> Result<Self, SparseError> {
        if data.len() != rows * cols {
            return Err(SparseError::DimensionMismatch);
        }
        let mut row_ptr = Vec::with_capacity(rows + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for i in 0..rows {
            for j in 0..cols {
                let v = data[i * cols + j];
                if v.is_nan() || v.is_infinite() {
                    return Err(SparseError::NotFinite);
                }
                if v.abs() > drop_tol {
                    col_idx.push(j);
                    values.push(v);
                }
            }
            row_ptr.push(col_idx.len());
        }
        Ok(Self {
            rows,
            cols,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Aus einer dynamischen dichten Matrix, siehe [`from_dense_slice`](Self::from_dense_slice).
    pub fn from_dense(a: &DynMatrix<T>, drop_tol: T) -> Result<Self, SparseError> {
        Self::from_dense_slice(a.rows(), a.cols(), a.as_slice(), drop_tol)
    }

    /// Aus einer dichten Matrix fester Größe.
    pub fn from_matrix<const R: usize, const C: usize>(
        a: &Matrix<T, R, C>,
        drop_tol: T,
    ) -> Result<Self, SparseError> {
        Self::from_dense_slice(R, C, a.as_array().as_flattened(), drop_tol)
    }

    /// Einheitsmatrix.
    pub fn identity(n: usize) -> Self {
        Self {
            rows: n,
            cols: n,
            row_ptr: (0..=n).collect(),
            col_idx: (0..n).collect(),
            values: vec![T::ONE; n],
        }
    }

    /// Geliehene Sicht auf dieselben Daten.
    pub fn view(&self) -> CsrView<'_, T> {
        // Invarianten sind durch alle Konstruktoren und Operationen gesichert
        CsrView::new(
            self.rows,
            self.cols,
            &self.row_ptr,
            &self.col_idx,
            &self.values,
        )
        .expect("CSR-Invarianten verletzt")
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
    pub fn row_ptr(&self) -> &[usize] {
        &self.row_ptr
    }

    /// Spaltenindizes aller Einträge.
    pub fn col_idx(&self) -> &[usize] {
        &self.col_idx
    }

    /// Werte aller Einträge.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Veränderliche Werte bei **festem Muster** (zum Beispiel für Vorkonditionierer, die das Muster behalten).
    pub fn values_mut(&mut self) -> &mut [T] {
        &mut self.values
    }

    /// Die Einträge `(Spalte, Wert)` der Zeile `i`. Panik bei `i >= rows`.
    pub fn row(&self, i: usize) -> impl Iterator<Item = (usize, T)> + '_ {
        let (a, b) = (self.row_ptr[i], self.row_ptr[i + 1]);
        self.col_idx[a..b]
            .iter()
            .copied()
            .zip(self.values[a..b].iter().copied())
    }

    /// Eintrag `(i, j)` (nicht gespeichert = 0); `None` außerhalb der Matrix.
    pub fn get(&self, i: usize, j: usize) -> Option<T> {
        self.view().get(i, j)
    }

    /// Die Diagonale (`min(rows, cols)` Einträge).
    pub fn diagonal(&self) -> Vec<T> {
        let mut d = vec![T::ZERO; self.rows.min(self.cols)];
        self.view().diagonal_into(&mut d).expect("Länge passt");
        d
    }

    /// `y = A·x`. Fehler bei falschen Längen.
    pub fn matvec(&self, x: &[T], y: &mut [T]) -> Result<(), SparseError> {
        self.view().matvec(x, y)
    }

    /// `y = Aᵀ·x`. Fehler bei falschen Längen.
    pub fn matvec_transposed(&self, x: &[T], y: &mut [T]) -> Result<(), SparseError> {
        self.view().matvec_transposed(x, y)
    }

    /// Symmetrieprüfung mit relativer Toleranz, siehe [`CsrView::is_symmetric`].
    pub fn is_symmetric(&self, tol: T) -> bool {
        self.view().is_symmetric(tol)
    }

    /// Die Transponierte (`O(nnz)`).
    pub fn transpose(&self) -> Self {
        let mut row_ptr = vec![0usize; self.cols + 1];
        for &j in &self.col_idx {
            row_ptr[j + 1] += 1;
        }
        for j in 0..self.cols {
            row_ptr[j + 1] += row_ptr[j];
        }
        let mut next = row_ptr.clone();
        let mut col_idx = vec![0usize; self.nnz()];
        let mut values = vec![T::ZERO; self.nnz()];
        for i in 0..self.rows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[k];
                let dst = next[j];
                col_idx[dst] = i;
                values[dst] = self.values[k];
                next[j] += 1;
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Multipliziert Zeile `i` mit `s[i]` (`s` hat `rows` Einträge).
    pub fn scale_rows(&mut self, s: &[T]) -> Result<(), SparseError> {
        if s.len() != self.rows {
            return Err(SparseError::DimensionMismatch);
        }
        for (i, &si) in s.iter().enumerate() {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                self.values[k] *= si;
            }
        }
        Ok(())
    }

    /// Multipliziert Spalte `j` mit `s[j]` (`s` hat `cols` Einträge).
    pub fn scale_cols(&mut self, s: &[T]) -> Result<(), SparseError> {
        if s.len() != self.cols {
            return Err(SparseError::DimensionMismatch);
        }
        for k in 0..self.nnz() {
            self.values[k] *= s[self.col_idx[k]];
        }
        Ok(())
    }

    /// Summe `self + other` (Muster = Vereinigung beider Muster). Fehler bei verschiedenen Größen.
    pub fn add(&self, other: &Self) -> Result<Self, SparseError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SparseError::DimensionMismatch);
        }
        let mut row_ptr = Vec::with_capacity(self.rows + 1);
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        row_ptr.push(0);
        for i in 0..self.rows {
            let (mut p, pe) = (self.row_ptr[i], self.row_ptr[i + 1]);
            let (mut q, qe) = (other.row_ptr[i], other.row_ptr[i + 1]);
            while p < pe || q < qe {
                let (jp, jq) = (
                    if p < pe { self.col_idx[p] } else { usize::MAX },
                    if q < qe { other.col_idx[q] } else { usize::MAX },
                );
                if jp == jq {
                    col_idx.push(jp);
                    values.push(self.values[p] + other.values[q]);
                    p += 1;
                    q += 1;
                } else if jp < jq {
                    col_idx.push(jp);
                    values.push(self.values[p]);
                    p += 1;
                } else {
                    col_idx.push(jq);
                    values.push(other.values[q]);
                    q += 1;
                }
            }
            row_ptr.push(col_idx.len());
        }
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Als dichte Matrix (Speicher `rows·cols`!).
    pub fn to_dense(&self) -> DynMatrix<T> {
        let mut data = vec![T::ZERO; self.rows * self.cols];
        for i in 0..self.rows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                data[i * self.cols + self.col_idx[k]] = self.values[k];
            }
        }
        DynMatrix::new(self.rows, self.cols, data).expect("Länge passt")
    }

    /// Prüft die Invarianten erneut (nach dem Umbau über `values_mut` bleibt die Struktur unverändert; nützlich in Tests).
    pub fn validate(&self) -> Result<(), SparseError> {
        validate(self.rows, self.cols, &self.row_ptr, &self.col_idx)
    }
}

impl<T: Real> LinearOperator<T> for CsrMatrix<T> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        self.view().apply(x, y);
    }
}
