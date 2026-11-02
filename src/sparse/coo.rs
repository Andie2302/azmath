use super::SparseError;
use crate::Real;
use alloc::vec::Vec;

/// Matrix in **Koordinatenform** (Tripletliste `(Zeile, Spalte, Wert)`), gedacht zum bequemen Aufbauen
/// (Assemblieren, zum Beispiel für FEM). Mehrfach vorkommende Positionen werden bei der Umwandlung in
/// [`CsrMatrix`](super::CsrMatrix) **summiert**.
///
/// ```
/// use azmath::sparse::{CooMatrix, CsrMatrix};
///
/// let mut coo = CooMatrix::new(2, 2);
/// coo.push(0, 0, 1.0f64).unwrap();
/// coo.push(0, 0, 2.0).unwrap(); // wird zu 3
/// coo.push(1, 0, 5.0).unwrap();
/// let a = CsrMatrix::from_coo(&coo);
/// assert_eq!(a.get(0, 0), Some(3.0));
/// assert_eq!(a.nnz(), 2);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CooMatrix<T> {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) triplets: Vec<(usize, usize, T)>,
}

impl<T: Real> CooMatrix<T> {
    /// Leere `rows × cols`-Matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            triplets: Vec::new(),
        }
    }

    /// Wie [`new`](Self::new) mit Platz für `capacity` Einträge.
    pub fn with_capacity(rows: usize, cols: usize, capacity: usize) -> Self {
        Self {
            rows,
            cols,
            triplets: Vec::with_capacity(capacity),
        }
    }

    /// Anzahl der Zeilen.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Anzahl der Spalten.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Anzahl der eingetragenen Tripel (vor dem Summieren von Duplikaten).
    pub fn len(&self) -> usize {
        self.triplets.len()
    }

    /// `true`, wenn noch nichts eingetragen wurde.
    pub fn is_empty(&self) -> bool {
        self.triplets.is_empty()
    }

    /// Trägt `v` an Position `(i, j)` ein (addiert sich zu bereits Eingetragenem).
    ///
    /// Fehler: [`SparseError::IndexOutOfBounds`], [`SparseError::NotFinite`].
    pub fn push(&mut self, i: usize, j: usize, v: T) -> Result<(), SparseError> {
        if i >= self.rows || j >= self.cols {
            return Err(SparseError::IndexOutOfBounds);
        }
        if v.is_nan() || v.is_infinite() {
            return Err(SparseError::NotFinite);
        }
        self.triplets.push((i, j, v));
        Ok(())
    }
}
