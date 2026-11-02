//! Gauß-Elimination als benannte, nachvollziehbare Funktionen.
//!
//! * `gauss_solve` löst `A·x = b` durch Vorwärtselimination mit Spaltenpivotsuche und Rückwärtseinsetzen.
//!   Das ist rechnerisch dasselbe wie die LU-Zerlegung (`solve`): LU speichert nur die Eliminationsfaktoren,
//!   um mehrere rechte Seiten wiederzuverwenden. Für mehrere `b` mit derselben Matrix ist
//!   [`LuDecomposition`](crate::linalg::LuDecomposition) die bessere Wahl.
//! * `rank` bestimmt den Rang durch Elimination, ohne dass eine singuläre Matrix ein Fehler ist.
//! * `row_echelon` liefert die Zeilenstufenform samt Pivotspalten (auch für rechteckige Matrizen).
//!
//! Ein Pivot gilt als null, wenn sein Betrag `≤ max(M, N)·ε·max|aᵢⱼ|` ist. Für die genauere
//! Rangbestimmung bei schlecht konditionierten Matrizen ist die SVD (`Svd::rank`) zuverlässiger.

use super::{LinalgError, Matrix, Vector};
use crate::Real;

fn finite<T: Real>(x: T) -> bool {
    !x.is_nan() && !x.is_infinite()
}

/// Vorwärtselimination auf zeilenweisen `rows × cols`-Daten in-place bis zur Zeilenstufenform. Zeilen- und
/// Spaltenvertauschungen: nur Zeilen. Optional wird eine rechte Seite `rhs` (Länge `rows`) mitgeführt.
/// `pivots` (Länge `≥ min(rows, cols)`) erhält die Pivotspalten; Rückgabe: der Rang.
pub(crate) fn eliminate<T: Real>(
    a: &mut [T],
    rows: usize,
    cols: usize,
    mut rhs: Option<&mut [T]>,
    pivots: &mut [usize],
) -> Result<usize, LinalgError> {
    let mut scale = T::ZERO;
    for &v in a.iter() {
        if !finite(v) {
            return Err(LinalgError::NotFinite);
        }
        scale = scale.max(v.abs());
    }
    if let Some(b) = rhs.as_deref() {
        if b.iter().any(|&v| !finite(v)) {
            return Err(LinalgError::NotFinite);
        }
    }
    let tol = T::from_f64(rows.max(cols) as f64) * T::EPSILON * scale;
    let mut r = 0;
    for c in 0..cols {
        if r == rows {
            break;
        }
        let mut p = r;
        let mut best = a[r * cols + c].abs();
        for i in (r + 1)..rows {
            let v = a[i * cols + c].abs();
            if v > best {
                best = v;
                p = i;
            }
        }
        if best <= tol {
            // Spalte ohne Pivot: Reste unterhalb sind numerisch null
            for i in r..rows {
                a[i * cols + c] = T::ZERO;
            }
            continue;
        }
        if p != r {
            for j in 0..cols {
                a.swap(r * cols + j, p * cols + j);
            }
            if let Some(b) = rhs.as_deref_mut() {
                b.swap(r, p);
            }
        }
        let pivot = a[r * cols + c];
        for i in (r + 1)..rows {
            let f = a[i * cols + c] / pivot;
            a[i * cols + c] = T::ZERO;
            if f != T::ZERO {
                for j in (c + 1)..cols {
                    a[i * cols + j] -= f * a[r * cols + j];
                }
                if let Some(b) = rhs.as_deref_mut() {
                    let br = b[r];
                    b[i] -= f * br;
                }
            }
        }
        pivots[r] = c;
        r += 1;
    }
    Ok(r)
}

/// Löst `A·x = b` für quadratisches `A` (`a`, `b` werden zerstört; Ergebnis in `b`).
pub(crate) fn gauss_solve_in_place<T: Real>(
    a: &mut [T],
    b: &mut [T],
    n: usize,
    pivots: &mut [usize],
) -> Result<(), LinalgError> {
    let rank = eliminate(a, n, n, Some(&mut *b), pivots)?;
    if rank < n {
        return Err(LinalgError::Singular);
    }
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * b[j];
        }
        b[i] = s / a[i * n + i];
    }
    Ok(())
}

/// Zeilenstufenform einer `M × N`-Matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowEchelon<T, const M: usize, const N: usize> {
    /// Die Matrix in Zeilenstufenform (Einträge unterhalb der Pivots sind 0).
    pub matrix: Matrix<T, M, N>,
    /// Rang = Anzahl der Pivots.
    pub rank: usize,
    pivot_cols: [usize; N],
}

impl<T, const M: usize, const N: usize> RowEchelon<T, M, N> {
    /// Die Spaltenindizes der Pivots (`rank` Stück, aufsteigend).
    pub fn pivots(&self) -> &[usize] {
        &self.pivot_cols[..self.rank]
    }
}

impl<T: Real, const N: usize> Matrix<T, N, N> {
    /// Löst `A·x = b` durch Gauß-Elimination mit Spaltenpivotsuche (siehe Moduldokumentation, gleiches
    /// Ergebnis wie [`solve`](Matrix::solve)).
    ///
    /// Fehler: [`LinalgError::Singular`], [`LinalgError::NotFinite`].
    ///
    /// ```
    /// use azmath::linalg::{Matrix, Vector};
    ///
    /// // 2x + y = 5,  x + 3y = 10
    /// let a = Matrix::new([[2.0f64, 1.0], [1.0, 3.0]]);
    /// let x = a.gauss_solve(&Vector::new([5.0, 10.0])).unwrap();
    /// assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 3.0).abs() < 1e-12);
    /// ```
    pub fn gauss_solve(&self, b: &Vector<T, N>) -> Result<Vector<T, N>, LinalgError> {
        let mut a = *self;
        let mut x = *b;
        let mut pivots = [0usize; N];
        gauss_solve_in_place(a.data.as_flattened_mut(), x.as_mut_slice(), N, &mut pivots)?;
        Ok(x)
    }
}

impl<T: Real, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Zeilenstufenform per Gauß-Elimination mit Spaltenpivotsuche (nur Zeilenvertauschungen).
    ///
    /// Fehler: [`LinalgError::NotFinite`].
    pub fn row_echelon(&self) -> Result<RowEchelon<T, M, N>, LinalgError> {
        let mut matrix = *self;
        let mut pivot_cols = [0usize; N];
        let rank = eliminate(matrix.data.as_flattened_mut(), M, N, None, &mut pivot_cols)?;
        Ok(RowEchelon {
            matrix,
            rank,
            pivot_cols,
        })
    }

    /// Rang per Gauß-Elimination (Pivots `≤ max(M,N)·ε·max|aᵢⱼ|` zählen als null). Eine singuläre Matrix ist
    /// kein Fehler; für schlecht konditionierte Fälle ist die SVD zuverlässiger.
    ///
    /// Fehler: [`LinalgError::NotFinite`].
    ///
    /// ```
    /// use azmath::linalg::Matrix;
    ///
    /// let a = Matrix::new([[1.0f64, 2.0], [2.0, 4.0], [3.0, 6.0]]);
    /// assert_eq!(a.rank().unwrap(), 1);
    /// ```
    pub fn rank(&self) -> Result<usize, LinalgError> {
        Ok(self.row_echelon()?.rank)
    }
}

#[cfg(feature = "alloc")]
mod dynamic {
    use super::{eliminate, gauss_solve_in_place};
    use crate::Real;
    use crate::linalg::{DynMatrix, DynVector, LinalgError};
    use alloc::vec;
    use alloc::vec::Vec;

    /// Zeilenstufenform einer dynamischen Matrix, siehe [`RowEchelon`](super::RowEchelon).
    #[derive(Debug, Clone, PartialEq)]
    pub struct DynRowEchelon<T> {
        /// Die Matrix in Zeilenstufenform.
        pub matrix: DynMatrix<T>,
        /// Rang = Anzahl der Pivots.
        pub rank: usize,
        /// Die Spaltenindizes der Pivots (aufsteigend).
        pub pivots: Vec<usize>,
    }

    impl<T: Real> DynMatrix<T> {
        /// Löst `A·x = b` durch Gauß-Elimination, siehe [`Matrix::gauss_solve`](crate::linalg::Matrix::gauss_solve).
        ///
        /// Fehler: [`LinalgError::DimensionMismatch`] (nicht quadratisch oder falsche Länge von `b`),
        /// [`LinalgError::Singular`], [`LinalgError::NotFinite`].
        pub fn gauss_solve(&self, b: &DynVector<T>) -> Result<DynVector<T>, LinalgError> {
            let n = self.rows();
            if n != self.cols() || b.len() != n {
                return Err(LinalgError::DimensionMismatch);
            }
            let mut a = self.as_slice().to_vec();
            let mut x = b.as_slice().to_vec();
            let mut pivots = vec![0usize; n];
            gauss_solve_in_place(&mut a, &mut x, n, &mut pivots)?;
            Ok(DynVector::new(x))
        }

        /// Zeilenstufenform, siehe [`Matrix::row_echelon`](crate::linalg::Matrix::row_echelon).
        pub fn row_echelon(&self) -> Result<DynRowEchelon<T>, LinalgError> {
            let (m, n) = (self.rows(), self.cols());
            let mut a = self.as_slice().to_vec();
            let mut pivots = vec![0usize; m.min(n)];
            let rank = eliminate(&mut a, m, n, None, &mut pivots)?;
            pivots.truncate(rank);
            Ok(DynRowEchelon {
                matrix: DynMatrix::new(m, n, a)?,
                rank,
                pivots,
            })
        }

        /// Rang per Gauß-Elimination, siehe [`Matrix::rank`](crate::linalg::Matrix::rank).
        pub fn rank(&self) -> Result<usize, LinalgError> {
            Ok(self.row_echelon()?.rank)
        }
    }
}

#[cfg(feature = "alloc")]
pub use dynamic::DynRowEchelon;
