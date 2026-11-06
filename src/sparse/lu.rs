//! Direkter Löser für allgemeine (nichtsymmetrische) dünn besetzte Matrizen: LU mit Zeilenpivotisierung.

use super::{CooMatrix, CsrMatrix, Ordering, minimum_degree_ordering, rcm_ordering};
use crate::Real;
use crate::linalg::LinalgError;
use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;

/// `R·P·A·Pᵀ = L·U` einer quadratischen dünn besetzten Matrix: `P` eine Umordnung zur Fill-in-Verringerung (symmetrisch auf
/// Zeilen und Spalten, bestimmt aus dem Muster von `A + Aᵀ`), `R` die Zeilenvertauschung der Pivotisierung (größter Betrag der
/// Spalte, „partial pivoting“). Spaltenweise links-schauendes Verfahren nach Gilbert und Peierls in einer einfachen Form
/// (Aufwand `O(n² + Flops)`).
///
/// ```
/// use azmath::sparse::{CooMatrix, CsrMatrix, Ordering, SparseLu};
///
/// // [[0, 2], [1, 1]] braucht Zeilenvertauschung
/// let mut coo = CooMatrix::new(2, 2);
/// coo.push(0, 1, 2.0f64).unwrap();
/// coo.push(1, 0, 1.0).unwrap();
/// coo.push(1, 1, 1.0).unwrap();
/// let lu = SparseLu::new(&CsrMatrix::from_coo(&coo), Ordering::Natural).unwrap();
/// let x = lu.solve(&[4.0, 3.0]).unwrap();
/// assert!((x[0] - 1.0).abs() < 1e-14 && (x[1] - 2.0).abs() < 1e-14);
/// ```
#[derive(Debug, Clone)]
pub struct SparseLu<T> {
    n: usize,
    /// `perm[neu] = alt`
    perm: Vec<usize>,
    /// `prow[k]` = Zeile (im umgeordneten System), die im Schritt `k` Pivotzeile wurde
    prow: Vec<usize>,
    /// `pinv[i]` = Schritt, in dem Zeile `i` Pivotzeile wurde
    pinv: Vec<usize>,
    /// `L`-Spalten ohne Diagonale: `(Zeile im umgeordneten System, Wert)`
    lcols: Vec<Vec<(usize, T)>>,
    /// `U`-Spalten ohne Diagonale: `(Schritt k < j, Wert)`
    ucols: Vec<Vec<(usize, T)>>,
    udiag: Vec<T>,
    symbolic: Rc<SparseLuSymbolic>,
    work: Vec<T>,
    valid: bool,
}

fn symmetric_pattern<T: Real>(a: &CsrMatrix<T>) -> CsrMatrix<T> {
    let n = a.rows();
    let mut coo = CooMatrix::with_capacity(n, n, 2 * a.nnz());
    for i in 0..n {
        for (j, _) in a.row(i) {
            let _ = coo.push(i, j, T::ONE);
            let _ = coo.push(j, i, T::ONE);
        }
    }
    CsrMatrix::from_coo(&coo)
}

/// Wertunabhängige Vorbereitung für pivotiertes Sparse-LU.
/// Speichert das exakte CSR-Muster, Fill-in-Umordnung und spaltenweise Quellindizes.
/// Zeilenpivots und L/U-Füllmuster bleiben wertabhängig und werden numerisch neu bestimmt.
#[derive(Debug, Clone)]
pub struct SparseLuSymbolic {
    n: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    perm: Vec<usize>,
    cols: Vec<Vec<(usize, usize)>>,
}
impl SparseLuSymbolic {
    /// Bereitet eine quadratische Struktur vor; Zahlenwerte sind hierfür unerheblich.
    pub fn analyze<T: Real>(a: &CsrMatrix<T>, ordering: Ordering) -> Result<Self, LinalgError> {
        let n = a.rows();
        if a.cols() != n {
            return Err(LinalgError::DimensionMismatch);
        }
        let perm = match ordering {
            Ordering::Natural => (0..n).collect(),
            Ordering::ReverseCuthillMcKee => rcm_ordering(&symmetric_pattern(a)),
            Ordering::MinimumDegree => minimum_degree_ordering(&symmetric_pattern(a)),
        };
        let mut inv = vec![0; n];
        for (i, &p) in perm.iter().enumerate() {
            inv[p] = i;
        }
        let mut cols = vec![Vec::new(); n];
        for r in 0..n {
            for src in a.row_ptr()[r]..a.row_ptr()[r + 1] {
                cols[inv[a.col_idx()[src]]].push((inv[r], src));
            }
        }
        Ok(Self {
            n,
            row_ptr: a.row_ptr().to_vec(),
            col_idx: a.col_idx().to_vec(),
            perm,
            cols,
        })
    }
    /// Dimension.
    pub fn n(&self) -> usize {
        self.n
    }
    /// Symmetrische Fill-in-Umordnung (nicht die numerische Zeilenpivotierung).
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }
    /// Erzeugt numerische Faktoren und deren wiederverwendbare Puffer.
    pub fn factorize<T: Real>(&self, a: &CsrMatrix<T>) -> Result<SparseLu<T>, LinalgError> {
        let n = self.n;
        let mut factor = SparseLu {
            n,
            perm: self.perm.clone(),
            prow: vec![0; n],
            pinv: vec![usize::MAX; n],
            lcols: vec![Vec::new(); n],
            ucols: vec![Vec::new(); n],
            udiag: vec![T::ZERO; n],
            symbolic: Rc::new(self.clone()),
            work: vec![T::ZERO; n],
            valid: false,
        };
        factor.refactor(a)?;
        Ok(factor)
    }
}

impl<T: Real> SparseLu<T> {
    /// Faktorisiert `a` (quadratisch). Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NotFinite`],
    /// [`LinalgError::Singular`] (kein von Null verschiedener Pivot in einer Spalte).
    pub fn new(a: &CsrMatrix<T>, ordering: Ordering) -> Result<Self, LinalgError> {
        SparseLuSymbolic::analyze(a, ordering)?.factorize(a)
    }
    /// Wertunabhängige Vorbereitung.
    pub fn symbolic(&self) -> &SparseLuSymbolic {
        &self.symbolic
    }
    /// Ob eine gültige numerische Faktorisierung vorliegt.
    pub fn is_valid(&self) -> bool {
        self.valid
    }
    /// Aktualisiert Zahlenwerte bei exakt gleichem CSR-Muster.
    /// Umordnung/Zugriffsmuster bleiben erhalten; numerische Pivots und Fill-in werden
    /// neu bestimmt. Spaltenkapazitäten werden wiederverwendet, können aber wachsen.
    /// Struktur-/Eingabefehler erhalten die Faktoren, numerische Fehler invalidieren sie.
    pub fn refactor(&mut self, a: &CsrMatrix<T>) -> Result<(), LinalgError> {
        let n = self.n;
        if a.rows() != n || a.cols() != n {
            return Err(LinalgError::DimensionMismatch);
        }
        if a.row_ptr() != self.symbolic.row_ptr || a.col_idx() != self.symbolic.col_idx {
            return Err(LinalgError::InvalidInput);
        }
        if a.values().iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        self.valid = false;
        self.work.fill(T::ZERO);
        self.pinv.fill(usize::MAX);
        for j in 0..n {
            self.lcols[j].clear();
            self.ucols[j].clear();
            for &(i, src) in &self.symbolic.cols[j] {
                self.work[i] = a.values()[src];
            }
            for k in 0..j {
                let u = self.work[self.prow[k]];
                if !u.is_finite() {
                    return Err(LinalgError::NotFinite);
                }
                if u != T::ZERO {
                    self.ucols[j].push((k, u));
                    for &(i, l) in &self.lcols[k] {
                        self.work[i] -= l * u;
                    }
                }
                self.work[self.prow[k]] = T::ZERO;
            }
            let mut pivot = usize::MAX;
            let mut best = T::ZERO;
            for i in 0..n {
                if self.pinv[i] == usize::MAX {
                    if !self.work[i].is_finite() {
                        return Err(LinalgError::NotFinite);
                    }
                    if self.work[i].abs() > best {
                        best = self.work[i].abs();
                        pivot = i;
                    }
                }
            }
            if pivot == usize::MAX {
                return Err(LinalgError::Singular);
            }
            let d = self.work[pivot];
            for i in 0..n {
                if self.pinv[i] == usize::MAX {
                    if i != pivot && self.work[i] != T::ZERO {
                        let l = self.work[i] / d;
                        if !l.is_finite() {
                            return Err(LinalgError::NotFinite);
                        }
                        self.lcols[j].push((i, l));
                    }
                    self.work[i] = T::ZERO;
                }
            }
            self.pinv[pivot] = j;
            self.prow[j] = pivot;
            self.udiag[j] = d;
        }
        self.valid = true;
        Ok(())
    }

    /// Dimension.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Einträge von `L` ohne die Einsen der Diagonale.
    pub fn nnz_l(&self) -> usize {
        self.lcols.iter().map(Vec::len).sum()
    }

    /// Einträge von `U` einschließlich der Diagonale.
    pub fn nnz_u(&self) -> usize {
        self.ucols.iter().map(Vec::len).sum::<usize>() + self.n
    }

    /// `ln|det A|`.
    pub fn log_abs_determinant(&self) -> T {
        self.udiag.iter().fold(T::ZERO, |s, v| s + v.abs().ln())
    }

    /// Vorzeichen von `det A` (`+1` oder `−1`).
    pub fn determinant_sign(&self) -> T {
        let mut neg = self.udiag.iter().filter(|v| **v < T::ZERO).count();
        // Parität der Zeilenvertauschung R (Zyklenzählung von prow)
        let mut seen = vec![false; self.n];
        for s in 0..self.n {
            if !seen[s] {
                let mut len = 0;
                let mut k = s;
                while !seen[k] {
                    seen[k] = true;
                    k = self.prow[k];
                    len += 1;
                }
                if len % 2 == 0 {
                    neg += 1;
                }
            }
        }
        if neg % 2 == 0 { T::ONE } else { -T::ONE }
    }

    /// Löst `A·x = b`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, LinalgError> {
        let n = self.n;
        if !self.valid {
            return Err(LinalgError::InvalidInput);
        }
        if b.iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        if b.len() != n {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut w: Vec<T> = (0..n).map(|i| b[self.perm[i]]).collect();
        let mut z = vec![T::ZERO; n];
        for k in 0..n {
            z[k] = w[self.prow[k]];
            for &(i, l) in &self.lcols[k] {
                w[i] -= l * z[k];
            }
        }
        for j in (0..n).rev() {
            z[j] /= self.udiag[j];
            let zj = z[j];
            for &(k, u) in &self.ucols[j] {
                z[k] -= u * zj;
            }
        }
        let mut x = vec![T::ZERO; n];
        for i in 0..n {
            x[self.perm[i]] = z[i];
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        Ok(x)
    }

    /// Löst `Aᵀ·x = b`.
    pub fn solve_transposed(&self, b: &[T]) -> Result<Vec<T>, LinalgError> {
        let n = self.n;
        if !self.valid {
            return Err(LinalgError::InvalidInput);
        }
        if b.iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        if b.len() != n {
            return Err(LinalgError::DimensionMismatch);
        }
        // C = Pa·A·Paᵀ  ⇒  Cᵀ = Pa·Aᵀ·Paᵀ: gleiche Umordnung wie in `solve`
        let c: Vec<T> = (0..n).map(|i| b[self.perm[i]]).collect();
        // Uᵀ u = c
        let mut u = vec![T::ZERO; n];
        for j in 0..n {
            let mut s = c[j];
            for &(k, ukj) in &self.ucols[j] {
                s -= ukj * u[k];
            }
            u[j] = s / self.udiag[j];
        }
        // Lᵀ w = u mit w_k = y[prow[k]]
        let mut w = vec![T::ZERO; n];
        for k in (0..n).rev() {
            let mut s = u[k];
            for &(i, l) in &self.lcols[k] {
                s -= l * w[self.pinv[i]];
            }
            w[k] = s;
        }
        let mut y = vec![T::ZERO; n];
        for k in 0..n {
            y[self.prow[k]] = w[k];
        }
        let mut x = vec![T::ZERO; n];
        for i in 0..n {
            x[self.perm[i]] = y[i];
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        Ok(x)
    }
}
