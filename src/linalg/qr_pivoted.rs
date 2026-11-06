//! Rangbestimmende Householder-QR mit Spaltenpivotierung.
use super::kernel::{
    qr_apply_q, qr_pivoted_decompose, qr_pivoted_rank, qr_pivoted_solve, qr_pivoted_tolerance,
    qr_residual_norm,
};
use super::{LinalgError, Matrix, Vector};
use crate::Real;

/// QR mit Spaltenpivotierung: `A[:, permutation] = Q·R`, für `M ≥ N`.
///
/// Auch rangdefiziente Matrizen können zerlegt werden. Der numerische Rang wird
/// über die führenden Diagonaleinträge von `R` geschätzt; dies ist keine SVD.
/// Starke Unterschiede der Spaltenskalierung beeinflussen den numerischen Rang.
/// Bei Bedarf die Spalten vorher äquilibrieren oder eine eigene absolute Schwelle wählen.
/// Für genaue Ergebnisse `std` oder `libm` verwenden, nicht `micromath`.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::linalg::{Matrix, Vector};
/// let a = Matrix::new([[0.0f64, 2.0], [0.0, 4.0], [0.0, 0.0]]);
/// let qr = a.qr_pivoted().unwrap();
/// assert_eq!(qr.rank(), 1);
/// assert_eq!(qr.permutation(), &[1, 0]);
/// let x = qr.solve(&Vector::new([6.0, 12.0, 1.0])).unwrap();
/// assert!(x[0] == 0.0 && (x[1] - 3.0).abs() < 1e-14);
/// # }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PivotedQr<T, const M: usize, const N: usize> {
    qr: Matrix<T, M, N>,
    tau: [T; N],
    permutation: [usize; N],
}

impl<T: Real, const M: usize, const N: usize> PivotedQr<T, M, N> {
    /// Zerlegt `a`; Fehler bei `M < N` oder nichtendlichen Daten/Zwischenergebnissen.
    /// Rangmangel ist kein Fehler.
    pub fn new(a: &Matrix<T, M, N>) -> Result<Self, LinalgError> {
        if M < N {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut qr = *a;
        let mut tau = [T::ZERO; N];
        let mut permutation = [0; N];
        qr_pivoted_decompose(qr.data.as_flattened_mut(), M, N, &mut tau, &mut permutation)?;
        Ok(Self {
            qr,
            tau,
            permutation,
        })
    }

    /// Ursprüngliche Spaltenindizes: Spalte `j` in `Q·R` ist Spalte `permutation[j]` in `A`.
    pub fn permutation(&self) -> &[usize; N] {
        &self.permutation
    }

    /// Absolute Standardschwelle `min(M·ε, 1)·|R₀₀|` (0 bei leerer Zerlegung).
    pub fn default_tolerance(&self) -> T {
        qr_pivoted_tolerance(self.qr.data.as_flattened(), M, N)
    }

    /// Geschätzter numerischer Rang mit der Standardschwelle.
    pub fn rank(&self) -> usize {
        let tol = self.default_tolerance();
        (0..N).take_while(|&i| self.qr[(i, i)].abs() > tol).count()
    }

    /// Größe des führenden Blocks mit `|Rᵢᵢ| > tol`.
    /// `tol` ist absolut, endlich und nichtnegativ, sonst `InvalidInput`.
    pub fn rank_with_tolerance(&self, tol: T) -> Result<usize, LinalgError> {
        qr_pivoted_rank(self.qr.data.as_flattened(), N, tol)
    }

    /// Löst das Ausgleichsproblem mit der Standardschwelle und prüft rechte Seite und Ergebnis.
    /// Bei Rangmangel: Basislösung mit freien pivotierten Koordinaten 0, im Allgemeinen
    /// **keine Lösung kleinster Norm** (dafür `Svd::solve`).
    pub fn solve(&self, b: &Vector<T, M>) -> Result<Vector<T, N>, LinalgError> {
        self.solve_with_tolerance(b, self.default_tolerance())
    }

    /// Wie `solve`, mit eigener absoluter Rangschwelle; siehe `rank_with_tolerance`.
    pub fn solve_with_tolerance(
        &self,
        b: &Vector<T, M>,
        tol: T,
    ) -> Result<Vector<T, N>, LinalgError> {
        let rank = self.rank_with_tolerance(tol)?;
        let mut y = b.data;
        let mut x = Vector::zeros();
        qr_pivoted_solve(
            self.qr.data.as_flattened(),
            M,
            N,
            &self.tau,
            &self.permutation,
            rank,
            &mut y,
            &mut x.data,
        )?;
        Ok(x)
    }

    /// Norm `‖A·x − b‖₂` der Basislösung mit der Standardschwelle.
    pub fn residual_norm(&self, b: &Vector<T, M>) -> Result<T, LinalgError> {
        let rank = self.rank();
        let mut y = b.data;
        let mut x = [T::ZERO; N];
        qr_pivoted_solve(
            self.qr.data.as_flattened(),
            M,
            N,
            &self.tau,
            &self.permutation,
            rank,
            &mut y,
            &mut x,
        )?;
        let norm = qr_residual_norm(&y, rank);
        if !norm.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        Ok(norm)
    }

    /// Obere Dreiecksmatrix `R` der pivotierten Zerlegung.
    pub fn r(&self) -> Matrix<T, N, N> {
        Matrix::from_fn(|i, j| if j >= i { self.qr[(i, j)] } else { T::ZERO })
    }

    /// Orthogonale Matrix `Q` (`M × M`).
    pub fn q(&self) -> Matrix<T, M, M> {
        let mut q = Matrix::zeros();
        for j in 0..M {
            let mut column = [T::ZERO; M];
            column[j] = T::ONE;
            qr_apply_q(self.qr.data.as_flattened(), M, N, &self.tau, &mut column);
            for i in 0..M {
                q[(i, j)] = column[i];
            }
        }
        q
    }
}

impl<T: Real, const M: usize, const N: usize> Matrix<T, M, N> {
    /// Householder-QR mit Spaltenpivotierung; erlaubt Rangmangel. Siehe [`PivotedQr`].
    pub fn qr_pivoted(&self) -> Result<PivotedQr<T, M, N>, LinalgError> {
        PivotedQr::new(self)
    }
}

#[cfg(feature = "alloc")]
mod dynamic {
    use super::*;
    use crate::linalg::{DynMatrix, DynVector};
    use alloc::{vec, vec::Vec};

    /// Dynamische Variante von [`PivotedQr`] (Feature `alloc`).
    #[derive(Debug, Clone, PartialEq)]
    pub struct DynPivotedQr<T> {
        m: usize,
        n: usize,
        qr: Vec<T>,
        tau: Vec<T>,
        permutation: Vec<usize>,
    }

    impl<T: Real> DynPivotedQr<T> {
        /// Zerlegt `a`, auch bei Rangmangel. Fehler wie bei [`PivotedQr::new`].
        pub fn new(a: &DynMatrix<T>) -> Result<Self, LinalgError> {
            let (m, n) = (a.rows(), a.cols());
            if m < n {
                return Err(LinalgError::DimensionMismatch);
            }
            let mut qr = a.as_slice().to_vec();
            let mut tau = vec![T::ZERO; n];
            let mut permutation = vec![0; n];
            qr_pivoted_decompose(&mut qr, m, n, &mut tau, &mut permutation)?;
            Ok(Self {
                m,
                n,
                qr,
                tau,
                permutation,
            })
        }

        /// Ursprüngliche Spaltenindizes in pivotierter Reihenfolge.
        pub fn permutation(&self) -> &[usize] {
            &self.permutation
        }

        /// Absolute Standardschwelle; siehe [`PivotedQr::default_tolerance`].
        pub fn default_tolerance(&self) -> T {
            qr_pivoted_tolerance(&self.qr, self.m, self.n)
        }

        /// Geschätzter numerischer Rang mit Standardschwelle.
        pub fn rank(&self) -> usize {
            let tol = self.default_tolerance();
            (0..self.n)
                .take_while(|&i| self.qr[i * self.n + i].abs() > tol)
                .count()
        }

        /// Numerischer Rang mit eigener absoluter Schwelle; Fehler bei ungültiger Schwelle.
        pub fn rank_with_tolerance(&self, tol: T) -> Result<usize, LinalgError> {
            qr_pivoted_rank(&self.qr, self.n, tol)
        }

        /// Kleinste-Quadrate-Basislösung; bei Rangmangel im Allgemeinen nicht kleinster Norm.
        pub fn solve(&self, b: &DynVector<T>) -> Result<DynVector<T>, LinalgError> {
            self.solve_with_tolerance(b, self.default_tolerance())
        }

        /// Basislösung mit eigener absoluter Rangschwelle.
        /// Fehler bei falscher Dimension, ungültiger Schwelle oder nichtendlichen Daten/Ergebnissen.
        pub fn solve_with_tolerance(
            &self,
            b: &DynVector<T>,
            tol: T,
        ) -> Result<DynVector<T>, LinalgError> {
            if b.len() != self.m {
                return Err(LinalgError::DimensionMismatch);
            }
            let rank = self.rank_with_tolerance(tol)?;
            let mut y = b.as_slice().to_vec();
            let mut x = vec![T::ZERO; self.n];
            qr_pivoted_solve(
                &self.qr,
                self.m,
                self.n,
                &self.tau,
                &self.permutation,
                rank,
                &mut y,
                &mut x,
            )?;
            Ok(DynVector::new(x))
        }

        /// Norm des Residuums der Basislösung mit Standardschwelle.
        pub fn residual_norm(&self, b: &DynVector<T>) -> Result<T, LinalgError> {
            if b.len() != self.m {
                return Err(LinalgError::DimensionMismatch);
            }
            let rank = self.rank();
            let mut y = b.as_slice().to_vec();
            let mut x = vec![T::ZERO; self.n];
            qr_pivoted_solve(
                &self.qr,
                self.m,
                self.n,
                &self.tau,
                &self.permutation,
                rank,
                &mut y,
                &mut x,
            )?;
            let norm = qr_residual_norm(&y, rank);
            if !norm.is_finite() {
                return Err(LinalgError::NotFinite);
            }
            Ok(norm)
        }

        /// Obere Dreiecksmatrix `R` (`n × n`).
        pub fn r(&self) -> DynMatrix<T> {
            let mut r = DynMatrix::zeros(self.n, self.n);
            for i in 0..self.n {
                for j in i..self.n {
                    r[(i, j)] = self.qr[i * self.n + j];
                }
            }
            r
        }

        /// Orthogonale Matrix `Q` (`m × m`).
        pub fn q(&self) -> DynMatrix<T> {
            let mut q = DynMatrix::zeros(self.m, self.m);
            let mut column = vec![T::ZERO; self.m];
            for j in 0..self.m {
                column.fill(T::ZERO);
                column[j] = T::ONE;
                qr_apply_q(&self.qr, self.m, self.n, &self.tau, &mut column);
                for i in 0..self.m {
                    q[(i, j)] = column[i];
                }
            }
            q
        }
    }

    impl<T: Real> DynMatrix<T> {
        /// Householder-QR mit Spaltenpivotierung, auch bei Rangmangel.
        pub fn qr_pivoted(&self) -> Result<DynPivotedQr<T>, LinalgError> {
            DynPivotedQr::new(self)
        }
    }
}

#[cfg(feature = "alloc")]
pub use dynamic::DynPivotedQr;
