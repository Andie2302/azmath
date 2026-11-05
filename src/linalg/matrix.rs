use core::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use super::iterative::{self, IterOptions, IterResult, Preconditioner};
use super::kernel::{check_diagonal, triangular_solve};
use super::{Cholesky, LinalgError, LuDecomposition, QrDecomposition, Vector};
use crate::Real;

/// Matrix fester Größe mit `R` Zeilen und `C` Spalten (liegt auf dem Stack).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Matrix<T, const R: usize, const C: usize> {
    pub(crate) data: [[T; C]; R],
}

impl<T: Real, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Erzeugt eine Matrix aus Zeilen.
    pub const fn new(data: [[T; C]; R]) -> Self {
        Self { data }
    }

    /// Die Nullmatrix.
    pub fn zeros() -> Self {
        Self {
            data: [[T::ZERO; C]; R],
        }
    }

    /// Erzeugt eine Matrix, deren Element `(i, j)` den Wert `f(i, j)` hat.
    pub fn from_fn(mut f: impl FnMut(usize, usize) -> T) -> Self {
        let mut m = Self::zeros();
        for i in 0..R {
            for j in 0..C {
                m.data[i][j] = f(i, j);
            }
        }
        m
    }

    /// Zugriff auf die zugrunde liegenden Zeilen.
    pub fn as_array(&self) -> &[[T; C]; R] {
        &self.data
    }

    /// Anzahl der Zeilen.
    pub const fn rows(&self) -> usize {
        R
    }

    /// Anzahl der Spalten.
    pub const fn cols(&self) -> usize {
        C
    }

    /// Die transponierte Matrix.
    pub fn transpose(&self) -> Matrix<T, C, R> {
        Matrix::<T, C, R>::from_fn(|i, j| self.data[j][i])
    }

    /// Multipliziert jedes Element mit `s`.
    pub fn scale(self, s: T) -> Self {
        let mut out = self;
        for row in out.data.iter_mut() {
            for v in row.iter_mut() {
                *v *= s;
            }
        }
        out
    }

    /// Größter Betrag aller Elemente (Maximumnorm); NaN wird weitergegeben.
    pub fn max_abs(&self) -> T {
        super::norm::max_abs(self.data.as_flattened())
    }
}

impl<T: Real, const N: usize> Matrix<T, N, N> {
    /// Die Einheitsmatrix.
    pub fn identity() -> Self {
        Self::from_fn(|i, j| if i == j { T::ONE } else { T::ZERO })
    }

    /// Spur (Summe der Diagonalelemente).
    pub fn trace(&self) -> T {
        let mut s = T::ZERO;
        for i in 0..N {
            s += self.data[i][i];
        }
        s
    }

    /// LU-Zerlegung mit Teilpivotisierung.
    pub fn lu(&self) -> Result<LuDecomposition<T, N>, LinalgError> {
        LuDecomposition::new(self)
    }

    /// Löst `self · x = b`.
    pub fn solve(&self, b: &Vector<T, N>) -> Result<Vector<T, N>, LinalgError> {
        Ok(self.lu()?.solve(b))
    }

    /// Dreieckslöser ohne Diagonalprüfung (Kern für alle `solve_*`-Varianten).
    fn triangular(
        &self,
        b: &Vector<T, N>,
        upper: bool,
        transpose: bool,
        unit: bool,
    ) -> Vector<T, N> {
        let mut x = *b;
        triangular_solve(
            self.data.as_flattened(),
            N,
            upper,
            transpose,
            unit,
            &mut x.data,
        );
        x
    }

    /// Wie `triangular`, prüft vorher aber, dass keine Diagonale 0 oder NaN ist.
    fn triangular_checked(
        &self,
        b: &Vector<T, N>,
        upper: bool,
        transpose: bool,
    ) -> Result<Vector<T, N>, LinalgError> {
        check_diagonal(self.data.as_flattened(), N)?;
        Ok(self.triangular(b, upper, transpose, false))
    }

    /// Löst `L·x = b` für die untere Dreiecksmatrix `self` (Vorwärtssubstitution).
    ///
    /// Die Einträge oberhalb der Diagonalen werden ignoriert. Fehler:
    /// [`LinalgError::Singular`], wenn ein Diagonalelement 0 oder NaN ist.
    pub fn solve_lower(&self, b: &Vector<T, N>) -> Result<Vector<T, N>, LinalgError> {
        self.triangular_checked(b, false, false)
    }

    /// Löst `U·x = b` für die obere Dreiecksmatrix `self` (Rückwärtssubstitution).
    ///
    /// Die Einträge unterhalb der Diagonalen werden ignoriert. Fehler:
    /// [`LinalgError::Singular`], wenn ein Diagonalelement 0 oder NaN ist.
    pub fn solve_upper(&self, b: &Vector<T, N>) -> Result<Vector<T, N>, LinalgError> {
        self.triangular_checked(b, true, false)
    }

    /// Wie [`solve_lower`](Self::solve_lower), aber mit Einheitsdiagonale: die Diagonale von
    /// `self` wird nicht gelesen und als 1 angenommen (wie bei `L` der LU-Zerlegung).
    pub fn solve_lower_unit(&self, b: &Vector<T, N>) -> Vector<T, N> {
        // Mit Einheitsdiagonale ist das System immer lösbar.
        self.triangular(b, false, false, true)
    }

    /// Wie [`solve_upper`](Self::solve_upper), aber mit Einheitsdiagonale.
    pub fn solve_upper_unit(&self, b: &Vector<T, N>) -> Vector<T, N> {
        self.triangular(b, true, false, true)
    }

    /// Löst `Lᵀ·x = b`, wobei `self` die untere Dreiecksmatrix `L` ist (ohne `L` zu transponieren).
    ///
    /// Das ist der zweite Schritt eines Cholesky-Lösers.
    pub fn solve_lower_transposed(&self, b: &Vector<T, N>) -> Result<Vector<T, N>, LinalgError> {
        self.triangular_checked(b, false, true)
    }

    /// Löst `Uᵀ·x = b`, wobei `self` die obere Dreiecksmatrix `U` ist (ohne `U` zu transponieren).
    pub fn solve_upper_transposed(&self, b: &Vector<T, N>) -> Result<Vector<T, N>, LinalgError> {
        self.triangular_checked(b, true, true)
    }

    /// Löst `self · x = b` für symmetrisch positiv definite Matrizen mit dem CG-Verfahren,
    /// Start bei `x = 0`. Liefert die Lösung und Angaben zur Iteration.
    ///
    /// Für kleine dichte Systeme ist [`solve_spd`](Self::solve_spd) (Cholesky) meist besser;
    /// CG lohnt sich vor allem matrixfrei über [`iterative::cg`] für große Systeme.
    pub fn solve_cg(
        &self,
        b: &Vector<T, N>,
        opts: IterOptions<T>,
    ) -> Result<(Vector<T, N>, IterResult<T>), LinalgError> {
        self.solve_pcg(&iterative::Identity, b, opts)
    }

    /// Wie [`solve_cg`](Self::solve_cg), aber mit Vorkonditionierer `m`
    /// (zum Beispiel [`iterative::Jacobi`]).
    pub fn solve_pcg<M: Preconditioner<T>>(
        &self,
        m: &M,
        b: &Vector<T, N>,
        opts: IterOptions<T>,
    ) -> Result<(Vector<T, N>, IterResult<T>), LinalgError> {
        let mut x = Vector::<T, N>::zeros();
        let mut work = [[T::ZERO; N]; 4];
        let res = iterative::pcg(self, m, &b.data, &mut x.data, opts, work.as_flattened_mut())?;
        Ok((x, res))
    }

    /// Löst `self · x = b` für beliebige reguläre (auch nichtsymmetrische) Matrizen mit dem
    /// BiCGSTAB-Verfahren, Start bei `x = 0`. Liefert die Lösung und Angaben zur Iteration.
    pub fn solve_bicgstab(
        &self,
        b: &Vector<T, N>,
        opts: IterOptions<T>,
    ) -> Result<(Vector<T, N>, IterResult<T>), LinalgError> {
        self.solve_pbicgstab(&iterative::Identity, b, opts)
    }

    /// Wie [`solve_bicgstab`](Self::solve_bicgstab), aber mit Vorkonditionierer `m`.
    pub fn solve_pbicgstab<M: Preconditioner<T>>(
        &self,
        m: &M,
        b: &Vector<T, N>,
        opts: IterOptions<T>,
    ) -> Result<(Vector<T, N>, IterResult<T>), LinalgError> {
        let mut x = Vector::<T, N>::zeros();
        let mut work = [[T::ZERO; N]; 8];
        let res =
            iterative::pbicgstab(self, m, &b.data, &mut x.data, opts, work.as_flattened_mut())?;
        Ok((x, res))
    }

    /// Cholesky-Zerlegung für symmetrisch positiv definite Matrizen.
    pub fn cholesky(&self) -> Result<Cholesky<T, N>, LinalgError> {
        Cholesky::new(self)
    }

    /// Löst `self · x = b` für symmetrisch positiv definite Matrizen (per Cholesky).
    pub fn solve_spd(&self, b: &Vector<T, N>) -> Result<Vector<T, N>, LinalgError> {
        Ok(self.cholesky()?.solve(b))
    }

    /// Determinante. Für singuläre Matrizen `0`, bei NaN/Unendlich NaN.
    pub fn determinant(&self) -> T {
        match self.lu() {
            Ok(lu) => lu.determinant(),
            Err(LinalgError::NotFinite) => T::NAN,
            Err(_) => T::ZERO,
        }
    }

    /// Inverse Matrix.
    pub fn inverse(&self) -> Result<Self, LinalgError> {
        Ok(self.lu()?.inverse())
    }
}

impl<T: Real, const M: usize, const N: usize> Matrix<T, M, N> {
    /// QR-Zerlegung (Householder); erfordert `M >= N` und vollen Spaltenrang.
    pub fn qr(&self) -> Result<QrDecomposition<T, M, N>, LinalgError> {
        QrDecomposition::new(self)
    }

    /// Kleinste-Quadrate-Lösung von `min ‖self · x − b‖₂` (per QR).
    /// [`LinalgError::NotFinite`] auch bei ungültiger rechter Seite oder nichtendlichem Ergebnis.
    pub fn least_squares(&self, b: &Vector<T, M>) -> Result<Vector<T, N>, LinalgError> {
        self.qr()?.try_solve(b)
    }
}

impl<T, const R: usize, const C: usize> Index<(usize, usize)> for Matrix<T, R, C> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &T {
        &self.data[i][j]
    }
}

impl<T, const R: usize, const C: usize> IndexMut<(usize, usize)> for Matrix<T, R, C> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        &mut self.data[i][j]
    }
}

impl<T: Real, const R: usize, const C: usize> Add for Matrix<T, R, C> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        for i in 0..R {
            for j in 0..C {
                self.data[i][j] += rhs.data[i][j];
            }
        }
        self
    }
}

impl<T: Real, const R: usize, const C: usize> Sub for Matrix<T, R, C> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        for i in 0..R {
            for j in 0..C {
                self.data[i][j] -= rhs.data[i][j];
            }
        }
        self
    }
}

impl<T: Real, const R: usize, const C: usize> Neg for Matrix<T, R, C> {
    type Output = Self;
    fn neg(self) -> Self {
        self.scale(-T::ONE)
    }
}

impl<T: Real, const R: usize, const C: usize> Mul<T> for Matrix<T, R, C> {
    type Output = Self;
    fn mul(self, s: T) -> Self {
        self.scale(s)
    }
}

impl<T: Real, const R: usize, const C: usize> Mul<Vector<T, C>> for Matrix<T, R, C> {
    type Output = Vector<T, R>;
    fn mul(self, v: Vector<T, C>) -> Vector<T, R> {
        let mut out = Vector::<T, R>::zeros();
        for i in 0..R {
            let mut s = T::ZERO;
            for j in 0..C {
                s += self.data[i][j] * v.data[j];
            }
            out.data[i] = s;
        }
        out
    }
}

impl<T: Real, const R: usize, const C: usize, const K: usize> Mul<Matrix<T, C, K>>
    for Matrix<T, R, C>
{
    type Output = Matrix<T, R, K>;
    fn mul(self, rhs: Matrix<T, C, K>) -> Matrix<T, R, K> {
        let mut out = Matrix::<T, R, K>::zeros();
        for i in 0..R {
            for j in 0..K {
                let mut s = T::ZERO;
                for k in 0..C {
                    s += self.data[i][k] * rhs.data[k][j];
                }
                out.data[i][j] = s;
            }
        }
        out
    }
}
