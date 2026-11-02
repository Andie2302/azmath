//! Matrixfreie lineare Operatoren: alles, was `y = A·x` berechnen kann.

use core::marker::PhantomData;

use super::Matrix;
use azmath_core::Real;

/// Ein linearer Operator `A` mit `rows() × cols()` Einträgen, der auf Slices wirkt.
///
/// Iterative Löser wie [`cg`](crate::iterative::cg) benötigen die Matrix nie explizit,
/// sondern nur das Produkt `A·x`. Damit lassen sich dichte Matrizen, Stencils (zum Beispiel das
/// diskrete Laplace) oder beliebige Closures verwenden, ohne Speicher für die Matrix.
pub trait LinearOperator<T: Real> {
    /// Anzahl der Zeilen (Länge von `y`).
    fn rows(&self) -> usize;
    /// Anzahl der Spalten (Länge von `x`).
    fn cols(&self) -> usize;
    /// Berechnet `y = A·x`. Es gilt `x.len() == cols()` und `y.len() == rows()`.
    fn apply(&self, x: &[T], y: &mut [T]);
}

impl<T: Real, A: LinearOperator<T> + ?Sized> LinearOperator<T> for &A {
    fn rows(&self) -> usize {
        (**self).rows()
    }
    fn cols(&self) -> usize {
        (**self).cols()
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        (**self).apply(x, y)
    }
}

impl<T: Real, const R: usize, const C: usize> LinearOperator<T> for Matrix<T, R, C> {
    fn rows(&self) -> usize {
        R
    }
    fn cols(&self) -> usize {
        C
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        for (yi, row) in y.iter_mut().zip(self.data.iter()) {
            let mut s = T::ZERO;
            for (a, xj) in row.iter().zip(x.iter()) {
                s += *a * *xj;
            }
            *yi = s;
        }
    }
}

/// Ein Operator, der durch eine Closure `|x, y| { y = A·x }` gegeben ist.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath_linalg::{FnOperator, LinearOperator};
///
/// // Verdopplung als Operator: y = 2·x
/// let op = FnOperator::new(3, 3, |x: &[f64], y: &mut [f64]| {
///     for (yi, xi) in y.iter_mut().zip(x) {
///         *yi = 2.0 * xi;
///     }
/// });
/// let mut y = [0.0; 3];
/// op.apply(&[1.0, 2.0, 3.0], &mut y);
/// assert_eq!(y, [2.0, 4.0, 6.0]);
/// # }
/// ```
pub struct FnOperator<T, F> {
    rows: usize,
    cols: usize,
    f: F,
    _marker: PhantomData<fn(T)>,
}

impl<T: Real, F: Fn(&[T], &mut [T])> FnOperator<T, F> {
    /// Erzeugt einen `rows × cols`-Operator aus der Closure `f(x, y)`, die `y = A·x` setzt.
    pub fn new(rows: usize, cols: usize, f: F) -> Self {
        Self {
            rows,
            cols,
            f,
            _marker: PhantomData,
        }
    }
}

impl<T: Real, F: Fn(&[T], &mut [T])> LinearOperator<T> for FnOperator<T, F> {
    fn rows(&self) -> usize {
        self.rows
    }
    fn cols(&self) -> usize {
        self.cols
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        (self.f)(x, y)
    }
}
