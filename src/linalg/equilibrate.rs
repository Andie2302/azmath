//! Äquilibrierung: Zeilen- und Spaltenskalierung `Â = R·A·C` (`R`, `C` Diagonalmatrizen) zur besseren
//! Konditionierung.
//!
//! | Verfahren | Ziel |
//! |---|---|
//! | `equilibrate_ruiz` | alle Zeilen- und Spaltenmaxima (∞-Norm) von `Â` gleich 1 |
//! | `equilibrate_sinkhorn` | Zeilensummen von `|Â|` gleich 1, Spaltensummen gleich `m/n` (doppelt stochastisch bei quadratischer Matrix) |
//!
//! Das lineare System `A·x = b` wird zu `(R·A·C)·y = R·b` mit `x = C·y`
//! ([`Equilibrated::scale_rhs`], [`Equilibrated::recover_solution`]).
//!
//! * Die Faktoren werden in Gleitkomma iteriert; mit `pow2` (Standard) werden sie am Ende auf **Zweierpotenzen**
//!   gerundet. Dann ist die Skalierung exakt (kein zusätzlicher Rundungsfehler), das Ziel gilt aber nur noch
//!   bis auf einen Faktor 2 (Zeilen- und Spaltenfaktor werden je um höchstens `√2` gerundet).
//! * Nullzeilen und Nullspalten behalten den Faktor 1.
//! * Eingang mit NaN/∞ → [`LinalgError::NotFinite`]; keine Konvergenz in `max_iter` Runden →
//!   [`LinalgError::NoConvergence`]. Sinkhorn konvergiert nur, wenn die Matrix „Gesamtstütze“ hat.

use super::{LinalgError, Matrix, Vector};
use crate::Real;

/// Einstellungen für die Äquilibrierung.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EquilibrateOptions<T> {
    /// Maximale Zahl der Runden.
    pub max_iter: usize,
    /// Toleranz für die Abweichung vom Ziel (relativ).
    pub tol: T,
    /// Faktoren am Ende auf Zweierpotenzen runden (exakte Skalierung).
    pub pow2: bool,
}

impl<T: Real> Default for EquilibrateOptions<T> {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: T::EPSILON.sqrt(),
            pow2: true,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Method {
    Ruiz,
    Sinkhorn,
}

fn finite<T: Real>(x: T) -> bool {
    !x.is_nan() && !x.is_infinite()
}

/// Rundet `s > 0` auf die nächste Zweierpotenz (im Logarithmus).
fn round_pow2<T: Real>(s: T) -> T {
    T::from_f64(2.0).powf(s.log2().round())
}

/// Maximum (`sum = false`) bzw. Summe (`sum = true`) der Beträge der skalierten Zeile `i`: `rᵢ·|aᵢⱼ|·cⱼ`.
fn row_stat<T: Real>(a: &[T], cols: usize, i: usize, r: &[T], c: &[T], sum: bool) -> T {
    let mut acc = T::ZERO;
    for j in 0..cols {
        let v = a[i * cols + j].abs() * c[j];
        acc = if sum { acc + v } else { acc.max(v) };
    }
    acc * r[i]
}

/// Wie [`row_stat`] für Spalte `j`.
fn col_stat<T: Real>(
    a: &[T],
    rows: usize,
    cols: usize,
    j: usize,
    r: &[T],
    c: &[T],
    sum: bool,
) -> T {
    let mut acc = T::ZERO;
    for i in 0..rows {
        let v = a[i * cols + j].abs() * r[i];
        acc = if sum { acc + v } else { acc.max(v) };
    }
    acc * c[j]
}

/// Gemeinsamer Kern auf einer zeilenweisen `rows × cols`-Matrix. Die Eingabe bleibt unverändert und es wird
/// nichts allokiert; berechnet werden die Skalen `r` und `c` (Länge `rows` bzw. `cols`). Rückgabe: Zahl der Runden.
pub(crate) fn equilibrate<T: Real>(
    a: &[T],
    rows: usize,
    cols: usize,
    r: &mut [T],
    c: &mut [T],
    method: Method,
    opts: &EquilibrateOptions<T>,
) -> Result<usize, LinalgError> {
    if a.iter().any(|&x| !finite(x)) {
        return Err(LinalgError::NotFinite);
    }
    r.fill(T::ONE);
    c.fill(T::ONE);
    let sum = method == Method::Sinkhorn;
    let nz_rows = (0..rows)
        .filter(|&i| a[i * cols..(i + 1) * cols].iter().any(|&x| x != T::ZERO))
        .count();
    let nz_cols = (0..cols)
        .filter(|&j| (0..rows).any(|i| a[i * cols + j] != T::ZERO))
        .count();
    // Ziel der Spaltensummen bei Sinkhorn (Zeilensummen: 1)
    let col_target = if sum && nz_cols > 0 {
        T::from_f64(nz_rows as f64) / T::from_f64(nz_cols as f64)
    } else {
        T::ONE
    };

    let mut iters = 0;
    loop {
        // Abweichung vom Ziel, ohne Zwischenspeicher
        let mut dev = T::ZERO;
        for i in 0..rows {
            let st = row_stat(a, cols, i, r, c, sum);
            if st != T::ZERO {
                dev = dev.max((st - T::ONE).abs());
            }
        }
        for j in 0..cols {
            let st = col_stat(a, rows, cols, j, r, c, sum);
            if st != T::ZERO {
                dev = dev.max((st / col_target - T::ONE).abs());
            }
        }
        if dev <= opts.tol {
            break;
        }
        if iters >= opts.max_iter {
            return Err(LinalgError::NoConvergence);
        }
        iters += 1;
        // Zeilen, dann Spalten (mit den bereits neuen Zeilenfaktoren)
        for i in 0..rows {
            let st = row_stat(a, cols, i, r, c, sum);
            if st != T::ZERO {
                r[i] = if sum { r[i] / st } else { r[i] / st.sqrt() };
            }
        }
        for j in 0..cols {
            let st = col_stat(a, rows, cols, j, r, c, sum);
            if st != T::ZERO {
                c[j] = if sum {
                    c[j] * col_target / st
                } else {
                    c[j] / st.sqrt()
                };
            }
        }
    }
    if opts.pow2 {
        for x in r.iter_mut().chain(c.iter_mut()) {
            *x = round_pow2(*x);
        }
    }
    if r.iter()
        .chain(c.iter())
        .any(|&x| !finite(x) || x == T::ZERO)
    {
        return Err(LinalgError::NotFinite);
    }
    Ok(iters)
}

/// Ergebnis einer Äquilibrierung: `matrix = diag(row_scale)·A·diag(col_scale)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equilibrated<T, const M: usize, const N: usize> {
    /// Die skalierte Matrix `R·A·C`.
    pub matrix: Matrix<T, M, N>,
    /// Zeilenfaktoren (Diagonale von `R`).
    pub row_scale: Vector<T, M>,
    /// Spaltenfaktoren (Diagonale von `C`).
    pub col_scale: Vector<T, N>,
    /// Zahl der benötigten Runden.
    pub iterations: usize,
}

impl<T: Real, const M: usize, const N: usize> Equilibrated<T, M, N> {
    /// Skaliert die rechte Seite: `R·b`.
    pub fn scale_rhs(&self, b: &Vector<T, M>) -> Vector<T, M> {
        let mut out = *b;
        for i in 0..M {
            out[i] *= self.row_scale[i];
        }
        out
    }

    /// Rechnet die Lösung `y` des skalierten Systems zurück: `x = C·y`.
    pub fn recover_solution(&self, y: &Vector<T, N>) -> Vector<T, N> {
        let mut out = *y;
        for j in 0..N {
            out[j] *= self.col_scale[j];
        }
        out
    }
}

impl<T: Real, const M: usize, const N: usize> Matrix<T, M, N> {
    fn equilibrate_with(
        &self,
        method: Method,
        opts: &EquilibrateOptions<T>,
    ) -> Result<Equilibrated<T, M, N>, LinalgError> {
        let mut row_scale = Vector::<T, M>::zeros();
        let mut col_scale = Vector::<T, N>::zeros();
        let iterations = equilibrate(
            self.data.as_flattened(),
            M,
            N,
            row_scale.as_mut_slice(),
            col_scale.as_mut_slice(),
            method,
            opts,
        )?;
        let matrix = Matrix::from_fn(|i, j| row_scale[i] * self[(i, j)] * col_scale[j]);
        Ok(Equilibrated {
            matrix,
            row_scale,
            col_scale,
            iterations,
        })
    }

    /// Ruiz-Äquilibrierung: Zeilen- und Spaltenmaxima der skalierten Matrix werden 1.
    ///
    /// ```
    /// # #[cfg(any(feature = "std", feature = "libm"))] {
    /// use azmath::linalg::{EquilibrateOptions, Matrix};
    ///
    /// let a = Matrix::new([[1.0e6f64, 2.0], [3.0, 4.0e-6]]);
    /// let e = a.equilibrate_ruiz(&EquilibrateOptions::default()).unwrap();
    /// assert!(e.matrix.max_abs() <= 2.0);
    /// # }
    /// ```
    pub fn equilibrate_ruiz(
        &self,
        opts: &EquilibrateOptions<T>,
    ) -> Result<Equilibrated<T, M, N>, LinalgError> {
        self.equilibrate_with(Method::Ruiz, opts)
    }

    /// Sinkhorn-Knopp-Äquilibrierung: Zeilensummen von `|Â|` werden 1, Spaltensummen `m/n`.
    pub fn equilibrate_sinkhorn(
        &self,
        opts: &EquilibrateOptions<T>,
    ) -> Result<Equilibrated<T, M, N>, LinalgError> {
        self.equilibrate_with(Method::Sinkhorn, opts)
    }
}

#[cfg(feature = "alloc")]
mod dynamic {
    use super::{EquilibrateOptions, Method, equilibrate};
    use crate::Real;
    use crate::linalg::{DynMatrix, DynVector, LinalgError};
    use alloc::vec;

    /// Ergebnis einer Äquilibrierung dynamischer Größe, siehe [`Equilibrated`](super::Equilibrated).
    #[derive(Debug, Clone, PartialEq)]
    pub struct DynEquilibrated<T> {
        /// Die skalierte Matrix `R·A·C`.
        pub matrix: DynMatrix<T>,
        /// Zeilenfaktoren.
        pub row_scale: DynVector<T>,
        /// Spaltenfaktoren.
        pub col_scale: DynVector<T>,
        /// Zahl der benötigten Runden.
        pub iterations: usize,
    }

    impl<T: Real> DynEquilibrated<T> {
        /// Skaliert die rechte Seite: `R·b`. Fehler bei falscher Länge.
        pub fn scale_rhs(&self, b: &DynVector<T>) -> Result<DynVector<T>, LinalgError> {
            if b.len() != self.row_scale.len() {
                return Err(LinalgError::DimensionMismatch);
            }
            Ok(DynVector::new(
                b.as_slice()
                    .iter()
                    .zip(self.row_scale.as_slice())
                    .map(|(&x, &s)| x * s)
                    .collect(),
            ))
        }

        /// Rechnet die Lösung `y` des skalierten Systems zurück: `x = C·y`. Fehler bei falscher Länge.
        pub fn recover_solution(&self, y: &DynVector<T>) -> Result<DynVector<T>, LinalgError> {
            if y.len() != self.col_scale.len() {
                return Err(LinalgError::DimensionMismatch);
            }
            Ok(DynVector::new(
                y.as_slice()
                    .iter()
                    .zip(self.col_scale.as_slice())
                    .map(|(&x, &s)| x * s)
                    .collect(),
            ))
        }
    }

    impl<T: Real> DynMatrix<T> {
        fn equilibrate_with(
            &self,
            method: Method,
            opts: &EquilibrateOptions<T>,
        ) -> Result<DynEquilibrated<T>, LinalgError> {
            let (m, n) = (self.rows(), self.cols());
            let mut r = vec![T::ONE; m];
            let mut c = vec![T::ONE; n];
            let iterations = equilibrate(self.as_slice(), m, n, &mut r, &mut c, method, opts)?;
            let mut data = self.as_slice().to_vec();
            for i in 0..m {
                for j in 0..n {
                    data[i * n + j] = r[i] * data[i * n + j] * c[j];
                }
            }
            Ok(DynEquilibrated {
                matrix: DynMatrix::new(m, n, data)?,
                row_scale: DynVector::new(r),
                col_scale: DynVector::new(c),
                iterations,
            })
        }

        /// Ruiz-Äquilibrierung, siehe [`Matrix::equilibrate_ruiz`](crate::linalg::Matrix::equilibrate_ruiz).
        pub fn equilibrate_ruiz(
            &self,
            opts: &EquilibrateOptions<T>,
        ) -> Result<DynEquilibrated<T>, LinalgError> {
            self.equilibrate_with(Method::Ruiz, opts)
        }

        /// Sinkhorn-Knopp-Äquilibrierung, siehe [`Matrix::equilibrate_sinkhorn`](crate::linalg::Matrix::equilibrate_sinkhorn).
        pub fn equilibrate_sinkhorn(
            &self,
            opts: &EquilibrateOptions<T>,
        ) -> Result<DynEquilibrated<T>, LinalgError> {
            self.equilibrate_with(Method::Sinkhorn, opts)
        }
    }
}

#[cfg(feature = "alloc")]
pub use dynamic::DynEquilibrated;
