use core::cell::RefCell;

use super::{
    MultigridError, MultigridOptions, check_options, grid_levels, oned, twod, work_len_1d,
    work_len_2d,
};
use azmath_core::Real;
use azmath_linalg::LinearOperator;
use azmath_linalg::iterative::Preconditioner;

/// Der 1D-Operator `−u''` als Differenzenstern `(2uᵢ − uᵢ₋₁ − uᵢ₊₁)/h²` mit `h = 1/(n+1)` und Nullrand,
/// als matrixfreier [`LinearOperator`]. Gedacht für [`cg`](azmath_linalg::iterative::cg) mit dem
/// [`MultigridPreconditioner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Poisson1d {
    n: usize,
}

impl Poisson1d {
    /// Operator für `n` innere Punkte (beliebiges `n ≥ 1`).
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl<T: Real> LinearOperator<T> for Poisson1d {
    fn rows(&self) -> usize {
        self.n
    }
    fn cols(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        let n = self.n;
        let h = T::ONE / T::from_f64((n + 1) as f64);
        let inv_h2 = T::ONE / (h * h);
        for i in 0..n {
            let l = if i > 0 { x[i - 1] } else { T::ZERO };
            let r = if i + 1 < n { x[i + 1] } else { T::ZERO };
            y[i] = (T::from_f64(2.0) * x[i] - l - r) * inv_h2;
        }
    }
}

/// Der 2D-Operator `−Δ` als 5-Punkte-Stern `(4uᵢⱼ − Nachbarn)/h²` auf dem `n × n`-Gitter mit `h = 1/(n+1)`
/// und Nullrand (zeilenweise gespeichert), als matrixfreier [`LinearOperator`] mit `n²` Zeilen und Spalten.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Poisson2d {
    n: usize,
}

impl Poisson2d {
    /// Operator für ein `n × n`-Gitter (beliebiges `n ≥ 1`).
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl<T: Real> LinearOperator<T> for Poisson2d {
    fn rows(&self) -> usize {
        self.n * self.n
    }
    fn cols(&self) -> usize {
        self.n * self.n
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        let n = self.n;
        let h = T::ONE / T::from_f64((n + 1) as f64);
        let inv_h2 = T::ONE / (h * h);
        let four = T::from_f64(4.0);
        for i in 0..n {
            for j in 0..n {
                let up = if i > 0 { x[(i - 1) * n + j] } else { T::ZERO };
                let down = if i + 1 < n {
                    x[(i + 1) * n + j]
                } else {
                    T::ZERO
                };
                let left = if j > 0 { x[i * n + j - 1] } else { T::ZERO };
                let right = if j + 1 < n { x[i * n + j + 1] } else { T::ZERO };
                y[i * n + j] = (four * x[i * n + j] - up - down - left - right) * inv_h2;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dim {
    One,
    Two,
}

/// Ein Multigrid-Zyklus als **Vorkonditionierer** `z = B·r ≈ A⁻¹·r` für den Poisson-Operator
/// ([`Poisson1d`] bzw. [`Poisson2d`]), geeignet für [`pcg`](azmath_linalg::iterative::pcg).
///
/// Der Zyklus startet bei 0 und ist mit gleich vielen Vor- und Nachglättungsschritten (Gauß-Seidel vorwärts
/// bzw. rückwärts, oder Jacobi) **symmetrisch positiv definit**, wie CG es verlangt. Deshalb ist
/// `pre_smooth == post_smooth` Pflicht. Mit diesem Vorkonditionierer braucht CG auf dem Poisson-Problem eine
/// von der Gittergröße praktisch unabhängige Zahl von Iterationen (typisch 6 bis 10).
///
/// Der Vorkonditionierer leiht sich seinen Arbeitsspeicher (`work_len_*`) und ist nicht
/// threadsicher (innere Veränderlichkeit über `RefCell`).
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath_linalg::iterative::{pcg, IterOptions};
/// use azmath_multigrid::{work_len_2d, MultigridOptions, MultigridPreconditioner, Poisson2d};
///
/// const N: usize = 31;
/// let a = Poisson2d::new(N);
/// let mut mg_work = [0.0f64; work_len_2d(N)];
/// let mg = MultigridPreconditioner::new_2d(N, MultigridOptions::default(), &mut mg_work).unwrap();
///
/// let b = [1.0; N * N];
/// let mut x = [0.0; N * N];
/// let mut cg_work = [0.0; 4 * N * N];
/// let r = pcg(&a, &mg, &b, &mut x, IterOptions::new(1e-10, 100), &mut cg_work).unwrap();
/// assert!(r.iterations <= 15); // ohne Vorkonditionierer: rund 100
/// # }
/// ```
pub struct MultigridPreconditioner<'a, T> {
    dim: Dim,
    n: usize,
    opts: MultigridOptions<T>,
    work: RefCell<&'a mut [T]>,
}

impl<'a, T: Real> MultigridPreconditioner<'a, T> {
    fn new(
        dim: Dim,
        n: usize,
        opts: MultigridOptions<T>,
        work: &'a mut [T],
    ) -> Result<Self, MultigridError> {
        if grid_levels(n).is_none() {
            return Err(MultigridError::InvalidGridSize);
        }
        let need = match dim {
            Dim::One => work_len_1d(n),
            Dim::Two => work_len_2d(n),
        };
        if work.len() < need || opts.pre_smooth != opts.post_smooth {
            return Err(MultigridError::InvalidInput);
        }
        check_options(&opts)?;
        Ok(Self {
            dim,
            n,
            opts,
            work: RefCell::new(work),
        })
    }

    /// Vorkonditionierer für [`Poisson1d`] mit `n = 2ᵏ − 1` Punkten; `work` mindestens
    /// [`work_len_1d`]`(n)` Elemente.
    ///
    /// Fehler: [`MultigridError::InvalidGridSize`], [`MultigridError::InvalidInput`]
    /// (zu kurzer `work`, `pre_smooth ≠ post_smooth` oder ungültiges Jacobi-Gewicht).
    pub fn new_1d(
        n: usize,
        opts: MultigridOptions<T>,
        work: &'a mut [T],
    ) -> Result<Self, MultigridError> {
        Self::new(Dim::One, n, opts, work)
    }

    /// Vorkonditionierer für [`Poisson2d`] mit einem `n × n`-Gitter, `n = 2ᵏ − 1`; `work` mindestens
    /// [`work_len_2d`]`(n)` Elemente. Fehler wie bei [`new_1d`](Self::new_1d).
    pub fn new_2d(
        n: usize,
        opts: MultigridOptions<T>,
        work: &'a mut [T],
    ) -> Result<Self, MultigridError> {
        Self::new(Dim::Two, n, opts, work)
    }
}

impl<T: Real> Preconditioner<T> for MultigridPreconditioner<'_, T> {
    fn apply(&self, r: &[T], z: &mut [T]) {
        let mut guard = self.work.borrow_mut();
        let work: &mut [T] = &mut guard;
        let (n, m) = match self.dim {
            Dim::One => (self.n, self.n),
            Dim::Two => (self.n, self.n * self.n),
        };
        let len = match self.dim {
            Dim::One => work_len_1d(n),
            Dim::Two => work_len_2d(n),
        };
        let buf = &mut work[..len];
        buf[..m].fill(T::ZERO);
        buf[m..2 * m].copy_from_slice(r);
        match self.dim {
            Dim::One => oned::run_cycle(buf, n, &self.opts),
            Dim::Two => twod::run_cycle(buf, n, &self.opts),
        }
        z.copy_from_slice(&buf[..m]);
    }
}
