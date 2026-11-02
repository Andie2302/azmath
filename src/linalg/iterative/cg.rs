use super::super::{LinalgError, LinearOperator};
use super::{Identity, IterOptions, IterResult, Preconditioner, axpy, dot};
use crate::Real;

/// Verfahren der konjugierten Gradienten (CG) für `A·x = b` mit symmetrisch positiv
/// definitem `A`.
///
/// - `a`: quadratischer Operator `n × n` (nur `A·x` wird benötigt).
/// - `b`: rechte Seite, `x`: Startnäherung beim Aufruf und Lösung beim Rückgabe.
/// - `work`: Arbeitsspeicher mit mindestens `4·n` Elementen.
///
/// Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NotPositiveDefinite`] (der
/// Operator ist nicht SPD), [`LinalgError::NotFinite`], [`LinalgError::NoConvergence`]. Bei
/// `NoConvergence` enthält `x` die letzte Näherung.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::linalg::iterative::{cg, IterOptions};
/// use azmath::linalg::Matrix;
///
/// let a = Matrix::new([[4.0f64, 1.0], [1.0, 3.0]]);
/// let b = [1.0, 2.0];
/// let mut x = [0.0; 2];
/// let mut work = [0.0; 8];
/// let r = cg(&a, &b, &mut x, IterOptions::new(1e-12, 50), &mut work).unwrap();
/// assert!(r.iterations <= 2);
/// assert!((x[0] - 1.0 / 11.0).abs() < 1e-10 && (x[1] - 7.0 / 11.0).abs() < 1e-10);
/// # }
/// ```
pub fn cg<T: Real, A: LinearOperator<T> + ?Sized>(
    a: &A,
    b: &[T],
    x: &mut [T],
    opts: IterOptions<T>,
    work: &mut [T],
) -> Result<IterResult<T>, LinalgError> {
    pcg(a, &Identity, b, x, opts, work)
}

/// Vorkonditioniertes CG (PCG) für `A·x = b` mit SPD-Operator `A` und SPD-Vorkonditionierer `M`.
///
/// Wie [`cg`], zusätzlich mit dem Vorkonditionierer `m` (siehe [`Preconditioner`]).
pub fn pcg<T, A, M>(
    a: &A,
    m: &M,
    b: &[T],
    x: &mut [T],
    opts: IterOptions<T>,
    work: &mut [T],
) -> Result<IterResult<T>, LinalgError>
where
    T: Real,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = b.len();
    if a.rows() != n || a.cols() != n || x.len() != n || work.len() < 4 * n {
        return Err(LinalgError::DimensionMismatch);
    }
    if n == 0 {
        return Ok(IterResult {
            iterations: 0,
            residual_norm: T::ZERO,
        });
    }

    let (r, rest) = work.split_at_mut(n);
    let (z, rest) = rest.split_at_mut(n);
    let (p, rest) = rest.split_at_mut(n);
    let q = &mut rest[..n];

    let b2 = dot(b, b);
    if b2.is_nan() || b2.is_infinite() {
        return Err(LinalgError::NotFinite);
    }
    if b2 == T::ZERO {
        // A·x = 0 hat (für regulär A) nur die Lösung x = 0.
        x.fill(T::ZERO);
        return Ok(IterResult {
            iterations: 0,
            residual_norm: T::ZERO,
        });
    }
    let stop = opts.tol * opts.tol * b2;

    // r = b − A·x
    a.apply(x, r);
    for (ri, bi) in r.iter_mut().zip(b) {
        *ri = *bi - *ri;
    }
    let mut r2 = dot(r, r);
    if r2.is_nan() || r2.is_infinite() {
        return Err(LinalgError::NotFinite);
    }

    m.apply(r, z);
    p.copy_from_slice(z);
    let mut rz = dot(r, z);

    for k in 0..opts.max_iter {
        if r2 <= stop {
            return Ok(IterResult {
                iterations: k,
                residual_norm: r2.sqrt(),
            });
        }
        a.apply(p, q);
        let pq = dot(p, q);
        if pq.is_nan() || pq.is_infinite() {
            return Err(LinalgError::NotFinite);
        }
        if pq <= T::ZERO {
            return Err(LinalgError::NotPositiveDefinite);
        }
        let alpha = rz / pq;
        axpy(alpha, p, x);
        axpy(-alpha, q, r);
        r2 = dot(r, r);

        m.apply(r, z);
        let rz_new = dot(r, z);
        let beta = rz_new / rz;
        rz = rz_new;
        for (pi, zi) in p.iter_mut().zip(z.iter()) {
            *pi = *zi + beta * *pi;
        }
    }

    if r2 <= stop {
        Ok(IterResult {
            iterations: opts.max_iter,
            residual_norm: r2.sqrt(),
        })
    } else {
        Err(LinalgError::NoConvergence)
    }
}
