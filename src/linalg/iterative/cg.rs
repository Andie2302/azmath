use super::super::{LinalgError, LinearOperator};
use super::{
    Identity, IterOptions, IterResult, Preconditioner, axpy, converged, dot, finite, norm,
    residual, scaled_axpy, validate,
};
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
/// [`LinalgError::InvalidInput`] bei negativer oder nichtendlicher Toleranz.
/// Erfolg und `residual_norm` werden mit dem neu berechneten Residuum bestätigt.
/// Eine dabei erkannte Abweichung von der Rekursion führt zu einem Neustart im verbleibenden Budget.
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
    validate(b, x, opts)?;
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

    let bnorm = norm(b);
    if !bnorm.is_finite() {
        return Err(LinalgError::NotFinite);
    }
    if bnorm == T::ZERO {
        x.fill(T::ZERO);
    }
    let rn = residual(a, b, x, r)?;
    if converged(rn, bnorm, opts.tol) {
        return Ok(IterResult {
            iterations: 0,
            residual_norm: rn,
        });
    }
    if opts.max_iter == 0 || bnorm == T::ZERO {
        return Err(LinalgError::NoConvergence);
    }
    for ri in r.iter_mut() {
        *ri /= bnorm;
    }
    m.apply(r, z);
    finite(z)?;
    p.copy_from_slice(z);
    let mut rz = dot(r, z);
    if !rz.is_finite() {
        return Err(LinalgError::NotFinite);
    }
    if rz <= T::ZERO {
        return Err(LinalgError::NotPositiveDefinite);
    }

    for k in 1..=opts.max_iter {
        a.apply(p, q);
        finite(q)?;
        let pq = dot(p, q);
        if !pq.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        if pq <= T::ZERO {
            return Err(LinalgError::NotPositiveDefinite);
        }
        let alpha = rz / pq;
        if !alpha.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        scaled_axpy(alpha, bnorm, p, x);
        finite(x)?;
        axpy(-alpha, q, r);
        let rn = norm(r);
        if !rn.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        let mut restart = false;
        if rn <= opts.tol || k == opts.max_iter {
            let actual = residual(a, b, x, q)?;
            if converged(actual, bnorm, opts.tol) {
                return Ok(IterResult {
                    iterations: k,
                    residual_norm: actual,
                });
            }
            if k == opts.max_iter {
                return Err(LinalgError::NoConvergence);
            }
            for (ri, qi) in r.iter_mut().zip(q.iter()) {
                *ri = *qi / bnorm;
            }
            restart = true;
        }
        m.apply(r, z);
        finite(z)?;
        let rz_new = dot(r, z);
        if !rz_new.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        if rz_new <= T::ZERO {
            return Err(LinalgError::NotPositiveDefinite);
        }
        let beta = if restart { T::ZERO } else { rz_new / rz };
        if !beta.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        rz = rz_new;
        for (pi, zi) in p.iter_mut().zip(z.iter()) {
            *pi = *zi + beta * *pi;
        }
        finite(p)?;
    }
    Err(LinalgError::NoConvergence)
}
