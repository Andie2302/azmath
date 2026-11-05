use super::super::{LinalgError, LinearOperator};
use super::{
    Identity, IterOptions, IterResult, Preconditioner, converged, dot, finite, norm, residual,
    scaled_axpy, validate,
};
use crate::Real;

/// BiCGSTAB (stabilisiertes Bi-konjugiertes Gradientenverfahren) für `A·x = b` mit beliebiger,
/// auch **nichtsymmetrischer** regulärer Matrix `A`.
///
/// - `a`: quadratischer Operator `n × n` (nur `A·x` wird benötigt, kein `Aᵀ·x`).
/// - `b`: rechte Seite, `x`: Startnäherung beim Aufruf und Lösung beim Rückgabe.
/// - `work`: Arbeitsspeicher mit mindestens `8·n` Elementen.
///
/// Anders als CG konvergiert BiCGSTAB nicht zwingend monoton und kann in seltenen Fällen
/// zusammenbrechen. Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NotFinite`],
/// [`LinalgError::Breakdown`] (Division durch 0 im Verfahren), [`LinalgError::NoConvergence`].
/// Bei `NoConvergence` und `Breakdown` enthält `x` die letzte Näherung.
/// [`LinalgError::InvalidInput`] bei negativer oder nichtendlicher Toleranz.
/// Vor jeder erfolgreichen Rückgabe wird das wahre Residuum neu berechnet.
/// Bei Abweichung vom rekursiven Residuum wird im verbleibenden Budget neu gestartet.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::linalg::iterative::{bicgstab, IterOptions};
/// use azmath::linalg::Matrix;
///
/// // nichtsymmetrisch
/// let a = Matrix::new([[4.0f64, 1.0], [2.0, 3.0]]);
/// let b = [1.0, 2.0];
/// let mut x = [0.0; 2];
/// let mut work = [0.0; 16];
/// bicgstab(&a, &b, &mut x, IterOptions::new(1e-12, 50), &mut work).unwrap();
/// assert!((x[0] - 0.1).abs() < 1e-10 && (x[1] - 0.6).abs() < 1e-10);
/// # }
/// ```
pub fn bicgstab<T: Real, A: LinearOperator<T> + ?Sized>(
    a: &A,
    b: &[T],
    x: &mut [T],
    opts: IterOptions<T>,
    work: &mut [T],
) -> Result<IterResult<T>, LinalgError> {
    pbicgstab(a, &Identity, b, x, opts, work)
}

/// Vorkonditioniertes BiCGSTAB (rechts vorkonditioniert) für `A·x = b`.
///
/// Wie [`bicgstab`], zusätzlich mit dem Vorkonditionierer `m` (siehe [`Preconditioner`]); der
/// Vorkonditionierer muss hier weder symmetrisch noch positiv definit sein.
pub fn pbicgstab<T, A, M>(
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
    if a.rows() != n || a.cols() != n || x.len() != n || work.len() < 8 * n {
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
    let (r_hat, rest) = rest.split_at_mut(n);
    let (p, rest) = rest.split_at_mut(n);
    let (v, rest) = rest.split_at_mut(n);
    let (s, rest) = rest.split_at_mut(n);
    let (t, rest) = rest.split_at_mut(n);
    let (p_hat, rest) = rest.split_at_mut(n);
    let s_hat = &mut rest[..n];

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
    r_hat.copy_from_slice(r);
    p.fill(T::ZERO);
    v.fill(T::ZERO);
    let (mut rho_old, mut alpha, mut omega) = (T::ONE, T::ONE, T::ONE);

    for k in 1..=opts.max_iter {
        let rho = dot(r_hat, r);
        if !rho.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        if rho == T::ZERO {
            return Err(LinalgError::Breakdown);
        }
        let beta = (rho / rho_old) * (alpha / omega);
        if !beta.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }
        finite(p)?;
        m.apply(p, p_hat);
        finite(p_hat)?;
        a.apply(p_hat, v);
        finite(v)?;
        let den = dot(r_hat, v);
        if !den.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        if den == T::ZERO {
            return Err(LinalgError::Breakdown);
        }
        alpha = rho / den;
        if !alpha.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }
        let snorm = norm(s);
        if !snorm.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        if snorm <= opts.tol {
            scaled_axpy(alpha, bnorm, p_hat, x);
            let actual = residual(a, b, x, r)?;
            if converged(actual, bnorm, opts.tol) {
                return Ok(IterResult {
                    iterations: k,
                    residual_norm: actual,
                });
            }
            for ri in r.iter_mut() {
                *ri /= bnorm;
            }
            r_hat.copy_from_slice(r);
            p.fill(T::ZERO);
            v.fill(T::ZERO);
            (rho_old, alpha, omega) = (T::ONE, T::ONE, T::ONE);
            continue;
        }
        m.apply(s, s_hat);
        finite(s_hat)?;
        a.apply(s_hat, t);
        finite(t)?;
        let tt = dot(t, t);
        if !tt.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        if tt == T::ZERO {
            return Err(LinalgError::Breakdown);
        }
        omega = dot(t, s) / tt;
        if !omega.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        if omega == T::ZERO {
            return Err(LinalgError::Breakdown);
        }
        scaled_axpy(alpha, bnorm, p_hat, x);
        scaled_axpy(omega, bnorm, s_hat, x);
        finite(x)?;
        for i in 0..n {
            r[i] = s[i] - omega * t[i];
        }
        let rn = norm(r);
        if !rn.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        if rn <= opts.tol || k == opts.max_iter {
            let actual = residual(a, b, x, s)?;
            if converged(actual, bnorm, opts.tol) {
                return Ok(IterResult {
                    iterations: k,
                    residual_norm: actual,
                });
            }
            if k == opts.max_iter {
                return Err(LinalgError::NoConvergence);
            }
            for (ri, si) in r.iter_mut().zip(s.iter()) {
                *ri = *si / bnorm;
            }
            r_hat.copy_from_slice(r);
            p.fill(T::ZERO);
            v.fill(T::ZERO);
            (rho_old, alpha, omega) = (T::ONE, T::ONE, T::ONE);
        } else {
            rho_old = rho;
        }
    }
    Err(LinalgError::NoConvergence)
}
