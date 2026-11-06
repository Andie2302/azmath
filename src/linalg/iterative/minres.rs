//! Short-recurrence MINRES (Paige–Saunders, SIAM J. Numer. Anal. 12, 1975).
use super::{
    Identity, IterOptions, IterResult, Preconditioner, converged, dot, finite, norm, residual,
    scaled_axpy, validate,
};
use crate::{
    Real,
    linalg::{LinalgError, LinearOperator},
};

/// Arbeitsspeicher für [`minres`] und [`pminres`]: `8*n` Elemente.
pub const fn minres_work_len(n: usize) -> usize {
    8 * n
}

/// MINRES für ein reelles **symmetrisches**, ggf. indefinites `A*x=b`.
///
/// Wie [`super::cg`], aber ohne positive Definitheit von A. `x` ist Startnäherung
/// und Ergebnis, `work` hat mindestens [`minres_work_len`]`(b.len())` Elemente.
/// Erfolg wird am **ursprünglichen**, neu berechneten Residuum bestätigt:
/// `||b-A*x|| <= tol*||b||`. Dafür fallen je Iteration zwei Operatoranwendungen an.
/// Die QR-Rekursion braucht konstanten Speicher und keine Allokation.
/// Bei einer erkannten Residuenlücke wird aus dem wahren Residuum im
/// verbleibenden Iterationsbudget neu gestartet.
///
/// Symmetrie ist eine Voraussetzung des Aufrufers und wird matrixfrei nicht
/// vollständig geprüft. Dies ist MINRES, nicht MINRES-QLP; eine Lösung mit
/// minimaler Norm oder eine Least-Squares-Rückgabe wird nicht garantiert.
///
/// Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::InvalidInput`]
/// (Toleranz), [`LinalgError::NotFinite`], [`LinalgError::Breakdown`]
/// (invarianter Krylov-Raum ohne ausreichendes Residuum),
/// [`LinalgError::NoConvergence`] (Iterationsbudget). Bei den letzten beiden
/// Fehlern bleibt die letzte Näherung in `x`.
///
/// ```
/// # #[cfg(any(feature="std",feature="libm"))] {
/// use azmath::linalg::{Matrix,iterative::{minres,IterOptions,minres_work_len}};
/// let a=Matrix::new([[0.0f64,1.0],[1.0,0.0]]);
/// let mut x=[0.0;2];
/// let mut work=[0.0;minres_work_len(2)];
/// let result=minres(&a,&[1.0,0.0],&mut x,IterOptions::new(1e-12,10),&mut work).unwrap();
/// assert!(result.residual_norm<1e-12);
/// assert!((x[1]-1.0).abs()<1e-12);
/// # }
/// ```
pub fn minres<T: Real, A: LinearOperator<T> + ?Sized>(
    a: &A,
    b: &[T],
    x: &mut [T],
    opts: IterOptions<T>,
    work: &mut [T],
) -> Result<IterResult<T>, LinalgError> {
    pminres(a, &Identity, b, x, opts, work)
}

// sqrt(r^T z) without squaring large/small vector entries. The sign check
// detects a violated SPD preconditioner assumption in the visited direction.
fn positive_norm<T: Real>(r: &[T], z: &[T]) -> Result<T, LinalgError> {
    finite(z)?;
    let rs = crate::linalg::norm::max_abs(r);
    let zs = crate::linalg::norm::max_abs(z);
    if rs == T::ZERO {
        return if zs == T::ZERO {
            Ok(T::ZERO)
        } else {
            Err(LinalgError::NotPositiveDefinite)
        };
    }
    if zs == T::ZERO {
        return Err(LinalgError::NotPositiveDefinite);
    }
    let value = r
        .iter()
        .zip(z)
        .fold(T::ZERO, |s, (&r, &z)| s + (r / rs) * (z / zs));
    if !value.is_finite() {
        return Err(LinalgError::NotFinite);
    }
    if value <= T::ZERO {
        return Err(LinalgError::NotPositiveDefinite);
    }
    let beta = value.sqrt() * rs.sqrt() * zs.sqrt();
    if !beta.is_finite() {
        return Err(LinalgError::NotFinite);
    }
    if beta == T::ZERO {
        return Err(LinalgError::Breakdown);
    }
    Ok(beta)
}

/// Vorkonditioniertes MINRES mit einem **linearen, symmetrisch positiv definiten**
/// Vorkonditionierer `M`; dessen [`Preconditioner::apply`] berechnet `M^-1*r`.
///
/// A darf indefinit sein, M nicht. Ein Jacobi-Vorkonditionierer aus der
/// vorzeichenbehafteten Diagonale eines indefiniten A ist daher meist ungeeignet;
/// z.B. eine strikt positive Betragsdiagonale verwenden. Auftretende nichtpositive
/// Formen `r^T M^-1 r` ergeben [`LinalgError::NotPositiveDefinite`]. Das ersetzt
/// keinen vollständigen Symmetrie-/SPD-Nachweis für einen beliebigen Operator.
/// Abbruchkriterium und Rückgabe beziehen sich wie bei [`minres`] auf A*x=b,
/// nicht nur auf das vorkonditionierte System.
pub fn pminres<T, A, M>(
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
    if a.rows() != n || a.cols() != n || x.len() != n || work.len() < minres_work_len(n) {
        return Err(LinalgError::DimensionMismatch);
    }
    validate(b, x, opts)?;
    if n == 0 {
        return Ok(IterResult {
            iterations: 0,
            residual_norm: T::ZERO,
        });
    }
    let (r_old, rest) = work.split_at_mut(n);
    let (r_cur, rest) = rest.split_at_mut(n);
    let (z, rest) = rest.split_at_mut(n);
    let (v, rest) = rest.split_at_mut(n);
    let (mut w_old, rest) = rest.split_at_mut(n);
    let (mut w_cur, rest) = rest.split_at_mut(n);
    let (mut w_new, rest) = rest.split_at_mut(n);
    let scratch = &mut rest[..n];
    let bnorm = norm(b);
    if !bnorm.is_finite() {
        return Err(LinalgError::NotFinite);
    }
    if bnorm == T::ZERO {
        x.fill(T::ZERO);
    }
    let mut scale = residual(a, b, x, r_cur)?;
    if converged(scale, bnorm, opts.tol) {
        return Ok(IterResult {
            iterations: 0,
            residual_norm: scale,
        });
    }
    if opts.max_iter == 0 || bnorm == T::ZERO {
        return Err(LinalgError::NoConvergence);
    }
    for ri in r_cur.iter_mut() {
        *ri /= scale;
    }
    r_old.copy_from_slice(r_cur);
    m.apply(r_cur, z);
    let mut beta = positive_norm(r_cur, z)?;
    let mut beta_start = beta;
    let mut cycle = 0usize;
    let mut old_beta = T::ZERO;
    let mut dbar = T::ZERO;
    let mut epsln = T::ZERO;
    let mut phibar = beta;
    let (mut cs, mut sn) = (-T::ONE, T::ZERO);
    w_old.fill(T::ZERO);
    w_cur.fill(T::ZERO);
    for k in 1..=opts.max_iter {
        for (vi, &zi) in v.iter_mut().zip(z.iter()) {
            *vi = zi / beta;
        }
        finite(v)?;
        a.apply(v, scratch);
        finite(scratch)?;
        if cycle > 0 {
            let factor = beta / old_beta;
            for (yi, &ri) in scratch.iter_mut().zip(r_old.iter()) {
                *yi -= factor * ri;
            }
        }
        let alpha = dot(v, scratch);
        if !alpha.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        let factor = alpha / beta;
        for (yi, &ri) in scratch.iter_mut().zip(r_cur.iter()) {
            *yi -= factor * ri;
        }
        finite(scratch)?;
        r_old.copy_from_slice(r_cur);
        r_cur.copy_from_slice(scratch);
        m.apply(r_cur, z);
        old_beta = beta;
        beta = positive_norm(r_cur, z)?;
        let old_eps = epsln;
        let delta = cs * dbar + sn * alpha;
        let gbar = sn * dbar - cs * alpha;
        epsln = sn * beta;
        dbar = -cs * beta;
        let gamma = norm(&[gbar, beta]);
        if !gamma.is_finite() {
            return Err(LinalgError::NotFinite);
        }
        if gamma == T::ZERO {
            return Err(LinalgError::Breakdown);
        }
        cs = gbar / gamma;
        sn = beta / gamma;
        let phi = cs * phibar;
        phibar *= sn;
        for i in 0..n {
            w_new[i] = (v[i] - old_eps * w_old[i] - delta * w_cur[i]) / gamma;
        }
        finite(w_new)?;
        scaled_axpy(phi, scale, w_new, x);
        finite(x)?;
        core::mem::swap(&mut w_old, &mut w_cur);
        core::mem::swap(&mut w_cur, &mut w_new);
        let actual = residual(a, b, x, scratch)?;
        if converged(actual, bnorm, opts.tol) {
            return Ok(IterResult {
                iterations: k,
                residual_norm: actual,
            });
        }
        if phibar.abs() <= opts.tol * beta_start && k < opts.max_iter {
            // The recurrence can report a tiny residual while the actual one
            // has stagnated in finite precision. Restart from that actual
            // residual, retaining the total iteration budget.
            scale = actual;
            for (ri, &value) in r_cur.iter_mut().zip(scratch.iter()) {
                *ri = value / scale;
            }
            r_old.copy_from_slice(r_cur);
            m.apply(r_cur, z);
            beta = positive_norm(r_cur, z)?;
            beta_start = beta;
            old_beta = T::ZERO;
            dbar = T::ZERO;
            epsln = T::ZERO;
            phibar = beta;
            cs = -T::ONE;
            sn = T::ZERO;
            w_old.fill(T::ZERO);
            w_cur.fill(T::ZERO);
            cycle = 0;
            continue;
        }
        if beta == T::ZERO {
            return Err(LinalgError::Breakdown);
        }
        cycle += 1;
    }
    Err(LinalgError::NoConvergence)
}
