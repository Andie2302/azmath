use super::super::{LinalgError, LinearOperator};
use super::{Identity, IterOptions, IterResult, Preconditioner, axpy, dot};
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

    let bad = |v: T| v.is_nan() || v.is_infinite();

    let b2 = dot(b, b);
    if bad(b2) {
        return Err(LinalgError::NotFinite);
    }
    if b2 == T::ZERO {
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
    if bad(r2) {
        return Err(LinalgError::NotFinite);
    }

    r_hat.copy_from_slice(r);
    p.fill(T::ZERO);
    v.fill(T::ZERO);
    let (mut rho_old, mut alpha, mut omega) = (T::ONE, T::ONE, T::ONE);

    for k in 0..opts.max_iter {
        if r2 <= stop {
            return Ok(IterResult {
                iterations: k,
                residual_norm: r2.sqrt(),
            });
        }

        let rho = dot(r_hat, r);
        if bad(rho) {
            return Err(LinalgError::NotFinite);
        }
        if rho == T::ZERO {
            return Err(LinalgError::Breakdown);
        }
        // p = r + β·(p − ω·v); im ersten Schritt ist p = v = 0, also p = r.
        let beta = (rho / rho_old) * (alpha / omega);
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        m.apply(p, p_hat);
        a.apply(p_hat, v);
        let den = dot(r_hat, v);
        if bad(den) {
            return Err(LinalgError::NotFinite);
        }
        if den == T::ZERO {
            return Err(LinalgError::Breakdown);
        }
        alpha = rho / den;

        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }
        let s2 = dot(s, s);
        if bad(s2) {
            return Err(LinalgError::NotFinite);
        }
        if s2 <= stop {
            // Halbschritt genügt bereits
            axpy(alpha, p_hat, x);
            return Ok(IterResult {
                iterations: k + 1,
                residual_norm: s2.sqrt(),
            });
        }

        m.apply(s, s_hat);
        a.apply(s_hat, t);
        let tt = dot(t, t);
        if bad(tt) {
            return Err(LinalgError::NotFinite);
        }
        if tt == T::ZERO {
            return Err(LinalgError::Breakdown);
        }
        omega = dot(t, s) / tt;
        if omega == T::ZERO {
            return Err(LinalgError::Breakdown);
        }

        axpy(alpha, p_hat, x);
        axpy(omega, s_hat, x);
        for i in 0..n {
            r[i] = s[i] - omega * t[i];
        }
        r2 = dot(r, r);
        if bad(r2) {
            return Err(LinalgError::NotFinite);
        }
        rho_old = rho;
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
