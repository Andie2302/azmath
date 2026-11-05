use super::super::{LinalgError, LinearOperator};
use super::{
    Identity, IterOptions, IterResult, Preconditioner, axpy, converged, dot, finite, norm,
    residual, validate,
};
use crate::Real;

/// Benötigter Arbeitsspeicher für [`gmres`]/[`pgmres`] bei Dimension `n` und Neustartlänge
/// `restart`.
///
/// Die Krylov-Dimension ist höchstens `n`; es wird `min(restart, n)` verwendet, größere Werte
/// brauchen also nicht mehr Speicher als `n`, aber die Funktion liefert dann denselben Bedarf wie
/// für `restart = n`.
pub const fn gmres_work_len(n: usize, restart: usize) -> usize {
    let m = if restart < n { restart } else { n };
    // V: (m+1) Basisvektoren, H: (m+1)×m, Givens cs, sn (2m), g (m+1), y (m), w und z (2n)
    (m + 1) * n + (m + 1) * m + 4 * m + 1 + 2 * n
}

/// GMRES mit Neustart (GMRES(m)) für `A·x = b` mit beliebiger regulärer Matrix `A`.
///
/// Minimiert in jedem Schritt die Residuumsnorm über dem Krylov-Raum, konvergiert deshalb monoton
/// in exakter Arithmetik für reguläre Systeme ohne Zusammenbruch. Rundungsfehler und
/// Stagnation können die erreichbare Genauigkeit begrenzen. Es braucht pro Arnoldi-Schritt
/// eine Anwendung von `A` (zusätzlich zu Residuenprüfungen) und funktioniert
/// auch für symmetrisch **indefinite** und **schiefsymmetrische** Matrizen, bei denen CG und
/// BiCGSTAB versagen. Dafür speichert es `restart + 1` Basisvektoren; nach `restart` Schritten wird
/// neu gestartet (kleineres `restart`: weniger Speicher, aber langsamere oder stagnierende
/// Konvergenz; `restart ≥ n` ist das volle GMRES). Beispiel 1D-Konvektions-Diffusion mit
/// `n = 100`: volles GMRES braucht 100 Anwendungen von `A`, GMRES(20) rund 500.
///
/// - `a`: quadratischer Operator `n × n` (nur `A·x` wird benötigt).
/// - `b`: rechte Seite, `x`: Startnäherung beim Aufruf und Lösung beim Rückgabe.
/// - `restart`: Neustartlänge, mindestens 1. `work`: mindestens [`gmres_work_len`]`(n, restart)`
///   Elemente.
/// - Abbruch bei `‖b − A·x‖ ≤ tol · ‖b‖`; `IterResult::iterations` zählt die Arnoldi-Schritte (zusätzliche Residuenprüfungen nicht mitgezählt).
///
/// Fehler: [`LinalgError::DimensionMismatch`] (auch bei `restart = 0` oder zu kleinem `work`),
/// [`LinalgError::NotFinite`], [`LinalgError::Singular`] (die Matrix ist singulär, das
/// Verfahren findet keine Lösung), [`LinalgError::NoConvergence`] (`x` enthält dann die letzte
/// Näherung). [`LinalgError::InvalidInput`] bei negativer oder nichtendlicher Toleranz.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::linalg::iterative::{gmres, gmres_work_len, IterOptions};
/// use azmath::linalg::Matrix;
///
/// // schiefsymmetrisch: dafür versagen CG und BiCGSTAB, GMRES nicht
/// let a = Matrix::new([[0.0f64, 1.0], [-1.0, 0.0]]);
/// let b = [1.0, 0.0];
/// let mut x = [0.0; 2];
/// let mut work = [0.0; gmres_work_len(2, 2)];
/// gmres(&a, &b, &mut x, 2, IterOptions::new(1e-12, 50), &mut work).unwrap();
/// // A·x = (x₁, −x₀) = (1, 0)  ->  x = (0, 1)
/// assert!(x[0].abs() < 1e-10 && (x[1] - 1.0).abs() < 1e-10);
/// # }
/// ```
pub fn gmres<T: Real, A: LinearOperator<T> + ?Sized>(
    a: &A,
    b: &[T],
    x: &mut [T],
    restart: usize,
    opts: IterOptions<T>,
    work: &mut [T],
) -> Result<IterResult<T>, LinalgError> {
    pgmres(a, &Identity, b, x, restart, opts, work)
}

/// GMRES(m) mit **Rechts-Vorkonditionierer** `m` (siehe [`Preconditioner`]): löst
/// `A·M⁻¹·u = b` und liefert `x = M⁻¹·u`. Der Vorkonditionierer muss nicht symmetrisch sein; das
/// gemessene Residuum bleibt das der ursprünglichen Gleichung `b − A·x`.
///
/// Ein Rechts-Vorkonditionierer wirkt auf die **Spalten** von `A`: Der Jacobi-Vorkonditionierer
/// (`M = diag(A)`) beseitigt daher eine ungleichmäßige *Spalten*skalierung, nicht aber eine
/// Zeilenskalierung (`A·M⁻¹ = S·C·S⁻¹` bleibt bei `A = S·B` schlecht konditioniert).
///
/// Parameter und Fehler wie bei [`gmres`].
pub fn pgmres<T, A, M>(
    a: &A,
    m: &M,
    b: &[T],
    x: &mut [T],
    restart: usize,
    opts: IterOptions<T>,
    work: &mut [T],
) -> Result<IterResult<T>, LinalgError>
where
    T: Real,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = b.len();
    if restart == 0
        || a.rows() != n
        || a.cols() != n
        || x.len() != n
        || work.len() < gmres_work_len(n, restart)
    {
        return Err(LinalgError::DimensionMismatch);
    }
    validate(b, x, opts)?;
    if n == 0 {
        return Ok(IterResult {
            iterations: 0,
            residual_norm: T::ZERO,
        });
    }
    let k = restart.min(n);

    let bad = |v: T| v.is_nan() || v.is_infinite();
    let bnorm = norm(b);
    if bad(bnorm) {
        return Err(LinalgError::NotFinite);
    }
    if bnorm == T::ZERO {
        x.fill(T::ZERO);
        let rn = residual(a, b, x, &mut work[..n])?;
        if rn != T::ZERO {
            return Err(LinalgError::NoConvergence);
        }
        return Ok(IterResult {
            iterations: 0,
            residual_norm: rn,
        });
    }

    // Arbeitsspeicher aufteilen
    let (basis, rest) = work.split_at_mut((k + 1) * n);
    let (h, rest) = rest.split_at_mut((k + 1) * k);
    let (cs, rest) = rest.split_at_mut(k);
    let (sn, rest) = rest.split_at_mut(k);
    let (g, rest) = rest.split_at_mut(k + 1);
    let (y, rest) = rest.split_at_mut(k);
    let (w, rest) = rest.split_at_mut(n);
    let z = &mut rest[..n];

    let mut iters = 0usize;
    loop {
        // w = b − A·x  (wahres Residuum)
        let beta = residual(a, b, x, w)?;
        if bad(beta) {
            return Err(LinalgError::NotFinite);
        }
        if converged(beta, bnorm, opts.tol) {
            return Ok(IterResult {
                iterations: iters,
                residual_norm: beta,
            });
        }
        if iters >= opts.max_iter {
            return Err(LinalgError::NoConvergence);
        }

        // v₀ = r/β,  g = (β, 0, …)
        for (v0, wi) in basis[..n].iter_mut().zip(w.iter()) {
            *v0 = *wi / beta;
        }
        g.fill(T::ZERO);
        g[0] = beta;

        let mut cols = 0usize; // Zahl der berechneten Spalten von H
        for j in 0..k {
            // z = M⁻¹·v_j,  w = A·z
            let (head, tail) = basis.split_at_mut((j + 1) * n);
            let vj = &head[j * n..];
            m.apply(vj, z);
            finite(z)?;
            a.apply(z, w);
            finite(w)?;
            iters += 1;

            // Modifiziertes Gram-Schmidt gegen v₀ … v_j
            for i in 0..=j {
                let vi = &head[i * n..(i + 1) * n];
                let hij = dot(w, vi);
                h[i * k + j] = hij;
                axpy(-hij, vi, w);
            }
            let hnext = norm(w);
            if bad(hnext) {
                return Err(LinalgError::NotFinite);
            }
            h[(j + 1) * k + j] = hnext;
            let lucky = hnext == T::ZERO;
            if !lucky {
                // v_{j+1} = w/h_{j+1,j}
                for (vn, wi) in tail[..n].iter_mut().zip(w.iter()) {
                    *vn = *wi / hnext;
                }
            }

            // bisherige Givens-Rotationen auf die neue Spalte anwenden
            for i in 0..j {
                let t = cs[i] * h[i * k + j] + sn[i] * h[(i + 1) * k + j];
                h[(i + 1) * k + j] = -sn[i] * h[i * k + j] + cs[i] * h[(i + 1) * k + j];
                h[i * k + j] = t;
            }
            // neue Rotation, die h_{j+1,j} annulliert
            let (p, q) = (h[j * k + j], h[(j + 1) * k + j]);
            let denom = p.hypot(q);
            if !denom.is_finite() {
                return Err(LinalgError::NotFinite);
            }
            if denom == T::ZERO {
                cs[j] = T::ONE;
                sn[j] = T::ZERO;
            } else {
                cs[j] = p / denom;
                sn[j] = q / denom;
            }
            h[j * k + j] = denom;
            h[(j + 1) * k + j] = T::ZERO;
            g[j + 1] = -sn[j] * g[j];
            g[j] = cs[j] * g[j];
            cols = j + 1;

            if converged(g[j + 1].abs(), bnorm, opts.tol) || lucky || iters >= opts.max_iter {
                break;
            }
        }

        // Rückwärtssubstitution: H[0..cols, 0..cols] · y = g[0..cols]
        for i in (0..cols).rev() {
            let mut s = g[i];
            for l in (i + 1)..cols {
                s -= h[i * k + l] * y[l];
            }
            let d = h[i * k + i];
            if d == T::ZERO {
                return Err(LinalgError::Singular);
            }
            y[i] = s / d;
        }
        // x += M⁻¹·(V·y)
        w.fill(T::ZERO);
        for (i, yi) in y[..cols].iter().enumerate() {
            axpy(*yi, &basis[i * n..(i + 1) * n], w);
        }
        m.apply(w, z);
        for (xi, zi) in x.iter_mut().zip(z.iter()) {
            *xi += *zi;
        }
        if x.iter().any(|v| bad(*v)) {
            return Err(LinalgError::NotFinite);
        }
        // Neustart: oben wird das wahre Residuum neu berechnet und geprüft.
    }
}

/// Arbeitsspeicher für [`pgmres_left`]: [`gmres_work_len`] plus `2·n`.
pub const fn pgmres_left_work_len(n: usize, restart: usize) -> usize {
    gmres_work_len(n, restart) + 2 * n
}

/// Der Operator `M⁻¹·A` mit Zwischenspeicher für das Produkt `A·x`.
struct LeftPreconditioned<'a, T, A: ?Sized, M: ?Sized> {
    a: &'a A,
    m: &'a M,
    tmp: core::cell::RefCell<&'a mut [T]>,
}

impl<T: Real, A: LinearOperator<T> + ?Sized, M: Preconditioner<T> + ?Sized> LinearOperator<T>
    for LeftPreconditioned<'_, T, A, M>
{
    fn rows(&self) -> usize {
        self.a.rows()
    }
    fn cols(&self) -> usize {
        self.a.cols()
    }
    fn apply(&self, x: &[T], y: &mut [T]) {
        let mut tmp = self.tmp.borrow_mut();
        self.a.apply(x, &mut tmp[..]);
        self.m.apply(&tmp[..], y);
    }
}

/// GMRES(m) mit **Links-Vorkonditionierer**: löst `M⁻¹·A·x = M⁻¹·b`.
///
/// Anders als bei [`pgmres`] (rechts) bezieht sich das Residuum auf das **vorkonditionierte** System: `residual_norm`
/// und die Toleranz `opts.tol` gelten für `‖M⁻¹(b − A·x)‖` relativ zu `‖M⁻¹b‖`. Ein Links-Vorkonditionierer wirkt auf
/// die *Zeilen* von `A` und kann deshalb eine Zeilenskalierung beseitigen, die ein Rechts-Vorkonditionierer
/// (z. B. Jacobi) nicht erreicht.
///
/// `work`: mindestens [`pgmres_left_work_len`]`(n, restart)` Elemente. Fehler wie bei [`gmres`].
pub fn pgmres_left<T, A, M>(
    a: &A,
    m: &M,
    b: &[T],
    x: &mut [T],
    restart: usize,
    opts: IterOptions<T>,
    work: &mut [T],
) -> Result<IterResult<T>, LinalgError>
where
    T: Real,
    A: LinearOperator<T> + ?Sized,
    M: Preconditioner<T> + ?Sized,
{
    let n = b.len();
    if restart == 0
        || a.rows() != n
        || a.cols() != n
        || x.len() != n
        || work.len() < pgmres_left_work_len(n, restart)
    {
        return Err(LinalgError::DimensionMismatch);
    }
    validate(b, x, opts)?;
    let (tmp, rest) = work.split_at_mut(n);
    let (b_hat, rest) = rest.split_at_mut(n);
    m.apply(b, b_hat);
    finite(b_hat)?;
    if norm(b_hat) == T::ZERO && norm(b) != T::ZERO {
        return Err(LinalgError::Breakdown);
    }
    let op = LeftPreconditioned {
        a,
        m,
        tmp: core::cell::RefCell::new(tmp),
    };
    gmres(&op, b_hat, x, restart, opts, rest)
}
