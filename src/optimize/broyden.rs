use super::{
    Jac, NoJacobian, OptimizeError, OptimizeOptions, OptimizeResult, Termination, dot,
    eval_jacobian, finite, norm,
};
use crate::Real;

/// Arbeitsspeicher von [`broyden`] und [`broyden_fd`] für `n` Unbekannte: `4·n² + 7·n` Elemente.
pub const fn broyden_work_len(n: usize) -> usize {
    4 * n * n + 7 * n
}

/// Invertiert die `n × n`-Matrix `j` per Gauß-Jordan mit Spaltenpivotsuche. `aug` hat Platz für `n × 2n`,
/// das Ergebnis steht in `inv`. `false` bei (numerisch) singulärer Matrix.
pub(crate) fn invert<T: Real>(j: &[T], n: usize, aug: &mut [T], inv: &mut [T]) -> bool {
    let w = 2 * n;
    let mut scale = T::ZERO;
    for r in 0..n {
        for c in 0..n {
            aug[r * w + c] = j[r * n + c];
            aug[r * w + n + c] = if r == c { T::ONE } else { T::ZERO };
            scale = scale.max(j[r * n + c].abs());
        }
    }
    let tol = T::from_f64(n as f64) * T::EPSILON * scale;
    for k in 0..n {
        let mut p = k;
        let mut best = aug[k * w + k].abs();
        for r in (k + 1)..n {
            let v = aug[r * w + k].abs();
            if v > best {
                best = v;
                p = r;
            }
        }
        if best.is_nan() || best <= tol {
            return false;
        }
        if p != k {
            for c in 0..w {
                aug.swap(k * w + c, p * w + c);
            }
        }
        let piv = aug[k * w + k];
        for c in 0..w {
            aug[k * w + c] /= piv;
        }
        for r in 0..n {
            if r != k {
                let f = aug[r * w + k];
                if f != T::ZERO {
                    for c in 0..w {
                        let v = aug[k * w + c];
                        aug[r * w + c] -= f * v;
                    }
                }
            }
        }
    }
    for r in 0..n {
        for c in 0..n {
            inv[r * n + c] = aug[r * w + n + c];
        }
    }
    true
}

fn broyden_core<T, F, J>(
    mut f: F,
    mut jac: Jac<J>,
    x: &mut [T],
    opts: OptimizeOptions<T>,
    work: &mut [T],
) -> Result<OptimizeResult<T>, OptimizeError>
where
    T: Real,
    F: FnMut(&[T], &mut [T]),
    J: FnMut(&[T], &mut [T]),
{
    let n = x.len();
    if n == 0 || work.len() < broyden_work_len(n) {
        return Err(OptimizeError::InvalidInput);
    }
    if x.iter().any(|&v| !finite(v)) {
        return Err(OptimizeError::NotFinite);
    }
    let (h, rest) = work.split_at_mut(n * n);
    let (jm, rest) = rest.split_at_mut(n * n);
    let (aug, rest) = rest.split_at_mut(2 * n * n);
    let (fx, rest) = rest.split_at_mut(n);
    let (f_new, rest) = rest.split_at_mut(n);
    let (dx, rest) = rest.split_at_mut(n);
    let (x_new, rest) = rest.split_at_mut(n);
    let (y, rest) = rest.split_at_mut(n);
    let (hy, rest) = rest.split_at_mut(n);
    let w = &mut rest[..n];

    f(x, fx);
    if fx.iter().any(|&v| !finite(v)) {
        return Err(OptimizeError::NotFinite);
    }
    let mut fnorm = norm(fx);
    let mut fresh = false; // H stammt gerade aus einer frischen Jacobi-Matrix
    for iter in 0..=opts.max_iter {
        if fnorm <= opts.tol_f {
            return Ok(OptimizeResult {
                iterations: iter,
                residual_norm: fnorm,
                termination: Termination::Residual,
            });
        }
        if iter == opts.max_iter {
            break;
        }
        if iter == 0 || (!fresh && h.iter().any(|&v| !finite(v))) {
            // Anfangsmatrix: J(x) aufstellen und invertieren (aug dient als Zwischenspeicher)
            let (scratch, _) = aug.split_at_mut(2 * n);
            eval_jacobian(&mut f, &mut jac, x, fx, jm, scratch);
            if jm.iter().any(|&v| !finite(v)) {
                return Err(OptimizeError::NotFinite);
            }
            if !invert(jm, n, aug, h) {
                return Err(OptimizeError::SingularJacobian);
            }
            fresh = true;
        }
        // Newton-artiger Schritt dx = −H·F mit Backtracking auf ‖F‖
        for i in 0..n {
            dx[i] = -dot(&h[i * n..(i + 1) * n], fx);
        }
        if norm(dx) <= opts.tol_x * (norm(x) + opts.tol_x) {
            // der volle Schritt ist unterhalb der Genauigkeit: fertig (Rundungsgrenze von ‖F‖ erreicht)
            return Ok(OptimizeResult {
                iterations: iter,
                residual_norm: fnorm,
                termination: Termination::Step,
            });
        }
        let mut t = T::ONE;
        let mut accepted = false;
        for _ in 0..30 {
            for i in 0..n {
                x_new[i] = x[i] + t * dx[i];
            }
            f(x_new, f_new);
            let fn_new = norm(f_new);
            if finite(fn_new) && fn_new < (T::ONE - T::from_f64(1e-4) * t) * fnorm {
                accepted = true;
                break;
            }
            t *= T::from_f64(0.5);
        }
        if !accepted {
            if fresh {
                return Err(OptimizeError::NoConvergence);
            }
            // Näherung verbraucht: Jacobi-Matrix neu aufstellen und noch einmal versuchen
            let (scratch, _) = aug.split_at_mut(2 * n);
            eval_jacobian(&mut f, &mut jac, x, fx, jm, scratch);
            if jm.iter().any(|&v| !finite(v)) {
                return Err(OptimizeError::NotFinite);
            }
            if !invert(jm, n, aug, h) {
                return Err(OptimizeError::SingularJacobian);
            }
            fresh = true;
            continue;
        }
        // Broyden-Update (inverse Form): H += (s − H·y)·(sᵀ·H) / (sᵀ·H·y)
        let mut step = T::ZERO;
        for i in 0..n {
            dx[i] *= t; // s
            y[i] = f_new[i] - fx[i];
            step = step.hypot(dx[i]);
        }
        for i in 0..n {
            hy[i] = dot(&h[i * n..(i + 1) * n], y);
        }
        let denom = dot(dx, hy);
        if denom.abs() > T::EPSILON * step * norm(hy) {
            for j in 0..n {
                let mut s = T::ZERO;
                for i in 0..n {
                    s += dx[i] * h[i * n + j];
                }
                w[j] = s;
            }
            for i in 0..n {
                let c = (dx[i] - hy[i]) / denom;
                for j in 0..n {
                    h[i * n + j] += c * w[j];
                }
            }
            fresh = false;
        }
        x.copy_from_slice(x_new);
        fx.copy_from_slice(f_new);
        fnorm = norm(fx);
        if step <= opts.tol_x * (norm(x) + opts.tol_x) {
            return Ok(OptimizeResult {
                iterations: iter + 1,
                residual_norm: fnorm,
                termination: Termination::Step,
            });
        }
    }
    Err(OptimizeError::NoConvergence)
}

/// Broyden-Verfahren („gutes“ Broyden) für das nichtlineare Gleichungssystem `F(x) = 0` mit `n`
/// Gleichungen und `n` Unbekannten.
///
/// Die Jacobi-Matrix wird **nur am Start** ausgewertet (und bei Stillstand erneuert); danach hält das Verfahren
/// eine Näherung der inversen Jacobi-Matrix und aktualisiert sie in jedem Schritt mit einer Rang-1-Korrektur
/// (Sherman-Morrison). Pro Schritt kostet das eine Auswertung von `F` und `O(n²)` Rechnung statt einer
/// Jacobi-Matrix und einer Faktorisierung wie beim Newton-Verfahren ([`newton_system`](super::newton_system)).
/// Die Konvergenz ist superlinear statt quadratisch. Eine Schrittweitensteuerung (Halbieren, bis `‖F‖` sinkt) macht
/// es robuster.
///
/// - `f(x, out)` schreibt `F(x)`; `jacobian(x, j)` die `n × n`-Jacobi-Matrix zeilenweise (nur für die Anfangsmatrix).
/// - `x`: Startwert und Ergebnis. `work`: mindestens [`broyden_work_len`]`(n)` Elemente.
/// - Abbruch über `tol_f` (`‖F‖`) oder `tol_x` (Schritt); Ergebnis wie bei den anderen Lösern ([`OptimizeResult`]).
///
/// Fehler: [`OptimizeError::InvalidInput`], [`OptimizeError::NotFinite`], [`OptimizeError::SingularJacobian`],
/// [`OptimizeError::NoConvergence`].
pub fn broyden<T: Real>(
    f: impl FnMut(&[T], &mut [T]),
    jacobian: impl FnMut(&[T], &mut [T]),
    x: &mut [T],
    opts: OptimizeOptions<T>,
    work: &mut [T],
) -> Result<OptimizeResult<T>, OptimizeError> {
    broyden_core(f, Jac::Analytic(jacobian), x, opts, work)
}

/// Wie [`broyden`], bestimmt die Anfangs-Jacobi-Matrix aber selbst per Vorwärtsdifferenzen.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::optimize::{broyden_fd, broyden_work_len, OptimizeOptions};
///
/// // Kreis x² + y² = 4 und Gerade y = x
/// let f = |x: &[f64], out: &mut [f64]| {
///     out[0] = x[0] * x[0] + x[1] * x[1] - 4.0;
///     out[1] = x[1] - x[0];
/// };
/// let mut x = [1.0, 0.5];
/// let mut work = [0.0; broyden_work_len(2)];
/// broyden_fd(f, &mut x, OptimizeOptions::default(), &mut work).unwrap();
/// assert!((x[0] - 2f64.sqrt()).abs() < 1e-7);
/// # }
/// ```
pub fn broyden_fd<T: Real>(
    f: impl FnMut(&[T], &mut [T]),
    x: &mut [T],
    opts: OptimizeOptions<T>,
    work: &mut [T],
) -> Result<OptimizeResult<T>, OptimizeError> {
    broyden_core(f, Jac::Numeric::<NoJacobian<T>>, x, opts, work)
}
