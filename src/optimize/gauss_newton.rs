use super::{
    Jac, NoJacobian, OptimizeError, OptimizeOptions, OptimizeResult, Termination, eval_jacobian,
    finite, norm,
};
use crate::Real;
use crate::linalg::LinalgError;
use crate::linalg::kernel::{qr_decompose, qr_solve};

/// Benötigter Arbeitsspeicher für [`gauss_newton`] und [`gauss_newton_fd`] bei `m` Residuen und
/// `n` Parametern.
pub const fn gauss_newton_work_len(m: usize, n: usize) -> usize {
    // Jacobi-Matrix (m·n), tau (n), r, y, r_try (3m), dx, x_try (2n), Differenzen-Puffer (m + n)
    m * n + 4 * m + 4 * n
}

/// Benötigter Arbeitsspeicher für [`newton_system`] und [`newton_system_fd`] bei `n` Gleichungen.
pub const fn newton_work_len(n: usize) -> usize {
    gauss_newton_work_len(n, n)
}

fn gn_core<T, R, J>(
    mut residual: R,
    mut jac: Jac<J>,
    x: &mut [T],
    m: usize,
    opts: OptimizeOptions<T>,
    work: &mut [T],
    root: bool,
) -> Result<OptimizeResult<T>, OptimizeError>
where
    T: Real,
    R: FnMut(&[T], &mut [T]),
    J: FnMut(&[T], &mut [T]),
{
    super::validate_options(opts)?;
    let n = x.len();
    if n == 0 || m < n || work.len() < gauss_newton_work_len(m, n) {
        return Err(OptimizeError::InvalidInput);
    }
    if x.iter().any(|v| !finite(*v)) {
        return Err(OptimizeError::NotFinite);
    }

    let (jm, rest) = work.split_at_mut(m * n);
    let (tau, rest) = rest.split_at_mut(n);
    let (r, rest) = rest.split_at_mut(m);
    let (y, rest) = rest.split_at_mut(m);
    let (r_try, rest) = rest.split_at_mut(m);
    let (dx, rest) = rest.split_at_mut(n);
    let (x_try, rest) = rest.split_at_mut(n);
    let fd = &mut rest[..m + n];

    residual(x, r);
    let mut cost = norm(r);
    if !finite(cost) {
        return Err(OptimizeError::NotFinite);
    }

    for iter in 0..=opts.max_iter {
        let rn = cost;
        if rn <= opts.tol_f {
            return Ok(OptimizeResult {
                iterations: iter,
                residual_norm: rn,
                termination: Termination::Residual,
            });
        }

        if iter == opts.max_iter {
            break;
        }
        eval_jacobian(&mut residual, &mut jac, x, r, jm, fd);
        if jm.iter().any(|&v| !finite(v)) {
            return Err(OptimizeError::NotFinite);
        }

        // Gradient Jᵀr
        if !root {
            let mut ginf = T::ZERO;
            for k in 0..n {
                let mut s = T::ZERO;
                for i in 0..m {
                    s += jm[i * n + k] * r[i];
                }
                if !finite(s) {
                    return Err(OptimizeError::NotFinite);
                }
                ginf = ginf.max(s.abs());
            }
            if ginf <= opts.tol_g * (T::ONE + rn) {
                return Ok(OptimizeResult {
                    iterations: iter,
                    residual_norm: rn,
                    termination: Termination::Gradient,
                });
            }
        }
        // Gauß-Newton-Schritt: min ‖J·δ + r‖  (per QR)
        match qr_decompose(jm, m, n, tau) {
            Ok(()) => {}
            Err(LinalgError::NotFinite) => return Err(OptimizeError::NotFinite),
            Err(_) => return Err(OptimizeError::SingularJacobian),
        }
        for (yi, ri) in y.iter_mut().zip(r.iter()) {
            *yi = -*ri;
        }
        qr_solve(jm, m, n, tau, y, dx);
        let step = norm(dx);
        if !finite(step) {
            return Err(OptimizeError::NotFinite);
        }

        // Auch kleine Schritte werden erst nach erfolgreicher Auswertung übernommen.
        // Schrittweitensteuerung: halbieren, bis das Residuum nicht mehr wächst
        let mut t = T::ONE;
        let mut accepted = false;
        // Kosten dürfen im Rahmen des Rundungsrauschens leicht schwanken (siehe Termination::Cost)
        let noise = T::from_f64(16.0) * T::EPSILON * cost;
        for _ in 0..30 {
            for k in 0..n {
                x_try[k] = x[k] + t * dx[k];
            }
            if x_try.iter().any(|&v| !finite(v)) {
                t *= T::from_f64(0.5);
                continue;
            }
            residual(x_try, r_try);
            let c = norm(r_try);
            if finite(c) && c <= cost + noise {
                x.copy_from_slice(x_try);
                r.copy_from_slice(r_try);
                cost = c;
                accepted = true;
                break;
            }
            t *= T::from_f64(0.5);
        }
        if !accepted {
            return Err(OptimizeError::NoConvergence);
        }
        if cost <= opts.tol_f {
            return super::root_result(iter + 1, cost, opts);
        }
        if t * step <= opts.tol_x * (norm(x) + opts.tol_x) {
            if root {
                return Err(OptimizeError::NoConvergence);
            }
            return Ok(OptimizeResult {
                iterations: iter + 1,
                residual_norm: cost,
                termination: Termination::Step,
            });
        }
    }
    Err(OptimizeError::NoConvergence)
}

/// Gauß-Newton-Verfahren für `min ‖r(x)‖₂` mit `m ≥ n` Residuen und `n` Parametern.
///
/// Jeder Schritt löst das linearisierte Ausgleichsproblem `min ‖J·δ + r‖` per QR-Zerlegung
/// (numerisch stabiler als die Normalengleichungen) und halbiert die Schrittweite, bis das
/// Residuum nicht mehr wächst. Das Verfahren konvergiert schnell, wenn das Residuum an der Lösung
/// klein ist, braucht aber einen brauchbaren Startwert und eine Jacobi-Matrix mit vollem
/// Spaltenrang; sonst ist [`levenberg_marquardt`](super::levenberg_marquardt) die bessere Wahl.
///
/// - `residual(x, r)` schreibt `r(x)`; `jacobian(x, j)` schreibt die Jacobi-Matrix zeilenweise
///   (`j[i*n + k] = ∂rᵢ/∂xₖ`).
/// - `x`: Startwert und Ergebnis (`n` Elemente), `m`: Anzahl der Residuen.
/// - `work`: mindestens [`gauss_newton_work_len`]`(m, n)` Elemente.
///
/// Fehler: [`OptimizeError::InvalidInput`], [`OptimizeError::NotFinite`],
/// [`OptimizeError::SingularJacobian`], [`OptimizeError::NoConvergence`].
pub fn gauss_newton<T: Real>(
    residual: impl FnMut(&[T], &mut [T]),
    jacobian: impl FnMut(&[T], &mut [T]),
    x: &mut [T],
    m: usize,
    opts: OptimizeOptions<T>,
    work: &mut [T],
) -> Result<OptimizeResult<T>, OptimizeError> {
    gn_core(residual, Jac::Analytic(jacobian), x, m, opts, work, false)
}

/// Wie [`gauss_newton`], bestimmt die Jacobi-Matrix aber selbst per Vorwärtsdifferenzen
/// (siehe [`numerical_jacobian`](super::numerical_jacobian)).
pub fn gauss_newton_fd<T: Real>(
    residual: impl FnMut(&[T], &mut [T]),
    x: &mut [T],
    m: usize,
    opts: OptimizeOptions<T>,
    work: &mut [T],
) -> Result<OptimizeResult<T>, OptimizeError> {
    gn_core(
        residual,
        Jac::Numeric::<NoJacobian<T>>,
        x,
        m,
        opts,
        work,
        false,
    )
}

/// Newton-Verfahren für das nichtlineare Gleichungssystem `F(x) = 0` mit `n` Gleichungen und
/// `n` Unbekannten (`n = x.len()`).
///
/// Quadratisch konvergent nahe einer Lösung mit regulärer Jacobi-Matrix; die Schrittweiten-
/// steuerung (Halbieren, bis `‖F‖` nicht wächst) macht es gegenüber dem reinen Newton-Verfahren
/// robuster. Eine singuläre Jacobi-Matrix ergibt [`OptimizeError::SingularJacobian`].
///
/// - `f(x, out)` schreibt `F(x)`; `jacobian(x, j)` die `n × n`-Jacobi-Matrix zeilenweise.
/// - `work`: mindestens [`newton_work_len`]`(n)` Elemente.
///
/// Erfolg nur bei `‖F(x)‖₂ ≤ tol_f` ([`Termination::Residual`]). Anders als bei
/// [`gauss_newton`] ergeben stationäre Punkte oder kleine Schritte mit größerem Residuum
/// [`OptimizeError::NoConvergence`]. `tol_g` wird bei Gleichungslösern nicht verwendet.
/// Alle Toleranzen müssen endlich und nichtnegativ sein.
///
/// Fehler wie bei [`gauss_newton`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::optimize::{newton_system, newton_work_len, OptimizeOptions};
///
/// // Schnittpunkt von Kreis x² + y² = 4 und Gerade y = x
/// let f = |v: &[f64], out: &mut [f64]| {
///     out[0] = v[0] * v[0] + v[1] * v[1] - 4.0;
///     out[1] = v[1] - v[0];
/// };
/// let jac = |v: &[f64], j: &mut [f64]| {
///     j.copy_from_slice(&[2.0 * v[0], 2.0 * v[1], -1.0, 1.0]);
/// };
/// let mut x = [1.0, 0.5];
/// let mut work = [0.0; newton_work_len(2)];
/// newton_system(f, jac, &mut x, OptimizeOptions::default(), &mut work).unwrap();
/// assert!((x[0] - 2f64.sqrt()).abs() < 1e-12 && (x[1] - 2f64.sqrt()).abs() < 1e-12);
/// # }
/// ```
pub fn newton_system<T: Real>(
    f: impl FnMut(&[T], &mut [T]),
    jacobian: impl FnMut(&[T], &mut [T]),
    x: &mut [T],
    opts: OptimizeOptions<T>,
    work: &mut [T],
) -> Result<OptimizeResult<T>, OptimizeError> {
    let n = x.len();
    gn_core(f, Jac::Analytic(jacobian), x, n, opts, work, true)
}

/// Wie [`newton_system`], bestimmt die Jacobi-Matrix aber selbst per Vorwärtsdifferenzen.
pub fn newton_system_fd<T: Real>(
    f: impl FnMut(&[T], &mut [T]),
    x: &mut [T],
    opts: OptimizeOptions<T>,
    work: &mut [T],
) -> Result<OptimizeResult<T>, OptimizeError> {
    let n = x.len();
    gn_core(f, Jac::Numeric::<NoJacobian<T>>, x, n, opts, work, true)
}
