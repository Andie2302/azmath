use super::{
    Jac, NoJacobian, OptimizeError, OptimizeOptions, OptimizeResult, Termination, dot,
    eval_jacobian, finite, norm,
};
use azmath_core::Real;
use azmath_linalg::LinalgError;
use azmath_linalg::kernel::{cholesky_decompose, cholesky_solve};

/// Benötigter Arbeitsspeicher für [`levenberg_marquardt`] und [`levenberg_marquardt_fd`] bei `m`
/// Residuen und `n` Parametern.
pub const fn levenberg_marquardt_work_len(m: usize, n: usize) -> usize {
    // J (m·n), r, r_try (2m), JᵀJ und Cholesky-Faktor (2n²), g, dx, x_try, d (4n), Differenzen (m + n)
    m * n + 3 * m + 2 * n * n + 5 * n
}

fn lm_core<T, R, J>(
    mut residual: R,
    mut jac: Jac<J>,
    x: &mut [T],
    m: usize,
    opts: OptimizeOptions<T>,
    work: &mut [T],
) -> Result<OptimizeResult<T>, OptimizeError>
where
    T: Real,
    R: FnMut(&[T], &mut [T]),
    J: FnMut(&[T], &mut [T]),
{
    let n = x.len();
    if n == 0 || m < n || work.len() < levenberg_marquardt_work_len(m, n) {
        return Err(OptimizeError::InvalidInput);
    }
    if x.iter().any(|v| !finite(*v)) {
        return Err(OptimizeError::NotFinite);
    }

    let (jm, rest) = work.split_at_mut(m * n);
    let (r, rest) = rest.split_at_mut(m);
    let (r_try, rest) = rest.split_at_mut(m);
    let (a, rest) = rest.split_at_mut(n * n);
    let (l, rest) = rest.split_at_mut(n * n);
    let (g, rest) = rest.split_at_mut(n);
    let (dx, rest) = rest.split_at_mut(n);
    let (x_try, rest) = rest.split_at_mut(n);
    let (d, rest) = rest.split_at_mut(n);
    let fd = &mut rest[..m + n];

    residual(x, r);
    let mut cost = dot(r, r);
    if !finite(cost) {
        return Err(OptimizeError::NotFinite);
    }

    let one_third = T::ONE / T::from_f64(3.0);
    let max_lambda = T::from_f64(1.0e16);
    let mut lambda = T::from_f64(1.0e-3);
    let mut nu = T::from_f64(2.0);

    for iter in 0..opts.max_iter {
        let rn = cost.sqrt();
        if rn <= opts.tol_f {
            return Ok(OptimizeResult {
                iterations: iter,
                residual_norm: rn,
                termination: Termination::Residual,
            });
        }

        eval_jacobian(&mut residual, &mut jac, x, r, jm, fd);

        // A = JᵀJ (exakt symmetrisch), g = Jᵀr
        for i in 0..n {
            for j in 0..=i {
                let mut s = T::ZERO;
                for k in 0..m {
                    s += jm[k * n + i] * jm[k * n + j];
                }
                a[i * n + j] = s;
                a[j * n + i] = s;
            }
        }
        let mut ginf = T::ZERO;
        for i in 0..n {
            let mut s = T::ZERO;
            for k in 0..m {
                s += jm[k * n + i] * r[k];
            }
            g[i] = s;
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
        if a.iter().any(|v| !finite(*v)) {
            return Err(OptimizeError::NotFinite);
        }

        // Marquardt-Skalierung D = diag(JᵀJ), nach unten begrenzt (verschwindende Spalten)
        let mut max_diag = T::ZERO;
        for i in 0..n {
            max_diag = max_diag.max(a[i * n + i]);
        }
        let floor = if max_diag > T::ZERO {
            max_diag * T::EPSILON
        } else {
            T::ONE
        };
        for i in 0..n {
            d[i] = a[i * n + i].max(floor);
        }

        // Dämpfung erhöhen, bis ein Schritt das Residuum verringert
        let mut accepted = false;
        for _ in 0..60 {
            l.copy_from_slice(a);
            for i in 0..n {
                l[i * n + i] += lambda * d[i];
            }
            match cholesky_decompose(l, n) {
                Ok(()) => {}
                Err(LinalgError::NotFinite) => return Err(OptimizeError::NotFinite),
                Err(_) => {
                    lambda *= nu;
                    nu *= T::from_f64(2.0);
                    if lambda > max_lambda {
                        return Err(OptimizeError::NoConvergence);
                    }
                    continue;
                }
            }
            // (A + λD)·δ = −g; die rechte Seite liegt vorübergehend in x_try
            for i in 0..n {
                x_try[i] = -g[i];
            }
            cholesky_solve(l, n, x_try, dx);

            let mut predicted = T::ZERO;
            for i in 0..n {
                predicted += dx[i] * (lambda * d[i] * dx[i] - g[i]);
            }
            for i in 0..n {
                x_try[i] = x[i] + dx[i];
            }
            residual(x_try, r_try);
            let cost_try = dot(r_try, r_try);

            // Kein überprüfbarer Fortschritt mehr (unterhalb des Rundungsrauschens der Kosten):
            // den Schritt übernehmen, sofern er nicht messbar schadet, und fertig.
            let noise = T::from_f64(16.0) * T::EPSILON * cost;
            if finite(cost_try)
                && predicted <= noise
                && cost_try <= cost + noise
                && lambda <= T::ONE
            {
                x.copy_from_slice(x_try);
                return Ok(OptimizeResult {
                    iterations: iter + 1,
                    residual_norm: cost_try.sqrt(),
                    termination: Termination::Cost,
                });
            }

            let rho = if finite(cost_try) && predicted > T::ZERO {
                (cost - cost_try) / predicted
            } else {
                -T::ONE
            };
            if rho > T::from_f64(1.0e-4) {
                let two_rho_minus_one = T::from_f64(2.0) * rho - T::ONE;
                let shrink = (T::ONE - two_rho_minus_one * two_rho_minus_one * two_rho_minus_one)
                    .max(one_third);
                lambda *= shrink;
                nu = T::from_f64(2.0);
                let small = norm(dx) <= opts.tol_x * (norm(x) + opts.tol_x);
                x.copy_from_slice(x_try);
                r.copy_from_slice(r_try);
                cost = cost_try;
                accepted = true;
                if small {
                    return Ok(OptimizeResult {
                        iterations: iter + 1,
                        residual_norm: cost.sqrt(),
                        termination: Termination::Step,
                    });
                }
                break;
            }
            lambda *= nu;
            nu *= T::from_f64(2.0);
            if lambda > max_lambda {
                return Err(OptimizeError::NoConvergence);
            }
        }
        if !accepted {
            return Err(OptimizeError::NoConvergence);
        }
    }
    Err(OptimizeError::NoConvergence)
}

/// Levenberg-Marquardt-Verfahren für `min ‖r(x)‖₂` mit `m ≥ n` Residuen und `n` Parametern.
///
/// Das Standardverfahren für nichtlineare Kurvenanpassung. Es dämpft den Gauß-Newton-Schritt
/// adaptiv (`(JᵀJ + λ·diag(JᵀJ))·δ = −Jᵀr`, Marquardt-Skalierung): bei weitem Startwert verhält es
/// sich wie ein vorsichtiger Gradientenabstieg, nahe der Lösung wie Gauß-Newton. Es kommt auch mit
/// Rangmangel der Jacobi-Matrix zurecht und ist deutlich robuster als
/// [`gauss_newton`](super::gauss_newton).
///
/// - `residual(x, r)` schreibt `r(x)`; `jacobian(x, j)` schreibt die Jacobi-Matrix zeilenweise
///   (`j[i*n + k] = ∂rᵢ/∂xₖ`).
/// - `x`: Startwert und Ergebnis (`n` Elemente), `m`: Anzahl der Residuen.
/// - `work`: mindestens [`levenberg_marquardt_work_len`]`(m, n)` Elemente.
///
/// Fehler: [`OptimizeError::InvalidInput`], [`OptimizeError::NotFinite`],
/// [`OptimizeError::NoConvergence`] (`x` enthält dann den letzten Stand).
pub fn levenberg_marquardt<T: Real>(
    residual: impl FnMut(&[T], &mut [T]),
    jacobian: impl FnMut(&[T], &mut [T]),
    x: &mut [T],
    m: usize,
    opts: OptimizeOptions<T>,
    work: &mut [T],
) -> Result<OptimizeResult<T>, OptimizeError> {
    lm_core(residual, Jac::Analytic(jacobian), x, m, opts, work)
}

/// Wie [`levenberg_marquardt`], bestimmt die Jacobi-Matrix aber selbst per Vorwärtsdifferenzen
/// (siehe [`numerical_jacobian`](super::numerical_jacobian)).
pub fn levenberg_marquardt_fd<T: Real>(
    residual: impl FnMut(&[T], &mut [T]),
    x: &mut [T],
    m: usize,
    opts: OptimizeOptions<T>,
    work: &mut [T],
) -> Result<OptimizeResult<T>, OptimizeError> {
    lm_core(residual, Jac::Numeric::<NoJacobian<T>>, x, m, opts, work)
}
