use super::broyden::invert;
use super::{
    Jac, NoJacobian, OptimizeError, OptimizeOptions, OptimizeResult, Termination, dot,
    eval_jacobian, finite, norm,
};
use crate::Real;

/// Arbeitsspeicher von [`dogleg`] und [`dogleg_fd`] für `n` Unbekannte: `4·n² + 10·n` Elemente.
pub const fn dogleg_work_len(n: usize) -> usize {
    4 * n * n + 10 * n
}

#[allow(clippy::needless_range_loop)]
fn dogleg_core<T, F, J>(
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
    super::validate_options(opts)?;
    let n = x.len();
    if n == 0 || work.len() < dogleg_work_len(n) {
        return Err(OptimizeError::InvalidInput);
    }
    if x.iter().any(|&v| !finite(v)) {
        return Err(OptimizeError::NotFinite);
    }
    let (jm, rest) = work.split_at_mut(n * n);
    let (jinv, rest) = rest.split_at_mut(n * n);
    let (aug, rest) = rest.split_at_mut(2 * n * n);
    let (fx, rest) = rest.split_at_mut(n);
    let (g, rest) = rest.split_at_mut(n);
    let (p_newton, rest) = rest.split_at_mut(n);
    let (p_cauchy, rest) = rest.split_at_mut(n);
    let (p, rest) = rest.split_at_mut(n);
    let (x_new, rest) = rest.split_at_mut(n);
    let (f_new, rest) = rest.split_at_mut(n);
    let (jg, rest) = rest.split_at_mut(n);
    let (jp, rest) = rest.split_at_mut(n);
    let _ = rest;

    let quarter = T::from_f64(0.25);
    let two = T::from_f64(2.0);

    f(x, fx);
    if fx.iter().any(|&v| !finite(v)) {
        return Err(OptimizeError::NotFinite);
    }
    let mut fnorm = norm(fx);
    if !finite(fnorm) {
        return Err(OptimizeError::NotFinite);
    }
    let mut model_scale = T::ONE;
    let mut radius = T::ZERO;
    let mut have_jacobian = false;
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
        if !have_jacobian {
            let (scratch, _) = aug.split_at_mut(2 * n);
            eval_jacobian(&mut f, &mut jac, x, fx, jm, scratch);
            if jm.iter().any(|&v| !finite(v)) {
                return Err(OptimizeError::NotFinite);
            }
            model_scale = jm
                .iter()
                .chain(fx.iter())
                .fold(T::ZERO, |s, v| s.max(v.abs()));
            for v in jm.iter_mut() {
                *v /= model_scale;
            }
            // g = (J/scale)ᵀ·(F/scale)
            for k in 0..n {
                let mut s = T::ZERO;
                for i in 0..n {
                    s += jm[i * n + k] * (fx[i] / model_scale);
                }
                g[k] = s;
            }
            have_jacobian = true;
        }
        let gnorm = norm(g);
        if !finite(gnorm) {
            return Err(OptimizeError::NotFinite);
        }
        if gnorm == T::ZERO {
            return Err(OptimizeError::NoConvergence);
        }
        // Newton-Schritt (falls J regulär)
        let newton_ok = invert(jm, n, aug, jinv);
        if newton_ok {
            for i in 0..n {
                p_newton[i] =
                    -(0..n).fold(T::ZERO, |s, k| s + jinv[i * n + k] * (fx[k] / model_scale));
            }
        }
        // Einheitsrichtung vermeidet das Quadrieren der (eventuell winzigen) Gradientenlänge.
        for i in 0..n {
            jg[i] = (0..n).fold(T::ZERO, |s, k| s + jm[i * n + k] * (g[k] / gnorm));
        }
        let jgnorm = norm(jg);
        let cauchy_len = (gnorm / jgnorm) / jgnorm;
        for i in 0..n {
            p_cauchy[i] = -cauchy_len * (g[i] / gnorm);
        }
        if radius == T::ZERO {
            // Anfangsradius: Newton-Länge, aber nicht über max(1, ‖x‖)
            let start = if newton_ok {
                norm(p_newton)
            } else {
                norm(p_cauchy)
            };
            radius = start.min(T::ONE.max(norm(x))).max(opts.tol_x);
        }
        // Dogleg-Pfad
        let pn_len = if newton_ok {
            norm(p_newton)
        } else {
            T::INFINITY
        };
        if pn_len <= radius {
            p.copy_from_slice(p_newton);
        } else {
            let pc_len = norm(p_cauchy);
            if pc_len >= radius || !newton_ok {
                for i in 0..n {
                    p[i] = -radius * (g[i] / gnorm);
                }
            } else {
                // τ ∈ (0, 1) mit ‖p_cauchy + τ·(p_newton − p_cauchy)‖ = radius
                let (mut a, mut b) = (T::ZERO, T::ZERO);
                for i in 0..n {
                    let d = p_newton[i] - p_cauchy[i];
                    a += d * d;
                    b += two * p_cauchy[i] * d;
                }
                let c = pc_len * pc_len - radius * radius;
                let tau = (-b + (b * b - T::from_f64(4.0) * a * c).max(T::ZERO).sqrt()) / (two * a);
                for i in 0..n {
                    p[i] = p_cauchy[i] + tau * (p_newton[i] - p_cauchy[i]);
                }
            }
        }
        let step = norm(p);
        for i in 0..n {
            x_new[i] = x[i] + p[i];
        }
        let fn_new = if x_new.iter().all(|&v| finite(v)) {
            f(x_new, f_new);
            norm(f_new)
        } else {
            T::INFINITY
        };
        // Abnahmen relativ zur alten Residuumsnorm statt Differenzen riesiger Quadrate.
        for i in 0..n {
            jp[i] = dot(&jm[i * n..(i + 1) * n], p) + fx[i] / model_scale;
        }
        let model_ratio = norm(jp) / (fnorm / model_scale);
        let actual_ratio = fn_new / fnorm;
        let predicted = (T::ONE - model_ratio) * (T::ONE + model_ratio);
        let actual = (T::ONE - actual_ratio) * (T::ONE + actual_ratio);
        let rho = if finite(fn_new) && finite(predicted) && predicted > T::ZERO {
            actual / predicted
        } else {
            -T::ONE
        };
        if rho < quarter {
            radius = quarter * step;
        } else if rho > T::from_f64(0.75) && step >= T::from_f64(0.9) * radius {
            radius *= two;
        }
        if rho > T::from_f64(1e-4) {
            x.copy_from_slice(x_new);
            fx.copy_from_slice(f_new);
            fnorm = fn_new;
            have_jacobian = false;
            if fnorm <= opts.tol_f {
                return super::root_result(iter + 1, fnorm, opts);
            }
            if step <= opts.tol_x * (norm(x) + opts.tol_x) {
                return Err(OptimizeError::NoConvergence);
            }
        } else if radius <= opts.tol_x * (norm(x) + opts.tol_x) {
            // Vertrauensbereich kollabiert: kein Fortschritt mehr möglich
            return Err(OptimizeError::NoConvergence);
        }
    }
    Err(OptimizeError::NoConvergence)
}

/// Dogleg-Verfahren (Powell, Vertrauensbereich) für das nichtlineare Gleichungssystem `F(x) = 0` mit `n`
/// Gleichungen und `n` Unbekannten.
///
/// In jedem Schritt wird zwischen Gauß-Newton-Schritt `−J⁻¹F` und Cauchy-Schritt (steilster Abstieg von `½‖F‖²`)
/// auf einem Kreisbogen („Hundebein“) innerhalb eines **Vertrauensradius** gewählt; der Radius wächst oder schrumpft
/// je nach Übereinstimmung von vorhergesagter und tatsächlicher Abnahme. Das ist robuster als das gedämpfte Newton-Verfahren
/// ([`newton_system`](super::newton_system)) bei schlechtem Startwert, und anders als dort kann eine singuläre Jacobi-Matrix
/// vorkommen: der Schritt fällt dann auf den Cauchy-Schritt zurück.
///
/// - `f(x, out)` schreibt `F(x)`; `jacobian(x, j)` die `n × n`-Jacobi-Matrix zeilenweise.
/// - `x`: Startwert und Ergebnis. `work`: mindestens [`dogleg_work_len`]`(n)` Elemente.
/// - Erfolg nur bei `‖F(x)‖₂ ≤ tol_f` ([`Termination::Residual`]). Stationäre Punkte mit
///   größerem Residuum und stagnierende Schritte ergeben [`OptimizeError::NoConvergence`].
///
/// Fehler: [`OptimizeError::InvalidInput`], [`OptimizeError::NotFinite`], [`OptimizeError::NoConvergence`].
pub fn dogleg<T: Real>(
    f: impl FnMut(&[T], &mut [T]),
    jacobian: impl FnMut(&[T], &mut [T]),
    x: &mut [T],
    opts: OptimizeOptions<T>,
    work: &mut [T],
) -> Result<OptimizeResult<T>, OptimizeError> {
    dogleg_core(f, Jac::Analytic(jacobian), x, opts, work)
}

/// Wie [`dogleg`], bestimmt die Jacobi-Matrix aber selbst per Vorwärtsdifferenzen.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::optimize::{dogleg_fd, dogleg_work_len, OptimizeOptions};
///
/// // arctan(x) = 0 mit Startwert 2: das reine Newton-Verfahren divergiert, Dogleg nicht
/// let mut x = [2.0f64];
/// let mut work = [0.0; dogleg_work_len(1)];
/// dogleg_fd(|x: &[f64], out: &mut [f64]| out[0] = x[0].atan(), &mut x, OptimizeOptions::default(), &mut work).unwrap();
/// assert!(x[0].abs() < 1e-8);
/// # }
/// ```
pub fn dogleg_fd<T: Real>(
    f: impl FnMut(&[T], &mut [T]),
    x: &mut [T],
    opts: OptimizeOptions<T>,
    work: &mut [T],
) -> Result<OptimizeResult<T>, OptimizeError> {
    dogleg_core(f, Jac::Numeric::<NoJacobian<T>>, x, opts, work)
}
