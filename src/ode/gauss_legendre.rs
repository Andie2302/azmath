use super::implicit::{Jac, NoJacobian};
use super::{OdeError, all_finite, finite, fixed_times, weighted_update};
use crate::Real;
use crate::optimize::{OptimizeOptions, newton_system, newton_system_fd, newton_work_len};

/// Arbeitselemente für [`gauss_legendre4`] und [`gauss_legendre4_fd`].
/// Bei nicht darstellbarer Größe wird `usize::MAX` zurückgegeben.
pub const fn gauss_legendre4_work_len(n: usize) -> usize {
    n.saturating_mul(n)
        .saturating_mul(5)
        .saturating_add(n.saturating_mul(21))
}

#[allow(clippy::too_many_arguments)]
fn integrate<T, F, J>(
    mut f: F,
    mut jac: Jac<J>,
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    options: OptimizeOptions<T>,
    work: &mut [T],
    mut observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError>
where
    T: Real,
    F: FnMut(T, &[T], &mut [T]),
    J: FnMut(T, &[T], &mut [T]),
{
    let n = y.len();
    let required = gauss_legendre4_work_len(n);
    if n == 0 || steps == 0 || required == usize::MAX || work.len() < required {
        return Err(OdeError::InvalidInput);
    }
    crate::optimize::validate_options(options)?;
    if !finite(t0) || !finite(t1) || !all_finite(y) {
        return Err(OdeError::NotFinite);
    }
    observer(t0, y);
    if t0 == t1 {
        return Ok(());
    }
    let (nw, rest) = work.split_at_mut(newton_work_len(2 * n));
    let (stages, rest) = rest.split_at_mut(2 * n);
    let (candidate, rest) = rest.split_at_mut(n);
    let (values, rest) = rest.split_at_mut(2 * n);
    let jtmp = &mut rest[..n * n];
    let d = T::from_f64(0.288_675_134_594_812_87);
    let half = T::from_f64(0.5);
    let quarter = T::from_f64(0.25);
    let c = [half - d, half + d];
    let a = [[quarter, quarter - d], [quarter + d, quarter]];
    for step in 0..steps {
        let (t, next, h) = fixed_times(t0, t1, step, steps)?;
        let times = [t + c[0] * h, t + c[1] * h];
        if !all_finite(&times) {
            return Err(OdeError::NotFinite);
        }
        for stage in stages.chunks_exact_mut(n) {
            stage.copy_from_slice(y);
        }
        // Unscaled stage-state residuals: tol_f is in state units.
        let residual = |z: &[T], out: &mut [T]| {
            for j in 0..2 {
                f(
                    times[j],
                    &z[j * n..(j + 1) * n],
                    &mut values[j * n..(j + 1) * n],
                );
            }
            for i in 0..2 {
                for k in 0..n {
                    out[i * n + k] =
                        z[i * n + k] - y[k] - h * (a[i][0] * values[k] + a[i][1] * values[n + k]);
                }
            }
        };
        match &mut jac {
            Jac::Analytic(jf) => {
                let derivative = |z: &[T], out: &mut [T]| {
                    for j in 0..2 {
                        jf(times[j], &z[j * n..(j + 1) * n], jtmp);
                        for i in 0..2 {
                            for r in 0..n {
                                for col in 0..n {
                                    let identity =
                                        if i == j && r == col { T::ONE } else { T::ZERO };
                                    out[(i * n + r) * (2 * n) + j * n + col] =
                                        identity - h * a[i][j] * jtmp[r * n + col];
                                }
                            }
                        }
                    }
                };
                newton_system(residual, derivative, stages, options, nw)?;
            }
            Jac::Numeric => {
                newton_system_fd(residual, stages, options, nw)?;
            }
        }
        for j in 0..2 {
            f(
                times[j],
                &stages[j * n..(j + 1) * n],
                &mut values[j * n..(j + 1) * n],
            );
        }
        if !all_finite(values) {
            return Err(OdeError::NotFinite);
        }
        for k in 0..n {
            candidate[k] = weighted_update(y[k], h, &[0.5, 0.5], |j| values[j * n + k]);
        }
        if !all_finite(candidate) {
            return Err(OdeError::NotFinite);
        }
        y.copy_from_slice(candidate);
        observer(next, y);
    }
    Ok(())
}

/// Zweistufiges implizites Gauß–Legendre-Runge–Kutta-Verfahren der Ordnung 4.
///
/// Integriert `y' = f(t, y)` mit `steps` festen Schritten von `t0` nach `t1`,
/// auch rückwärts. `jacobian` schreibt `∂f/∂y` zeilenweise (`n × n`). Die
/// gekoppelten `2n` Stufenzustände werden mit [`newton_system`] gelöst;
/// `options.tol_f` begrenzt die euklidische Norm des unskalierten Stufenresiduums
/// in Zustandseinheiten. Für Ordnung 4 muss der Newton-Fehler ausreichend kleiner
/// als der Diskretisierungsfehler sein. Es gibt keine adaptive Fehlerschätzung.
///
/// Das Verfahren ist A-stabil, aber nicht L-stabil. Für kanonische Hamilton-Systeme
/// ist es bei exakt gelösten Stufengleichungen symplektisch und zeitumkehrbar,
/// auch für nicht separierbare Hamilton-Funktionen. Newton-Toleranzen und Rundung
/// begrenzen diese Eigenschaften. Allgemeine nichtlineare Energien werden nicht
/// exakt erhalten; quadratische Invarianten werden in exakter Arithmetik erhalten.
///
/// `work` braucht mindestens [`gauss_legendre4_work_len`]`(y.len())` Elemente.
/// Der Beobachter erhält den Anfang und jeden vollständigen Schritt. Bei Fehlern
/// bleibt `y` beim letzten vollständigen Schritt. Ein leeres Zeitintervall ruft
/// nach der Eingabeprüfung nur den Beobachter auf, keine rechte Seite/Jacobi-Matrix.
///
/// Fehler: [`OdeError::InvalidInput`] (auch ungültige Newton-Toleranzen),
/// [`OdeError::NotFinite`], [`OdeError::StepSizeTooSmall`],
/// [`OdeError::NewtonFailed`]. Callbacks müssen alle Ausgabeelemente überschreiben.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::ode::{gauss_legendre4, gauss_legendre4_work_len};
/// use azmath::optimize::OptimizeOptions;
/// let mut y = [1.0_f64, 0.0];
/// let mut work = [0.0; gauss_legendre4_work_len(2)];
/// gauss_legendre4(
///     |_, y, out| { out[0] = y[1]; out[1] = -y[0]; },
///     |_, _, j| j.copy_from_slice(&[0.0, 1.0, -1.0, 0.0]),
///     0.0, 1.0, &mut y, 10, OptimizeOptions::default(), &mut work, |_, _| {}
/// ).unwrap();
/// assert!((y[0] - 1.0_f64.cos()).abs() < 2e-7);
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn gauss_legendre4<T: Real>(
    f: impl FnMut(T, &[T], &mut [T]),
    jacobian: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    options: OptimizeOptions<T>,
    work: &mut [T],
    observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError> {
    integrate(
        f,
        Jac::Analytic(jacobian),
        t0,
        t1,
        y,
        steps,
        options,
        work,
        observer,
    )
}

/// Wie [`gauss_legendre4`], mit Vorwärtsdifferenzen des gekoppelten Stufensystems.
/// Die erreichbare Genauigkeit hängt zusätzlich von der numerischen Ableitung ab.
#[allow(clippy::too_many_arguments)]
pub fn gauss_legendre4_fd<T: Real>(
    f: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    options: OptimizeOptions<T>,
    work: &mut [T],
    observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError> {
    integrate(
        f,
        Jac::Numeric::<NoJacobian<T>>,
        t0,
        t1,
        y,
        steps,
        options,
        work,
        observer,
    )
}
