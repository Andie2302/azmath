use super::{OdeError, all_finite, finite, from_usize};
use azmath_core::Real;
use azmath_optimize::{OptimizeOptions, newton_system, newton_system_fd, newton_work_len};

/// Benötigter Arbeitsspeicher für die impliziten Verfahren bei `n` Zustandsgrößen.
pub const fn implicit_work_len(n: usize) -> usize {
    // Newton-Arbeitsspeicher plus y_alt, f_alt, f_tmp (3n)
    newton_work_len(n) + 3 * n
}

enum Jac<J> {
    Analytic(J),
    Numeric,
}

/// Platzhalter-Typ für `Jac::Numeric` (wird nie aufgerufen).
type NoJacobian<T> = fn(T, &[T], &mut [T]);

/// Gemeinsamer Kern des θ-Verfahrens: `yₙ₊₁ = yₙ + h·((1−θ)·f(tₙ, yₙ) + θ·f(tₙ₊₁, yₙ₊₁))`.
/// `θ = 1`: implizites Euler-Verfahren, `θ = ½`: implizite Trapezregel.
#[allow(clippy::too_many_arguments)]
fn theta_core<T, F, J>(
    mut f: F,
    mut jac: Jac<J>,
    theta: T,
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    newton: OptimizeOptions<T>,
    work: &mut [T],
    mut observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError>
where
    T: Real,
    F: FnMut(T, &[T], &mut [T]),
    J: FnMut(T, &[T], &mut [T]),
{
    let n = y.len();
    if n == 0 || steps == 0 || work.len() < implicit_work_len(n) {
        return Err(OdeError::InvalidInput);
    }
    if !finite(t0) || !finite(t1) || !all_finite(y) {
        return Err(OdeError::NotFinite);
    }

    let (newton_work, rest) = work.split_at_mut(newton_work_len(n));
    let (y_old, rest) = rest.split_at_mut(n);
    let (f_old, rest) = rest.split_at_mut(n);
    let f_tmp = &mut rest[..n];

    let h = (t1 - t0) / from_usize::<T>(steps);
    let explicit_part = T::ONE - theta;
    observer(t0, y);
    for i in 0..steps {
        let t = t0 + h * from_usize::<T>(i);
        let t_next = t0 + h * from_usize::<T>(i + 1);
        y_old.copy_from_slice(y);
        if explicit_part != T::ZERO {
            f(t, y_old, f_old);
            if !all_finite(f_old) {
                return Err(OdeError::NotFinite);
            }
        }

        // Gesucht ist z = yₙ₊₁ mit g(z) = z − yₙ − h·((1−θ)·fₙ + θ·f(tₙ₊₁, z)) = 0
        let g = |z: &[T], out: &mut [T]| {
            f(t_next, z, f_tmp);
            for k in 0..n {
                let rhs = explicit_part * f_old[k] + theta * f_tmp[k];
                out[k] = z[k] - y_old[k] - h * rhs;
            }
        };
        // Startwert für Newton ist y (= yₙ)
        match &mut jac {
            Jac::Analytic(j) => {
                // ∂g/∂z = I − θ·h·∂f/∂y
                let scale = theta * h;
                let jg = |z: &[T], out: &mut [T]| {
                    j(t_next, z, out);
                    for v in out.iter_mut() {
                        *v = -scale * *v;
                    }
                    for k in 0..n {
                        out[k * n + k] += T::ONE;
                    }
                };
                newton_system(g, jg, y, newton, newton_work)?;
            }
            Jac::Numeric => {
                newton_system_fd(g, y, newton, newton_work)?;
            }
        }
        if !all_finite(y) {
            return Err(OdeError::NotFinite);
        }
        observer(t_next, y);
    }
    Ok(())
}

/// Implizites (rückwärtiges) Euler-Verfahren mit `steps` gleich großen Schritten von `t0` nach `t1`
/// (Ordnung 1, `yₙ₊₁ = yₙ + h·f(tₙ₊₁, yₙ₊₁)`).
///
/// Unbedingt stabil für abklingende Lösungen (L-stabil), daher das robuste Verfahren für **steife**
/// Probleme, bei denen explizite Verfahren winzige Schritte brauchen. Jeder Schritt löst das
/// nichtlineare Gleichungssystem für `yₙ₊₁` mit dem Newton-Verfahren
/// ([`newton_system`](azmath_optimize::newton_system)).
///
/// - `jacobian(t, y, j)` schreibt `∂f/∂y` (`n × n`, zeilenweise: `j[i*n + k] = ∂fᵢ/∂yₖ`).
///   Ohne Ableitung: [`implicit_euler_fd`] bestimmt sie per Differenzen.
/// - `newton`: Einstellungen für das Newton-Verfahren (`OptimizeOptions::default()` passt meist).
/// - `work`: mindestens [`implicit_work_len`]`(n)` Elemente.
/// - `observer(t, y)` wird mit dem Anfangszustand und nach jedem Schritt aufgerufen.
///
/// Fehler: [`OdeError::InvalidInput`], [`OdeError::NotFinite`], [`OdeError::NewtonFailed`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath_ode::{implicit_euler, implicit_work_len};
/// use azmath_optimize::OptimizeOptions;
///
/// // steif: y' = −1000·y, y(0) = 1. Das explizite Euler-Verfahren wäre bei h = 0.05 instabil.
/// let f = |_t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -1000.0 * y[0];
/// let jac = |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = -1000.0;
/// let mut y = [1.0];
/// let mut work = [0.0; implicit_work_len(1)];
/// implicit_euler(f, jac, 0.0, 1.0, &mut y, 20, OptimizeOptions::default(), &mut work, |_, _| {})
///     .unwrap();
/// assert!(y[0] > 0.0 && y[0] < 1e-6); // klingt ab, ohne zu explodieren
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn implicit_euler<T: Real>(
    f: impl FnMut(T, &[T], &mut [T]),
    jacobian: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    newton: OptimizeOptions<T>,
    work: &mut [T],
    observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError> {
    theta_core(
        f,
        Jac::Analytic(jacobian),
        T::ONE,
        t0,
        t1,
        y,
        steps,
        newton,
        work,
        observer,
    )
}

/// Wie [`implicit_euler`], bestimmt `∂f/∂y` aber selbst per Vorwärtsdifferenzen.
#[allow(clippy::too_many_arguments)]
pub fn implicit_euler_fd<T: Real>(
    f: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    newton: OptimizeOptions<T>,
    work: &mut [T],
    observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError> {
    theta_core(
        f,
        Jac::Numeric::<NoJacobian<T>>,
        T::ONE,
        t0,
        t1,
        y,
        steps,
        newton,
        work,
        observer,
    )
}

/// Implizite Trapezregel (Crank-Nicolson) mit `steps` gleich großen Schritten von `t0` nach `t1`
/// (Ordnung 2, `yₙ₊₁ = yₙ + h/2·(f(tₙ, yₙ) + f(tₙ₊₁, yₙ₊₁))`).
///
/// A-stabil und genauer als das implizite Euler-Verfahren, dämpft sehr steife Anteile aber nicht
/// so stark (sie können schwach oszillieren). Parameter wie bei [`implicit_euler`].
#[allow(clippy::too_many_arguments)]
pub fn implicit_trapezoid<T: Real>(
    f: impl FnMut(T, &[T], &mut [T]),
    jacobian: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    newton: OptimizeOptions<T>,
    work: &mut [T],
    observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError> {
    theta_core(
        f,
        Jac::Analytic(jacobian),
        T::from_f64(0.5),
        t0,
        t1,
        y,
        steps,
        newton,
        work,
        observer,
    )
}

/// Wie [`implicit_trapezoid`], bestimmt `∂f/∂y` aber selbst per Vorwärtsdifferenzen.
#[allow(clippy::too_many_arguments)]
pub fn implicit_trapezoid_fd<T: Real>(
    f: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    newton: OptimizeOptions<T>,
    work: &mut [T],
    observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError> {
    theta_core(
        f,
        Jac::Numeric::<NoJacobian<T>>,
        T::from_f64(0.5),
        t0,
        t1,
        y,
        steps,
        newton,
        work,
        observer,
    )
}
