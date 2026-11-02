#![allow(clippy::needless_range_loop)]

use super::implicit::{Jac, NoJacobian};
use super::{OdeError, all_finite, finite, from_usize};
use crate::Real;
use crate::optimize::{invert, numerical_jacobian};

/// Benötigter Arbeitsspeicher für [`ros2`] und [`ros2_fd`] bei `n` Zustandsgrößen.
pub const fn ros2_work_len(n: usize) -> usize {
    // Erweitertes System der Größe m = n + 1 (die Zeit wird als Zustand mitgeführt):
    // Matrix, Inverse, Gauß-Jordan (2m²), Jacobi-Matrix (n²), sechs Vektoren der Länge m, f(t+δ) (n), Differenzen-Puffer (2n)
    let m = n + 1;
    4 * m * m + n * n + 6 * m + 3 * n
}

#[allow(clippy::too_many_arguments)]
fn ros2_core<T, F, J>(
    mut f: F,
    mut jac: Jac<J>,
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    work: &mut [T],
    mut observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError>
where
    T: Real,
    F: FnMut(T, &[T], &mut [T]),
    J: FnMut(T, &[T], &mut [T]),
{
    let n = y.len();
    if n == 0 || steps == 0 || work.len() < ros2_work_len(n) {
        return Err(OdeError::InvalidInput);
    }
    if !finite(t0) || !finite(t1) || !all_finite(y) {
        return Err(OdeError::NotFinite);
    }
    let m = n + 1;
    let (a, rest) = work.split_at_mut(m * m);
    let (ainv, rest) = rest.split_at_mut(m * m);
    let (aug, rest) = rest.split_at_mut(2 * m * m);
    let (jn, rest) = rest.split_at_mut(n * n);
    let (f0, rest) = rest.split_at_mut(m);
    let (f1, rest) = rest.split_at_mut(m);
    let (k1, rest) = rest.split_at_mut(m);
    let (k2, rest) = rest.split_at_mut(m);
    let (stage, rest) = rest.split_at_mut(m);
    let (rhs, rest) = rest.split_at_mut(m);
    let (fplus, rest) = rest.split_at_mut(n);
    let fd = &mut rest[..2 * n];

    let h = (t1 - t0) / from_usize::<T>(steps);
    let gamma = T::ONE + T::ONE / T::from_f64(core::f64::consts::SQRT_2);
    let gh = gamma * h;
    let sqrt_eps = T::EPSILON.sqrt();
    let (c1, c2) = (T::from_f64(1.5), T::from_f64(0.5));
    let two = T::from_f64(2.0);

    observer(t0, y);
    for i in 0..steps {
        let t = t0 + h * from_usize::<T>(i);
        f(t, y, &mut f0[..n]);
        f0[n] = T::ONE;
        if !all_finite(&f0[..n]) {
            return Err(OdeError::NotFinite);
        }
        match &mut jac {
            Jac::Analytic(jf) => jf(t, y, jn),
            Jac::Numeric => {
                let f_ref = &mut f;
                numerical_jacobian(|x: &[T], r: &mut [T]| f_ref(t, x, r), y, &f0[..n], jn, fd);
            }
        }
        // Zeitableitung ∂f/∂t per Vorwärtsdifferenz
        let delta = sqrt_eps * T::ONE.max(t.abs());
        f(t + delta, y, fplus);
        // A = I − γh·Ĵ mit Ĵ = [[J, f_t], [0, 0]]
        for r in 0..n {
            for c in 0..n {
                let id = if r == c { T::ONE } else { T::ZERO };
                a[r * m + c] = id - gh * jn[r * n + c];
            }
            a[r * m + n] = -gh * (fplus[r] - f0[r]) / delta;
        }
        for c in 0..n {
            a[n * m + c] = T::ZERO;
        }
        a[n * m + n] = T::ONE;
        if !invert(a, m, aug, ainv) {
            return Err(OdeError::NewtonFailed);
        }
        // k1 = A⁻¹·f̂(ŷ)
        for r in 0..m {
            let mut s = T::ZERO;
            for c in 0..m {
                s += ainv[r * m + c] * f0[c];
            }
            k1[r] = s;
        }
        // zweite Stufe an ŷ + h·k1
        for k in 0..n {
            stage[k] = y[k] + h * k1[k];
        }
        f(t + h * k1[n], &stage[..n], &mut f1[..n]);
        f1[n] = T::ONE;
        for r in 0..m {
            rhs[r] = f1[r] - two * k1[r];
        }
        for r in 0..m {
            let mut s = T::ZERO;
            for c in 0..m {
                s += ainv[r * m + c] * rhs[c];
            }
            k2[r] = s;
        }
        for k in 0..n {
            y[k] += h * (c1 * k1[k] + c2 * k2[k]);
        }
        if !all_finite(y) {
            return Err(OdeError::NotFinite);
        }
        observer(t0 + h * from_usize::<T>(i + 1), y);
    }
    Ok(())
}

/// Linear-implizites Rosenbrock-Verfahren **ROS2** (Verwer et al., Ordnung 2, L-stabil, `γ = 1 + 1/√2`) mit `steps` gleich
/// großen Schritten von `t0` nach `t1`.
///
/// Statt eines Newton-Verfahrens löst jeder Schritt nur **lineare** Gleichungssysteme mit `I − γ·h·J` (hier über die
/// Inverse), wertet also `f` zweimal und die Jacobi-Matrix einmal aus – kein Iterieren, kein `NewtonFailed` durch
/// Nichtkonvergenz. Gut für steife Probleme mäßiger Genauigkeit; für hohe Genauigkeit [`radau5`](super::radau5).
/// Die Zeitabhängigkeit von `f` wird über eine Vorwärtsdifferenz `∂f/∂t` berücksichtigt (die Zeit läuft als
/// zusätzlicher Zustand mit, deshalb ist der Speicherbedarf `O(n²)` für `m = n + 1`).
///
/// Dies ist ein Verfahren der Rosenbrock-Familie, **nicht** das höhere Verfahren RODAS (Ordnung 4) der ROADMAP.
///
/// - `jacobian(t, y, j)` schreibt `∂f/∂y` (`n × n`, zeilenweise). Ohne Ableitung: [`ros2_fd`].
/// - `work`: mindestens [`ros2_work_len`]`(n)`. `observer(t, y)` wird am Anfang und nach jedem Schritt aufgerufen.
///
/// Fehler: [`OdeError::InvalidInput`], [`OdeError::NotFinite`], [`OdeError::NewtonFailed`] (die Matrix `I − γhJ` ist singulär).
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::ode::{ros2, ros2_work_len};
///
/// // y' = −y, y(0) = 1
/// let mut y = [1.0];
/// let mut work = [0.0; ros2_work_len(1)];
/// ros2(
///     |_t, y: &[f64], dy: &mut [f64]| dy[0] = -y[0],
///     |_t, _y: &[f64], j: &mut [f64]| j[0] = -1.0,
///     0.0, 1.0, &mut y, 100, &mut work, |_, _| {},
/// ).unwrap();
/// assert!((y[0] - (-1.0f64).exp()).abs() < 1e-4);
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn ros2<T: Real>(
    f: impl FnMut(T, &[T], &mut [T]),
    jacobian: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    work: &mut [T],
    observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError> {
    ros2_core(f, Jac::Analytic(jacobian), t0, t1, y, steps, work, observer)
}

/// Wie [`ros2`], bestimmt `∂f/∂y` aber selbst per Vorwärtsdifferenzen.
#[allow(clippy::too_many_arguments)]
pub fn ros2_fd<T: Real>(
    f: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    work: &mut [T],
    observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError> {
    ros2_core(
        f,
        Jac::Numeric::<NoJacobian<T>>,
        t0,
        t1,
        y,
        steps,
        work,
        observer,
    )
}
