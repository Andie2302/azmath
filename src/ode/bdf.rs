use super::implicit::{Jac, NoJacobian};
use super::radau::{radau_step, radau_work_len};
use super::{OdeError, all_finite, finite, from_usize};
use crate::Real;
use crate::optimize::{OptimizeOptions, newton_system, newton_system_fd, newton_work_len};

/// Höchste unterstützte Ordnung der BDF-Verfahren.
const MAX_ORDER: usize = 5;

/// Benötigter Arbeitsspeicher für [`bdf`] und [`bdf_fd`] bei `n` Zustandsgrößen.
pub const fn bdf_work_len(n: usize) -> usize {
    // Newton (n) + Verlauf (5n) + f-Zwischenspeicher (n) + Anlaufrechnung mit Radau IIA
    newton_work_len(n) + MAX_ORDER * n + n + radau_work_len(n)
}

/// Koeffizienten `(α₀ … α_{p−1}, β)` der BDF-Formel `yₙ₊₁ = Σⱼ αⱼ·yₙ₋ⱼ + β·h·f(tₙ₊₁, yₙ₊₁)`.
fn coefficients(p: usize) -> ([f64; MAX_ORDER], f64) {
    match p {
        1 => ([1.0, 0.0, 0.0, 0.0, 0.0], 1.0),
        2 => ([4.0 / 3.0, -1.0 / 3.0, 0.0, 0.0, 0.0], 2.0 / 3.0),
        3 => ([18.0 / 11.0, -9.0 / 11.0, 2.0 / 11.0, 0.0, 0.0], 6.0 / 11.0),
        4 => (
            [48.0 / 25.0, -36.0 / 25.0, 16.0 / 25.0, -3.0 / 25.0, 0.0],
            12.0 / 25.0,
        ),
        _ => (
            [
                300.0 / 137.0,
                -300.0 / 137.0,
                200.0 / 137.0,
                -75.0 / 137.0,
                12.0 / 137.0,
            ],
            60.0 / 137.0,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn bdf_core<T, F, J>(
    mut f: F,
    mut jac: Jac<J>,
    order: usize,
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
    if n == 0 || steps == 0 || order == 0 || order > MAX_ORDER || work.len() < bdf_work_len(n) {
        return Err(OdeError::InvalidInput);
    }
    if !finite(t0) || !finite(t1) || !all_finite(y) {
        return Err(OdeError::NotFinite);
    }
    let (newton_work, rest) = work.split_at_mut(newton_work_len(n));
    let (hist, rest) = rest.split_at_mut(MAX_ORDER * n);
    let (f_tmp, rest) = rest.split_at_mut(n);
    let radau_work = &mut rest[..radau_work_len(n)];

    let h = (t1 - t0) / from_usize::<T>(steps);
    let (alpha_f64, beta_f64) = coefficients(order);
    let alpha: [T; MAX_ORDER] = alpha_f64.map(T::from_f64);
    let beta = T::from_f64(beta_f64);

    // hist[j·n..] = y an tₙ₋ⱼ
    hist[..n].copy_from_slice(y);
    observer(t0, y);
    for i in 0..steps {
        let t = t0 + h * from_usize::<T>(i);
        let t_next = t0 + h * from_usize::<T>(i + 1);
        if i + 1 < order {
            // Anlaufrechnung: Radau IIA (Ordnung 5, L-stabil) liefert die fehlenden Startwerte
            radau_step(&mut f, &mut jac, t, h, y, newton, radau_work)?;
        } else {
            // z − Σ αⱼ·yₙ₋ⱼ − β·h·f(tₙ₊₁, z) = 0
            let hist_ref: &[T] = hist;
            let g = |z: &[T], out: &mut [T]| {
                f(t_next, z, f_tmp);
                for k in 0..n {
                    let mut s = T::ZERO;
                    for j in 0..order {
                        s += alpha[j] * hist_ref[j * n + k];
                    }
                    out[k] = z[k] - s - beta * h * f_tmp[k];
                }
            };
            match &mut jac {
                Jac::Analytic(jf) => {
                    let scale = beta * h;
                    let jg = |z: &[T], out: &mut [T]| {
                        jf(t_next, z, out);
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
        }
        if !all_finite(y) {
            return Err(OdeError::NotFinite);
        }
        hist.copy_within(0..(MAX_ORDER - 1) * n, n);
        hist[..n].copy_from_slice(y);
        observer(t_next, y);
    }
    Ok(())
}

/// BDF-Verfahren (rückwärtige Differenzenformeln) der Ordnung `order ∈ 1..=5` mit `steps` gleich großen Schritten von `t0`
/// nach `t1`.
///
/// `yₙ₊₁ = Σⱼ αⱼ·yₙ₋ⱼ + β·h·f(tₙ₊₁, yₙ₊₁)`; jeder Schritt löst die implizite Gleichung mit Newton. BDF1 ist das implizite
/// Euler-Verfahren; BDF1 und BDF2 sind L-stabil, BDF3 bis BDF5 sind nur A(α)-stabil (Winkel `86°`, `73°`, `51°`):
/// für Probleme mit **stark oszillierenden** schnellen Anteilen sind sie ungeeignet. Pro Schritt wird nur **eine**
/// implizite Gleichung gelöst, das ist billiger als Radau IIA bei großen Systemen.
///
/// Die ersten `order − 1` Schritte rechnet [`radau5`](super::radau5) (Ordnung 5), damit die Startwerte die Ordnung
/// nicht verderben. **Feste Schrittweite**; Schrittweiten- und Ordnungssteuerung sind noch offen.
///
/// - `jacobian(t, y, j)` schreibt `∂f/∂y` (`n × n`, zeilenweise). Ohne Ableitung: [`bdf_fd`].
/// - `work`: mindestens [`bdf_work_len`]`(n)`. `observer(t, y)` wird am Anfang und nach jedem Schritt aufgerufen.
///
/// Fehler: [`OdeError::InvalidInput`] (auch `order` außerhalb von `1..=5`), [`OdeError::NotFinite`], [`OdeError::NewtonFailed`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::ode::{bdf, bdf_work_len};
/// use azmath::optimize::OptimizeOptions;
///
/// // y' = −y, y(0) = 1, BDF3
/// let mut y = [1.0];
/// let mut work = [0.0; bdf_work_len(1)];
/// bdf(
///     |_t, y: &[f64], dy: &mut [f64]| dy[0] = -y[0],
///     |_t, _y: &[f64], j: &mut [f64]| j[0] = -1.0,
///     3, 0.0, 1.0, &mut y, 200, OptimizeOptions::default(), &mut work, |_, _| {},
/// ).unwrap();
/// assert!((y[0] - (-1.0f64).exp()).abs() < 1e-6);
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn bdf<T: Real>(
    f: impl FnMut(T, &[T], &mut [T]),
    jacobian: impl FnMut(T, &[T], &mut [T]),
    order: usize,
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    newton: OptimizeOptions<T>,
    work: &mut [T],
    observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError> {
    bdf_core(
        f,
        Jac::Analytic(jacobian),
        order,
        t0,
        t1,
        y,
        steps,
        newton,
        work,
        observer,
    )
}

/// Wie [`bdf`], bestimmt die Jacobi-Matrix aber selbst per Vorwärtsdifferenzen.
#[allow(clippy::too_many_arguments)]
pub fn bdf_fd<T: Real>(
    f: impl FnMut(T, &[T], &mut [T]),
    order: usize,
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    newton: OptimizeOptions<T>,
    work: &mut [T],
    observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError> {
    bdf_core(
        f,
        Jac::Numeric::<NoJacobian<T>>,
        order,
        t0,
        t1,
        y,
        steps,
        newton,
        work,
        observer,
    )
}
