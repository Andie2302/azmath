use super::implicit::{Jac, NoJacobian};
use super::{OdeError, all_finite, finite, from_usize};
use crate::Real;
use crate::optimize::{OptimizeOptions, newton_system, newton_system_fd, newton_work_len};

/// √6 (als Konstante, weil `f64::sqrt` in `no_std` nicht zur Verfügung steht).
const SQRT6: f64 = 2.449_489_742_783_178;

/// Knoten `c` und Matrix `A` des Radau-IIA-Verfahrens mit drei Stufen (Ordnung 5).
fn tableau() -> ([f64; 3], [[f64; 3]; 3]) {
    let s = SQRT6;
    (
        [(4.0 - s) / 10.0, (4.0 + s) / 10.0, 1.0],
        [
            [
                (88.0 - 7.0 * s) / 360.0,
                (296.0 - 169.0 * s) / 1800.0,
                (-2.0 + 3.0 * s) / 225.0,
            ],
            [
                (296.0 + 169.0 * s) / 1800.0,
                (88.0 + 7.0 * s) / 360.0,
                (-2.0 - 3.0 * s) / 225.0,
            ],
            [(16.0 - s) / 36.0, (16.0 + s) / 36.0, 1.0 / 9.0],
        ],
    )
}

/// Benötigter Arbeitsspeicher für [`radau5`] und [`radau5_fd`] bei `n` Zustandsgrößen.
pub const fn radau_work_len(n: usize) -> usize {
    // Newton auf 3n Unbekannten + Stufenwerte (3n) + alter Zustand (n) + f an den Stufen (3n) + Jacobi-Block (n²)
    newton_work_len(3 * n) + 3 * n + n + 3 * n + n * n
}

/// Ein Radau-IIA-Schritt der Größe `h` von `t` aus; `y` wird fortgeschrieben.
pub(super) fn radau_step<T, F, J>(
    f: &mut F,
    jac: &mut Jac<J>,
    t: T,
    h: T,
    y: &mut [T],
    newton: OptimizeOptions<T>,
    work: &mut [T],
) -> Result<(), OdeError>
where
    T: Real,
    F: FnMut(T, &[T], &mut [T]),
    J: FnMut(T, &[T], &mut [T]),
{
    let n = y.len();
    let (newton_work, rest) = work.split_at_mut(newton_work_len(3 * n));
    let (stage, rest) = rest.split_at_mut(3 * n);
    let (y_old, rest) = rest.split_at_mut(n);
    let (fbuf, rest) = rest.split_at_mut(3 * n);
    let jtmp = &mut rest[..n * n];

    let (c_f64, a_f64) = tableau();
    let c: [T; 3] = c_f64.map(T::from_f64);
    let a: [[T; 3]; 3] = a_f64.map(|row| row.map(T::from_f64));

    y_old.copy_from_slice(y);
    for i in 0..3 {
        stage[i * n..(i + 1) * n].copy_from_slice(y_old);
    }

    // G(Y)ᵢ = Yᵢ − yₙ − h·Σⱼ aᵢⱼ·f(t + cⱼh, Yⱼ)
    let g = |z: &[T], out: &mut [T]| {
        for j in 0..3 {
            f(
                t + c[j] * h,
                &z[j * n..(j + 1) * n],
                &mut fbuf[j * n..(j + 1) * n],
            );
        }
        for i in 0..3 {
            for k in 0..n {
                let mut s = T::ZERO;
                for j in 0..3 {
                    s += a[i][j] * fbuf[j * n + k];
                }
                out[i * n + k] = z[i * n + k] - y_old[k] - h * s;
            }
        }
    };
    match jac {
        Jac::Analytic(jf) => {
            // Block (i, j) = δᵢⱼ·I − h·aᵢⱼ·J(t + cⱼh, Yⱼ)
            let jg = |z: &[T], out: &mut [T]| {
                let w = 3 * n;
                for j in 0..3 {
                    jf(t + c[j] * h, &z[j * n..(j + 1) * n], jtmp);
                    for i in 0..3 {
                        for r in 0..n {
                            for col in 0..n {
                                let mut v = -h * a[i][j] * jtmp[r * n + col];
                                if i == j && r == col {
                                    v += T::ONE;
                                }
                                out[(i * n + r) * w + j * n + col] = v;
                            }
                        }
                    }
                }
            };
            newton_system(g, jg, stage, newton, newton_work)?;
        }
        Jac::Numeric => {
            newton_system_fd(g, stage, newton, newton_work)?;
        }
    }
    y.copy_from_slice(&stage[2 * n..3 * n]); // steif genau: yₙ₊₁ = Y₃
    if !all_finite(y) {
        return Err(OdeError::NotFinite);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn radau_core<T, F, J>(
    mut f: F,
    mut jac: Jac<J>,
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
    if n == 0 || steps == 0 || work.len() < radau_work_len(n) {
        return Err(OdeError::InvalidInput);
    }
    if !finite(t0) || !finite(t1) || !all_finite(y) {
        return Err(OdeError::NotFinite);
    }
    let h = (t1 - t0) / from_usize::<T>(steps);
    observer(t0, y);
    for i in 0..steps {
        let t = t0 + h * from_usize::<T>(i);
        radau_step(&mut f, &mut jac, t, h, y, newton, work)?;
        observer(t0 + h * from_usize::<T>(i + 1), y);
    }
    Ok(())
}

/// Radau-IIA-Verfahren mit drei Stufen (**Ordnung 5**, L-stabil, steif genau) mit `steps` gleich großen Schritten von `t0`
/// nach `t1`.
///
/// Jeder Schritt löst das gekoppelte nichtlineare System für die drei Stufenwerte (`3·n` Unbekannte) mit dem
/// Newton-Verfahren ([`newton_system`]). Radau IIA ist das Standardverfahren für **steife** Probleme hoher Genauigkeit;
/// wegen der Ordnung 5 reichen oft wenige Schritte. Die Schrittweite ist fest (adaptiv: siehe Ausbau in der ROADMAP).
///
/// - `jacobian(t, y, j)` schreibt `∂f/∂y` (`n × n`, zeilenweise). Ohne Ableitung: [`radau5_fd`].
/// - `newton`: Einstellungen für Newton (`OptimizeOptions::default()` passt meist).
/// - `work`: mindestens [`radau_work_len`]`(n)` Elemente. `observer(t, y)` wird am Anfang und nach jedem Schritt aufgerufen.
///
/// Fehler: [`OdeError::InvalidInput`], [`OdeError::NotFinite`], [`OdeError::NewtonFailed`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::ode::{radau5, radau_work_len};
/// use azmath::optimize::OptimizeOptions;
///
/// // steif: y' = −1000·(y − cos t), nach kurzer Anlaufzeit y ≈ cos t
/// let f = |t: f64, y: &[f64], dy: &mut [f64]| dy[0] = -1000.0 * (y[0] - t.cos());
/// let jac = |_t: f64, _y: &[f64], j: &mut [f64]| j[0] = -1000.0;
/// let mut y = [1.0];
/// let mut work = [0.0; radau_work_len(1)];
/// radau5(f, jac, 0.0, 1.0, &mut y, 20, OptimizeOptions::default(), &mut work, |_, _| {}).unwrap();
/// assert!((y[0] - 1.0f64.cos()).abs() < 2e-3);
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn radau5<T: Real>(
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
    radau_core(
        f,
        Jac::Analytic(jacobian),
        t0,
        t1,
        y,
        steps,
        newton,
        work,
        observer,
    )
}

/// Wie [`radau5`], bestimmt die Jacobi-Matrix aber selbst per Vorwärtsdifferenzen (auf dem gekoppelten `3n`-System).
#[allow(clippy::too_many_arguments)]
pub fn radau5_fd<T: Real>(
    f: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    steps: usize,
    newton: OptimizeOptions<T>,
    work: &mut [T],
    observer: impl FnMut(T, &[T]),
) -> Result<(), OdeError> {
    radau_core(
        f,
        Jac::Numeric::<NoJacobian<T>>,
        t0,
        t1,
        y,
        steps,
        newton,
        work,
        observer,
    )
}
