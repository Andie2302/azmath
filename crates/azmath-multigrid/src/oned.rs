use super::{
    MultigridError, MultigridOptions, MultigridResult, Smoother, SolveOptions, check_options,
    finite_slice, gamma, grid_levels,
};
use azmath_core::Real;

/// Benötigter Arbeitsspeicher für die 1D-Funktionen bei `n = 2ᵏ − 1` inneren Punkten.
///
/// Pro Ebene werden drei Vektoren gehalten (Näherung, rechte Seite, Residuum); für ungültiges `n`
/// ergibt sich ein nicht verwendbarer Wert.
pub const fn work_len_1d(n: usize) -> usize {
    let mut total = 0;
    let mut m = n;
    while m >= 1 {
        total += 3 * m;
        m = (m - 1) / 2;
    }
    total
}

fn omega_or_gs<T: Real>(s: Smoother<T>) -> Option<T> {
    match s {
        Smoother::WeightedJacobi(w) => Some(w),
        Smoother::GaussSeidel => None,
    }
}

/// `r = f − A·u` mit `A = (2u_i − u_{i−1} − u_{i+1})/h²`, `h2 = h²`.
fn residual<T: Real>(u: &[T], f: &[T], h2: T, r: &mut [T]) {
    let n = u.len();
    for i in 0..n {
        let l = if i > 0 { u[i - 1] } else { T::ZERO };
        let rr = if i + 1 < n { u[i + 1] } else { T::ZERO };
        r[i] = f[i] - (T::from_f64(2.0) * u[i] - l - rr) / h2;
    }
}

fn smooth<T: Real>(
    u: &mut [T],
    f: &[T],
    r: &mut [T],
    h2: T,
    sweeps: usize,
    forward: bool,
    s: Smoother<T>,
) {
    let n = u.len();
    let two = T::from_f64(2.0);
    for _ in 0..sweeps {
        if let Some(omega) = omega_or_gs(s) {
            residual(u, f, h2, r);
            for i in 0..n {
                u[i] += omega * r[i] * h2 / two;
            }
        } else if forward {
            for i in 0..n {
                let l = if i > 0 { u[i - 1] } else { T::ZERO };
                let rr = if i + 1 < n { u[i + 1] } else { T::ZERO };
                u[i] = (f[i] * h2 + l + rr) / two;
            }
        } else {
            for i in (0..n).rev() {
                let l = if i > 0 { u[i - 1] } else { T::ZERO };
                let rr = if i + 1 < n { u[i + 1] } else { T::ZERO };
                u[i] = (f[i] * h2 + l + rr) / two;
            }
        }
    }
}

/// Ein Zyklus auf dem Puffer `buf = [u, f, r | gröbere Ebenen …]` der Größe `n`.
pub(crate) fn run_cycle<T: Real>(buf: &mut [T], n: usize, opts: &MultigridOptions<T>) {
    let h = T::ONE / T::from_f64((n + 1) as f64);
    let h2 = h * h;
    let (u, rest) = buf.split_at_mut(n);
    let (f, rest) = rest.split_at_mut(n);
    let (r, rest) = rest.split_at_mut(n);

    if n == 1 {
        // gröbstes Gitter: 2·u/h² = f
        u[0] = f[0] * h2 / T::from_f64(2.0);
        return;
    }

    smooth(u, f, r, h2, opts.pre_smooth, true, opts.smoother);
    residual(u, f, h2, r);

    // Restriktion (Full Weighting): r_c[i] = (r[2i] + 2·r[2i+1] + r[2i+2]) / 4
    let nc = (n - 1) / 2;
    {
        let (uc, tail) = rest.split_at_mut(nc);
        let fc = &mut tail[..nc];
        for i in 0..nc {
            fc[i] = (r[2 * i] + T::from_f64(2.0) * r[2 * i + 1] + r[2 * i + 2]) / T::from_f64(4.0);
            uc[i] = T::ZERO;
        }
    }
    for _ in 0..gamma(opts.cycle) {
        run_cycle(&mut *rest, nc, opts);
    }

    // Prolongation (lineare Interpolation) und Korrektur
    let uc = &rest[..nc];
    for i in 0..nc {
        u[2 * i + 1] += uc[i];
    }
    let half = T::from_f64(0.5);
    for i in 0..=nc {
        let left = if i > 0 { uc[i - 1] } else { T::ZERO };
        let right = if i < nc { uc[i] } else { T::ZERO };
        u[2 * i] += half * (left + right);
    }

    smooth(u, f, r, h2, opts.post_smooth, false, opts.smoother);
}

fn check<T: Real>(
    f: &[T],
    u: &[T],
    n: usize,
    opts: &MultigridOptions<T>,
    work: &[T],
) -> Result<(), MultigridError> {
    if grid_levels(n).is_none() {
        return Err(MultigridError::InvalidGridSize);
    }
    if f.len() != n || u.len() != n || work.len() < work_len_1d(n) {
        return Err(MultigridError::InvalidInput);
    }
    check_options(opts)?;
    if !finite_slice(f) || !finite_slice(u) {
        return Err(MultigridError::NotFinite);
    }
    Ok(())
}

fn residual_norm<T: Real>(u: &[T], f: &[T], n: usize) -> T {
    let h = T::ONE / T::from_f64((n + 1) as f64);
    let h2 = h * h;
    let mut s = T::ZERO;
    for i in 0..n {
        let l = if i > 0 { u[i - 1] } else { T::ZERO };
        let rr = if i + 1 < n { u[i + 1] } else { T::ZERO };
        let r = f[i] - (T::from_f64(2.0) * u[i] - l - rr) / h2;
        s += r * r;
    }
    s.sqrt()
}

/// Ein einzelner Multigrid-Zyklus für `−u'' = f` mit `n = 2ᵏ − 1` inneren Punkten: verbessert die
/// Näherung `u` in-place.
///
/// `work` hat mindestens [`work_len_1d`]`(n)` Elemente. Fehler: [`MultigridError::InvalidGridSize`],
/// [`MultigridError::InvalidInput`], [`MultigridError::NotFinite`].
pub fn cycle_1d<T: Real>(
    f: &[T],
    u: &mut [T],
    n: usize,
    opts: MultigridOptions<T>,
    work: &mut [T],
) -> Result<(), MultigridError> {
    check(f, u, n, &opts, work)?;
    let buf = &mut work[..work_len_1d(n)];
    buf[..n].copy_from_slice(u);
    buf[n..2 * n].copy_from_slice(f);
    run_cycle(buf, n, &opts);
    u.copy_from_slice(&buf[..n]);
    if !finite_slice(u) {
        return Err(MultigridError::NotFinite);
    }
    Ok(())
}

/// Löst `−u'' = f` mit `n = 2ᵏ − 1` inneren Punkten durch wiederholte Multigrid-Zyklen, bis
/// `‖f − A·u‖ ≤ tol·‖f‖` gilt.
///
/// - `f`: rechte Seite (`n` Werte), `u`: Startnäherung und Ergebnis (`n` Werte).
/// - `work`: mindestens [`work_len_1d`]`(n)` Elemente.
///
/// Fehler: [`MultigridError::InvalidGridSize`], [`MultigridError::InvalidInput`],
/// [`MultigridError::NotFinite`], [`MultigridError::NoConvergence`].
pub fn solve_1d<T: Real>(
    f: &[T],
    u: &mut [T],
    n: usize,
    opts: MultigridOptions<T>,
    solve: SolveOptions<T>,
    work: &mut [T],
) -> Result<MultigridResult<T>, MultigridError> {
    check(f, u, n, &opts, work)?;
    let fnorm = f.iter().fold(T::ZERO, |s, v| s + *v * *v).sqrt();
    if fnorm == T::ZERO {
        u.fill(T::ZERO);
        return Ok(MultigridResult {
            cycles: 0,
            residual_norm: T::ZERO,
        });
    }
    let stop = solve.tol * fnorm;
    let buf = &mut work[..work_len_1d(n)];
    buf[n..2 * n].copy_from_slice(f);
    buf[..n].copy_from_slice(u);
    for cycles in 0..=solve.max_cycles {
        let res = residual_norm(&buf[..n], f, n);
        if res.is_nan() || res.is_infinite() {
            return Err(MultigridError::NotFinite);
        }
        if res <= stop {
            u.copy_from_slice(&buf[..n]);
            return Ok(MultigridResult {
                cycles,
                residual_norm: res,
            });
        }
        if cycles == solve.max_cycles {
            break;
        }
        run_cycle(buf, n, &opts);
    }
    u.copy_from_slice(&buf[..n]);
    Err(MultigridError::NoConvergence)
}
