use super::{
    MultigridError, MultigridOptions, MultigridResult, Smoother, SolveOptions, check_options,
    finite_slice, gamma, grid_levels,
};
use azmath_core::Real;

/// Benötigter Arbeitsspeicher für die 2D-Funktionen bei einem `n × n`-Gitter mit `n = 2ᵏ − 1`.
///
/// Pro Ebene werden drei Gitter gehalten (Näherung, rechte Seite, Residuum); für ungültiges `n` ergibt
/// sich ein nicht verwendbarer Wert.
pub const fn work_len_2d(n: usize) -> usize {
    let mut total = 0;
    let mut m = n;
    while m >= 1 {
        total += 3 * m * m;
        m = (m - 1) / 2;
    }
    total
}

/// Nachbarwerte `(oben, unten, links, rechts)` von `(i, j)` mit Nullrand.
#[inline(always)]
fn neighbors<T: Real>(u: &[T], n: usize, i: usize, j: usize) -> T {
    let up = if i > 0 { u[(i - 1) * n + j] } else { T::ZERO };
    let down = if i + 1 < n {
        u[(i + 1) * n + j]
    } else {
        T::ZERO
    };
    let left = if j > 0 { u[i * n + j - 1] } else { T::ZERO };
    let right = if j + 1 < n { u[i * n + j + 1] } else { T::ZERO };
    up + down + left + right
}

/// `r = f − A·u` mit dem 5-Punkte-Stern `(4u − Nachbarn)/h²`.
fn residual<T: Real>(u: &[T], f: &[T], n: usize, h2: T, r: &mut [T]) {
    let four = T::from_f64(4.0);
    for i in 0..n {
        for j in 0..n {
            let k = i * n + j;
            r[k] = f[k] - (four * u[k] - neighbors(u, n, i, j)) / h2;
        }
    }
}

fn gs_point<T: Real>(u: &mut [T], f: &[T], n: usize, h2: T, i: usize, j: usize) {
    let k = i * n + j;
    u[k] = (f[k] * h2 + neighbors(u, n, i, j)) / T::from_f64(4.0);
}

#[allow(clippy::too_many_arguments)]
fn smooth<T: Real>(
    u: &mut [T],
    f: &[T],
    r: &mut [T],
    n: usize,
    h2: T,
    sweeps: usize,
    forward: bool,
    s: Smoother<T>,
) {
    for _ in 0..sweeps {
        match s {
            Smoother::WeightedJacobi(omega) => {
                residual(u, f, n, h2, r);
                for k in 0..n * n {
                    u[k] += omega * r[k] * h2 / T::from_f64(4.0);
                }
            }
            Smoother::GaussSeidel => {
                if forward {
                    for i in 0..n {
                        for j in 0..n {
                            gs_point(u, f, n, h2, i, j);
                        }
                    }
                } else {
                    for i in (0..n).rev() {
                        for j in (0..n).rev() {
                            gs_point(u, f, n, h2, i, j);
                        }
                    }
                }
            }
        }
    }
}

/// Ein Zyklus auf dem Puffer `buf = [u, f, r | gröbere Ebenen …]` mit Gittern der Größe `n × n`.
pub(crate) fn run_cycle<T: Real>(buf: &mut [T], n: usize, opts: &MultigridOptions<T>) {
    let h = T::ONE / T::from_f64((n + 1) as f64);
    let h2 = h * h;
    let m = n * n;
    let (u, rest) = buf.split_at_mut(m);
    let (f, rest) = rest.split_at_mut(m);
    let (r, rest) = rest.split_at_mut(m);

    if n == 1 {
        // gröbstes Gitter: 4·u/h² = f
        u[0] = f[0] * h2 / T::from_f64(4.0);
        return;
    }

    smooth(u, f, r, n, h2, opts.pre_smooth, true, opts.smoother);
    residual(u, f, n, h2, r);

    // Restriktion (Full Weighting, 9-Punkte): Gewichte 1 2 1 / 2 4 2 / 1 2 1, geteilt durch 16
    let nc = (n - 1) / 2;
    {
        let (uc, tail) = rest.split_at_mut(nc * nc);
        let fc = &mut tail[..nc * nc];
        let at = |a: usize, b: usize| r[a * n + b];
        let two = T::from_f64(2.0);
        let four = T::from_f64(4.0);
        let sixteen = T::from_f64(16.0);
        for i in 0..nc {
            for j in 0..nc {
                let (a, b) = (2 * i, 2 * j);
                let corners = at(a, b) + at(a, b + 2) + at(a + 2, b) + at(a + 2, b + 2);
                let edges = at(a, b + 1) + at(a + 1, b) + at(a + 1, b + 2) + at(a + 2, b + 1);
                fc[i * nc + j] = (corners + two * edges + four * at(a + 1, b + 1)) / sixteen;
                uc[i * nc + j] = T::ZERO;
            }
        }
    }
    for _ in 0..gamma(opts.cycle) {
        run_cycle(&mut *rest, nc, opts);
    }

    // Prolongation (bilineare Interpolation) und Korrektur
    let uc = &rest[..nc * nc];
    let coarse = |i: isize, j: isize| -> T {
        if i < 0 || j < 0 || i as usize >= nc || j as usize >= nc {
            T::ZERO
        } else {
            uc[i as usize * nc + j as usize]
        }
    };
    let half = T::from_f64(0.5);
    for a in 0..n {
        // ungerade Zeile: direkt über einer Grobzeile; gerade: Mittel zweier Grobzeilen
        let rows: [(isize, T); 2] = if a % 2 == 1 {
            [(((a - 1) / 2) as isize, T::ONE), (0, T::ZERO)]
        } else {
            [((a / 2) as isize - 1, half), ((a / 2) as isize, half)]
        };
        for b in 0..n {
            let cols: [(isize, T); 2] = if b % 2 == 1 {
                [(((b - 1) / 2) as isize, T::ONE), (0, T::ZERO)]
            } else {
                [((b / 2) as isize - 1, half), ((b / 2) as isize, half)]
            };
            let mut corr = T::ZERO;
            for &(ri, rw) in &rows {
                if rw == T::ZERO {
                    continue;
                }
                for &(ci, cw) in &cols {
                    if cw != T::ZERO {
                        corr += rw * cw * coarse(ri, ci);
                    }
                }
            }
            u[a * n + b] += corr;
        }
    }

    smooth(u, f, r, n, h2, opts.post_smooth, false, opts.smoother);
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
    if f.len() != n * n || u.len() != n * n || work.len() < work_len_2d(n) {
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
    let four = T::from_f64(4.0);
    let mut s = T::ZERO;
    for i in 0..n {
        for j in 0..n {
            let k = i * n + j;
            let r = f[k] - (four * u[k] - neighbors(u, n, i, j)) / h2;
            s += r * r;
        }
    }
    s.sqrt()
}

/// Ein einzelner Multigrid-Zyklus für `−Δu = f` auf dem `n × n`-Gitter (`n = 2ᵏ − 1`, zeilenweise
/// gespeichert): verbessert die Näherung `u` in-place. `work` hat mindestens [`work_len_2d`]`(n)`
/// Elemente.
///
/// Fehler: [`MultigridError::InvalidGridSize`], [`MultigridError::InvalidInput`],
/// [`MultigridError::NotFinite`].
pub fn cycle_2d<T: Real>(
    f: &[T],
    u: &mut [T],
    n: usize,
    opts: MultigridOptions<T>,
    work: &mut [T],
) -> Result<(), MultigridError> {
    check(f, u, n, &opts, work)?;
    let m = n * n;
    let buf = &mut work[..work_len_2d(n)];
    buf[..m].copy_from_slice(u);
    buf[m..2 * m].copy_from_slice(f);
    run_cycle(buf, n, &opts);
    u.copy_from_slice(&buf[..m]);
    if !finite_slice(u) {
        return Err(MultigridError::NotFinite);
    }
    Ok(())
}

/// Löst `−Δu = f` auf dem `n × n`-Gitter (`n = 2ᵏ − 1`) durch wiederholte Multigrid-Zyklen, bis
/// `‖f − A·u‖ ≤ tol·‖f‖` gilt. Der 5-Punkte-Stern ist `(4uᵢⱼ − Nachbarn)/h²`, `u[i·n + j]`.
///
/// Parameter und Fehler wie bei [`solve_1d`](super::solve_1d); `work`: mindestens
/// [`work_len_2d`]`(n)` Elemente.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath_multigrid::{solve_2d, work_len_2d, MultigridOptions, SolveOptions};
///
/// const N: usize = 15;
/// let f = [1.0f64; N * N];
/// let mut u = [0.0; N * N];
/// let mut work = [0.0; work_len_2d(N)];
/// let r = solve_2d(&f, &mut u, N, MultigridOptions::default(), SolveOptions::default(), &mut work)
///     .unwrap();
/// // −Δu = 1 auf dem Einheitsquadrat: Maximum in der Mitte, u(½, ½) ≈ 0,0737
/// assert!((u[(N / 2) * N + N / 2] - 0.0737).abs() < 1e-3 && r.cycles <= 15);
/// # }
/// ```
pub fn solve_2d<T: Real>(
    f: &[T],
    u: &mut [T],
    n: usize,
    opts: MultigridOptions<T>,
    solve: SolveOptions<T>,
    work: &mut [T],
) -> Result<MultigridResult<T>, MultigridError> {
    check(f, u, n, &opts, work)?;
    let m = n * n;
    let fnorm = f.iter().fold(T::ZERO, |s, v| s + *v * *v).sqrt();
    if fnorm == T::ZERO {
        u.fill(T::ZERO);
        return Ok(MultigridResult {
            cycles: 0,
            residual_norm: T::ZERO,
        });
    }
    let stop = solve.tol * fnorm;
    let buf = &mut work[..work_len_2d(n)];
    buf[m..2 * m].copy_from_slice(f);
    buf[..m].copy_from_slice(u);
    for cycles in 0..=solve.max_cycles {
        let res = residual_norm(&buf[..m], f, n);
        if res.is_nan() || res.is_infinite() {
            return Err(MultigridError::NotFinite);
        }
        if res <= stop {
            u.copy_from_slice(&buf[..m]);
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
    u.copy_from_slice(&buf[..m]);
    Err(MultigridError::NoConvergence)
}
