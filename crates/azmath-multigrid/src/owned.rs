//! Komfortfunktionen mit eigener Speicherverwaltung (Feature `alloc`).

use alloc::vec;
use alloc::vec::Vec;

use super::{
    MultigridError, MultigridOptions, MultigridResult, SolveOptions, solve_1d, solve_2d,
    work_len_1d, work_len_2d,
};
use azmath_core::Real;

/// Wie [`solve_1d`], startet bei `u = 0` und legt Lösung und Arbeitsspeicher selbst an (Feature `alloc`).
pub fn solve_1d_vec<T: Real>(
    f: &[T],
    opts: MultigridOptions<T>,
    solve: SolveOptions<T>,
) -> Result<(Vec<T>, MultigridResult<T>), MultigridError> {
    let n = f.len();
    if super::grid_levels(n).is_none() {
        return Err(MultigridError::InvalidGridSize);
    }
    let mut u = vec![T::ZERO; n];
    let mut work = vec![T::ZERO; work_len_1d(n)];
    let r = solve_1d(f, &mut u, n, opts, solve, &mut work)?;
    Ok((u, r))
}

/// Wie [`solve_2d`] für ein `n × n`-Gitter, startet bei `u = 0` und legt Lösung und Arbeitsspeicher selbst
/// an (Feature `alloc`). `f.len()` muss `n²` sein.
pub fn solve_2d_vec<T: Real>(
    f: &[T],
    n: usize,
    opts: MultigridOptions<T>,
    solve: SolveOptions<T>,
) -> Result<(Vec<T>, MultigridResult<T>), MultigridError> {
    if super::grid_levels(n).is_none() {
        return Err(MultigridError::InvalidGridSize);
    }
    let mut u = vec![T::ZERO; f.len()];
    let mut work = vec![T::ZERO; work_len_2d(n)];
    let r = solve_2d(f, &mut u, n, opts, solve, &mut work)?;
    Ok((u, r))
}
