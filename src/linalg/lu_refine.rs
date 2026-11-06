//! Gemeinsame Diagnostik für feste und dynamische LU-Zerlegungen.
use super::{LinalgError, kernel::triangular_solve};
use crate::Real;

/// Einstellungen für iterative LU-Verfeinerung.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefinementOptions<T> {
    /// Gewünschter normweiser Rückwärtsfehler (endlich und nichtnegativ).
    pub tolerance: T,
    /// Maximale Anzahl berechneter Korrekturen; 0 erlaubt reine Diagnostik.
    pub max_iter: usize,
}
impl<T: Real> Default for RefinementOptions<T> {
    fn default() -> Self {
        Self {
            tolerance: T::from_f64(8.0) * T::EPSILON,
            max_iter: 5,
        }
    }
}
/// Abbruchgrund der iterativen Verfeinerung.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefinementTermination {
    /// Gewünschter Rückwärtsfehler erreicht.
    Tolerance,
    /// Keine Verbesserung des Rückwärtsfehlers mehr; beste Lösung bleibt erhalten.
    Stagnation,
    /// Maximale Zahl von Korrekturen erreicht.
    MaxIterations,
}
/// Diagnose der Verfeinerung; kleiner Rückwärtsfehler garantiert keinen kleinen Vorwärtsfehler.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RefinementResult<T> {
    /// Anzahl übernommener Korrekturen.
    pub iterations: usize,
    /// Rückwärtsfehler der übergebenen Startlösung.
    pub initial_backward_error: T,
    /// Normweiser Rückwärtsfehler `‖b−Ax‖∞/(‖A‖∞‖x‖∞+‖b‖∞)` der besten Lösung.
    pub backward_error: T,
    /// Grund für den Abbruch (kein Konvergenzzertifikat für den Vorwärtsfehler).
    pub termination: RefinementTermination,
}

fn max_abs<T: Real>(v: &[T]) -> T {
    v.iter().fold(T::ZERO, |s, v| s.max(v.abs()))
}
fn matrix_norm<T: Real>(a: &[T], n: usize, scale: T) -> T {
    let mut norm = T::ZERO;
    for i in 0..n {
        let mut row = T::ZERO;
        for j in 0..n {
            row += (a[i * n + j] / scale).abs();
        }
        norm = norm.max(row);
    }
    norm
}
// Löse mit A/scale oder dessen Transponierter, ohne neue Zerlegung und ohne
// eine explizite Inverse. L bleibt gleich, U wird beim Zugriff skaliert.
fn scaled_solve<T: Real>(
    lu: &[T],
    n: usize,
    perm: &[usize],
    scale: T,
    b: &[T],
    out: &mut [T],
    tmp: &mut [T],
    transpose: bool,
) -> Result<(), LinalgError> {
    if transpose {
        tmp.copy_from_slice(b);
        for i in 0..n {
            let mut s = tmp[i];
            for j in 0..i {
                s -= (lu[j * n + i] / scale) * tmp[j];
            }
            tmp[i] = s / (lu[i * n + i] / scale);
        }
        triangular_solve(lu, n, false, true, true, tmp);
        for i in 0..n {
            out[perm[i]] = tmp[i];
        }
    } else {
        for i in 0..n {
            tmp[i] = b[perm[i]];
        }
        triangular_solve(lu, n, false, false, true, tmp);
        for i in (0..n).rev() {
            let mut s = tmp[i];
            for j in i + 1..n {
                s -= (lu[i * n + j] / scale) * tmp[j];
            }
            tmp[i] = s / (lu[i * n + i] / scale);
        }
        out.copy_from_slice(tmp);
    }
    if out.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    Ok(())
}

/// Hager-Schätzung von ‖(A/scale)⁻ᵀ‖₁ mit zusätzlichem alternierenden Testvektor.
pub(crate) fn rcond<T: Real>(
    a: &[T],
    lu: &[T],
    n: usize,
    perm: &[usize],
    x: &mut [T],
    y: &mut [T],
    z: &mut [T],
    tmp: &mut [T],
) -> Result<T, LinalgError> {
    if n == 0 {
        return Ok(T::ONE);
    }
    let scale = max_abs(a);
    let anorm = matrix_norm(a, n, scale);
    x.fill(T::ONE / T::from_usize(n));
    let mut estimate = T::ZERO;
    let mut previous = n;
    for _ in 0..8 {
        if scaled_solve(lu, n, perm, scale, x, y, tmp, true).is_err() {
            return Ok(T::ZERO);
        }
        let sum = y.iter().fold(T::ZERO, |s, v| s + v.abs());
        if sum <= estimate {
            break;
        }
        estimate = sum;
        for v in y.iter_mut() {
            *v = if *v >= T::ZERO { T::ONE } else { -T::ONE };
        }
        if scaled_solve(lu, n, perm, scale, y, z, tmp, false).is_err() {
            return Ok(T::ZERO);
        }
        let mut j = 0;
        for i in 1..n {
            if z[i].abs() > z[j].abs() {
                j = i;
            }
        }
        let dot = z
            .iter()
            .zip(x.iter())
            .fold(T::ZERO, |s, (a, b)| s + *a * *b);
        if z[j].abs() <= dot || j == previous {
            break;
        }
        previous = j;
        x.fill(T::ZERO);
        x[j] = T::ONE;
    }
    for (i, v) in x.iter_mut().enumerate() {
        let weight = if n == 1 {
            T::ONE
        } else {
            T::ONE + T::from_usize(i) / T::from_usize(n - 1)
        };
        *v = if i % 2 == 0 { weight } else { -weight };
    }
    if scaled_solve(lu, n, perm, scale, x, y, tmp, true).is_err() {
        return Ok(T::ZERO);
    }
    let alt =
        y.iter().fold(T::ZERO, |s, v| s + v.abs()) / x.iter().fold(T::ZERO, |s, v| s + v.abs());
    estimate = estimate.max(alt);
    Ok(((T::ONE / anorm) / estimate).min(T::ONE).max(T::ZERO))
}

// Kompensierte Summation einschließlich Produktfehler via mul_add.
fn residual<T: Real>(
    a: &[T],
    n: usize,
    scale: T,
    b: &[T],
    x: &[T],
    r: &mut [T],
) -> Result<T, LinalgError> {
    let mut bmax = T::ZERO;
    for i in 0..n {
        let mut sum = b[i] / scale;
        bmax = bmax.max(sum.abs());
        let mut correction = (-sum).mul_add(scale, b[i]) / scale;
        for j in 0..n {
            let value = -(a[i * n + j] / scale);
            let division_error = value.mul_add(scale, a[i * n + j]) / scale;
            let product = value * x[j];
            let error = value.mul_add(x[j], -product);
            let next = sum + product;
            correction += if sum.abs() >= product.abs() {
                (sum - next) + product
            } else {
                (product - next) + sum
            };
            correction += error - division_error * x[j];
            sum = next;
        }
        r[i] = sum + correction;
    }
    if r.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    let xmax = max_abs(x);
    let vscale = xmax.max(bmax);
    if vscale == T::ZERO {
        return Ok(T::ZERO);
    }
    Ok((max_abs(r) / vscale) / (matrix_norm(a, n, scale) * (xmax / vscale) + bmax / vscale))
}

pub(crate) fn refine<T: Real>(
    a: &[T],
    lu: &[T],
    n: usize,
    perm: &[usize],
    b: &[T],
    x: &mut [T],
    opts: RefinementOptions<T>,
    r: &mut [T],
    delta: &mut [T],
    trial: &mut [T],
    tmp: &mut [T],
) -> Result<RefinementResult<T>, LinalgError> {
    if !opts.tolerance.is_finite() || opts.tolerance < T::ZERO {
        return Err(LinalgError::InvalidInput);
    }
    if b.iter().chain(x.iter()).any(|v| !v.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    let scale = if n == 0 { T::ONE } else { max_abs(a) };
    let mut error = residual(a, n, scale, b, x, r)?;
    let initial = error;
    let mut iterations = 0;
    let mut termination = RefinementTermination::MaxIterations;
    for _ in 0..opts.max_iter {
        if error <= opts.tolerance {
            termination = RefinementTermination::Tolerance;
            break;
        }
        scaled_solve(lu, n, perm, scale, r, delta, tmp, false)?;
        for i in 0..n {
            trial[i] = x[i] + delta[i];
        }
        let candidate = residual(a, n, scale, b, trial, r)?;
        if candidate >= error {
            termination = RefinementTermination::Stagnation;
            break;
        }
        x.copy_from_slice(trial);
        error = candidate;
        iterations += 1;
    }
    if error <= opts.tolerance {
        termination = RefinementTermination::Tolerance;
    }
    Ok(RefinementResult {
        iterations,
        initial_backward_error: initial,
        backward_error: error,
        termination,
    })
}
