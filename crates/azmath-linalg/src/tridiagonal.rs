//! Tridiagonale Gleichungssysteme (Thomas-Algorithmus) in `O(n)`.

use super::LinalgError;
use azmath_core::Real;

/// Benötigter Arbeitsspeicher für [`solve_tridiagonal`] bei `n` Unbekannten.
pub const fn tridiagonal_work_len(n: usize) -> usize {
    n
}

/// Benötigter Arbeitsspeicher für [`solve_cyclic_tridiagonal`] bei `n` Unbekannten.
pub const fn cyclic_tridiagonal_work_len(n: usize) -> usize {
    4 * n
}

fn bad<T: Real>(v: T) -> bool {
    v.is_nan() || v.is_infinite()
}

/// Löst das tridiagonale System `A·x = d` mit dem Thomas-Algorithmus (Gauß-Elimination ohne
/// Pivotsuche) in `O(n)` Zeit.
///
/// Zeile `i` lautet `a[i]·x[i−1] + b[i]·x[i] + c[i]·x[i+1] = d[i]`; `a[0]` und `c[n−1]` werden
/// ignoriert. Alle Slices haben `n` Elemente, `scratch` mindestens [`tridiagonal_work_len`]`(n)`.
///
/// Der Algorithmus ist ohne Pivotsuche stabil, wenn die Matrix **diagonaldominant** oder symmetrisch
/// positiv definit ist (typisch bei Splines und Differenzenverfahren). Sonst kann er ungenau
/// werden; dann ist [`Matrix::solve`](super::Matrix::solve) (LU mit Pivotsuche) sicherer.
///
/// Fehler: [`LinalgError::DimensionMismatch`] (Längen), [`LinalgError::Singular`] (Pivot 0),
/// [`LinalgError::NotFinite`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath_linalg::{solve_tridiagonal, tridiagonal_work_len};
///
/// // 2·x₀ − x₁ = 1,  −x₀ + 2·x₁ − x₂ = 0,  −x₁ + 2·x₂ = 1  ->  x = (1, 1, 1)
/// let (a, b, c) = ([0.0, -1.0, -1.0], [2.0, 2.0, 2.0], [-1.0, -1.0, 0.0]);
/// let d = [1.0, 0.0, 1.0];
/// let (mut x, mut scratch) = ([0.0; 3], [0.0; tridiagonal_work_len(3)]);
/// solve_tridiagonal(&a, &b, &c, &d, &mut x, &mut scratch).unwrap();
/// assert!(x.iter().all(|v| (v - 1.0f64).abs() < 1e-14));
/// # }
/// ```
pub fn solve_tridiagonal<T: Real>(
    a: &[T],
    b: &[T],
    c: &[T],
    d: &[T],
    x: &mut [T],
    scratch: &mut [T],
) -> Result<(), LinalgError> {
    let n = b.len();
    if a.len() != n || c.len() != n || d.len() != n || x.len() != n || scratch.len() < n {
        return Err(LinalgError::DimensionMismatch);
    }
    if n == 0 {
        return Ok(());
    }
    let cp = &mut scratch[..n];
    if b[0] == T::ZERO || bad(b[0]) {
        return Err(if bad(b[0]) {
            LinalgError::NotFinite
        } else {
            LinalgError::Singular
        });
    }
    cp[0] = c[0] / b[0];
    x[0] = d[0] / b[0];
    for i in 1..n {
        let m = b[i] - a[i] * cp[i - 1];
        if bad(m) {
            return Err(LinalgError::NotFinite);
        }
        if m == T::ZERO {
            return Err(LinalgError::Singular);
        }
        cp[i] = c[i] / m;
        x[i] = (d[i] - a[i] * x[i - 1]) / m;
    }
    for i in (0..n - 1).rev() {
        let next = x[i + 1];
        x[i] -= cp[i] * next;
    }
    if x.iter().any(|v| bad(*v)) {
        return Err(LinalgError::NotFinite);
    }
    Ok(())
}

/// Löst ein **zyklisch** tridiagonales System (periodische Randbedingungen, zusätzlich zwei
/// Eckeinträge) mit der Sherman-Morrison-Formel in `O(n)`.
///
/// Zeile `i` lautet `a[i]·x[i−1] + b[i]·x[i] + c[i]·x[i+1] = d[i]` mit **zyklischen** Indizes:
/// `a[0]` gehört zu `x[n−1]` und `c[n−1]` zu `x[0]`. Alle Slices haben `n ≥ 2` Elemente, `work`
/// mindestens [`cyclic_tridiagonal_work_len`]`(n)`. Gilt, wie [`solve_tridiagonal`], für
/// diagonaldominante Matrizen.
///
/// Fehler: [`LinalgError::DimensionMismatch`] (Längen oder `n < 2`), [`LinalgError::Singular`],
/// [`LinalgError::NotFinite`].
pub fn solve_cyclic_tridiagonal<T: Real>(
    a: &[T],
    b: &[T],
    c: &[T],
    d: &[T],
    x: &mut [T],
    work: &mut [T],
) -> Result<(), LinalgError> {
    let n = b.len();
    if n < 2
        || a.len() != n
        || c.len() != n
        || d.len() != n
        || x.len() != n
        || work.len() < cyclic_tridiagonal_work_len(n)
    {
        return Err(LinalgError::DimensionMismatch);
    }
    let (bb, rest) = work.split_at_mut(n);
    let (u, rest) = rest.split_at_mut(n);
    let (z, rest) = rest.split_at_mut(n);
    let scratch = &mut rest[..n];

    let (alpha, beta) = (c[n - 1], a[0]);
    let gamma = -b[0];
    if gamma == T::ZERO || bad(gamma) {
        return Err(if bad(gamma) {
            LinalgError::NotFinite
        } else {
            LinalgError::Singular
        });
    }
    bb.copy_from_slice(b);
    bb[0] = b[0] - gamma;
    bb[n - 1] = b[n - 1] - alpha * beta / gamma;

    // y = T⁻¹·d (in x), z = T⁻¹·u
    solve_tridiagonal(a, bb, c, d, x, scratch)?;
    u.fill(T::ZERO);
    u[0] = gamma;
    u[n - 1] += alpha;
    solve_tridiagonal(a, bb, c, u, z, scratch)?;

    let denom = T::ONE + z[0] + beta * z[n - 1] / gamma;
    if denom == T::ZERO || bad(denom) {
        return Err(if bad(denom) {
            LinalgError::NotFinite
        } else {
            LinalgError::Singular
        });
    }
    let factor = (x[0] + beta * x[n - 1] / gamma) / denom;
    for (xi, zi) in x.iter_mut().zip(z.iter()) {
        *xi -= factor * *zi;
    }
    if x.iter().any(|v| bad(*v)) {
        return Err(LinalgError::NotFinite);
    }
    Ok(())
}
