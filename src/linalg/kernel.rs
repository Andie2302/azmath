//! Gemeinsame LU-Algorithmen auf flachen, zeilenweise gespeicherten Slices.
//! Werden von den festen (`Matrix`) und dynamischen (`DynMatrix`) Typen genutzt.

#![allow(clippy::needless_range_loop)]

use super::LinalgError;
use crate::Real;

/// Löst in-place ein Dreieckssystem; `x` enthält beim Aufruf die rechte Seite `b`.
///
/// - `upper`: `a` wird als obere (`true`) bzw. untere (`false`) Dreiecksmatrix `T` gelesen,
///   das jeweils andere Dreieck wird ignoriert.
/// - `transpose`: gelöst wird `Tᵀ·x = b` statt `T·x = b`.
/// - `unit`: Die Diagonale gilt als 1 und wird nicht gelesen.
///
/// Die Diagonale wird nicht auf 0 geprüft (siehe [`check_diagonal`]).
pub fn triangular_solve<T: Real>(
    a: &[T],
    n: usize,
    upper: bool,
    transpose: bool,
    unit: bool,
    x: &mut [T],
) {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(x.len(), n);
    // Element (i, j) der effektiven Matrix E = T bzw. Tᵀ
    let at = |i: usize, j: usize| {
        if transpose {
            a[j * n + i]
        } else {
            a[i * n + j]
        }
    };

    if upper == transpose {
        // E ist untere Dreiecksmatrix: Vorwärtssubstitution
        for i in 0..n {
            let mut s = x[i];
            for j in 0..i {
                s -= at(i, j) * x[j];
            }
            x[i] = if unit { s } else { s / at(i, i) };
        }
    } else {
        // E ist obere Dreiecksmatrix: Rückwärtssubstitution
        for i in (0..n).rev() {
            let mut s = x[i];
            for j in (i + 1)..n {
                s -= at(i, j) * x[j];
            }
            x[i] = if unit { s } else { s / at(i, i) };
        }
    }
}

/// Prüft, dass keine Diagonale 0 oder NaN ist (sonst ist das Dreieckssystem nicht eindeutig lösbar).
pub fn check_diagonal<T: Real>(a: &[T], n: usize) -> Result<(), LinalgError> {
    for i in 0..n {
        let d = a[i * n + i];
        if d == T::ZERO || d.is_nan() {
            return Err(LinalgError::Singular);
        }
    }
    Ok(())
}

/// Zerlegt die `n x n`-Matrix `a` (zeilenweise) in-place in `P·A = L·U`.
///
/// `L` (Einheitsdiagonale, nicht gespeichert) steht unterhalb, `U` auf und oberhalb der
/// Diagonalen. `perm[i]` ist der ursprüngliche Zeilenindex der Zeile `i`.
/// Rückgabe: `true`, wenn die Anzahl der Zeilenvertauschungen ungerade ist.
pub fn lu_decompose<T: Real>(
    a: &mut [T],
    n: usize,
    perm: &mut [usize],
) -> Result<bool, LinalgError> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(perm.len(), n);

    for (i, p) in perm.iter_mut().enumerate() {
        *p = i;
    }
    if n == 0 {
        return Ok(false);
    }

    let mut scale = T::ZERO;
    for &v in a.iter() {
        if v.is_nan() || v.is_infinite() {
            return Err(LinalgError::NotFinite);
        }
        scale = scale.max(v.abs());
    }
    if scale == T::ZERO {
        return Err(LinalgError::Singular);
    }
    let tol = T::from_f64(n as f64) * T::EPSILON * scale;

    let mut odd = false;
    for k in 0..n {
        // Pivotsuche: betragsgrößtes Element der Spalte k ab Zeile k.
        let mut p = k;
        let mut best = a[k * n + k].abs();
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > best {
                best = v;
                p = i;
            }
        }
        if best.is_nan() || best <= tol {
            return Err(LinalgError::Singular);
        }
        if p != k {
            for j in 0..n {
                a.swap(k * n + j, p * n + j);
            }
            perm.swap(k, p);
            odd = !odd;
        }

        let pivot = a[k * n + k];
        for i in (k + 1)..n {
            let f = a[i * n + k] / pivot;
            a[i * n + k] = f;
            for j in (k + 1)..n {
                a[i * n + j] -= f * a[k * n + j];
                if !a[i * n + j].is_finite() {
                    return Err(LinalgError::NotFinite);
                }
            }
        }
    }
    Ok(odd)
}

/// Löst `A·x = b` mit der Zerlegung aus [`lu_decompose`]. `x` und `b` haben Länge `n`.
pub fn lu_solve<T: Real>(lu: &[T], n: usize, perm: &[usize], b: &[T], x: &mut [T]) {
    debug_assert_eq!(lu.len(), n * n);
    for i in 0..n {
        x[i] = b[perm[i]];
    }
    triangular_solve(lu, n, false, false, true, x); // L hat Einheitsdiagonale
    triangular_solve(lu, n, true, false, false, x); // U
}

/// Determinante aus der Zerlegung: Produkt der Diagonale mit Vorzeichen der Permutation.
pub fn lu_determinant<T: Real>(lu: &[T], n: usize, odd: bool) -> T {
    let mut det = if odd { -T::ONE } else { T::ONE };
    for i in 0..n {
        det *= lu[i * n + i];
    }
    det
}

// ============================================================================
// Cholesky
// ============================================================================

/// Zerlegt die symmetrisch positiv definite `n x n`-Matrix `a` (zeilenweise) in-place in
/// `A = L·Lᵀ`. `L` steht danach in der unteren Dreiecksmatrix, oberhalb der Diagonalen steht 0.
pub fn cholesky_decompose<T: Real>(a: &mut [T], n: usize) -> Result<(), LinalgError> {
    debug_assert_eq!(a.len(), n * n);
    if n == 0 {
        return Ok(());
    }

    let mut scale = T::ZERO;
    let mut max_diag = T::ZERO;
    for (idx, &v) in a.iter().enumerate() {
        if v.is_nan() || v.is_infinite() {
            return Err(LinalgError::NotFinite);
        }
        scale = scale.max(v.abs());
        if idx % (n + 1) == 0 {
            max_diag = max_diag.max(v.abs());
        }
    }

    let nf = T::from_f64(n as f64);
    let sym_tol = T::from_f64(8.0) * nf * T::EPSILON * scale;
    for i in 0..n {
        for j in 0..i {
            if (a[i * n + j] - a[j * n + i]).abs() > sym_tol {
                return Err(LinalgError::NotSymmetric);
            }
        }
    }

    let pd_tol = nf * T::EPSILON * max_diag;
    for j in 0..n {
        let mut s = a[j * n + j];
        for k in 0..j {
            s -= a[j * n + k] * a[j * n + k];
        }
        if s.is_nan() || s <= pd_tol {
            return Err(LinalgError::NotPositiveDefinite);
        }
        let d = s.sqrt();
        a[j * n + j] = d;
        for i in (j + 1)..n {
            let mut t = a[i * n + j];
            for k in 0..j {
                t -= a[i * n + k] * a[j * n + k];
            }
            a[i * n + j] = t / d;
        }
    }

    for i in 0..n {
        for j in (i + 1)..n {
            a[i * n + j] = T::ZERO;
        }
    }
    Ok(())
}

/// Löst `A·x = b` mit `A = L·Lᵀ` (Vorwärts-, dann Rückwärtssubstitution).
pub fn cholesky_solve<T: Real>(l: &[T], n: usize, b: &[T], x: &mut [T]) {
    debug_assert_eq!(l.len(), n * n);
    x.copy_from_slice(b);
    triangular_solve(l, n, false, false, false, x); // L·y = b
    triangular_solve(l, n, false, true, false, x); // Lᵀ·x = y
}

/// Determinante aus `L`: Quadrat des Produkts der Diagonale.
pub fn cholesky_determinant<T: Real>(l: &[T], n: usize) -> T {
    let mut p = T::ONE;
    for i in 0..n {
        p *= l[i * n + i];
    }
    p * p
}

// ============================================================================
// QR (Householder)
// ============================================================================

/// QR-Zerlegung der `m x n`-Matrix `a` (`m >= n`, zeilenweise) in-place mit Householder-Spiegelungen.
///
/// `R` steht auf und oberhalb der Diagonalen. Unterhalb der Diagonalen stehen die
/// Householder-Vektoren `w` (mit `w[k] = 1` implizit), `tau[k]` die zugehörigen Faktoren:
/// `H_k = I - tau[k]·w·wᵀ` und `Q = H_0·H_1·…·H_{n-1}`.
///
/// Fehler: [`LinalgError::Singular`] bei Rangdefizit (kleines oder fehlendes `R_kk`).
pub fn qr_decompose<T: Real>(
    a: &mut [T],
    m: usize,
    n: usize,
    tau: &mut [T],
) -> Result<(), LinalgError> {
    debug_assert_eq!(a.len(), m * n);
    debug_assert_eq!(tau.len(), n);
    debug_assert!(m >= n);

    for &v in a.iter() {
        if v.is_nan() || v.is_infinite() {
            return Err(LinalgError::NotFinite);
        }
    }

    for k in 0..n {
        let mut norm2 = T::ZERO;
        for i in k..m {
            norm2 += a[i * n + k] * a[i * n + k];
        }
        // Auf norm2 prüfen (exakte Arithmetik): sqrt(0) ist bei Näherungs-Backends nicht exakt 0.
        if norm2 == T::ZERO {
            tau[k] = T::ZERO;
            continue;
        }
        let norm = norm2.sqrt();
        let akk = a[k * n + k];
        let alpha = if akk > T::ZERO { -norm } else { norm };
        let v0 = akk - alpha;
        let mut vn2 = v0 * v0;
        for i in (k + 1)..m {
            vn2 += a[i * n + k] * a[i * n + k];
        }
        let t = T::from_f64(2.0) * v0 * v0 / vn2;
        tau[k] = t;
        for i in (k + 1)..m {
            a[i * n + k] /= v0;
        }
        for j in (k + 1)..n {
            let mut s = a[k * n + j];
            for i in (k + 1)..m {
                s += a[i * n + k] * a[i * n + j];
            }
            s *= t;
            a[k * n + j] -= s;
            for i in (k + 1)..m {
                a[i * n + j] -= s * a[i * n + k];
            }
        }
        a[k * n + k] = alpha;
    }

    let mut rmax = T::ZERO;
    for k in 0..n {
        rmax = rmax.max(a[k * n + k].abs());
    }
    let tol = T::from_f64(m as f64) * T::EPSILON * rmax;
    for k in 0..n {
        let d = a[k * n + k].abs();
        if d.is_nan() || d <= tol {
            return Err(LinalgError::Singular);
        }
    }
    Ok(())
}

/// Wendet eine Householder-Spiegelung `H_k` auf `b` an.
fn apply_reflector<T: Real>(a: &[T], n: usize, m: usize, tau: &[T], k: usize, b: &mut [T]) {
    if tau[k] == T::ZERO {
        return;
    }
    let mut s = b[k];
    for i in (k + 1)..m {
        s += a[i * n + k] * b[i];
    }
    s *= tau[k];
    b[k] -= s;
    for i in (k + 1)..m {
        b[i] -= s * a[i * n + k];
    }
}

/// Berechnet `Qᵀ·b` in-place (`b` hat Länge `m`).
pub fn qr_apply_qt<T: Real>(a: &[T], m: usize, n: usize, tau: &[T], b: &mut [T]) {
    for k in 0..n {
        apply_reflector(a, n, m, tau, k, b);
    }
}

/// Berechnet `Q·b` in-place (`b` hat Länge `m`).
pub fn qr_apply_q<T: Real>(a: &[T], m: usize, n: usize, tau: &[T], b: &mut [T]) {
    for k in (0..n).rev() {
        apply_reflector(a, n, m, tau, k, b);
    }
}

/// Löst das Ausgleichsproblem `min ‖A·x − b‖₂`. `y` ist eine Kopie von `b` (Länge `m`) und
/// enthält danach `Qᵀ·b`; `x` hat Länge `n`.
pub fn qr_solve<T: Real>(a: &[T], m: usize, n: usize, tau: &[T], y: &mut [T], x: &mut [T]) {
    qr_apply_qt(a, m, n, tau, y);
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= a[i * n + j] * x[j];
        }
        x[i] = s / a[i * n + i];
    }
}

/// Euklidische Norm des Residuums aus `Qᵀ·b` (Einträge `n..m`).
pub fn qr_residual_norm<T: Real>(y: &[T], n: usize) -> T {
    let mut s = T::ZERO;
    for &v in &y[n..] {
        s += v * v;
    }
    s.sqrt()
}
