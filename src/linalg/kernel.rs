//! Gemeinsame LU-Algorithmen auf flachen, zeilenweise gespeicherten Slices.
//! Werden von den festen (`Matrix`) und dynamischen (`DynMatrix`) Typen genutzt.

#![allow(clippy::needless_range_loop)]

use super::LinalgError;
use crate::Real;

/// Zerlegt die `n x n`-Matrix `a` (zeilenweise) in-place in `P·A = L·U`.
///
/// `L` (Einheitsdiagonale, nicht gespeichert) steht unterhalb, `U` auf und oberhalb der
/// Diagonalen. `perm[i]` ist der ursprüngliche Zeilenindex der Zeile `i`.
/// Rückgabe: `true`, wenn die Anzahl der Zeilenvertauschungen ungerade ist.
pub(crate) fn lu_decompose<T: Real>(
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
            }
        }
    }
    Ok(odd)
}

/// Löst `A·x = b` mit der Zerlegung aus [`lu_decompose`]. `x` und `b` haben Länge `n`.
pub(crate) fn lu_solve<T: Real>(lu: &[T], n: usize, perm: &[usize], b: &[T], x: &mut [T]) {
    debug_assert_eq!(lu.len(), n * n);
    for i in 0..n {
        x[i] = b[perm[i]];
    }
    // Vorwärtssubstitution (L hat Einheitsdiagonale).
    for i in 0..n {
        let mut s = x[i];
        for j in 0..i {
            s -= lu[i * n + j] * x[j];
        }
        x[i] = s;
    }
    // Rückwärtssubstitution.
    for i in (0..n).rev() {
        let mut s = x[i];
        for j in (i + 1)..n {
            s -= lu[i * n + j] * x[j];
        }
        x[i] = s / lu[i * n + i];
    }
}

/// Determinante aus der Zerlegung: Produkt der Diagonale mit Vorzeichen der Permutation.
pub(crate) fn lu_determinant<T: Real>(lu: &[T], n: usize, odd: bool) -> T {
    let mut det = if odd { -T::ONE } else { T::ONE };
    for i in 0..n {
        det *= lu[i * n + i];
    }
    det
}
