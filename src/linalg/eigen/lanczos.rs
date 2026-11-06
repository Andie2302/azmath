#![allow(clippy::needless_range_loop)]

//! Krylov-Zerlegungen für große Operatoren: Lanczos (symmetrisch) und Arnoldi (allgemein).
//!
//! Beide arbeiten auf Slices ohne Allokator. Die orthonormale Basis liegt **zeilenweise** im Arbeitsspeicher:
//! Zeile `j` (`work[j·n .. (j+1)·n]`) ist der Basisvektor `q_j`. Mit voller Reorthogonalisierung bleibt die
//! Basis auch bei vielen Schritten orthogonal; der Preis ist Speicher `O(k·n)` und Aufwand `O(k²·n)`.

use super::MAX_SWEEPS;
use super::kernel::jacobi_eigen;
use crate::Real;
use crate::linalg::{LinalgError, LinearOperator};

fn bad<T: Real>(v: T) -> bool {
    v.is_nan() || v.is_infinite()
}

fn dot<T: Real>(a: &[T], b: &[T]) -> T {
    a.iter().zip(b).fold(T::ZERO, |s, (x, y)| s + *x * *y)
}

/// Arbeitsspeicher für [`lanczos`]: `(k + 1)·n` Elemente (Basis plus Hilfsvektor).
pub const fn lanczos_work_len(n: usize, k: usize) -> usize {
    (k + 1) * n
}

/// Arbeitsspeicher für [`arnoldi`]: `(k + 2)·n` Elemente (Basis aus `k + 1` Vektoren plus Hilfsvektor).
pub const fn arnoldi_work_len(n: usize, k: usize) -> usize {
    (k + 2) * n
}

/// Lanczos-Verfahren für einen **symmetrischen** Operator: baut eine orthonormale Basis `Q = [q₀ … q_{m−1}]` des
/// Krylov-Raums und die Tridiagonalmatrix `T = Qᵀ·A·Q` mit Diagonale `alpha` und Nebendiagonale `beta`:
/// `A·q_j = β_{j−1}·q_{j−1} + α_j·q_j + β_j·q_{j+1}`.
///
/// - `start`: Startvektor (ungleich 0, wird normiert). `k`: gewünschte Schrittzahl (höchstens `n`).
/// - `alpha`, `beta`: Ausgabe, mindestens `k` Elemente; gültig sind `alpha[..m]` und `beta[..m]`, wobei
///   `beta[m − 1]` die Norm des Restvektors ist (Maß für die Güte der Ritz-Werte).
/// - `work`: mindestens [`lanczos_work_len`]`(n, k)`. Danach steht die Basis in `work[..m·n]`.
/// - Rückgabe: die tatsächliche Schrittzahl `m ≤ k`; bei einem invarianten Unterraum endet das Verfahren früher.
///
/// Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NotFinite`], [`LinalgError::Breakdown`] (Startvektor 0).
pub fn lanczos<T: Real, A: LinearOperator<T> + ?Sized>(
    op: &A,
    start: &[T],
    k: usize,
    alpha: &mut [T],
    beta: &mut [T],
    work: &mut [T],
) -> Result<usize, LinalgError> {
    let n = start.len();
    if op.rows() != n
        || op.cols() != n
        || k == 0
        || k > n
        || alpha.len() < k
        || beta.len() < k
        || work.len() < lanczos_work_len(n, k)
    {
        return Err(LinalgError::DimensionMismatch);
    }
    let (basis, rest) = work.split_at_mut(k * n);
    let w = &mut rest[..n];
    let norm0 = dot(start, start).sqrt();
    if bad(norm0) {
        return Err(LinalgError::NotFinite);
    }
    if norm0 == T::ZERO {
        return Err(LinalgError::Breakdown);
    }
    for i in 0..n {
        basis[i] = start[i] / norm0;
    }
    let mut steps = k;
    for j in 0..k {
        let (done, cur) = basis.split_at_mut(j * n);
        let qj = &cur[..n];
        op.apply(qj, w);
        let a = dot(qj, w);
        if bad(a) {
            return Err(LinalgError::NotFinite);
        }
        alpha[j] = a;
        // volle Reorthogonalisierung (zweimal) gegen q₀ … q_j
        for _ in 0..2 {
            for i in 0..=j {
                let qi = if i < j { &done[i * n..(i + 1) * n] } else { qj };
                let c = dot(qi, w);
                for t in 0..n {
                    w[t] -= c * qi[t];
                }
            }
        }
        let b = dot(w, w).sqrt();
        if bad(b) {
            return Err(LinalgError::NotFinite);
        }
        beta[j] = b;
        let scale = a.abs() + if j > 0 { beta[j - 1] } else { T::ZERO };
        if b <= T::from_f64(n as f64) * T::EPSILON * scale + T::MIN_POSITIVE {
            beta[j] = T::ZERO;
            steps = j + 1;
            break;
        }
        if j + 1 < k {
            let next = &mut cur[n..2 * n];
            for t in 0..n {
                next[t] = w[t] / b;
            }
        }
    }
    Ok(steps)
}

/// Arbeitsspeicher für [`lanczos_ritz`]: `2·m²` Elemente bei `m` Lanczos-Schritten.
pub const fn lanczos_ritz_work_len(m: usize) -> usize {
    2 * m * m
}

/// Ritz-Werte aus einer Lanczos-Zerlegung: die Eigenwerte der `m × m`-Tridiagonalmatrix (`alpha`, `beta`), **aufsteigend**
/// in `values[..m]`. In `residuals[..m]` steht für jeden Ritz-Wert die Abschätzung `|β_{m−1}·s_{m−1,i}|` für den Fehler
/// `‖A·y − θ·y‖` des zugehörigen Ritz-Paars (`s` = Eigenvektor der Tridiagonalmatrix).
///
/// Die extremen Ritz-Werte nähern die extremen Eigenwerte von `A` meist schon nach wenigen Schritten an.
/// `work`: mindestens [`lanczos_ritz_work_len`]`(m)`.
///
/// Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NoConvergence`].
pub fn lanczos_ritz<T: Real>(
    alpha: &[T],
    beta: &[T],
    m: usize,
    values: &mut [T],
    residuals: &mut [T],
    work: &mut [T],
) -> Result<(), LinalgError> {
    if alpha.len() < m
        || beta.len() < m
        || values.len() < m
        || residuals.len() < m
        || work.len() < lanczos_ritz_work_len(m)
    {
        return Err(LinalgError::DimensionMismatch);
    }
    if m == 0 {
        return Ok(());
    }
    let (t, v) = work.split_at_mut(m * m);
    let v = &mut v[..m * m];
    t.fill(T::ZERO);
    for i in 0..m {
        t[i * m + i] = alpha[i];
        if i + 1 < m {
            t[i * m + i + 1] = beta[i];
            t[(i + 1) * m + i] = beta[i];
        }
    }
    jacobi_eigen(t, v, m, MAX_SWEEPS)?;
    for i in 0..m {
        values[i] = t[i * m + i];
        residuals[i] = (beta[m - 1] * v[(m - 1) * m + i]).abs();
    }
    Ok(())
}

/// Arnoldi-Verfahren für einen **beliebigen** Operator: baut eine orthonormale Basis `V = [v₀ … v_m]` und die obere
/// Hessenberg-Matrix `H` (`(m+1) × m`) mit `A·V_m = V_{m+1}·H`.
///
/// - `start`: Startvektor (ungleich 0, wird normiert). `k`: gewünschte Schrittzahl (höchstens `n`).
/// - `h`: Ausgabe, zeilenweise `(k+1) × k` (`h[i·k + j]`), mindestens `(k+1)·k` Elemente; gültig ist der Block
///   `(m+1) × m`, der Rest ist 0.
/// - `work`: mindestens [`arnoldi_work_len`]`(n, k)`. Danach steht die Basis in `work[..(m+1)·n]` (Zeile `j` = `v_j`).
///   Bei frühem Abbruch ist der zusätzliche Vektor `v_m` null, nicht normiert.
/// - Rückgabe: die tatsächliche Schrittzahl `m ≤ k`; bei einem invarianten Unterraum (`h_{m,m−1} = 0`) endet das
///   Verfahren früher.
///
/// Die Eigenwerte von `H` (Ritz-Werte) zu berechnen setzt einen nichtsymmetrischen Eigenwertlöser voraus, der erst
/// noch folgt. Für lineare Systeme siehe [`gmres`](crate::linalg::iterative::gmres), das dieselbe Zerlegung nutzt.
///
/// Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NotFinite`], [`LinalgError::Breakdown`] (Startvektor 0).
pub fn arnoldi<T: Real, A: LinearOperator<T> + ?Sized>(
    op: &A,
    start: &[T],
    k: usize,
    h: &mut [T],
    work: &mut [T],
) -> Result<usize, LinalgError> {
    let n = start.len();
    if op.rows() != n
        || op.cols() != n
        || k == 0
        || k > n
        || h.len() < (k + 1) * k
        || work.len() < arnoldi_work_len(n, k)
    {
        return Err(LinalgError::DimensionMismatch);
    }
    let (basis, rest) = work.split_at_mut((k + 1) * n);
    let w = &mut rest[..n];
    h[..(k + 1) * k].fill(T::ZERO);
    basis.fill(T::ZERO);
    let norm0 = crate::linalg::norm::frobenius(start);
    if bad(norm0) {
        return Err(LinalgError::NotFinite);
    }
    if norm0 == T::ZERO {
        return Err(LinalgError::Breakdown);
    }
    for i in 0..n {
        basis[i] = start[i] / norm0;
    }
    let mut steps = k;
    for j in 0..k {
        let (done, cur) = basis.split_at_mut(j * n);
        let vj = &cur[..n];
        op.apply(vj, w);
        let mut anorm = T::ZERO;
        // Gram-Schmidt gegen v₀ … v_j, zweimal (die Koeffizienten werden in H aufsummiert)
        for _ in 0..2 {
            for i in 0..=j {
                let vi = if i < j { &done[i * n..(i + 1) * n] } else { vj };
                let c = dot(vi, w);
                h[i * k + j] += c;
                anorm = anorm.max(c.abs());
                for t in 0..n {
                    w[t] -= c * vi[t];
                }
            }
        }
        let b = crate::linalg::norm::frobenius(w);
        if bad(b) {
            return Err(LinalgError::NotFinite);
        }
        if b <= T::from_f64(n as f64) * T::EPSILON * anorm + T::MIN_POSITIVE {
            steps = j + 1;
            break;
        }
        h[(j + 1) * k + j] = b;
        let next = &mut cur[n..2 * n];
        for t in 0..n {
            next[t] = w[t] / b;
        }
    }
    Ok(steps)
}
