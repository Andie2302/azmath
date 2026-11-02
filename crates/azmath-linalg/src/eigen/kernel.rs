//! Gemeinsame Kernalgorithmen auf flachen, zeilenweise gespeicherten Slices.

#![allow(clippy::needless_range_loop)]

use super::super::LinalgError;
use azmath_core::Real;

fn bad<T: Real>(v: T) -> bool {
    v.is_nan() || v.is_infinite()
}

/// Zyklisches Jacobi-Verfahren für die symmetrische `n × n`-Matrix `a` (zeilenweise).
///
/// Nach dem Aufruf stehen die Eigenwerte **aufsteigend sortiert** auf der Diagonale von `a`
/// (der Rest ist 0 bzw. Rundungsrauschen) und die zugehörigen Eigenvektoren als **Spalten** von `v`.
/// Rückgabe: Anzahl der Sweeps.
pub(crate) fn jacobi_eigen<T: Real>(
    a: &mut [T],
    v: &mut [T],
    n: usize,
    max_sweeps: usize,
) -> Result<usize, LinalgError> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(v.len(), n * n);

    let mut fro2 = T::ZERO;
    let mut scale = T::ZERO;
    for &x in a.iter() {
        if bad(x) {
            return Err(LinalgError::NotFinite);
        }
        fro2 += x * x;
        scale = scale.max(x.abs());
    }
    let sym_tol = T::from_f64(8.0) * T::from_f64(n as f64) * T::EPSILON * scale;
    for i in 0..n {
        for j in 0..i {
            if (a[i * n + j] - a[j * n + i]).abs() > sym_tol {
                return Err(LinalgError::NotSymmetric);
            }
        }
    }

    v.fill(T::ZERO);
    for i in 0..n {
        v[i * n + i] = T::ONE;
    }

    let fro = fro2.sqrt();
    let two = T::from_f64(2.0);
    let mut sweeps = 0;
    let mut converged = n <= 1;
    while !converged && sweeps < max_sweeps {
        // Quadratsumme der Nebendiagonalelemente
        let mut off = T::ZERO;
        for p in 0..n {
            for q in p + 1..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off.sqrt() <= T::EPSILON * fro {
            converged = true;
            break;
        }
        sweeps += 1;
        for p in 0..n - 1 {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == T::ZERO {
                    continue;
                }
                let (app, aqq) = (a[p * n + p], a[q * n + q]);
                let theta = (aqq - app) / (two * apq);
                let t = if theta == T::ZERO {
                    T::ONE
                } else {
                    let root = (theta * theta + T::ONE).sqrt();
                    theta.signum() / (theta.abs() + root)
                };
                let c = T::ONE / (t * t + T::ONE).sqrt();
                let s = t * c;

                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = T::ZERO;
                a[q * n + p] = T::ZERO;
                for r in 0..n {
                    if r != p && r != q {
                        let (arp, arq) = (a[r * n + p], a[r * n + q]);
                        let (np, nq) = (c * arp - s * arq, s * arp + c * arq);
                        a[r * n + p] = np;
                        a[p * n + r] = np;
                        a[r * n + q] = nq;
                        a[q * n + r] = nq;
                    }
                    let (vrp, vrq) = (v[r * n + p], v[r * n + q]);
                    v[r * n + p] = c * vrp - s * vrq;
                    v[r * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }
    if !converged {
        // Letzte Prüfung nach dem letzten Sweep
        let mut off = T::ZERO;
        for p in 0..n {
            for q in p + 1..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off.sqrt() > T::from_f64(4.0) * T::EPSILON * fro {
            return Err(LinalgError::NoConvergence);
        }
    }

    // Eigenwerte aufsteigend sortieren (Auswahlsortierung, vertauscht auch die Spalten von v)
    for i in 0..n {
        let mut k = i;
        for j in i + 1..n {
            if a[j * n + j] < a[k * n + k] {
                k = j;
            }
        }
        if k != i {
            let (di, dk) = (a[i * n + i], a[k * n + k]);
            a[i * n + i] = dk;
            a[k * n + k] = di;
            for r in 0..n {
                v.swap(r * n + i, r * n + k);
            }
        }
    }
    Ok(sweeps)
}

/// Einseitiges Jacobi-Verfahren (Hestenes) für die SVD der `m × n`-Matrix `u` (`m ≥ n`, zeilenweise).
///
/// Danach enthält `u` die linken Singulärvektoren als Spalten (für Singulärwert 0 ist die Spalte 0),
/// `s` die Singulärwerte **absteigend** und `v` (`n × n`) die rechten Singulärvektoren als Spalten, mit
/// `A = U·diag(s)·Vᵀ`. Rückgabe: Anzahl der Sweeps.
pub(crate) fn jacobi_svd<T: Real>(
    u: &mut [T],
    s: &mut [T],
    v: &mut [T],
    m: usize,
    n: usize,
    max_sweeps: usize,
) -> Result<usize, LinalgError> {
    debug_assert_eq!(u.len(), m * n);
    debug_assert_eq!(s.len(), n);
    debug_assert_eq!(v.len(), n * n);
    debug_assert!(m >= n);

    if u.iter().any(|x| bad(*x)) {
        return Err(LinalgError::NotFinite);
    }
    v.fill(T::ZERO);
    for i in 0..n {
        v[i * n + i] = T::ONE;
    }

    let tol = T::EPSILON * T::from_f64(m as f64).sqrt();
    let two = T::from_f64(2.0);
    let mut sweeps = 0;
    let mut converged = n <= 1;
    while !converged && sweeps < max_sweeps {
        let mut rotated = false;
        for p in 0..n - 1 {
            for q in p + 1..n {
                let (mut alpha, mut beta, mut gamma) = (T::ZERO, T::ZERO, T::ZERO);
                for i in 0..m {
                    let (up, uq) = (u[i * n + p], u[i * n + q]);
                    alpha += up * up;
                    beta += uq * uq;
                    gamma += up * uq;
                }
                if gamma == T::ZERO || alpha == T::ZERO || beta == T::ZERO {
                    continue;
                }
                if gamma.abs() <= tol * (alpha * beta).sqrt() {
                    continue;
                }
                rotated = true;
                let zeta = (beta - alpha) / (two * gamma);
                let t = zeta.signum() / (zeta.abs() + (T::ONE + zeta * zeta).sqrt());
                let c = T::ONE / (T::ONE + t * t).sqrt();
                let sn = c * t;
                for i in 0..m {
                    let (up, uq) = (u[i * n + p], u[i * n + q]);
                    u[i * n + p] = c * up - sn * uq;
                    u[i * n + q] = sn * up + c * uq;
                }
                for i in 0..n {
                    let (vp, vq) = (v[i * n + p], v[i * n + q]);
                    v[i * n + p] = c * vp - sn * vq;
                    v[i * n + q] = sn * vp + c * vq;
                }
            }
        }
        sweeps += 1;
        if !rotated {
            converged = true;
        }
    }
    if !converged {
        return Err(LinalgError::NoConvergence);
    }

    // Singulärwerte = Spaltennormen; Spalten normieren
    for j in 0..n {
        let mut norm2 = T::ZERO;
        for i in 0..m {
            norm2 += u[i * n + j] * u[i * n + j];
        }
        let sigma = norm2.sqrt();
        s[j] = sigma;
        if sigma > T::ZERO {
            for i in 0..m {
                u[i * n + j] /= sigma;
            }
        }
    }
    // absteigend sortieren
    for i in 0..n {
        let mut k = i;
        for j in i + 1..n {
            if s[j] > s[k] {
                k = j;
            }
        }
        if k != i {
            s.swap(i, k);
            for r in 0..m {
                u.swap(r * n + i, r * n + k);
            }
            for r in 0..n {
                v.swap(r * n + i, r * n + k);
            }
        }
    }
    Ok(sweeps)
}
