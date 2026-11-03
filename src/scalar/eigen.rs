//! Eigenwerte und Eigenvektoren komplexer Matrizen: Hessenberg-Reduktion und QR-Verfahren mit Wilkinson-Shift.

use super::ScalarLu;
use super::qr::householder;
use crate::Real;
use crate::complex::Complex;
use crate::linalg::LinalgError;
use alloc::vec;
use alloc::vec::Vec;

type C<T> = Complex<T>;

fn cabs<T: Real>(z: C<T>) -> T {
    z.abs()
}

/// Reduziert `h` (`n×n`, zeilenweise) durch unitäre Ähnlichkeitstransformationen auf obere Hessenberg-Gestalt.
fn hessenberg<T: Real>(n: usize, h: &mut [C<T>]) {
    if n < 3 {
        return;
    }
    for k in 0..n - 2 {
        let x: Vec<C<T>> = (k + 1..n).map(|i| h[i * n + k]).collect();
        let Some((v, beta, alpha)) = householder(&x) else {
            continue;
        };
        let b = C::from_real(beta);
        // links: H[k+1.., :] −= β·v·(vᴴ·H[k+1.., :])
        for j in 0..n {
            let mut s = C::ZERO;
            for (idx, i) in (k + 1..n).enumerate() {
                s += v[idx].conj() * h[i * n + j];
            }
            let f = s * b;
            for (idx, i) in (k + 1..n).enumerate() {
                h[i * n + j] -= f * v[idx];
            }
        }
        // rechts: H[:, k+1..] −= β·(H[:, k+1..]·v)·vᴴ
        for i in 0..n {
            let mut s = C::ZERO;
            for (idx, j) in (k + 1..n).enumerate() {
                s += h[i * n + j] * v[idx];
            }
            let f = s * b;
            for (idx, j) in (k + 1..n).enumerate() {
                h[i * n + j] -= f * v[idx].conj();
            }
        }
        h[(k + 1) * n + k] = alpha;
        for i in k + 2..n {
            h[i * n + k] = C::ZERO;
        }
    }
}

/// Alle Eigenwerte einer komplexen `n×n`-Matrix `a` (zeilenweise), in der Reihenfolge der Deflation.
///
/// Verfahren: Hessenberg-Reduktion, dann QR-Iteration mit Givens-Drehungen und Wilkinson-Shift (Ausnahmeshift nach 10 und 20
/// Schritten). Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NotFinite`], [`LinalgError::NoConvergence`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::complex::Complex;
/// use azmath::scalar::complex_eigenvalues;
/// // Drehung um 90°: Eigenwerte ±i
/// let a = [Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)];
/// let ev = complex_eigenvalues(2, &a).unwrap();
/// assert!(ev.iter().all(|z| (z.abs() - 1.0).abs() < 1e-12 && z.re.abs() < 1e-12));
/// # }
/// ```
pub fn complex_eigenvalues<T: Real>(n: usize, a: &[C<T>]) -> Result<Vec<C<T>>, LinalgError> {
    if a.len() != n * n {
        return Err(LinalgError::DimensionMismatch);
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    let mut h = a.to_vec();
    hessenberg(n, &mut h);
    let mut eig = vec![C::ZERO; n];
    let eps = T::EPSILON;
    let half = T::from_f64(0.5);
    let mut hi = n;
    let mut iter = 0usize;
    while hi > 0 {
        let top = hi - 1;
        // kleine Nebendiagonale suchen
        let mut l = top;
        while l > 0 {
            let s = cabs(h[(l - 1) * n + l - 1]) + cabs(h[l * n + l]);
            let s = if s == T::ZERO { T::ONE } else { s };
            if cabs(h[l * n + l - 1]) <= eps * s {
                h[l * n + l - 1] = C::ZERO;
                break;
            }
            l -= 1;
        }
        if l == top {
            eig[top] = h[top * n + top];
            hi -= 1;
            iter = 0;
            continue;
        }
        iter += 1;
        if iter > 60 + 30 * n {
            return Err(LinalgError::NoConvergence);
        }
        // Shift
        let mu = if iter % 10 == 0 {
            let extra = cabs(h[top * n + top - 1])
                + if top >= 2 {
                    cabs(h[(top - 1) * n + top - 2])
                } else {
                    T::ZERO
                };
            h[top * n + top] + C::from_real(extra)
        } else {
            let (aa, bb, cc, dd) = (
                h[(top - 1) * n + top - 1],
                h[(top - 1) * n + top],
                h[top * n + top - 1],
                h[top * n + top],
            );
            let tr = (aa + dd) * C::from_real(half);
            let diff = (aa - dd) * C::from_real(half);
            let root = (diff * diff + bb * cc).sqrt();
            let (m1, m2) = (tr + root, tr - root);
            if cabs(m1 - dd) <= cabs(m2 - dd) {
                m1
            } else {
                m2
            }
        };
        for i in l..=top {
            h[i * n + i] -= mu;
        }
        let mut rot: Vec<(C<T>, C<T>)> = Vec::with_capacity(top - l);
        for k in l..top {
            let (x, y) = (h[k * n + k], h[(k + 1) * n + k]);
            let r = x.abs().hypot(y.abs());
            let (c, s) = if r == T::ZERO {
                (C::ONE, C::ZERO)
            } else {
                (x / C::from_real(r), y / C::from_real(r))
            };
            for j in k..=top {
                let (u, v) = (h[k * n + j], h[(k + 1) * n + j]);
                h[k * n + j] = c.conj() * u + s.conj() * v;
                h[(k + 1) * n + j] = -(s * u) + c * v;
            }
            rot.push((c, s));
        }
        for (idx, k) in (l..top).enumerate() {
            let (c, s) = rot[idx];
            let last = (k + 2).min(top);
            for i in l..=last {
                let (u, v) = (h[i * n + k], h[i * n + k + 1]);
                h[i * n + k] = u * c + v * s;
                h[i * n + k + 1] = -(u * s.conj()) + v * c.conj();
            }
        }
        for i in l..=top {
            h[i * n + i] += mu;
        }
    }
    Ok(eig)
}

/// Eigenwerte einer hermiteschen Matrix (reell, aufsteigend sortiert). Fehler zusätzlich [`LinalgError::NotSymmetric`], wenn
/// `a` nicht hermitesch ist.
pub fn hermitian_eigenvalues<T: Real>(n: usize, a: &[C<T>]) -> Result<Vec<T>, LinalgError> {
    if a.len() != n * n {
        return Err(LinalgError::DimensionMismatch);
    }
    let mut scale = T::ZERO;
    for v in a {
        scale = scale.max(v.abs());
    }
    let tol = T::EPSILON * T::from_f64(100.0) * scale;
    for i in 0..n {
        for j in 0..=i {
            if (a[i * n + j] - a[j * n + i].conj()).abs() > tol {
                return Err(LinalgError::NotSymmetric);
            }
        }
    }
    let ev = complex_eigenvalues(n, a)?;
    let mut r: Vec<T> = ev.iter().map(|z| z.re).collect();
    r.sort_by(|x, y| x.partial_cmp(y).unwrap_or(core::cmp::Ordering::Equal));
    Ok(r)
}

/// Eigenvektor (Norm 1) zum (näherungsweise bekannten) Eigenwert `lambda` durch inverse Iteration mit [`ScalarLu`].
pub fn complex_eigenvector<T: Real>(
    n: usize,
    a: &[C<T>],
    lambda: C<T>,
) -> Result<Vec<C<T>>, LinalgError> {
    if a.len() != n * n {
        return Err(LinalgError::DimensionMismatch);
    }
    let mut scale = T::ZERO;
    for v in a {
        scale = scale.max(v.abs());
    }
    let shift = lambda + C::from_real(T::EPSILON * T::from_f64(10.0) * (scale + T::ONE));
    let mut m = a.to_vec();
    for i in 0..n {
        m[i * n + i] -= shift;
    }
    let lu = ScalarLu::new(n, &m)?;
    let mut v: Vec<C<T>> = (0..n)
        .map(|i| C::new(T::ONE + T::from_usize(i) * T::from_f64(0.01), T::ZERO))
        .collect();
    for _ in 0..4 {
        let mut w = lu.solve(&v)?;
        let nrm = w.iter().fold(T::ZERO, |s, z| s + z.norm_sqr()).sqrt();
        if !(nrm > T::ZERO) || !nrm.is_finite() {
            return Err(LinalgError::Singular);
        }
        for z in w.iter_mut() {
            *z /= C::from_real(nrm);
        }
        v = w;
    }
    Ok(v)
}

/// Komplexe Schur-Zerlegung `A = Z·T·Zᴴ` mit oberer Dreiecksmatrix `T` (Eigenwerte auf der Diagonale) und unitärem `Z`.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexSchur<T> {
    /// Dimension.
    pub n: usize,
    /// Obere Dreiecksmatrix (zeilenweise).
    pub t: Vec<C<T>>,
    /// Unitäre Matrix `Z` (zeilenweise).
    pub z: Vec<C<T>>,
}

/// Berechnet die komplexe Schur-Zerlegung von `a` (`n×n`, zeilenweise): Hessenberg-Reduktion und QR-Iteration mit Wilkinson-Shift,
/// alle Transformationen werden akkumuliert. Fehler wie bei [`complex_eigenvalues`].
pub fn complex_schur<T: Real>(n: usize, a: &[C<T>]) -> Result<ComplexSchur<T>, LinalgError> {
    if a.len() != n * n {
        return Err(LinalgError::DimensionMismatch);
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    let mut h = a.to_vec();
    let mut z = vec![C::ZERO; n * n];
    for i in 0..n {
        z[i * n + i] = C::ONE;
    }
    // Hessenberg-Reduktion mit Akkumulation
    if n >= 3 {
        for k in 0..n - 2 {
            let x: Vec<C<T>> = (k + 1..n).map(|i| h[i * n + k]).collect();
            let Some((v, beta, alpha)) = householder(&x) else {
                continue;
            };
            let b = C::from_real(beta);
            for j in 0..n {
                let mut s = C::ZERO;
                for (idx, i) in (k + 1..n).enumerate() {
                    s += v[idx].conj() * h[i * n + j];
                }
                let f = s * b;
                for (idx, i) in (k + 1..n).enumerate() {
                    h[i * n + j] -= f * v[idx];
                }
            }
            for mat in [&mut h, &mut z] {
                for i in 0..n {
                    let mut s = C::ZERO;
                    for (idx, j) in (k + 1..n).enumerate() {
                        s += mat[i * n + j] * v[idx];
                    }
                    let f = s * b;
                    for (idx, j) in (k + 1..n).enumerate() {
                        mat[i * n + j] -= f * v[idx].conj();
                    }
                }
            }
            h[(k + 1) * n + k] = alpha;
            for i in k + 2..n {
                h[i * n + k] = C::ZERO;
            }
        }
    }
    let eps = T::EPSILON;
    let half = T::from_f64(0.5);
    let mut hi = n;
    let mut iter = 0usize;
    while hi > 0 {
        let top = hi - 1;
        let mut l = top;
        while l > 0 {
            let s = cabs(h[(l - 1) * n + l - 1]) + cabs(h[l * n + l]);
            let s = if s == T::ZERO { T::ONE } else { s };
            if cabs(h[l * n + l - 1]) <= eps * s {
                h[l * n + l - 1] = C::ZERO;
                break;
            }
            l -= 1;
        }
        if l == top {
            hi -= 1;
            iter = 0;
            continue;
        }
        iter += 1;
        if iter > 60 + 30 * n {
            return Err(LinalgError::NoConvergence);
        }
        let mu = if iter % 10 == 0 {
            let extra = cabs(h[top * n + top - 1])
                + if top >= 2 {
                    cabs(h[(top - 1) * n + top - 2])
                } else {
                    T::ZERO
                };
            h[top * n + top] + C::from_real(extra)
        } else {
            let (aa, bb, cc, dd) = (
                h[(top - 1) * n + top - 1],
                h[(top - 1) * n + top],
                h[top * n + top - 1],
                h[top * n + top],
            );
            let tr = (aa + dd) * C::from_real(half);
            let diff = (aa - dd) * C::from_real(half);
            let root = (diff * diff + bb * cc).sqrt();
            let (m1, m2) = (tr + root, tr - root);
            if cabs(m1 - dd) <= cabs(m2 - dd) {
                m1
            } else {
                m2
            }
        };
        for i in l..=top {
            h[i * n + i] -= mu;
        }
        let mut rot: Vec<(C<T>, C<T>)> = Vec::with_capacity(top - l);
        for k in l..top {
            let (x, y) = (h[k * n + k], h[(k + 1) * n + k]);
            let r = x.abs().hypot(y.abs());
            let (c, s) = if r == T::ZERO {
                (C::ONE, C::ZERO)
            } else {
                (x / C::from_real(r), y / C::from_real(r))
            };
            for j in k..n {
                let (u, v) = (h[k * n + j], h[(k + 1) * n + j]);
                h[k * n + j] = c.conj() * u + s.conj() * v;
                h[(k + 1) * n + j] = -(s * u) + c * v;
            }
            rot.push((c, s));
        }
        for (idx, k) in (l..top).enumerate() {
            let (c, s) = rot[idx];
            let last = (k + 2).min(top);
            for i in 0..=last {
                let (u, v) = (h[i * n + k], h[i * n + k + 1]);
                h[i * n + k] = u * c + v * s;
                h[i * n + k + 1] = -(u * s.conj()) + v * c.conj();
            }
            for i in 0..n {
                let (u, v) = (z[i * n + k], z[i * n + k + 1]);
                z[i * n + k] = u * c + v * s;
                z[i * n + k + 1] = -(u * s.conj()) + v * c.conj();
            }
        }
        for i in l..=top {
            h[i * n + i] += mu;
        }
    }
    for i in 1..n {
        for j in 0..i {
            h[i * n + j] = C::ZERO;
        }
    }
    Ok(ComplexSchur { n, t: h, z })
}

/// Eigenwerte und Eigenvektoren (Spalten, Norm 1) einer komplexen Matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ComplexEigen<T> {
    /// Eigenwerte.
    pub values: Vec<C<T>>,
    /// Eigenvektormatrix (`n×n`, zeilenweise): Spalte `k` gehört zu `values[k]`.
    pub vectors: Vec<C<T>>,
}

/// Vollständige Eigenzerlegung über die Schur-Form: Eigenvektoren von `T` durch Rückwärtseinsetzen, dann Rücktransformation mit
/// `Z`. Bei (fast) mehrfachen Eigenwerten defekter Matrizen sind die Vektoren näherungsweise parallel.
pub fn complex_eigen<T: Real>(n: usize, a: &[C<T>]) -> Result<ComplexEigen<T>, LinalgError> {
    let s = complex_schur(n, a)?;
    let t = &s.t;
    let mut scale = T::ZERO;
    for v in t {
        scale = scale.max(v.abs());
    }
    let small = T::EPSILON * (scale + T::ONE);
    let values: Vec<C<T>> = (0..n).map(|i| t[i * n + i]).collect();
    let mut vectors = vec![C::ZERO; n * n];
    for k in 0..n {
        let mut x = vec![C::ZERO; n];
        x[k] = C::ONE;
        for j in (0..k).rev() {
            let mut sum = C::ZERO;
            for i in j + 1..=k {
                sum += t[j * n + i] * x[i];
            }
            let mut den = t[j * n + j] - t[k * n + k];
            if den.abs() < small {
                den = C::from_real(small);
            }
            x[j] = -(sum / den);
        }
        let mut v = vec![C::ZERO; n];
        for i in 0..n {
            for j in 0..=k {
                v[i] += s.z[i * n + j] * x[j];
            }
        }
        let nrm = v.iter().fold(T::ZERO, |acc, z| acc + z.norm_sqr()).sqrt();
        for i in 0..n {
            vectors[i * n + k] = v[i] / C::from_real(nrm);
        }
    }
    Ok(ComplexEigen { values, vectors })
}

/// Eigenzerlegung einer reellen Matrix (die Eigenwerte und -vektoren sind im Allgemeinen komplex).
pub fn real_eigen<T: Real>(
    a: &crate::linalg::DynMatrix<T>,
) -> Result<ComplexEigen<T>, LinalgError> {
    let n = a.rows();
    if a.cols() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    let c: Vec<C<T>> = a.as_slice().iter().map(|&v| C::from_real(v)).collect();
    complex_eigen(n, &c)
}
