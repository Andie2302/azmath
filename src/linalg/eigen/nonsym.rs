#![allow(clippy::needless_range_loop)]

//! Eigenwerte **nichtsymmetrischer** reeller Matrizen: Balancieren, Reduktion auf Hessenberg-Form und
//! QR-Algorithmus mit Francis-Doppelshift (Verfahren „hqr“ aus EISPACK). Die Eigenwerte können komplex sein und treten
//! bei reellen Matrizen als konjugierte Paare auf.

use crate::Real;
use crate::complex::Complex;
use crate::linalg::{LinalgError, Matrix};

fn bad<T: Real>(v: T) -> bool {
    v.is_nan() || v.is_infinite()
}

/// `a` mit dem Vorzeichen von `b` (wie `SIGN` in den klassischen Numerik-Texten; `b = 0` zählt positiv).
fn sign<T: Real>(a: T, b: T) -> T {
    if b >= T::ZERO { a.abs() } else { -a.abs() }
}

/// Balancieren: ähnliche Skalierung `D⁻¹·A·D` mit Zweierpotenzen, die Zeilen- und Spaltennormen angleicht
/// (verbessert die Genauigkeit der Eigenwerte, ohne sie zu ändern).
pub(crate) fn balance<T: Real>(a: &mut [T], n: usize) {
    let radix = T::from_f64(2.0);
    let sqrdx = T::from_f64(4.0);
    let mut done = false;
    while !done {
        done = true;
        for i in 0..n {
            let (mut r, mut c) = (T::ZERO, T::ZERO);
            for j in 0..n {
                if j != i {
                    c += a[j * n + i].abs();
                    r += a[i * n + j].abs();
                }
            }
            if c != T::ZERO && r != T::ZERO {
                let mut g = r / radix;
                let mut f = T::ONE;
                let s = c + r;
                while c < g {
                    f *= radix;
                    c *= sqrdx;
                }
                g = r * radix;
                while c > g {
                    f /= radix;
                    c /= sqrdx;
                }
                if (c + r) / f < T::from_f64(0.95) * s {
                    done = false;
                    let g = T::ONE / f;
                    for j in 0..n {
                        a[i * n + j] *= g;
                    }
                    for j in 0..n {
                        a[j * n + i] *= f;
                    }
                }
            }
        }
    }
}

/// Reduktion auf obere Hessenberg-Form durch stabilisierte Eliminationsschritte (mit Zeilen-/Spaltenvertauschung);
/// ähnliche Transformation, die Eigenwerte bleiben erhalten. Unterhalb der ersten Nebendiagonale steht danach 0.
pub(crate) fn hessenberg<T: Real>(a: &mut [T], n: usize) {
    for m in 1..n.saturating_sub(1) {
        let mut x = T::ZERO;
        let mut i = m;
        for j in m..n {
            if a[j * n + m - 1].abs() > x.abs() {
                x = a[j * n + m - 1];
                i = j;
            }
        }
        if i != m {
            for j in (m - 1)..n {
                a.swap(i * n + j, m * n + j);
            }
            for j in 0..n {
                a.swap(j * n + i, j * n + m);
            }
        }
        if x != T::ZERO {
            for i in (m + 1)..n {
                let mut y = a[i * n + m - 1];
                if y != T::ZERO {
                    y /= x;
                    a[i * n + m - 1] = y;
                    for j in m..n {
                        let amj = a[m * n + j];
                        a[i * n + j] -= y * amj;
                    }
                    for j in 0..n {
                        let aji = a[j * n + i];
                        a[j * n + m] += y * aji;
                    }
                }
            }
        }
    }
    for i in 2..n {
        for j in 0..(i - 1) {
            a[i * n + j] = T::ZERO;
        }
    }
}

/// QR-Algorithmus mit Doppelshift auf einer oberen Hessenberg-Matrix (zerstört `a`). Schreibt Real- und Imaginärteile der
/// Eigenwerte nach `out`.
pub(crate) fn hqr<T: Real>(
    a: &mut [T],
    n: usize,
    out: &mut [Complex<T>],
) -> Result<(), LinalgError> {
    if n == 0 {
        return Ok(());
    }
    let at = |i: isize, j: isize| (i as usize) * n + j as usize;
    let half = T::from_f64(0.5);
    let mut anorm = T::ZERO;
    for i in 0..n {
        for j in i.saturating_sub(1)..n {
            anorm += a[i * n + j].abs();
        }
    }
    let mut nn = n as isize - 1;
    let mut t = T::ZERO;
    while nn >= 0 {
        let mut its = 0;
        loop {
            // kleines Subdiagonalelement suchen
            let mut l = nn;
            while l >= 1 {
                let mut s = a[at(l - 1, l - 1)].abs() + a[at(l, l)].abs();
                if s == T::ZERO {
                    s = anorm;
                }
                if a[at(l, l - 1)].abs() + s == s {
                    a[at(l, l - 1)] = T::ZERO;
                    break;
                }
                l -= 1;
            }
            let mut x = a[at(nn, nn)];
            if l == nn {
                // ein Eigenwert
                out[nn as usize] = Complex::new(x + t, T::ZERO);
                nn -= 1;
                break;
            }
            let mut y = a[at(nn - 1, nn - 1)];
            let mut w = a[at(nn, nn - 1)] * a[at(nn - 1, nn)];
            if l == nn - 1 {
                // zwei Eigenwerte
                let p = half * (y - x);
                let q = p * p + w;
                let z = q.abs().sqrt();
                x += t;
                let (i1, i0) = ((nn - 1) as usize, nn as usize);
                if q >= T::ZERO {
                    let z = p + sign(z, p);
                    let first = x + z;
                    let second = if z != T::ZERO { x - w / z } else { first };
                    out[i1] = Complex::new(first, T::ZERO);
                    out[i0] = Complex::new(second, T::ZERO);
                } else {
                    out[i1] = Complex::new(x + p, z);
                    out[i0] = Complex::new(x + p, -z);
                }
                nn -= 2;
                break;
            }
            if its == 60 {
                return Err(LinalgError::NoConvergence);
            }
            if its == 10 || its == 20 {
                // außergewöhnlicher Shift
                t += x;
                for i in 0..=nn {
                    a[at(i, i)] -= x;
                }
                let s = a[at(nn, nn - 1)].abs() + a[at(nn - 1, nn - 2)].abs();
                x = T::from_f64(0.75) * s;
                y = x;
                w = T::from_f64(-0.4375) * s * s;
            }
            its += 1;
            let (mut p, mut q, mut r) = (T::ZERO, T::ZERO, T::ZERO);
            let mut m = nn - 2;
            while m >= l {
                let z = a[at(m, m)];
                let rr = x - z;
                let ss = y - z;
                p = (rr * ss - w) / a[at(m + 1, m)] + a[at(m, m + 1)];
                q = a[at(m + 1, m + 1)] - z - rr - ss;
                r = a[at(m + 2, m + 1)];
                let s = p.abs() + q.abs() + r.abs();
                p /= s;
                q /= s;
                r /= s;
                if m == l {
                    break;
                }
                let u = a[at(m, m - 1)].abs() * (q.abs() + r.abs());
                let v = p.abs() * (a[at(m - 1, m - 1)].abs() + z.abs() + a[at(m + 1, m + 1)].abs());
                if u + v == v {
                    break;
                }
                m -= 1;
            }
            for i in (m + 2)..=nn {
                a[at(i, i - 2)] = T::ZERO;
                if i != m + 2 {
                    a[at(i, i - 3)] = T::ZERO;
                }
            }
            for k in m..=(nn - 1) {
                if k != m {
                    p = a[at(k, k - 1)];
                    q = a[at(k + 1, k - 1)];
                    r = if k != nn - 1 {
                        a[at(k + 2, k - 1)]
                    } else {
                        T::ZERO
                    };
                    x = p.abs() + q.abs() + r.abs();
                    if x != T::ZERO {
                        p /= x;
                        q /= x;
                        r /= x;
                    }
                }
                let s = sign((p * p + q * q + r * r).sqrt(), p);
                if s != T::ZERO {
                    if k == m {
                        if l != m {
                            a[at(k, k - 1)] = -a[at(k, k - 1)];
                        }
                    } else {
                        a[at(k, k - 1)] = -s * x;
                    }
                    p += s;
                    x = p / s;
                    y = q / s;
                    let z = r / s;
                    q /= p;
                    r /= p;
                    for j in k..=nn {
                        let mut pp = a[at(k, j)] + q * a[at(k + 1, j)];
                        if k != nn - 1 {
                            pp += r * a[at(k + 2, j)];
                            a[at(k + 2, j)] -= pp * z;
                        }
                        a[at(k + 1, j)] -= pp * y;
                        a[at(k, j)] -= pp * x;
                    }
                    let mmin = if nn < k + 3 { nn } else { k + 3 };
                    for i in l..=mmin {
                        let mut pp = x * a[at(i, k)] + y * a[at(i, k + 1)];
                        if k != nn - 1 {
                            pp += z * a[at(i, k + 2)];
                            a[at(i, k + 2)] -= pp * r;
                        }
                        a[at(i, k + 1)] -= pp * q;
                        a[at(i, k)] -= pp;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Alle Eigenwerte einer `n × n`-Matrix (zeilenweise, wird zerstört) nach `out` (Länge `n`).
pub(crate) fn eigenvalues_in_place<T: Real>(
    a: &mut [T],
    n: usize,
    out: &mut [Complex<T>],
) -> Result<(), LinalgError> {
    if a.iter().any(|&v| bad(v)) {
        return Err(LinalgError::NotFinite);
    }
    balance(a, n);
    hessenberg(a, n);
    hqr(a, n, out)
}

impl<T: Real, const N: usize> Matrix<T, N, N> {
    /// Obere Hessenberg-Form: eine ähnliche Matrix (gleiche Eigenwerte) mit Nullen unterhalb der ersten Nebendiagonale,
    /// berechnet durch stabilisierte Elimination mit Pivotsuche. Erster Schritt des QR-Algorithmus für
    /// [`eigenvalues`](Self::eigenvalues).
    pub fn hessenberg_form(&self) -> Matrix<T, N, N> {
        let mut h = *self;
        hessenberg(h.data.as_flattened_mut(), N);
        h
    }

    /// **Alle** Eigenwerte einer beliebigen reellen quadratischen Matrix (auch nichtsymmetrisch, auch komplex). Verfahren:
    /// Balancieren, Hessenberg-Form, QR-Algorithmus mit Francis-Doppelshift. Komplexe Eigenwerte treten als konjugierte Paare auf;
    /// die Reihenfolge ist nicht festgelegt. Aufwand `O(n³)`. Berechnet werden nur Eigenwerte, keine Eigenvektoren.
    ///
    /// Fehler: [`LinalgError::NotFinite`], [`LinalgError::NoConvergence`].
    ///
    /// ```
    /// # #[cfg(any(feature = "std", feature = "libm"))] {
    /// use azmath::linalg::Matrix;
    ///
    /// // Drehung um 90°: Eigenwerte ±i
    /// let r = Matrix::new([[0.0f64, -1.0], [1.0, 0.0]]);
    /// let ev = r.eigenvalues().unwrap();
    /// assert!(ev.iter().all(|z| z.re.abs() < 1e-14 && (z.im.abs() - 1.0).abs() < 1e-14));
    /// # }
    /// ```
    pub fn eigenvalues(&self) -> Result<[Complex<T>; N], LinalgError> {
        let mut a = *self;
        let mut out = [Complex::ZERO; N];
        eigenvalues_in_place(a.data.as_flattened_mut(), N, &mut out)?;
        Ok(out)
    }
}

#[cfg(feature = "alloc")]
mod dynamic {
    use super::{eigenvalues_in_place, hessenberg};
    use crate::Real;
    use crate::complex::Complex;
    use crate::linalg::{DynMatrix, LinalgError};
    use alloc::vec;
    use alloc::vec::Vec;

    impl<T: Real> DynMatrix<T> {
        /// Obere Hessenberg-Form, siehe [`Matrix::hessenberg_form`](crate::linalg::Matrix::hessenberg_form).
        /// Fehler: [`LinalgError::DimensionMismatch`] bei nicht quadratischer Matrix.
        pub fn hessenberg_form(&self) -> Result<DynMatrix<T>, LinalgError> {
            let n = self.rows();
            if n != self.cols() {
                return Err(LinalgError::DimensionMismatch);
            }
            let mut a = self.as_slice().to_vec();
            hessenberg(&mut a, n);
            DynMatrix::new(n, n, a)
        }

        /// Alle Eigenwerte, siehe [`Matrix::eigenvalues`](crate::linalg::Matrix::eigenvalues).
        /// Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NotFinite`], [`LinalgError::NoConvergence`].
        pub fn eigenvalues(&self) -> Result<Vec<Complex<T>>, LinalgError> {
            let n = self.rows();
            if n != self.cols() {
                return Err(LinalgError::DimensionMismatch);
            }
            let mut a = self.as_slice().to_vec();
            let mut out = vec![Complex::ZERO; n];
            eigenvalues_in_place(&mut a, n, &mut out)?;
            Ok(out)
        }
    }
}
