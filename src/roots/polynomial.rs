//! Nullstellen von Polynomen: Aberth-Ehrlich und Durand-Kerner.
//!
//! Beide Verfahren berechnen **alle** (komplexen) Nullstellen gleichzeitig. Koeffizienten stehen in
//! **aufsteigender** Reihenfolge: `p(x) = c[0] + c[1]·x + … + c[n]·xⁿ`.

use super::{RootError, RootOptions};
use crate::Real;
use crate::complex::Complex;

/// Ergebnis der Polynom-Nullstellensuche.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolyRootsResult<T> {
    /// Grad des Polynoms (nach Entfernen führender Nullkoeffizienten) = Zahl der Nullstellen in `roots[..degree]`.
    pub degree: usize,
    /// Anzahl der Iterationsrunden (0, wenn nur der Nullstellen bei 0 oder Grad ≤ 1 vorlag).
    pub iterations: usize,
    /// Größte relative Korrektur der letzten Runde.
    pub last_correction: T,
}

/// Wertet `c[0] + c[1]·x + … + c[n]·xⁿ` nach dem Horner-Schema aus (`c` leer ergibt 0).
///
/// ```
/// use azmath::roots::polyval;
/// // 2 + 3x + x² an x = 2
/// assert_eq!(polyval(&[2.0f64, 3.0, 1.0], 2.0), 12.0);
/// ```
pub fn polyval<T: Real>(c: &[T], x: T) -> T {
    c.iter().rev().fold(T::ZERO, |acc, &ck| acc * x + ck)
}

/// Wie [`polyval`] mit komplexem Argument und komplexen Koeffizienten.
pub fn polyval_complex<T: Real>(c: &[Complex<T>], z: Complex<T>) -> Complex<T> {
    c.iter().rev().fold(Complex::ZERO, |acc, &ck| acc * z + ck)
}

#[derive(Clone, Copy, PartialEq)]
enum Method {
    Aberth,
    DurandKerner,
}

fn finite_c<T: Real>(z: Complex<T>) -> bool {
    z.is_finite()
}

/// Kern: `coef(k)` liefert den Koeffizienten von `x^k`, `len` ist die Länge der Koeffizientenliste.
fn solve<T: Real>(
    coef: impl Fn(usize) -> Complex<T>,
    len: usize,
    roots: &mut [Complex<T>],
    opts: RootOptions<T>,
    method: Method,
) -> Result<PolyRootsResult<T>, RootError> {
    if (0..len).any(|k| !finite_c(coef(k))) {
        return Err(RootError::NotFinite);
    }
    let zero = |z: Complex<T>| z.re == T::ZERO && z.im == T::ZERO;
    let Some(hi) = (0..len).rev().find(|&k| !zero(coef(k))) else {
        return Err(RootError::InvalidInput); // Nullpolynom
    };
    let lo = (0..len).find(|&k| !zero(coef(k))).unwrap_or(0);
    if roots.len() < hi {
        return Err(RootError::InvalidInput);
    }
    for r in roots[..lo].iter_mut() {
        *r = Complex::ZERO; // Nullstellen bei 0
    }
    let d = hi - lo; // Grad des verbleibenden Polynoms
    let mut result = PolyRootsResult {
        degree: hi,
        iterations: 0,
        last_correction: T::ZERO,
    };
    if d == 0 {
        return Ok(result);
    }
    let z = &mut roots[lo..hi];
    if d == 1 {
        z[0] = -(coef(lo) / coef(hi));
        return Ok(result);
    }

    // Startwerte: Kreis mit dem Radius |c_lo / c_hi|^(1/d), mit versetztem Winkel gegen Symmetrie
    let lead = coef(hi);
    let radius = (coef(lo).abs() / lead.abs()).powf(T::ONE / T::from_f64(d as f64));
    let radius = if radius > T::ZERO && !radius.is_infinite() {
        radius
    } else {
        T::ONE
    };
    let two_pi = T::from_f64(2.0) * T::PI;
    for (k, zk) in z.iter_mut().enumerate() {
        let angle = two_pi * T::from_f64(k as f64) / T::from_f64(d as f64) + T::from_f64(0.4);
        *zk = Complex::from_polar(radius, angle);
    }

    let tiny = T::MIN_POSITIVE;
    for iter in 1..=opts.max_iter {
        let mut max_rel = T::ZERO;
        for i in 0..d {
            // p(z_i) und p'(z_i) in einem Horner-Durchgang
            let zi = z[i];
            let (mut p, mut dp) = (coef(hi), Complex::ZERO);
            for k in (lo..hi).rev() {
                dp = dp * zi + p;
                p = p * zi + coef(k);
            }
            if !finite_c(p) || !finite_c(dp) {
                return Err(RootError::NotFinite);
            }
            if zero(p) {
                continue;
            }
            let w = match method {
                Method::Aberth => {
                    let ratio = if zero(dp) {
                        p / Complex::from_real(tiny)
                    } else {
                        p / dp
                    };
                    let mut s = Complex::ZERO;
                    for (j, &zj) in z.iter().enumerate() {
                        if j != i {
                            let diff = zi - zj;
                            if !zero(diff) {
                                s += diff.recip();
                            }
                        }
                    }
                    ratio / (Complex::ONE - ratio * s)
                }
                Method::DurandKerner => {
                    let mut den = lead;
                    for (j, &zj) in z.iter().enumerate() {
                        if j != i {
                            den *= zi - zj;
                        }
                    }
                    p / den
                }
            };
            if !finite_c(w) {
                return Err(RootError::NotFinite);
            }
            z[i] = zi - w;
            let rel = w.abs() / z[i].abs().max(tiny);
            max_rel = max_rel.max(rel);
        }
        result.iterations = iter;
        result.last_correction = max_rel;
        if max_rel <= opts.tol {
            return Ok(result);
        }
    }
    Err(RootError::NoConvergence)
}

/// Alle Nullstellen eines Polynoms mit **reellen** Koeffizienten nach **Aberth-Ehrlich**: gleichzeitige
/// Newton-Iteration mit Abstoßung zwischen den Näherungen (kubische Konvergenz bei einfachen Nullstellen).
///
/// - `coeffs`: aufsteigend, `p(x) = c[0] + c[1]·x + …`. Führende Nullkoeffizienten werden ignoriert, Nullstellen bei 0
///   (niedrige Nullkoeffizienten) exakt als `0` ausgegeben.
/// - `roots`: Ausgabe, mindestens so lang wie der Grad; die Nullstellen stehen in `roots[..degree]`. Reihenfolge
///   unspezifiziert; bei reellen Koeffizienten treten komplexe Nullstellen als konjugierte Paare auf (nur bis auf
///   Rundungsfehler).
/// - Abbruch, wenn alle relativen Korrekturen `≤ opts.tol` sind. **Mehrfache Nullstellen** konvergieren nur linear
///   und mit Genauigkeit etwa `ε^(1/m)`; dafür `opts.tol` entsprechend größer wählen (zum Beispiel `1e-7`). Ebenso
///   bei hohem Grad mit schlecht konditionierten Koeffizienten (Grad 10 mit Nullstellen 1 bis 10: `tol ≈ 1e-9`).
/// - Keine Allokation. Sehr große oder sehr kleine Koeffizienten können im Horner-Schema über- oder unterlaufen.
///
/// Fehler: [`RootError::InvalidInput`] (Nullpolynom, `roots` zu kurz), [`RootError::NotFinite`],
/// [`RootError::NoConvergence`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::complex::Complex;
/// use azmath::roots::{aberth, RootOptions};
///
/// // x² + 1 = 0  →  ±i
/// let mut z = [Complex::new(0.0f64, 0.0); 2];
/// let r = aberth(&[1.0, 0.0, 1.0], &mut z, RootOptions::default()).unwrap();
/// assert_eq!(r.degree, 2);
/// assert!(z.iter().all(|r| r.re.abs() < 1e-14 && (r.im.abs() - 1.0).abs() < 1e-14));
/// # }
/// ```
pub fn aberth<T: Real>(
    coeffs: &[T],
    roots: &mut [Complex<T>],
    opts: RootOptions<T>,
) -> Result<PolyRootsResult<T>, RootError> {
    solve(
        |k| Complex::from_real(coeffs[k]),
        coeffs.len(),
        roots,
        opts,
        Method::Aberth,
    )
}

/// Wie [`aberth`] für **komplexe** Koeffizienten.
pub fn aberth_complex<T: Real>(
    coeffs: &[Complex<T>],
    roots: &mut [Complex<T>],
    opts: RootOptions<T>,
) -> Result<PolyRootsResult<T>, RootError> {
    solve(|k| coeffs[k], coeffs.len(), roots, opts, Method::Aberth)
}

/// Alle Nullstellen eines Polynoms mit reellen Koeffizienten nach **Durand-Kerner** (Weierstraß): einfacher als
/// Aberth, nur quadratisch konvergent und ohne Ableitung. Parameter und Fehler wie bei [`aberth`].
pub fn durand_kerner<T: Real>(
    coeffs: &[T],
    roots: &mut [Complex<T>],
    opts: RootOptions<T>,
) -> Result<PolyRootsResult<T>, RootError> {
    solve(
        |k| Complex::from_real(coeffs[k]),
        coeffs.len(),
        roots,
        opts,
        Method::DurandKerner,
    )
}

/// Wie [`durand_kerner`] für **komplexe** Koeffizienten.
pub fn durand_kerner_complex<T: Real>(
    coeffs: &[Complex<T>],
    roots: &mut [Complex<T>],
    opts: RootOptions<T>,
) -> Result<PolyRootsResult<T>, RootError> {
    solve(
        |k| coeffs[k],
        coeffs.len(),
        roots,
        opts,
        Method::DurandKerner,
    )
}
