//! Gauß-Quadraturregeln beliebiger Ordnung nach Golub-Welsch und Tensorprodukt-Kubatur.

use super::{IntegrateError, finite};
use crate::Real;
use crate::special::ln_gamma;

/// Familie orthogonaler Polynome (Gewichtsfunktion und Intervall).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Family<T> {
    /// Legendre: Gewicht 1 auf `[−1, 1]`.
    Legendre,
    /// Tschebyscheff 1. Art: Gewicht `1/√(1−x²)` auf `(−1, 1)`.
    ChebyshevT,
    /// Tschebyscheff 2. Art: Gewicht `√(1−x²)` auf `[−1, 1]`.
    ChebyshevU,
    /// Laguerre mit Parameter `α > −1`: Gewicht `x^α·e^(−x)` auf `[0, ∞)`.
    Laguerre(T),
    /// Hermite: Gewicht `e^(−x²)` auf `(−∞, ∞)`.
    Hermite,
    /// Jacobi mit `α, β > −1`: Gewicht `(1−x)^α·(1+x)^β` auf `[−1, 1]`.
    Jacobi(T, T),
}

/// Arbeitsspeicher für [`gauss_rule`] bei `n` Knoten.
pub const fn gauss_quad_work_len(n: usize) -> usize {
    n
}

/// Knoten und Gewichte der `n`-Punkte-Gauß-Regel der Familie `family`: `∫ w(x)·f(x) dx ≈ Σ weights[i]·f(nodes[i])`, exakt für
/// Polynome bis Grad `2n − 1`. Die Knoten sind aufsteigend sortiert.
///
/// Berechnung über die Eigenwerte der Jacobi-Matrix (Golub-Welsch, implizites QL). `nodes`, `weights`: mindestens `n`
/// Elemente; `work`: mindestens [`gauss_quad_work_len`]`(n)`.
///
/// Fehler: [`IntegrateError::InvalidInput`] (`n = 0`, zu kurze Slices, Parameter `≤ −1`), [`IntegrateError::NoConvergence`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::integrate::{gauss_rule, Family};
///
/// let (mut x, mut w, mut work) = ([0.0f64; 3], [0.0; 3], [0.0; 3]);
/// gauss_rule(Family::Legendre, 3, &mut x, &mut w, &mut work).unwrap();
/// let s: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(4)).sum();
/// assert!((s - 2.0 / 5.0).abs() < 1e-13);
/// # }
/// ```
pub fn gauss_rule<T: Real>(
    family: Family<T>,
    n: usize,
    nodes: &mut [T],
    weights: &mut [T],
    work: &mut [T],
) -> Result<(), IntegrateError> {
    if n == 0 || nodes.len() < n || weights.len() < n || work.len() < n {
        return Err(IntegrateError::InvalidInput);
    }
    let one = T::ONE;
    let two = T::from_f64(2.0);
    let four = T::from_f64(4.0);
    let fk = |k: usize| T::from_usize(k);
    let d = &mut nodes[..n];
    let e = &mut work[..n];
    let mu0: T;
    match family {
        Family::Legendre => {
            mu0 = two;
            for k in 0..n {
                d[k] = T::ZERO;
                let kk = fk(k + 1);
                e[k] = (kk * kk / (four * kk * kk - one)).sqrt();
            }
        }
        Family::ChebyshevT => {
            mu0 = T::PI;
            for k in 0..n {
                d[k] = T::ZERO;
                e[k] = if k == 0 {
                    T::from_f64(0.5).sqrt()
                } else {
                    T::from_f64(0.5)
                };
            }
        }
        Family::ChebyshevU => {
            mu0 = T::PI / two;
            for k in 0..n {
                d[k] = T::ZERO;
                e[k] = T::from_f64(0.5);
            }
        }
        Family::Laguerre(a) => {
            if !(a > -one) || !finite(a) {
                return Err(IntegrateError::InvalidInput);
            }
            mu0 = ln_gamma(a + one).exp();
            for k in 0..n {
                d[k] = two * fk(k) + a + one;
                let kk = fk(k + 1);
                e[k] = (kk * (kk + a)).sqrt();
            }
        }
        Family::Hermite => {
            mu0 = T::PI.sqrt();
            for k in 0..n {
                d[k] = T::ZERO;
                e[k] = (fk(k + 1) / two).sqrt();
            }
        }
        Family::Jacobi(a, b) => {
            if !(a > -one) || !(b > -one) || !finite(a) || !finite(b) {
                return Err(IntegrateError::InvalidInput);
            }
            let ab = a + b;
            mu0 = ((ab + one) * two.ln() + ln_gamma(a + one) + ln_gamma(b + one)
                - ln_gamma(ab + two))
            .exp();
            d[0] = (b - a) / (ab + two);
            for k in 1..n {
                let kk = fk(k);
                let s = two * kk + ab;
                d[k] = (b * b - a * a) / (s * (s + two));
            }
            for k in 0..n {
                let kk = fk(k + 1);
                e[k] = if k == 0 {
                    (four * (one + a) * (one + b)
                        / ((two + ab) * (two + ab) * (T::from_f64(3.0) + ab)))
                        .sqrt()
                } else {
                    let s = two * kk + ab;
                    (four * kk * (kk + a) * (kk + b) * (kk + ab) / (s * s * (s + one) * (s - one)))
                        .sqrt()
                };
            }
        }
    }
    e[n - 1] = T::ZERO;
    let z = &mut weights[..n];
    z.iter_mut().for_each(|v| *v = T::ZERO);
    z[0] = one;
    tridiagonal_ql_first_row(d, e, z)?;
    for v in z.iter_mut() {
        *v = mu0 * *v * *v;
    }
    // aufsteigend sortieren (Einfügesortierung)
    for i in 1..n {
        let mut j = i;
        while j > 0 && d[j - 1] > d[j] {
            d.swap(j - 1, j);
            z.swap(j - 1, j);
            j -= 1;
        }
    }
    Ok(())
}

/// Implizites QL-Verfahren für die symmetrische Tridiagonalmatrix (`d` Diagonale, `e[i]` koppelt `i` und `i+1`, `e[n−1] = 0`);
/// die Rotationen werden nur auf den Vektor `z` angewendet (erste Zeile der Eigenvektormatrix).
fn tridiagonal_ql_first_row<T: Real>(
    d: &mut [T],
    e: &mut [T],
    z: &mut [T],
) -> Result<(), IntegrateError> {
    let n = d.len();
    let eps = T::EPSILON;
    for l in 0..n {
        let mut iter = 0;
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= eps * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            iter += 1;
            if iter > 100 {
                return Err(IntegrateError::NoConvergence);
            }
            let mut g = (d[l + 1] - d[l]) / (T::from_f64(2.0) * e[l]);
            let mut r = g.hypot(T::ONE);
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let (mut s, mut c, mut p) = (T::ONE, T::ONE, T::ZERO);
            let mut underflow = false;
            let mut i = m;
            while i > l {
                i -= 1;
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot(g);
                e[i + 1] = r;
                if r == T::ZERO {
                    d[i + 1] -= p;
                    e[m] = T::ZERO;
                    underflow = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + T::from_f64(2.0) * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                let f = z[i + 1];
                z[i + 1] = s * z[i] + c * f;
                z[i] = c * z[i] - s * f;
            }
            if underflow {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = T::ZERO;
        }
    }
    Ok(())
}

/// Summe `Σ weights[i]·f(nodes[i])`.
pub fn gauss_apply<T: Real>(mut f: impl FnMut(T) -> T, nodes: &[T], weights: &[T]) -> T {
    let mut s = T::ZERO;
    for (x, w) in nodes.iter().zip(weights) {
        s += *w * f(*x);
    }
    s
}

/// Arbeitsspeicher für [`tensor_gauss`] bei `dim` Dimensionen und `n` Punkten je Dimension.
pub const fn tensor_gauss_work_len(dim: usize, n: usize) -> usize {
    3 * n + dim
}

/// Integral von `f` über den Quader `∏[lo_i, hi_i]` mit der Tensorproduktregel aus `n`-Punkte-Gauß-Legendre-Regeln
/// (`n^dim` Auswertungen, exakt für Polynome mit Grad `≤ 2n−1` je Variable).
///
/// `work`: mindestens [`tensor_gauss_work_len`]`(dim, n)`. Für hohe Dimensionen: `stats::mc_integrate`.
///
/// Fehler: [`IntegrateError::InvalidInput`], [`IntegrateError::NotFinite`], [`IntegrateError::NoConvergence`].
pub fn tensor_gauss<T: Real>(
    mut f: impl FnMut(&[T]) -> T,
    lo: &[T],
    hi: &[T],
    n: usize,
    work: &mut [T],
) -> Result<T, IntegrateError> {
    let dim = lo.len();
    if dim == 0 || hi.len() != dim || n == 0 || work.len() < tensor_gauss_work_len(dim, n) {
        return Err(IntegrateError::InvalidInput);
    }
    if lo.iter().chain(hi).any(|&v| !finite(v)) {
        return Err(IntegrateError::NotFinite);
    }
    let (nodes, rest) = work.split_at_mut(n);
    let (weights, rest) = rest.split_at_mut(n);
    let (tmp, rest) = rest.split_at_mut(n);
    let point = &mut rest[..dim];
    gauss_rule(Family::Legendre, n, nodes, weights, tmp)?;
    let half = T::from_f64(0.5);
    fn rec<T: Real>(
        k: usize,
        f: &mut impl FnMut(&[T]) -> T,
        lo: &[T],
        hi: &[T],
        nodes: &[T],
        weights: &[T],
        point: &mut [T],
        half: T,
    ) -> T {
        if k == lo.len() {
            return f(point);
        }
        let c = (hi[k] + lo[k]) * half;
        let r = (hi[k] - lo[k]) * half;
        let mut s = T::ZERO;
        for (x, w) in nodes.iter().zip(weights) {
            point[k] = c + r * *x;
            s += *w * rec(k + 1, f, lo, hi, nodes, weights, point, half);
        }
        s * r
    }
    let v = rec(0, &mut f, lo, hi, nodes, weights, point, half);
    if finite(v) {
        Ok(v)
    } else {
        Err(IntegrateError::NotFinite)
    }
}
