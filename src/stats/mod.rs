//! Statistik auf Slices: Momente, Kovarianz, Korrelation, Quantile und Monte-Carlo-Integration.
//!
//! Leere Eingaben oder zu kurze Stichproben liefern `NaN`. Varianz und Kovarianz sind die **Stichprobenwerte**
//! (Nenner `n − 1`).

use crate::Real;
use crate::random::Rng;

/// Arithmetisches Mittel.
pub fn mean<T: Real>(x: &[T]) -> T {
    if x.is_empty() {
        return T::NAN;
    }
    let mut s = T::ZERO;
    for &v in x {
        s += v;
    }
    s / T::from_usize(x.len())
}

/// Stichprobenvarianz (Welford, numerisch stabil).
pub fn variance<T: Real>(x: &[T]) -> T {
    if x.len() < 2 {
        return T::NAN;
    }
    let (mut m, mut m2) = (T::ZERO, T::ZERO);
    for (i, &v) in x.iter().enumerate() {
        let d = v - m;
        m += d / T::from_usize(i + 1);
        m2 += d * (v - m);
    }
    m2 / T::from_usize(x.len() - 1)
}

/// Stichproben-Standardabweichung.
pub fn std_dev<T: Real>(x: &[T]) -> T {
    variance(x).sqrt()
}

/// Stichprobenkovarianz zweier gleich langer Reihen (`NaN` bei ungleicher Länge).
pub fn covariance<T: Real>(x: &[T], y: &[T]) -> T {
    if x.len() != y.len() || x.len() < 2 {
        return T::NAN;
    }
    let (mx, my) = (mean(x), mean(y));
    let mut s = T::ZERO;
    for i in 0..x.len() {
        s += (x[i] - mx) * (y[i] - my);
    }
    s / T::from_usize(x.len() - 1)
}

/// Pearson-Korrelationskoeffizient.
pub fn correlation<T: Real>(x: &[T], y: &[T]) -> T {
    covariance(x, y) / (std_dev(x) * std_dev(y))
}

/// Quantil `q ∈ [0, 1]` einer **aufsteigend sortierten** Reihe (lineare Interpolation, wie NumPy-Standard).
pub fn quantile_sorted<T: Real>(sorted: &[T], q: T) -> T {
    let n = sorted.len();
    if n == 0 || q.is_nan() || q < T::ZERO || q > T::ONE {
        return T::NAN;
    }
    let pos = q * T::from_usize(n - 1);
    let lo = pos.floor();
    let i = (lo.to_f64() as usize).min(n - 1);
    let frac = pos - lo;
    if i + 1 < n {
        sorted[i] + (sorted[i + 1] - sorted[i]) * frac
    } else {
        sorted[i]
    }
}

/// Quantil einer beliebigen Reihe (sortiert eine Kopie; NaN führt zu `NaN`).
#[cfg(feature = "alloc")]
pub fn quantile<T: Real>(x: &[T], q: T) -> T {
    if x.iter().any(|v| v.is_nan()) {
        return T::NAN;
    }
    let mut v = alloc::vec::Vec::from(x);
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
    quantile_sorted(&v, q)
}

/// Median.
#[cfg(feature = "alloc")]
pub fn median<T: Real>(x: &[T]) -> T {
    quantile(x, T::from_f64(0.5))
}

/// Monte-Carlo-Integral von `f` über den Quader `∏[lo_i, hi_i]` mit `n` Zufallspunkten.
/// Liefert `(Schätzwert, Standardfehler)`. `work`: mindestens `lo.len()` Elemente.
pub fn mc_integrate<T: Real, R: Rng>(
    mut f: impl FnMut(&[T]) -> T,
    lo: &[T],
    hi: &[T],
    n: usize,
    rng: &mut R,
    work: &mut [T],
) -> (T, T) {
    let d = lo.len();
    assert!(
        hi.len() == d && work.len() >= d && n >= 2,
        "mc_integrate: ungültige Eingaben"
    );
    let mut volume = T::ONE;
    for i in 0..d {
        volume *= hi[i] - lo[i];
    }
    let (mut m, mut m2) = (T::ZERO, T::ZERO);
    for k in 0..n {
        for i in 0..d {
            work[i] = rng.uniform_range(lo[i], hi[i]);
        }
        let v = f(&work[..d]);
        let delta = v - m;
        m += delta / T::from_usize(k + 1);
        m2 += delta * (v - m);
    }
    let var = m2 / T::from_usize(n - 1);
    (volume * m, volume * (var / T::from_usize(n)).sqrt())
}
