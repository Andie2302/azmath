use super::{InterpError, finite};
use azmath_core::Real;

/// Wertet das **eine** Polynom vom Grad `n − 1` durch alle `n` Punkte `(x[i], y[i])` an der Stelle
/// `xq` aus (Neville-Verfahren, `O(n²)`).
///
/// Nur für wenige, gut verteilte Punkte sinnvoll: Bei vielen äquidistanten Stützstellen
/// schwingt das Polynom am Rand stark (Runge-Phänomen); dann sind [`CubicSpline`](super::CubicSpline)
/// oder [`Pchip`](super::Pchip) die bessere Wahl. Die Stützstellen müssen paarweise verschieden sein,
/// brauchen aber nicht sortiert zu sein.
///
/// - `work`: mindestens `n` Elemente.
///
/// Fehler: [`InterpError::InvalidInput`] (Längen, kein Punkt, zu kurzer `work`),
/// [`InterpError::NotIncreasing`] (doppelte Stützstelle), [`InterpError::NotFinite`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath_interp::lagrange;
///
/// // Parabel durch (0,1), (1,2), (3,10): y = x² + 1
/// let (x, y) = ([0.0, 1.0, 3.0], [1.0, 2.0, 10.0]);
/// let mut work = [0.0; 3];
/// assert!((lagrange(&x, &y, 2.0, &mut work).unwrap() - 5.0f64).abs() < 1e-12);
/// # }
/// ```
pub fn lagrange<T: Real>(x: &[T], y: &[T], xq: T, work: &mut [T]) -> Result<T, InterpError> {
    let n = x.len();
    if n == 0 || y.len() != n || work.len() < n {
        return Err(InterpError::InvalidInput);
    }
    if x.iter().chain(y.iter()).any(|v| !finite(*v)) || !finite(xq) {
        return Err(InterpError::NotFinite);
    }
    let p = &mut work[..n];
    p.copy_from_slice(y);
    // Neville: p[i] ← Wert des Interpolationspolynoms durch x[i..=i+level]
    for level in 1..n {
        for i in 0..n - level {
            let denom = x[i] - x[i + level];
            if denom == T::ZERO {
                return Err(InterpError::NotIncreasing);
            }
            p[i] = ((xq - x[i + level]) * p[i] + (x[i] - xq) * p[i + 1]) / denom;
        }
    }
    Ok(p[0])
}
