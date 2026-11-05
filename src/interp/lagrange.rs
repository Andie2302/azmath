use super::{InterpError, blend, finite, fraction};
use crate::Real;

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
/// use azmath::interp::lagrange;
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
    for i in 0..n {
        for j in 0..i {
            if x[i] == x[j] {
                return Err(InterpError::NotIncreasing);
            }
        }
    }
    if let Some(i) = x.iter().position(|v| *v == xq) {
        return Ok(y[i]);
    }
    let p = &mut work[..n];
    p.copy_from_slice(y);
    for level in 1..n {
        for i in 0..n - level {
            let t = fraction(x[i], x[i + level], xq);
            p[i] = blend(p[i], p[i + 1], t);
            if !finite(p[i]) {
                return Err(InterpError::NotFinite);
            }
        }
    }
    Ok(p[0])
}
