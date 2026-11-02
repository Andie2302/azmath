use super::{InterpError, check_nodes};
use crate::Real;

/// Index `i` des Teilintervalls `[x[i], x[i+1]]`, das `xq` enthält (`0 ≤ i ≤ n − 2`).
///
/// Außerhalb des Bereichs wird das erste bzw. letzte Teilintervall geliefert (Extrapolation).
/// Binäre Suche in `O(log n)`. `x` muss aufsteigend sein und mindestens 2 Elemente haben.
pub fn locate<T: Real>(x: &[T], xq: T) -> usize {
    let n = x.len();
    debug_assert!(n >= 2);
    // Anzahl der Stützstellen ≤ xq
    let k = x.partition_point(|v| *v <= xq);
    k.saturating_sub(1).min(n - 2)
}

/// Stückweise lineare Interpolation der Punkte `(x[i], y[i])` an der Stelle `xq`.
///
/// Außerhalb von `[x[0], x[n−1]]` wird die erste bzw. letzte Gerade fortgesetzt.
///
/// Fehler: [`InterpError::InvalidInput`] (Längen, weniger als 2 Punkte),
/// [`InterpError::NotIncreasing`], [`InterpError::NotFinite`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::interp::linear;
///
/// let (x, y) = ([0.0, 1.0, 3.0], [0.0, 2.0, 0.0]);
/// assert_eq!(linear(&x, &y, 0.5).unwrap(), 1.0);
/// assert_eq!(linear(&x, &y, 2.0).unwrap(), 1.0);
/// # }
/// ```
pub fn linear<T: Real>(x: &[T], y: &[T], xq: T) -> Result<T, InterpError> {
    check_nodes(x, y)?;
    let i = locate(x, xq);
    let t = (xq - x[i]) / (x[i + 1] - x[i]);
    Ok(y[i] + t * (y[i + 1] - y[i]))
}
