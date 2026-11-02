use super::{IntegrateError, finite, from_usize};
use azmath_core::Real;

/// Zusammengesetzte Trapezregel mit `n` gleich breiten Teilintervallen.
///
/// Exakt für lineare Funktionen, der Fehler fällt mit `h²`. Für glatte, periodische Funktionen
/// über eine ganze Periode ist sie überraschend genau.
///
/// Fehler: [`IntegrateError::InvalidInput`] bei `n = 0`, [`IntegrateError::NotFinite`].
pub fn trapezoid<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    n: usize,
) -> Result<T, IntegrateError> {
    if n == 0 {
        return Err(IntegrateError::InvalidInput);
    }
    if !finite(a) || !finite(b) {
        return Err(IntegrateError::NotFinite);
    }
    let h = (b - a) / from_usize::<T>(n);
    let mut sum = T::from_f64(0.5) * (f(a) + f(b));
    for i in 1..n {
        sum += f(a + h * from_usize::<T>(i));
    }
    let r = sum * h;
    if finite(r) {
        Ok(r)
    } else {
        Err(IntegrateError::NotFinite)
    }
}

/// Zusammengesetzte Simpson-Regel mit `n` Teilintervallen (`n` muss gerade sein).
///
/// Exakt bis Grad 3, der Fehler fällt mit `h⁴`.
///
/// Fehler: [`IntegrateError::InvalidInput`] bei `n = 0` oder ungeradem `n`,
/// [`IntegrateError::NotFinite`].
pub fn simpson<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    n: usize,
) -> Result<T, IntegrateError> {
    if n == 0 || n % 2 != 0 {
        return Err(IntegrateError::InvalidInput);
    }
    if !finite(a) || !finite(b) {
        return Err(IntegrateError::NotFinite);
    }
    let h = (b - a) / from_usize::<T>(n);
    let mut odd = T::ZERO;
    let mut even = T::ZERO;
    for i in 1..n {
        let y = f(a + h * from_usize::<T>(i));
        if i % 2 == 1 {
            odd += y;
        } else {
            even += y;
        }
    }
    let r = (f(a) + f(b) + T::from_f64(4.0) * odd + T::from_f64(2.0) * even) * h / T::from_f64(3.0);
    if finite(r) {
        Ok(r)
    } else {
        Err(IntegrateError::NotFinite)
    }
}

/// Trapezregel für **gleichmäßig verteilte Messwerte** `y` im Abstand `h`.
///
/// Fehler: [`IntegrateError::InvalidInput`] bei weniger als 2 Werten, [`IntegrateError::NotFinite`].
pub fn trapezoid_samples<T: Real>(y: &[T], h: T) -> Result<T, IntegrateError> {
    if y.len() < 2 {
        return Err(IntegrateError::InvalidInput);
    }
    let mut sum = T::from_f64(0.5) * (y[0] + y[y.len() - 1]);
    for v in &y[1..y.len() - 1] {
        sum += *v;
    }
    let r = sum * h;
    if finite(r) {
        Ok(r)
    } else {
        Err(IntegrateError::NotFinite)
    }
}

/// Simpson-Regel für **gleichmäßig verteilte Messwerte** `y` im Abstand `h`.
///
/// Es sind eine ungerade Anzahl Werte (≥ 3), also eine gerade Zahl an Teilintervallen nötig.
///
/// Fehler: [`IntegrateError::InvalidInput`], [`IntegrateError::NotFinite`].
pub fn simpson_samples<T: Real>(y: &[T], h: T) -> Result<T, IntegrateError> {
    let n = y.len();
    if n < 3 || n % 2 == 0 {
        return Err(IntegrateError::InvalidInput);
    }
    let mut sum = y[0] + y[n - 1];
    for (i, v) in y[1..n - 1].iter().enumerate() {
        // i = 0 entspricht Stützstelle 1 (ungerade)
        sum += if i % 2 == 0 {
            T::from_f64(4.0) * *v
        } else {
            T::from_f64(2.0) * *v
        };
    }
    let r = sum * h / T::from_f64(3.0);
    if finite(r) {
        Ok(r)
    } else {
        Err(IntegrateError::NotFinite)
    }
}

/// Trapezregel für **beliebig verteilte Messwerte**: Stützstellen `x` (aufsteigend oder
/// absteigend) und Werte `y`.
///
/// Fehler: [`IntegrateError::InvalidInput`] bei verschieden langen Reihen oder weniger als
/// 2 Punkten, [`IntegrateError::NotFinite`].
pub fn trapezoid_xy<T: Real>(x: &[T], y: &[T]) -> Result<T, IntegrateError> {
    if x.len() != y.len() || x.len() < 2 {
        return Err(IntegrateError::InvalidInput);
    }
    let mut sum = T::ZERO;
    for i in 1..x.len() {
        sum += T::from_f64(0.5) * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
    }
    if finite(sum) {
        Ok(sum)
    } else {
        Err(IntegrateError::NotFinite)
    }
}
