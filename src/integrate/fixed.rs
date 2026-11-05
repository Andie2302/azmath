use super::{
    IntegrateError, WeightedSum, checked, finite, from_usize, integral_scale, mul3, point,
};
use crate::Real;

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
    if a == b {
        return Ok(T::ZERO);
    }
    let weight = T::ONE / from_usize::<T>(n);
    let mut sum = WeightedSum::new();
    for i in 0..=n {
        let x = if i == n {
            b
        } else {
            point(a, b, from_usize::<T>(i) / from_usize::<T>(n))
        };
        sum.add(
            f(x),
            if i == 0 || i == n {
                weight * T::from_f64(0.5)
            } else {
                weight
            },
        )?;
    }
    checked(integral_scale(sum.value(), a, b))
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
    if a == b {
        return Ok(T::ZERO);
    }
    let weight = T::ONE / (T::from_f64(3.0) * from_usize::<T>(n));
    let mut sum = WeightedSum::new();
    for i in 0..=n {
        let x = if i == n {
            b
        } else {
            point(a, b, from_usize::<T>(i) / from_usize::<T>(n))
        };
        let factor = if i == 0 || i == n {
            T::ONE
        } else if i % 2 == 1 {
            T::from_f64(4.0)
        } else {
            T::from_f64(2.0)
        };
        sum.add(f(x), weight * factor)?;
    }
    checked(integral_scale(sum.value(), a, b))
}

/// Trapezregel für **gleichmäßig verteilte Messwerte** `y` im Abstand `h`.
///
/// Fehler: [`IntegrateError::InvalidInput`] bei weniger als 2 Werten, [`IntegrateError::NotFinite`].
pub fn trapezoid_samples<T: Real>(y: &[T], h: T) -> Result<T, IntegrateError> {
    if y.len() < 2 {
        return Err(IntegrateError::InvalidInput);
    }
    if !finite(h) {
        return Err(IntegrateError::NotFinite);
    }
    let n = y.len() - 1;
    let weight = T::ONE / from_usize::<T>(n);
    let mut sum = WeightedSum::new();
    for (i, &value) in y.iter().enumerate() {
        sum.add(
            value,
            if i == 0 || i == n {
                weight * T::from_f64(0.5)
            } else {
                weight
            },
        )?;
    }
    checked(mul3(sum.value(), h, from_usize::<T>(n)))
}

/// Simpson-Regel für **gleichmäßig verteilte Messwerte** `y` im Abstand `h`.
///
/// Es sind eine ungerade Anzahl Werte (≥ 3), also eine gerade Zahl an Teilintervallen nötig.
///
/// Fehler: [`IntegrateError::InvalidInput`], [`IntegrateError::NotFinite`].
pub fn simpson_samples<T: Real>(y: &[T], h: T) -> Result<T, IntegrateError> {
    if y.len() < 3 || y.len() % 2 == 0 {
        return Err(IntegrateError::InvalidInput);
    }
    if !finite(h) {
        return Err(IntegrateError::NotFinite);
    }
    let n = y.len() - 1;
    let weight = T::ONE / (T::from_f64(3.0) * from_usize::<T>(n));
    let mut sum = WeightedSum::new();
    for (i, &value) in y.iter().enumerate() {
        let factor = if i == 0 || i == n {
            T::ONE
        } else if i % 2 == 1 {
            T::from_f64(4.0)
        } else {
            T::from_f64(2.0)
        };
        sum.add(value, weight * factor)?;
    }
    checked(mul3(sum.value(), h, from_usize::<T>(n)))
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
    if x.iter().chain(y.iter()).any(|v| !finite(*v)) {
        return Err(IntegrateError::NotFinite);
    }
    let mut total = WeightedSum::new();
    for i in 1..x.len() {
        let mut mean = WeightedSum::new();
        mean.add(y[i - 1], T::from_f64(0.5))?;
        mean.add(y[i], T::from_f64(0.5))?;
        total.add(integral_scale(mean.value(), x[i - 1], x[i]), T::ONE)?;
    }
    checked(total.value())
}
