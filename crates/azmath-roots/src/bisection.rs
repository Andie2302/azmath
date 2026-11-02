use super::{RootError, RootOptions, RootResult, finite, scale};
use azmath_core::Real;

/// Bisektion: halbiert das Intervall `[a, b]` mit Vorzeichenwechsel, bis es klein genug ist.
///
/// Langsam (ein Bit pro Schritt), konvergiert aber immer, sobald `f(a)` und `f(b)` verschiedene
/// Vorzeichen haben. Die Reihenfolge `a < b` ist nicht nötig. Ein Vorzeichenwechsel kann auch
/// eine Polstelle statt einer Nullstelle einschließen (zum Beispiel `1/x`); `f_value` im
/// Ergebnis zeigt das.
///
/// Fehler: [`RootError::NotBracketed`], [`RootError::NotFinite`], [`RootError::NoConvergence`].
pub fn bisection<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: RootOptions<T>,
) -> Result<RootResult<T>, RootError> {
    let (mut lo, mut hi) = (a, b);
    let mut flo = f(lo);
    let fhi = f(hi);
    if !finite(lo) || !finite(hi) || !finite(flo) || !finite(fhi) {
        return Err(RootError::NotFinite);
    }
    if flo == T::ZERO {
        return Ok(RootResult {
            root: lo,
            f_value: flo,
            iterations: 0,
        });
    }
    if fhi == T::ZERO {
        return Ok(RootResult {
            root: hi,
            f_value: fhi,
            iterations: 0,
        });
    }
    if (flo < T::ZERO) == (fhi < T::ZERO) {
        return Err(RootError::NotBracketed);
    }

    let half = T::from_f64(0.5);
    for k in 1..=opts.max_iter {
        let mid = lo + half * (hi - lo);
        let fmid = f(mid);
        if !finite(fmid) {
            return Err(RootError::NotFinite);
        }
        if fmid == T::ZERO || (hi - lo).abs() <= opts.tol * scale(mid) {
            return Ok(RootResult {
                root: mid,
                f_value: fmid,
                iterations: k,
            });
        }
        if (fmid < T::ZERO) == (flo < T::ZERO) {
            lo = mid;
            flo = fmid;
        } else {
            hi = mid;
        }
    }
    Err(RootError::NoConvergence)
}
