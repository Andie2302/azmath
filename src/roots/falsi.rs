use super::{RootError, RootOptions, RootResult, finite, scale};
use crate::Real;

/// Gemeinsamer Kern von Regula Falsi und Illinois-Verfahren.
fn false_position<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: RootOptions<T>,
    illinois: bool,
) -> Result<RootResult<T>, RootError> {
    let (mut lo, mut hi) = (a, b);
    let mut flo = f(lo);
    let mut fhi = f(hi);
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
    // Welche Seite zuletzt ersetzt wurde: -1 = lo, +1 = hi, 0 = noch keine
    let mut side = 0i8;
    let mut prev = hi;
    for k in 1..=opts.max_iter {
        let mut x = hi - fhi * (hi - lo) / (fhi - flo);
        let (min, max) = if lo < hi { (lo, hi) } else { (hi, lo) };
        if !finite(x) || x < min || x > max {
            x = lo + half * (hi - lo); // Rundung: zurück auf die Mitte
        }
        let fx = f(x);
        if !finite(fx) {
            return Err(RootError::NotFinite);
        }
        let step = (x - prev).abs();
        prev = x;
        if fx == T::ZERO {
            return Ok(RootResult {
                root: x,
                f_value: fx,
                iterations: k,
            });
        }
        if (fx < T::ZERO) == (flo < T::ZERO) {
            lo = x;
            flo = fx;
            if illinois && side == -1 {
                fhi *= half;
            }
            side = -1;
        } else {
            hi = x;
            fhi = fx;
            if illinois && side == 1 {
                flo *= half;
            }
            side = 1;
        }
        if (hi - lo).abs() <= opts.tol * scale(x) || (k > 1 && step <= opts.tol * scale(x)) {
            return Ok(RootResult {
                root: x,
                f_value: fx,
                iterations: k,
            });
        }
    }
    Err(RootError::NoConvergence)
}

/// Regula Falsi (Methode der falschen Position): Sekante durch die Intervallenden, Einschluss bleibt erhalten.
///
/// Sicher wie die Bisektion, aber bei konvexen Funktionen bleibt ein Ende stehen und die Konvergenz wird
/// **nur linear mit schlechter Rate** (siehe [`illinois`] als Verbesserung). Abbruch, wenn das Intervall
/// oder der letzte Schritt `≤ tol·max(1,|x|)` ist; wegen des einseitigen Verhaltens kann der Fehler dann ein
/// Vielfaches der Toleranz betragen.
///
/// Fehler: [`RootError::NotBracketed`], [`RootError::NotFinite`], [`RootError::NoConvergence`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::roots::{regula_falsi, RootOptions};
///
/// let r = regula_falsi(|x: f64| x * x - 2.0, 0.0, 2.0, RootOptions::default()).unwrap();
/// assert!((r.root - 2f64.sqrt()).abs() < 1e-12);
/// # }
/// ```
pub fn regula_falsi<T: Real>(
    f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: RootOptions<T>,
) -> Result<RootResult<T>, RootError> {
    false_position(f, a, b, opts, false)
}

/// Illinois-Verfahren: Regula Falsi, die bei zweimal derselben Seite den Funktionswert des stehenbleibenden
/// Endes halbiert. Das behebt die einseitige Stagnation; die Konvergenzordnung ist etwa `1,44`.
///
/// Fehler: [`RootError::NotBracketed`], [`RootError::NotFinite`], [`RootError::NoConvergence`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::roots::{illinois, RootOptions};
///
/// let r = illinois(|x: f64| x.powi(10) - 1.0, 0.0, 1.3, RootOptions::default()).unwrap();
/// assert!((r.root - 1.0).abs() < 1e-12);
/// # }
/// ```
pub fn illinois<T: Real>(
    f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: RootOptions<T>,
) -> Result<RootResult<T>, RootError> {
    false_position(f, a, b, opts, true)
}
