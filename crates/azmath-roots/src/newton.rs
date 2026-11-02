use super::{RootError, RootOptions, RootResult, finite, scale};
use azmath_core::Real;

/// Newton-Verfahren: `x ← x − f(x)/f'(x)`.
///
/// Quadratisch konvergent nahe einer einfachen Nullstelle, aber ohne Garantie: ein schlechter
/// Startwert oder eine flache Stelle kann zur Divergenz führen. Ist ein Intervall mit
/// Vorzeichenwechsel bekannt, ist [`brent`](super::brent) die sicherere Wahl.
///
/// - `f`: die Funktion, `df`: ihre Ableitung, `x0`: Startwert.
///
/// Fehler: [`RootError::ZeroDerivative`] (`f'(x) = 0`), [`RootError::NotFinite`],
/// [`RootError::NoConvergence`].
pub fn newton<T: Real>(
    mut f: impl FnMut(T) -> T,
    mut df: impl FnMut(T) -> T,
    x0: T,
    opts: RootOptions<T>,
) -> Result<RootResult<T>, RootError> {
    if !finite(x0) {
        return Err(RootError::NotFinite);
    }
    let mut x = x0;
    for k in 0..opts.max_iter {
        let fx = f(x);
        if !finite(fx) {
            return Err(RootError::NotFinite);
        }
        if fx == T::ZERO {
            return Ok(RootResult {
                root: x,
                f_value: fx,
                iterations: k,
            });
        }
        let dfx = df(x);
        if !finite(dfx) {
            return Err(RootError::NotFinite);
        }
        if dfx == T::ZERO {
            return Err(RootError::ZeroDerivative);
        }
        let dx = fx / dfx;
        let x_new = x - dx;
        if !finite(x_new) {
            return Err(RootError::NotFinite);
        }
        if dx.abs() <= opts.tol * scale(x_new) {
            let f_value = f(x_new);
            return Ok(RootResult {
                root: x_new,
                f_value,
                iterations: k + 1,
            });
        }
        x = x_new;
    }
    Err(RootError::NoConvergence)
}

/// Sekantenverfahren: wie Newton, aber mit dem Differenzenquotient aus den letzten beiden
/// Näherungen statt der Ableitung.
///
/// Braucht zwei verschiedene Startwerte `x0` und `x1` (kein Vorzeichenwechsel nötig). Die
/// Konvergenzordnung ist etwa 1,6. Wie bei Newton gibt es keine Konvergenzgarantie.
///
/// Fehler: [`RootError::ZeroDerivative`] (`f(x0) = f(x1)`), [`RootError::NotFinite`],
/// [`RootError::NoConvergence`].
pub fn secant<T: Real>(
    mut f: impl FnMut(T) -> T,
    x0: T,
    x1: T,
    opts: RootOptions<T>,
) -> Result<RootResult<T>, RootError> {
    let (mut xa, mut xb) = (x0, x1);
    let mut fa = f(xa);
    let mut fb = f(xb);
    if !finite(xa) || !finite(xb) || !finite(fa) || !finite(fb) {
        return Err(RootError::NotFinite);
    }
    if fa == T::ZERO {
        return Ok(RootResult {
            root: xa,
            f_value: fa,
            iterations: 0,
        });
    }
    for k in 1..=opts.max_iter {
        if fb == T::ZERO {
            return Ok(RootResult {
                root: xb,
                f_value: fb,
                iterations: k - 1,
            });
        }
        if fb == fa {
            return Err(RootError::ZeroDerivative);
        }
        let x_new = xb - fb * (xb - xa) / (fb - fa);
        if !finite(x_new) {
            return Err(RootError::NotFinite);
        }
        let f_new = f(x_new);
        if !finite(f_new) {
            return Err(RootError::NotFinite);
        }
        let step = (x_new - xb).abs();
        xa = xb;
        fa = fb;
        xb = x_new;
        fb = f_new;
        if step <= opts.tol * scale(xb) || fb == T::ZERO {
            return Ok(RootResult {
                root: xb,
                f_value: fb,
                iterations: k,
            });
        }
    }
    Err(RootError::NoConvergence)
}
