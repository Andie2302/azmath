use super::{RootError, RootOptions, RootResult, close_points, finite, scale, validate_options};
use crate::Real;

/// Newton-Verfahren: `x ← x − f(x)/f'(x)`.
///
/// Quadratisch konvergent nahe einer einfachen Nullstelle, aber ohne Garantie: ein schlechter
/// Startwert oder eine flache Stelle kann zur Divergenz führen. Ist ein Intervall mit
/// Vorzeichenwechsel bekannt, ist [`brent`](super::brent) die sicherere Wahl.
///
/// - `f`: die Funktion, `df`: ihre Ableitung, `x0`: Startwert.
///
/// Fehler: [`RootError::ZeroDerivative`] (`f'(x) = 0`), [`RootError::NotFinite`],
/// [`RootError::NoConvergence`]. Negative oder nichtendliche Toleranzen ergeben
/// [`RootError::InvalidInput`].
pub fn newton<T: Real>(
    mut f: impl FnMut(T) -> T,
    mut df: impl FnMut(T) -> T,
    x0: T,
    opts: RootOptions<T>,
) -> Result<RootResult<T>, RootError> {
    validate_options(opts)?;
    if !finite(x0) {
        return Err(RootError::NotFinite);
    }
    let mut x = x0;
    let mut fx = f(x);
    if !finite(fx) {
        return Err(RootError::NotFinite);
    }
    if fx == T::ZERO {
        return Ok(RootResult {
            root: x,
            f_value: fx,
            iterations: 0,
        });
    }
    for k in 1..=opts.max_iter {
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
        let f_new = f(x_new);
        if !finite(f_new) {
            return Err(RootError::NotFinite);
        }
        if f_new == T::ZERO {
            return Ok(RootResult {
                root: x_new,
                f_value: f_new,
                iterations: k,
            });
        }
        if x_new == x {
            // Bei positiver Lagentoleranz kann der berechnete Newton-Schritt bereits
            // klein genug sein, obwohl er beim Addieren weggerundet wird.
            if opts.tol > T::ZERO && dx != T::ZERO && dx.abs() / scale(x_new) <= opts.tol {
                return Ok(RootResult {
                    root: x_new,
                    f_value: f_new,
                    iterations: k,
                });
            }
            return Err(RootError::NoConvergence);
        }
        if close_points(x_new, x, x_new, opts.tol) {
            return Ok(RootResult {
                root: x_new,
                f_value: f_new,
                iterations: k,
            });
        }
        x = x_new;
        fx = f_new;
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
/// [`RootError::NoConvergence`]. Negative oder nichtendliche Toleranzen ergeben
/// [`RootError::InvalidInput`].
pub fn secant<T: Real>(
    mut f: impl FnMut(T) -> T,
    x0: T,
    x1: T,
    opts: RootOptions<T>,
) -> Result<RootResult<T>, RootError> {
    validate_options(opts)?;
    if !finite(x0) || !finite(x1) {
        return Err(RootError::NotFinite);
    }
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
    if fb == T::ZERO {
        return Ok(RootResult {
            root: xb,
            f_value: fb,
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
        // Funktionswerte zuerst normieren: weder Produkt noch Differenz müssen
        // im ursprünglichen Wertebereich darstellbar sein.
        let fscale = fa.abs().max(fb.abs());
        let (u, v) = (fa / fscale, fb / fscale);
        let denominator = v - u;
        if denominator == T::ZERO {
            return Err(RootError::ZeroDerivative);
        }
        let weight = v / denominator;
        let x_new = if weight >= T::ZERO && weight <= T::ONE {
            if (xa < T::ZERO) == (xb < T::ZERO) {
                xb + weight * (xa - xb)
            } else {
                (T::ONE - weight) * xb + weight * xa
            }
        } else {
            let proposed = xb + weight * (xa - xb);
            if finite(proposed) {
                proposed
            } else {
                let xscale = xa.abs().max(xb.abs());
                (xb / xscale + weight * (xa / xscale - xb / xscale)) * xscale
            }
        };
        if !finite(x_new) {
            return Err(RootError::NotFinite);
        }
        let f_new = f(x_new);
        if !finite(f_new) {
            return Err(RootError::NotFinite);
        }
        let predicted_step = (weight * (xa - xb)).abs();
        if f_new != T::ZERO && x_new == xb {
            if opts.tol > T::ZERO
                && predicted_step > T::ZERO
                && finite(predicted_step)
                && predicted_step / scale(x_new) <= opts.tol
            {
                return Ok(RootResult {
                    root: x_new,
                    f_value: f_new,
                    iterations: k,
                });
            }
            return Err(RootError::NoConvergence);
        }
        let small_step = close_points(x_new, xb, x_new, opts.tol);
        xa = xb;
        fa = fb;
        xb = x_new;
        fb = f_new;
        if small_step || fb == T::ZERO {
            return Ok(RootResult {
                root: xb,
                f_value: fb,
                iterations: k,
            });
        }
    }
    Err(RootError::NoConvergence)
}
