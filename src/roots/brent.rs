use super::{RootError, RootOptions, RootResult, finite, midpoint, scale, validate_options};
use crate::Real;

/// Verfahren von Brent: kombiniert Bisektion, Sekante und inverse quadratische Interpolation.
///
/// So sicher wie die Bisektion (der Vorzeichenwechsel bleibt immer eingeschlossen), meist aber
/// um ein Vielfaches schneller. Die erste Wahl, wenn ein Intervall mit Vorzeichenwechsel bekannt
/// ist. Die Reihenfolge `a < b` ist nicht nötig.
///
/// Fehler: [`RootError::NotBracketed`], [`RootError::NotFinite`], [`RootError::NoConvergence`],
/// [`RootError::InvalidInput`] bei negativer oder nichtendlicher Toleranz.
pub fn brent<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: RootOptions<T>,
) -> Result<RootResult<T>, RootError> {
    validate_options(opts)?;
    if !finite(a) || !finite(b) {
        return Err(RootError::NotFinite);
    }
    let (mut a, mut b) = (a, b);
    let mut fa = f(a);
    let mut fb = f(b);
    if !finite(a) || !finite(b) || !finite(fa) || !finite(fb) {
        return Err(RootError::NotFinite);
    }
    if fa == T::ZERO {
        return Ok(RootResult {
            root: a,
            f_value: fa,
            iterations: 0,
        });
    }
    if fb == T::ZERO {
        return Ok(RootResult {
            root: b,
            f_value: fb,
            iterations: 0,
        });
    }
    if (fa < T::ZERO) == (fb < T::ZERO) {
        return Err(RootError::NotBracketed);
    }

    let two = T::from_f64(2.0);
    let three = T::from_f64(3.0);
    let half = T::from_f64(0.5);

    let (mut c, mut fc) = (b, fb);
    let mut d = b - a;
    let mut e = d;

    for k in 1..=opts.max_iter {
        // c soll immer auf der anderen Seite der Nullstelle liegen als b.
        if (fb < T::ZERO) == (fc < T::ZERO) {
            c = a;
            fc = fa;
            d = b - a;
            e = d;
        }
        // b ist die beste Näherung (kleinster |f|).
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        let tol1 = two * T::EPSILON * b.abs() + half * opts.tol * scale(b);
        let xm = midpoint(c, b) - b;
        if (opts.tol > T::ZERO && xm.abs() <= tol1) || fb == T::ZERO {
            return Ok(RootResult {
                root: b,
                f_value: fb,
                iterations: k,
            });
        }

        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            // Interpolation versuchen
            let s = fb / fa;
            let (mut p, mut q);
            if a == c {
                // Sekante (lineare Interpolation)
                p = two * xm * s;
                q = T::ONE - s;
            } else {
                // inverse quadratische Interpolation
                let qq = fa / fc;
                let r = fb / fc;
                p = s * (two * xm * qq * (qq - r) - (b - a) * (r - T::ONE));
                q = (qq - T::ONE) * (r - T::ONE) * (s - T::ONE);
            }
            if p > T::ZERO {
                q = -q;
            }
            p = p.abs();
            let min1 = three * xm * q - (tol1 * q).abs();
            let min2 = (e * q).abs();
            if finite(p) && finite(q) && q != T::ZERO && finite(p / q) && two * p < min1.min(min2) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            // Bisektion
            d = xm;
            e = d;
        }

        a = b;
        fa = fb;
        let old_b = b;
        let proposed = if d.abs() > tol1 {
            b + d
        } else if xm >= T::ZERO {
            b + tol1
        } else {
            b - tol1
        };
        // Eine fehlgeschlagene Interpolation darf keinen nichtendlichen Callback auslösen.
        b = if finite(proposed)
            && proposed >= old_b.min(c)
            && proposed <= old_b.max(c)
            && proposed != old_b
        {
            proposed
        } else {
            midpoint(old_b, c)
        };
        if b == old_b {
            return Err(RootError::NoConvergence);
        }
        fb = f(b);
        if !finite(fb) {
            return Err(RootError::NotFinite);
        }
        if fb == T::ZERO {
            return Ok(RootResult {
                root: b,
                f_value: fb,
                iterations: k,
            });
        }
    }
    Err(RootError::NoConvergence)
}
