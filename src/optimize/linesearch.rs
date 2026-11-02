use super::{dot, finite};
use crate::Real;

/// Einstellungen der Wolfe-Schrittweitensuche.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WolfeOptions<T> {
    /// Konstante der Armijo-Bedingung (hinreichender Abstieg), `0 < c1 < c2 < 1`.
    pub c1: T,
    /// Konstante der Krümmungsbedingung; `0,9` für Quasi-Newton, `0,1` für CG-Verfahren.
    pub c2: T,
    /// Größte zulässige Schrittweite.
    pub alpha_max: T,
    /// Höchstzahl der Funktionsauswertungen.
    pub max_evals: usize,
}

impl<T: Real> Default for WolfeOptions<T> {
    /// `c1 = 1e-4`, `c2 = 0,9`, `alpha_max = 1e6`, höchstens 40 Auswertungen.
    fn default() -> Self {
        Self {
            c1: T::from_f64(1e-4),
            c2: T::from_f64(0.9),
            alpha_max: T::from_f64(1e6),
            max_evals: 40,
        }
    }
}

/// Ergebnis der Schrittweitensuche. Der Punkt `x + alpha·d` und sein Gradient stehen in `x_new` und `g_new`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineSearchResult<T> {
    /// Gefundene Schrittweite.
    pub alpha: T,
    /// Funktionswert an `x + alpha·d`.
    pub f_value: T,
    /// Zahl der Auswertungen von `f` (und `grad`).
    pub evaluations: usize,
}

/// Fehler der Schrittweitensuche.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineSearchError {
    /// `d` ist keine Abstiegsrichtung (`∇f·d ≥ 0`) oder die Parameter sind ungültig.
    NotDescent,
    /// NaN oder Unendlich in `f` oder im Gradienten.
    NotFinite,
    /// Keine Schrittweite mit starken Wolfe-Bedingungen gefunden (Auswertungsgrenze, Intervall zu klein).
    Failed,
}

impl core::fmt::Display for LineSearchError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            LineSearchError::NotDescent => "keine Abstiegsrichtung",
            LineSearchError::NotFinite => "NaN oder unendlicher Wert aufgetreten",
            LineSearchError::Failed => "keine passende Schrittweite gefunden",
        })
    }
}

impl core::error::Error for LineSearchError {}

/// Auswertung entlang der Richtung: `φ(α) = f(x + α·d)` und `φ'(α) = ∇f(x + α·d)·d`.
struct Probe<'a, T, F, G> {
    f: &'a mut F,
    grad: &'a mut G,
    x: &'a [T],
    d: &'a [T],
    x_new: &'a mut [T],
    g_new: &'a mut [T],
    evals: usize,
}

impl<T: Real, F: FnMut(&[T]) -> T, G: FnMut(&[T], &mut [T])> Probe<'_, T, F, G> {
    fn eval(&mut self, alpha: T) -> Result<(T, T), LineSearchError> {
        for i in 0..self.x.len() {
            self.x_new[i] = self.x[i] + alpha * self.d[i];
        }
        let fa = (self.f)(self.x_new);
        (self.grad)(self.x_new, self.g_new);
        self.evals += 1;
        let da = dot(self.g_new, self.d);
        if !finite(fa) || !finite(da) {
            return Err(LineSearchError::NotFinite);
        }
        Ok((fa, da))
    }
}

/// Schrittweitensuche mit den **starken Wolfe-Bedingungen** (Nocedal & Wright, Alg. 3.5/3.6):
///
/// * hinreichender Abstieg: `f(x + αd) ≤ f(x) + c1·α·∇f·d`
/// * Krümmung: `|∇f(x + αd)·d| ≤ c2·|∇f·d|`
///
/// Sie sichern bei BFGS `sᵀy > 0` und damit eine positiv definite Matrix.
///
/// - `f0`, `g0`: Funktionswert und Gradient an `x`; `d`: Abstiegsrichtung (`g0·d < 0`);
///   `alpha0`: erste Probeschrittweite (bei Quasi-Newton meist 1).
/// - `x_new`, `g_new` (Länge `n`) sind Arbeitsspeicher und enthalten bei Erfolg `x + α·d` und `∇f` dort.
///
/// Fehler: [`LineSearchError`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::optimize::{line_search_wolfe, WolfeOptions};
///
/// // f(x) = x², Start bei x = 3, Richtung −f' = −6
/// let (mut xn, mut gn) = ([0.0], [0.0]);
/// let r = line_search_wolfe(
///     |x: &[f64]| x[0] * x[0],
///     |x: &[f64], g: &mut [f64]| g[0] = 2.0 * x[0],
///     &[3.0], 9.0, &[6.0], &[-6.0], 1.0, &WolfeOptions::default(), &mut xn, &mut gn,
/// ).unwrap();
/// assert!(r.f_value < 9.0);
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn line_search_wolfe<T: Real>(
    mut f: impl FnMut(&[T]) -> T,
    mut grad: impl FnMut(&[T], &mut [T]),
    x: &[T],
    f0: T,
    g0: &[T],
    d: &[T],
    alpha0: T,
    opts: &WolfeOptions<T>,
    x_new: &mut [T],
    g_new: &mut [T],
) -> Result<LineSearchResult<T>, LineSearchError> {
    let n = x.len();
    if g0.len() != n || d.len() != n || x_new.len() != n || g_new.len() != n {
        return Err(LineSearchError::NotDescent);
    }
    let dphi0 = dot(g0, d);
    if !finite(f0) || !finite(dphi0) {
        return Err(LineSearchError::NotFinite);
    }
    let params_ok = opts.c1 > T::ZERO && opts.c1 < opts.c2 && opts.c2 < T::ONE && alpha0 > T::ZERO;
    if dphi0 >= T::ZERO || !params_ok {
        return Err(LineSearchError::NotDescent);
    }
    let mut p = Probe {
        f: &mut f,
        grad: &mut grad,
        x,
        d,
        x_new,
        g_new,
        evals: 0,
    };
    let armijo = |a: T, fa: T| fa <= f0 + opts.c1 * a * dphi0;
    let curvature = |da: T| da.abs() <= -opts.c2 * dphi0;
    let two = T::from_f64(2.0);

    let (mut a_prev, mut f_prev, mut d_prev) = (T::ZERO, f0, dphi0);
    let mut a = alpha0.min(opts.alpha_max);
    // Einschluss suchen
    let (mut lo, mut flo, mut dlo, mut hi, mut fhi);
    loop {
        if p.evals >= opts.max_evals {
            return Err(LineSearchError::Failed);
        }
        let (fa, da) = p.eval(a)?;
        if !armijo(a, fa) || (p.evals > 1 && fa >= f_prev) {
            (lo, flo, dlo, hi, fhi) = (a_prev, f_prev, d_prev, a, fa);
            break;
        }
        if curvature(da) {
            return Ok(LineSearchResult {
                alpha: a,
                f_value: fa,
                evaluations: p.evals,
            });
        }
        if da >= T::ZERO {
            (lo, flo, dlo, hi, fhi) = (a, fa, da, a_prev, f_prev);
            break;
        }
        if a >= opts.alpha_max {
            return Err(LineSearchError::Failed);
        }
        (a_prev, f_prev, d_prev) = (a, fa, da);
        a = (a * two).min(opts.alpha_max);
    }
    // Zoom: Intervall zwischen lo und hi (lo hat den kleineren Wert und erfüllt Armijo)
    loop {
        if p.evals >= opts.max_evals {
            return Err(LineSearchError::Failed);
        }
        let width = hi - lo;
        if width.abs() <= T::EPSILON * lo.abs().max(hi.abs()) {
            return Err(LineSearchError::Failed);
        }
        // quadratische Interpolation mit Absicherung gegen Randnähe, sonst Mitte
        let denom = two * (fhi - flo - dlo * width);
        let mut a = if denom != T::ZERO {
            lo - dlo * width * width / denom
        } else {
            lo + width / two
        };
        let (mn, mx) = if lo < hi { (lo, hi) } else { (hi, lo) };
        let margin = (mx - mn) * T::from_f64(0.1);
        if !finite(a) || a < mn + margin || a > mx - margin {
            a = lo + width / two;
        }
        let (fa, da) = p.eval(a)?;
        if !armijo(a, fa) || fa >= flo {
            (hi, fhi) = (a, fa);
        } else {
            if curvature(da) {
                return Ok(LineSearchResult {
                    alpha: a,
                    f_value: fa,
                    evaluations: p.evals,
                });
            }
            if da * (hi - lo) >= T::ZERO {
                (hi, fhi) = (lo, flo);
            }
            (lo, flo, dlo) = (a, fa, da);
        }
    }
}
