use super::{
    InterpError, check_nodes, finite, fraction, line_value, locate, slope, weighted, width_product,
};
use crate::Real;

/// Stückweise kubische Hermite-Interpolation mit **Formerhaltung** (PCHIP, nach Fritsch und
/// Carlson): Auf Intervallen, in denen die Daten monoton sind, ist es auch die Interpolierende, und
/// lokale Extrema der Daten bleiben die einzigen der Kurve, ein Überschwingen tritt nicht auf.
///
/// Dafür ist sie nur einmal stetig differenzierbar (die zweite Ableitung springt). Die Steigungen
/// werden lokal aus den Nachbarpunkten bestimmt, es ist also kein Gleichungssystem zu lösen. Die Daten
/// werden **geliehen**, die Steigungen liegen in einem vom Aufrufer übergebenen Slice.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::interp::Pchip;
///
/// // monotone Daten mit Sprung: ein kubischer Spline würde überschwingen
/// let x = [0.0, 1.0, 2.0, 3.0, 4.0];
/// let y = [0.0, 0.0, 1.0, 1.0, 1.0];
/// let mut d = [0.0; 5];
/// let p = Pchip::new(&x, &y, &mut d).unwrap();
/// for k in 0..=40 {
///     let v = p.value(k as f64 * 0.1);
///     assert!((0.0..=1.0).contains(&v), "Überschwinger {v}");
/// }
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Pchip<'a, T> {
    x: &'a [T],
    y: &'a [T],
    /// Steigungen an den Stützstellen
    d: &'a [T],
}

fn sign<T: Real>(v: T) -> i8 {
    if v > T::ZERO {
        1
    } else if v < T::ZERO {
        -1
    } else {
        0
    }
}

/// Dreipunkt-Randsteigung mit Formerhaltung. `h0, h1` Intervallbreiten am Rand, `d0, d1` die
/// zugehörigen Sekantensteigungen (von außen nach innen).
fn end_slope<T: Real>(h0: T, h1: T, d0: T, d1: T) -> T {
    let hscale = h0.max(h1);
    let a = (h0 / hscale) / ((h0 / hscale) + (h1 / hscale));
    let s = weighted(&[d0, d1], &[T::ONE + a, -a]);
    if s.is_nan() {
        return T::NAN;
    }
    if sign(s) != sign(d0) {
        T::ZERO
    } else if sign(d0) != sign(d1) && s.abs() > T::from_f64(3.0) * d0.abs() {
        T::from_f64(3.0) * d0
    } else {
        s
    }
}

impl<'a, T: Real> Pchip<'a, T> {
    /// Berechnet die Steigungen.
    ///
    /// - `x`: streng aufsteigende Stützstellen, `y`: Werte (gleiche Länge `n ≥ 2`).
    /// - `d`: Ausgabespeicher für die Steigungen (genau `n` Elemente).
    ///
    /// Fehler: [`InterpError::InvalidInput`], [`InterpError::NotIncreasing`],
    /// [`InterpError::NotFinite`].
    pub fn new(x: &'a [T], y: &'a [T], d: &'a mut [T]) -> Result<Self, InterpError> {
        check_nodes(x, y)?;
        let n = x.len();
        if d.len() != n {
            return Err(InterpError::InvalidInput);
        }
        let halve = (0..n - 1).any(|i| (x[i + 1] - x[i]).is_infinite());
        let h = |i: usize| {
            if halve {
                crate::integrate::half_width(x[i], x[i + 1])
            } else {
                x[i + 1] - x[i]
            }
        };
        let delta = |i: usize| slope(x[i], x[i + 1], y[i], y[i + 1]);
        if (0..n - 1).any(|i| !finite(delta(i))) {
            return Err(InterpError::NotFinite);
        }
        if n == 2 {
            d[0] = delta(0);
            d[1] = delta(0);
        } else {
            for (k, dk) in d.iter_mut().enumerate().take(n - 1).skip(1) {
                let (d0, d1) = (delta(k - 1), delta(k));
                *dk = if sign(d0) * sign(d1) <= 0 {
                    T::ZERO
                } else {
                    let scale = h(k).max(h(k - 1));
                    let (a, b) = (h(k - 1) / scale, h(k) / scale);
                    let w1 = (T::from_f64(2.0) * b + a) / (T::from_f64(3.0) * (a + b));
                    let w2 = T::ONE - w1;
                    let small = d0.abs().min(d1.abs());
                    let magnitude = small / (w1 * (small / d0.abs()) + w2 * (small / d1.abs()));
                    if d0 < T::ZERO { -magnitude } else { magnitude }
                };
            }
            d[0] = end_slope(h(0), h(1), delta(0), delta(1));
            d[n - 1] = end_slope(h(n - 2), h(n - 3), delta(n - 2), delta(n - 3));
        }
        if d.iter().any(|v| !finite(*v)) {
            return Err(InterpError::NotFinite);
        }
        let d: &'a [T] = d;
        Ok(Self { x, y, d })
    }

    /// Ohne Prüfung aus bereits berechneten Teilen zusammensetzen.
    #[cfg(feature = "alloc")]
    pub(crate) fn from_parts(x: &'a [T], y: &'a [T], d: &'a [T]) -> Self {
        Self { x, y, d }
    }

    /// Die Stützstellen.
    pub fn nodes(&self) -> &'a [T] {
        self.x
    }

    /// Die Stützwerte.
    pub fn values(&self) -> &'a [T] {
        self.y
    }

    /// Die berechneten Steigungen an den Stützstellen.
    pub fn slopes(&self) -> &'a [T] {
        self.d
    }

    /// Der Definitionsbereich `(x₀, xₙ₋₁)`.
    pub fn domain(&self) -> (T, T) {
        (self.x[0], self.x[self.x.len() - 1])
    }

    /// `true`, wenn `xq` im Definitionsbereich liegt (sonst wird extrapoliert).
    pub fn contains(&self, xq: T) -> bool {
        let (lo, hi) = self.domain();
        xq >= lo && xq <= hi
    }

    fn piece_value(&self, i: usize, xq: T) -> T {
        let (x0, x1) = (self.x[i], self.x[i + 1]);
        let (y0, y1) = (self.y[i], self.y[i + 1]);
        if xq == x0 {
            return y0;
        }
        if xq == x1 {
            return y1;
        }
        let delta = slope(x0, x1, y0, y1);
        let (d0, d1) = (self.d[i], self.d[i + 1]);
        if d0 == delta && d1 == delta {
            return line_value(x0, x1, y0, y1, xq);
        }
        let t = fraction(x0, x1, xq);
        let u = T::ONE - t;
        let two = T::from_f64(2.0);
        let value = weighted(
            &[
                y0,
                y1,
                width_product(x0, x1, d0, t * u * u),
                width_product(x0, x1, d1, -t * t * u),
            ],
            &[
                (T::ONE + two * t) * u * u,
                t * t * (T::from_f64(3.0) - two * t),
                T::ONE,
                T::ONE,
            ],
        );
        if !finite(value) {
            return T::NAN;
        }
        if t >= T::ZERO && t <= T::ONE {
            value.max(y0.min(y1)).min(y0.max(y1))
        } else {
            value
        }
    }

    /// Wert `P(xq)`. Außerhalb des Bereichs wird das erste bzw. letzte Stück fortgesetzt.
    /// Nichtendliche Abfragen ergeben NaN.
    pub fn value(&self, xq: T) -> T {
        if !finite(xq) {
            return T::NAN;
        }
        self.piece_value(locate(self.x, xq), xq)
    }

    /// Erste Ableitung `P'(xq)` (stetig). Nichtendliche Abfragen ergeben NaN.
    pub fn derivative(&self, xq: T) -> T {
        if !finite(xq) {
            return T::NAN;
        }
        let i = locate(self.x, xq);
        if xq == self.x[i] {
            return self.d[i];
        }
        if xq == self.x[i + 1] {
            return self.d[i + 1];
        }
        let delta = slope(self.x[i], self.x[i + 1], self.y[i], self.y[i + 1]);
        let (d0, d1) = (self.d[i], self.d[i + 1]);
        if d0 == delta && d1 == delta {
            return delta;
        }
        let t = fraction(self.x[i], self.x[i + 1], xq);
        weighted(
            &[delta, d0, d1],
            &[
                T::from_f64(6.0) * t * (T::ONE - t),
                T::ONE - T::from_f64(4.0) * t + T::from_f64(3.0) * t * t,
                T::from_f64(3.0) * t * t - T::from_f64(2.0) * t,
            ],
        )
    }

    /// Integral von `P` über `[a, b]` (exakt). `a > b` ergibt das negative Integral.
    pub fn integral(&self, a: T, b: T) -> T {
        if !finite(a) || !finite(b) {
            return T::NAN;
        }
        if a == b {
            return T::ZERO;
        }
        if a > b {
            return -self.integral(b, a);
        }
        let (ia, ib) = (locate(self.x, a), locate(self.x, b));
        if ia == ib {
            return self.piece_integral(ia, a, b);
        }
        let mut sum = self.piece_integral(ia, a, self.x[ia + 1]);
        for i in ia + 1..ib {
            sum += self.piece_integral(i, self.x[i], self.x[i + 1]);
        }
        sum + self.piece_integral(ib, self.x[ib], b)
    }

    fn piece_integral(&self, i: usize, a: T, b: T) -> T {
        super::piece_integral(a, b, |q| self.piece_value(i, q))
    }
}
