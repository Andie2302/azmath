use super::{InterpError, check_nodes, locate};
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
    let s = ((T::from_f64(2.0) * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
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
        let h = |i: usize| x[i + 1] - x[i];
        let delta = |i: usize| (y[i + 1] - y[i]) / h(i);
        if n == 2 {
            d[0] = delta(0);
            d[1] = delta(0);
        } else {
            for (k, dk) in d.iter_mut().enumerate().take(n - 1).skip(1) {
                let (d0, d1) = (delta(k - 1), delta(k));
                *dk = if sign(d0) * sign(d1) <= 0 {
                    T::ZERO
                } else {
                    let w1 = T::from_f64(2.0) * h(k) + h(k - 1);
                    let w2 = h(k) + T::from_f64(2.0) * h(k - 1);
                    (w1 + w2) / (w1 / d0 + w2 / d1)
                };
            }
            d[0] = end_slope(h(0), h(1), delta(0), delta(1));
            d[n - 1] = end_slope(h(n - 2), h(n - 3), delta(n - 2), delta(n - 3));
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

    /// Koeffizienten `(c1, c2, c3)` des Teilstücks `i`: `P(xᵢ + s) = yᵢ + c1·s + c2·s² + c3·s³`.
    fn piece(&self, i: usize) -> (T, T, T) {
        let h = self.x[i + 1] - self.x[i];
        let delta = (self.y[i + 1] - self.y[i]) / h;
        let (d0, d1) = (self.d[i], self.d[i + 1]);
        let c2 = (T::from_f64(3.0) * delta - T::from_f64(2.0) * d0 - d1) / h;
        let c3 = (d0 + d1 - T::from_f64(2.0) * delta) / (h * h);
        (d0, c2, c3)
    }

    /// Wert `P(xq)`. Außerhalb des Bereichs wird das erste bzw. letzte Stück fortgesetzt.
    pub fn value(&self, xq: T) -> T {
        let i = locate(self.x, xq);
        let (c1, c2, c3) = self.piece(i);
        let s = xq - self.x[i];
        self.y[i] + s * (c1 + s * (c2 + s * c3))
    }

    /// Erste Ableitung `P'(xq)` (stetig).
    pub fn derivative(&self, xq: T) -> T {
        let i = locate(self.x, xq);
        let (c1, c2, c3) = self.piece(i);
        let s = xq - self.x[i];
        c1 + s * (T::from_f64(2.0) * c2 + T::from_f64(3.0) * c3 * s)
    }

    /// Integral von `P` über `[a, b]` (exakt). `a > b` ergibt das negative Integral.
    pub fn integral(&self, a: T, b: T) -> T {
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

    fn piece_integral(&self, i: usize, s: T, e: T) -> T {
        let (c1, c2, c3) = self.piece(i);
        let yi = self.y[i];
        let g = |t: T| {
            t * (yi
                + t * (c1 / T::from_f64(2.0)
                    + t * (c2 / T::from_f64(3.0) + t * c3 / T::from_f64(4.0))))
        };
        g(e - self.x[i]) - g(s - self.x[i])
    }
}
