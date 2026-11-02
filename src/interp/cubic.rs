use super::{InterpError, check_nodes, locate};
use crate::Real;
use crate::linalg::{solve_cyclic_tridiagonal, solve_tridiagonal};

/// Benötigter Arbeitsspeicher für [`CubicSpline::new`] bei `n` Stützstellen.
pub const fn cubic_spline_work_len(n: usize) -> usize {
    8 * n
}

/// Randbedingung eines kubischen Splines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition<T> {
    /// Natürlicher Spline: zweite Ableitung 0 an beiden Enden. Die einfachste Wahl, aber nur
    /// `O(h²)` genau am Rand, wenn die Funktion dort nicht verschwindende Krümmung hat.
    Natural,
    /// Eingespannter Spline: vorgegebene erste Ableitungen `left` bei `x₀` und `right` bei `xₙ₋₁`.
    /// Am genauesten, wenn die Steigungen bekannt sind.
    Clamped {
        /// Erste Ableitung am linken Rand.
        left: T,
        /// Erste Ableitung am rechten Rand.
        right: T,
    },
    /// Not-a-knot: dritte Ableitung stetig an `x₁` und `xₙ₋₂`. Gute Standardwahl ohne Wissen über den
    /// Rand; reproduziert Polynome bis Grad 3 exakt. Braucht mindestens 3 Punkte (bei 2 wird die
    /// Gerade verwendet).
    NotAKnot,
    /// Periodisch: Wert, erste und zweite Ableitung stimmen an beiden Enden überein. Es muss
    /// `y[0] == y[n−1]` gelten; mindestens 3 Punkte.
    Periodic,
}

/// Kubischer Spline durch die Punkte `(x[i], y[i])`: zweimal stetig differenzierbar, stückweise ein
/// Polynom dritten Grades.
///
/// Die Stützwerte werden **geliehen**, die berechneten zweiten Ableitungen (die den Spline
/// festlegen) liegen in einem vom Aufrufer übergebenen Slice; die Konstruktion löst ein
/// tridiagonales System in `O(n)`. Siehe auch das [Modul](super).
#[derive(Debug, Clone, Copy)]
pub struct CubicSpline<'a, T> {
    x: &'a [T],
    y: &'a [T],
    /// zweite Ableitungen an den Stützstellen
    m: &'a [T],
}

impl<'a, T: Real> CubicSpline<'a, T> {
    /// Berechnet den Spline.
    ///
    /// - `x`: streng aufsteigende Stützstellen, `y`: Werte (gleiche Länge `n ≥ 2`).
    /// - `m`: Ausgabespeicher für die zweiten Ableitungen (genau `n` Elemente).
    /// - `work`: mindestens [`cubic_spline_work_len`]`(n)` Elemente (nur während der Konstruktion).
    ///
    /// Fehler: [`InterpError::InvalidInput`], [`InterpError::NotIncreasing`],
    /// [`InterpError::NotFinite`], [`InterpError::NotPeriodic`], [`InterpError::Singular`].
    pub fn new(
        x: &'a [T],
        y: &'a [T],
        boundary: BoundaryCondition<T>,
        m: &'a mut [T],
        work: &mut [T],
    ) -> Result<Self, InterpError> {
        check_nodes(x, y)?;
        let n = x.len();
        if m.len() != n || work.len() < cubic_spline_work_len(n) {
            return Err(InterpError::InvalidInput);
        }
        if let BoundaryCondition::Clamped { left, right } = boundary {
            if !super::finite(left) || !super::finite(right) {
                return Err(InterpError::NotFinite);
            }
        }
        second_derivatives(x, y, boundary, m, work)?;
        let m: &'a [T] = m;
        Ok(Self { x, y, m })
    }

    /// Ohne Prüfung aus bereits berechneten Teilen zusammensetzen.
    #[cfg(feature = "alloc")]
    pub(crate) fn from_parts(x: &'a [T], y: &'a [T], m: &'a [T]) -> Self {
        Self { x, y, m }
    }

    /// Die Stützstellen.
    pub fn nodes(&self) -> &'a [T] {
        self.x
    }

    /// Die Stützwerte.
    pub fn values(&self) -> &'a [T] {
        self.y
    }

    /// Die zweiten Ableitungen an den Stützstellen.
    pub fn second_derivatives(&self) -> &'a [T] {
        self.m
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

    /// Koeffizienten `(h, b, c, d)` des Teilstücks `i`: `S(xᵢ + t) = yᵢ + b·t + c·t² + d·t³`.
    fn piece(&self, i: usize) -> (T, T, T, T) {
        let h = self.x[i + 1] - self.x[i];
        let (m0, m1) = (self.m[i], self.m[i + 1]);
        let six = T::from_f64(6.0);
        let b = (self.y[i + 1] - self.y[i]) / h - h * (T::from_f64(2.0) * m0 + m1) / six;
        let c = T::from_f64(0.5) * m0;
        let d = (m1 - m0) / (six * h);
        (h, b, c, d)
    }

    /// Wert `S(xq)`. Außerhalb des Bereichs wird das erste bzw. letzte Polynom fortgesetzt.
    pub fn value(&self, xq: T) -> T {
        let i = locate(self.x, xq);
        let (_, b, c, d) = self.piece(i);
        let t = xq - self.x[i];
        self.y[i] + t * (b + t * (c + t * d))
    }

    /// Erste Ableitung `S'(xq)`.
    pub fn derivative(&self, xq: T) -> T {
        let i = locate(self.x, xq);
        let (_, b, c, d) = self.piece(i);
        let t = xq - self.x[i];
        b + t * (T::from_f64(2.0) * c + T::from_f64(3.0) * d * t)
    }

    /// Zweite Ableitung `S''(xq)` (stückweise linear und stetig).
    pub fn second_derivative(&self, xq: T) -> T {
        let i = locate(self.x, xq);
        let h = self.x[i + 1] - self.x[i];
        let t = xq - self.x[i];
        self.m[i] + (self.m[i + 1] - self.m[i]) * t / h
    }

    /// Wertet den Spline an allen Stellen `xs` aus und schreibt die Ergebnisse nach `out`
    /// (gleiche Länge).
    ///
    /// # Panics
    ///
    /// Wenn `xs` und `out` verschieden lang sind.
    pub fn values_at(&self, xs: &[T], out: &mut [T]) {
        assert_eq!(xs.len(), out.len(), "values_at: Längen verschieden");
        for (o, xq) in out.iter_mut().zip(xs) {
            *o = self.value(*xq);
        }
    }

    /// Integral von `S` über `[a, b]` (exakt, Teilstück für Teilstück). `a > b` ergibt das negative
    /// Integral.
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

    /// Integral des Teilstücks `i` von `s` bis `e` (absolute Positionen).
    fn piece_integral(&self, i: usize, s: T, e: T) -> T {
        let (_, b, c, d) = self.piece(i);
        let yi = self.y[i];
        let g = |t: T| {
            t * (yi
                + t * (b / T::from_f64(2.0)
                    + t * (c / T::from_f64(3.0) + t * d / T::from_f64(4.0))))
        };
        g(e - self.x[i]) - g(s - self.x[i])
    }
}

/// Berechnet die zweiten Ableitungen `m` des Splines.
fn second_derivatives<T: Real>(
    x: &[T],
    y: &[T],
    boundary: BoundaryCondition<T>,
    m: &mut [T],
    work: &mut [T],
) -> Result<(), InterpError> {
    let n = x.len();
    let h = |i: usize| x[i + 1] - x[i];
    let delta = |i: usize| (y[i + 1] - y[i]) / h(i);
    let two = T::from_f64(2.0);
    let six = T::from_f64(6.0);

    if let BoundaryCondition::Periodic = boundary {
        if n < 3 {
            return Err(InterpError::InvalidInput);
        }
        if y[0] != y[n - 1] {
            return Err(InterpError::NotPeriodic);
        }
        let nn = n - 1; // Unbekannte m₀ … m_{n−2}, m_{n−1} = m₀
        let (a, rest) = work.split_at_mut(nn);
        let (b, rest) = rest.split_at_mut(nn);
        let (c, rest) = rest.split_at_mut(nn);
        let (d, cyc) = rest.split_at_mut(nn);
        for i in 0..nn {
            let prev = (i + nn - 1) % nn;
            a[i] = h(prev);
            b[i] = two * (h(prev) + h(i));
            c[i] = h(i);
            d[i] = six * (delta(i) - delta(prev));
        }
        solve_cyclic_tridiagonal(a, b, c, d, &mut m[..nn], cyc)?;
        m[n - 1] = m[0];
        return Ok(());
    }

    // Sonderfälle ohne Gleichungssystem
    if n == 2
        && matches!(
            boundary,
            BoundaryCondition::Natural | BoundaryCondition::NotAKnot
        )
    {
        m.fill(T::ZERO);
        return Ok(());
    }
    if n == 3 && matches!(boundary, BoundaryCondition::NotAKnot) {
        // dasselbe Polynom auf beiden Stücken: die Parabel durch die drei Punkte
        let dd = (delta(1) - delta(0)) / (x[2] - x[0]);
        m.fill(two * dd);
        return Ok(());
    }

    let (a, rest) = work.split_at_mut(n);
    let (b, rest) = rest.split_at_mut(n);
    let (c, rest) = rest.split_at_mut(n);
    let (d, scratch) = rest.split_at_mut(n);
    a.fill(T::ZERO);
    b.fill(T::ZERO);
    c.fill(T::ZERO);
    d.fill(T::ZERO);
    for i in 1..n - 1 {
        a[i] = h(i - 1);
        b[i] = two * (h(i - 1) + h(i));
        c[i] = h(i);
        d[i] = six * (delta(i) - delta(i - 1));
    }

    match boundary {
        BoundaryCondition::Natural => {
            b[0] = T::ONE;
            b[n - 1] = T::ONE;
        }
        BoundaryCondition::Clamped { left, right } => {
            b[0] = two * h(0);
            c[0] = h(0);
            d[0] = six * (delta(0) - left);
            a[n - 1] = h(n - 2);
            b[n - 1] = two * h(n - 2);
            d[n - 1] = six * (right - delta(n - 2));
        }
        BoundaryCondition::NotAKnot => {
            // m₀ und m_{n−1} über die Stetigkeit der dritten Ableitung eliminieren; übrig bleibt ein
            // tridiagonales System für m₁ … m_{n−2}.
            let (h0, h1) = (h(0), h(1));
            b[1] += h0 * (h0 + h1) / h1;
            c[1] -= h0 * h0 / h1;
            let (hp, hq) = (h(n - 3), h(n - 2));
            b[n - 2] += hq * (hp + hq) / hp;
            a[n - 2] -= hq * hq / hp;
            solve_tridiagonal(
                &a[1..n - 1],
                &b[1..n - 1],
                &c[1..n - 1],
                &d[1..n - 1],
                &mut m[1..n - 1],
                scratch,
            )?;
            m[0] = ((h0 + h1) * m[1] - h0 * m[2]) / h1;
            m[n - 1] = ((hp + hq) * m[n - 2] - hq * m[n - 3]) / hp;
            return Ok(());
        }
        BoundaryCondition::Periodic => unreachable!("oben behandelt"),
    }
    solve_tridiagonal(a, b, c, d, m, scratch)?;
    Ok(())
}
