use super::{
    InterpError, blend, check_nodes, finite, fraction, line_value, locate, slope, weighted,
    width_product,
};
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
        if (0..n - 1).any(|i| !finite(slope(x[i], x[i + 1], y[i], y[i + 1]))) {
            return Err(InterpError::NotFinite);
        }
        second_derivatives(x, y, boundary, m, work)?;
        if m.iter().any(|v| !finite(*v)) {
            return Err(InterpError::NotFinite);
        }
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

    fn piece_value(&self, i: usize, xq: T) -> T {
        let (x0, x1) = (self.x[i], self.x[i + 1]);
        let (y0, y1) = (self.y[i], self.y[i + 1]);
        if xq == x0 {
            return y0;
        }
        if xq == x1 {
            return y1;
        }
        let (m0, m1) = (self.m[i], self.m[i + 1]);
        if m0 == T::ZERO && m1 == T::ZERO {
            return line_value(x0, x1, y0, y1, xq);
        }
        let t = fraction(x0, x1, xq);
        let u = T::ONE - t;
        let six = T::from_f64(6.0);
        let c0 = width_product(
            x0,
            x1,
            width_product(x0, x1, m0, (u * u * u - u) / six),
            T::ONE,
        );
        let c1 = width_product(
            x0,
            x1,
            width_product(x0, x1, m1, (t * t * t - t) / six),
            T::ONE,
        );
        weighted(&[blend(y0, y1, t), c0, c1], &[T::ONE; 3])
    }

    /// Wert `S(xq)`; außerhalb wird das Randpolynom fortgesetzt. Nichtendliche Abfragen ergeben NaN.
    pub fn value(&self, xq: T) -> T {
        if !finite(xq) {
            return T::NAN;
        }
        self.piece_value(locate(self.x, xq), xq)
    }

    /// Erste Ableitung `S'(xq)`. Nichtendliche Abfragen ergeben NaN.
    pub fn derivative(&self, xq: T) -> T {
        if !finite(xq) {
            return T::NAN;
        }
        let i = locate(self.x, xq);
        let (x0, x1) = (self.x[i], self.x[i + 1]);
        let delta = slope(x0, x1, self.y[i], self.y[i + 1]);
        if self.m[i] == T::ZERO && self.m[i + 1] == T::ZERO {
            return delta;
        }
        let t = fraction(x0, x1, xq);
        let u = T::ONE - t;
        let (three, six) = (T::from_f64(3.0), T::from_f64(6.0));
        weighted(
            &[
                delta,
                width_product(x0, x1, self.m[i], (T::ONE - three * u * u) / six),
                width_product(x0, x1, self.m[i + 1], (three * t * t - T::ONE) / six),
            ],
            &[T::ONE; 3],
        )
    }

    /// Zweite Ableitung `S''(xq)` (stückweise linear und stetig). Nichtendliche Abfragen ergeben NaN.
    pub fn second_derivative(&self, xq: T) -> T {
        if !finite(xq) {
            return T::NAN;
        }
        let i = locate(self.x, xq);
        blend(
            self.m[i],
            self.m[i + 1],
            fraction(self.x[i], self.x[i + 1], xq),
        )
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

    /// Integral des Teilstücks `i` von `s` bis `e` (absolute Positionen).
    fn piece_integral(&self, i: usize, a: T, b: T) -> T {
        super::piece_integral(a, b, |q| self.piece_value(i, q))
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
    let halve = (0..n - 1).any(|i| (x[i + 1] - x[i]).is_infinite());
    let h = |i: usize| {
        if halve {
            crate::integrate::half_width(x[i], x[i + 1])
        } else {
            x[i + 1] - x[i]
        }
    };
    let delta = |i: usize| slope(x[i], x[i + 1], y[i], y[i + 1]);
    let rhs_factor = if halve {
        T::from_f64(3.0)
    } else {
        T::from_f64(6.0)
    };
    let weights = |left: T, right: T| {
        let scale = left.max(right);
        let (a, b) = (left / scale, right / scale);
        (a / (a + b), b / (a + b))
    };
    let rhs =
        |before: T, after: T, left: T, right: T| rhs_factor * slope(-left, right, before, after);
    if matches!(boundary, BoundaryCondition::Periodic) {
        if n < 3 {
            return Err(InterpError::InvalidInput);
        }
        if y[0] != y[n - 1] {
            return Err(InterpError::NotPeriodic);
        }
    }
    let d0 = delta(0);
    let is_line = (1..n - 1).all(|i| delta(i) == d0);
    let line_boundary = match boundary {
        BoundaryCondition::Natural | BoundaryCondition::NotAKnot => true,
        BoundaryCondition::Clamped { left, right } => left == d0 && right == d0,
        BoundaryCondition::Periodic => d0 == T::ZERO,
    };
    if is_line && line_boundary {
        m.fill(T::ZERO);
        return Ok(());
    }
    let two = T::from_f64(2.0);

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
            let (left, right) = weights(h(prev), h(i));
            a[i] = left;
            b[i] = two;
            c[i] = right;
            d[i] = rhs(delta(prev), delta(i), h(prev), h(i));
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
        let dd = slope(x[0], x[2], delta(0), delta(1));
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
        let (left, right) = weights(h(i - 1), h(i));
        a[i] = left;
        b[i] = two;
        c[i] = right;
        d[i] = rhs(delta(i - 1), delta(i), h(i - 1), h(i));
    }

    match boundary {
        BoundaryCondition::Natural => {
            b[0] = T::ONE;
            b[n - 1] = T::ONE;
        }
        BoundaryCondition::Clamped { left, right } => {
            b[0] = two;
            c[0] = T::ONE;
            d[0] = rhs_factor * slope(T::ZERO, h(0), left, delta(0));
            a[n - 1] = T::ONE;
            b[n - 1] = two;
            d[n - 1] = rhs_factor * slope(T::ZERO, h(n - 2), delta(n - 2), right);
        }
        BoundaryCondition::NotAKnot => {
            // m₀ und m_{n−1} über die Stetigkeit der dritten Ableitung eliminieren; übrig bleibt ein
            // tridiagonales System für m₁ … m_{n−2}.
            let (h0, h1) = (h(0), h(1));
            let ratio = h0 / h1;
            b[1] += a[1] * (T::ONE + ratio);
            c[1] -= a[1] * ratio;
            let (hp, hq) = (h(n - 3), h(n - 2));
            let ratio = hq / hp;
            b[n - 2] += c[n - 2] * (T::ONE + ratio);
            a[n - 2] -= c[n - 2] * ratio;
            solve_tridiagonal(
                &a[1..n - 1],
                &b[1..n - 1],
                &c[1..n - 1],
                &d[1..n - 1],
                &mut m[1..n - 1],
                scratch,
            )?;
            m[0] = weighted(&[m[1], m[2]], &[T::ONE + h0 / h1, -h0 / h1]);
            m[n - 1] = weighted(&[m[n - 2], m[n - 3]], &[T::ONE + hq / hp, -hq / hp]);
            return Ok(());
        }
        BoundaryCondition::Periodic => unreachable!("oben behandelt"),
    }
    solve_tridiagonal(a, b, c, d, m, scratch)?;
    Ok(())
}
