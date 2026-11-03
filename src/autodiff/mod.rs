//! Automatisches Differenzieren im Vorwärtsmodus, ohne Allokator.
//!
//! * [`Jet`]: Wert plus Gradient fester Größe `N` auf dem Stack; erfüllt [`Real`], läuft also durch jeden
//!   generischen Code.
//! * `Tape`, `Var`, `gradient_ad` (Feature `alloc`): Rückwärtsmodus, Gradient mit einem Rückwärtsdurchlauf.
//! * [`VectorFn`]: vektorwertige Funktion, generisch über den Zahlentyp (`f64` oder `Jet`).
//! * [`jacobian_ad`], [`hessian_ad`]: Adapter, die die unveränderten Löser aus [`optimize`](crate::optimize) mit
//!   exakten Ableitungen versorgen (Entwurf 0001, D1–D4).
//!
//! **Vergleiche:** `==` und `<` vergleichen nur den **Wert**; [`Jet::same`] vergleicht Wert und Gradient.
//! `is_nan`/`is_infinite` schlagen auch bei nicht endlichem Gradienten an.
//!
//! **Nicht glatte Funktionen** (`abs`, `min`, `max`, `floor`, …): Die Ableitung folgt dem gewählten Ast, an
//! Knickstellen ist sie willkürlich.
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::Real;
//! use azmath::autodiff::{Jet, VectorFn, ad_scratch_len, jacobian_ad};
//!
//! struct F;
//! impl VectorFn for F {
//!     fn eval<S: Real>(&self, x: &[S], out: &mut [S]) {
//!         out[0] = x[0] * x[1]; // d/dx0 = x1, d/dx1 = x0
//!     }
//! }
//! let mut scratch = [Jet::<f64, 2>::constant(0.0); ad_scratch_len(1, 2)];
//! let mut jac = [0.0; 2];
//! jacobian_ad::<f64, 2, _>(&F, &[3.0, 5.0], 1, &mut jac, &mut scratch);
//! assert_eq!(jac, [5.0, 3.0]);
//! # }
//! ```

#![allow(clippy::needless_range_loop)] // parallele Gradientfelder: Indexschleifen sind hier klarer

#[cfg(feature = "alloc")]
mod reverse;
#[cfg(feature = "alloc")]
pub use reverse::{Tape, Var, gradient_ad};

use crate::Real;
use core::cmp::Ordering;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ===========================================================================
// Jet<T, N>
// ===========================================================================

/// Wert plus Gradient fester Größe `N`. Kommutativer Ring mit Nilpotenten, kein Körper:
/// `(0, d) · (0, d') = (0, 0)`.
///
/// Gleichheit und Ordnung beziehen sich nur auf den **Wert** (wie bei Ceres); `same` vergleicht alles.
#[derive(Clone, Copy, Debug)]
pub struct Jet<T: Real, const N: usize> {
    /// Funktionswert.
    pub v: T,
    /// Gradient bezüglich der `N` unabhängigen Variablen.
    pub d: [T; N],
}

impl<T: Real, const N: usize> Jet<T, N> {
    /// Konstante: Gradient 0.
    pub fn constant(v: T) -> Self {
        Self { v, d: [T::ZERO; N] }
    }

    /// Unabhängige Variable Nummer `i`: Gradient = i-ter Einheitsvektor.
    pub fn variable(v: T, i: usize) -> Self {
        let mut d = [T::ZERO; N];
        d[i] = T::ONE;
        Self { v, d }
    }

    /// Kettenregel: `f(self)` mit Funktionswert `f` und Ableitung `df` an der Stelle `self.v`.
    pub fn chain(self, f: T, df: T) -> Self {
        let mut d = self.d;
        for k in d.iter_mut() {
            *k *= df;
        }
        Self { v: f, d }
    }

    /// Geprüfter Kehrwert: kein Inverses für Nullteiler, also für `value == 0`.
    pub fn checked_recip(&self) -> Option<Self> {
        if self.v == T::ZERO {
            None
        } else {
            Some(Self::ONE / *self)
        }
    }

    /// Vollständiger Vergleich (Wert **und** Gradient).
    pub fn same(&self, o: &Self) -> bool {
        self.v == o.v && self.d == o.d
    }
}

impl<T: Real, const N: usize> PartialEq for Jet<T, N> {
    fn eq(&self, o: &Self) -> bool {
        self.v == o.v
    }
}
impl<T: Real, const N: usize> PartialOrd for Jet<T, N> {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        self.v.partial_cmp(&o.v)
    }
}

impl<T: Real, const N: usize> Add for Jet<T, N> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        let mut d = self.d;
        for k in 0..N {
            d[k] += o.d[k];
        }
        Self { v: self.v + o.v, d }
    }
}
impl<T: Real, const N: usize> Sub for Jet<T, N> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        let mut d = self.d;
        for k in 0..N {
            d[k] -= o.d[k];
        }
        Self { v: self.v - o.v, d }
    }
}
impl<T: Real, const N: usize> Mul for Jet<T, N> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        let mut d = [T::ZERO; N];
        for k in 0..N {
            d[k] = self.v * o.d[k] + o.v * self.d[k];
        }
        Self { v: self.v * o.v, d }
    }
}
impl<T: Real, const N: usize> Div for Jet<T, N> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let q = self.v / o.v;
        let mut d = [T::ZERO; N];
        for k in 0..N {
            d[k] = (self.d[k] - q * o.d[k]) / o.v;
        }
        Self { v: q, d }
    }
}
impl<T: Real, const N: usize> Neg for Jet<T, N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut d = self.d;
        for k in d.iter_mut() {
            *k = -*k;
        }
        Self { v: -self.v, d }
    }
}
impl<T: Real, const N: usize> AddAssign for Jet<T, N> {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl<T: Real, const N: usize> SubAssign for Jet<T, N> {
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}
impl<T: Real, const N: usize> MulAssign for Jet<T, N> {
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}
impl<T: Real, const N: usize> DivAssign for Jet<T, N> {
    fn div_assign(&mut self, o: Self) {
        *self = *self / o;
    }
}

impl<T: Real, const N: usize> Real for Jet<T, N> {
    const ZERO: Self = Jet {
        v: T::ZERO,
        d: [T::ZERO; N],
    };
    const ONE: Self = Jet {
        v: T::ONE,
        d: [T::ZERO; N],
    };
    const PI: Self = Jet {
        v: T::PI,
        d: [T::ZERO; N],
    };
    const E: Self = Jet {
        v: T::E,
        d: [T::ZERO; N],
    };
    const EPSILON: Self = Jet {
        v: T::EPSILON,
        d: [T::ZERO; N],
    };
    const NAN: Self = Jet {
        v: T::NAN,
        d: [T::ZERO; N],
    };
    const INFINITY: Self = Jet {
        v: T::INFINITY,
        d: [T::ZERO; N],
    };
    const NEG_INFINITY: Self = Jet {
        v: T::NEG_INFINITY,
        d: [T::ZERO; N],
    };
    const MIN_POSITIVE: Self = Jet {
        v: T::MIN_POSITIVE,
        d: [T::ZERO; N],
    };

    fn from_f64(v: f64) -> Self {
        Self::constant(T::from_f64(v))
    }

    fn to_f64(self) -> f64 {
        self.v.to_f64()
    }

    fn abs(self) -> Self {
        self.chain(self.v.abs(), self.v.signum())
    }
    /// NaN in Wert **oder** Gradient (damit Löser auch nicht endliche Ableitungen erkennen).
    fn is_nan(self) -> bool {
        self.v.is_nan() || self.d.iter().any(|x| x.is_nan())
    }
    fn is_infinite(self) -> bool {
        self.v.is_infinite() || self.d.iter().any(|x| x.is_infinite())
    }
    fn min(self, o: Self) -> Self {
        if self.v.is_nan() {
            o
        } else if o.v.is_nan() || self.v <= o.v {
            self
        } else {
            o
        }
    }
    fn max(self, o: Self) -> Self {
        if self.v.is_nan() {
            o
        } else if o.v.is_nan() || self.v >= o.v {
            self
        } else {
            o
        }
    }
    fn signum(self) -> Self {
        Self::constant(self.v.signum())
    }

    fn sqrt(self) -> Self {
        let r = self.v.sqrt();
        self.chain(r, T::ONE / (T::from_f64(2.0) * r))
    }
    fn sin(self) -> Self {
        self.chain(self.v.sin(), self.v.cos())
    }
    fn cos(self) -> Self {
        self.chain(self.v.cos(), -self.v.sin())
    }
    fn tan(self) -> Self {
        let t = self.v.tan();
        self.chain(t, T::ONE + t * t)
    }
    fn asin(self) -> Self {
        self.chain(self.v.asin(), T::ONE / (T::ONE - self.v * self.v).sqrt())
    }
    fn acos(self) -> Self {
        self.chain(self.v.acos(), -T::ONE / (T::ONE - self.v * self.v).sqrt())
    }
    fn atan(self) -> Self {
        self.chain(self.v.atan(), T::ONE / (T::ONE + self.v * self.v))
    }
    /// `self` ist `y`, `o` ist `x`.
    fn atan2(self, o: Self) -> Self {
        let den = o.v * o.v + self.v * self.v;
        let mut d = [T::ZERO; N];
        for k in 0..N {
            d[k] = (o.v * self.d[k] - self.v * o.d[k]) / den;
        }
        Self {
            v: self.v.atan2(o.v),
            d,
        }
    }
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.chain(e, e)
    }
    fn ln(self) -> Self {
        self.chain(self.v.ln(), T::ONE / self.v)
    }
    fn log10(self) -> Self {
        self.chain(
            self.v.log10(),
            T::ONE / (self.v * T::from_f64(core::f64::consts::LN_10)),
        )
    }
    fn log2(self) -> Self {
        self.chain(
            self.v.log2(),
            T::ONE / (self.v * T::from_f64(core::f64::consts::LN_2)),
        )
    }
    fn powf(self, n: Self) -> Self {
        let value = self.v.powf(n.v);
        let dx = n.v * self.v.powf(n.v - T::ONE);
        let ln = self.v.ln();
        let mut d = [T::ZERO; N];
        for k in 0..N {
            d[k] = dx * self.d[k];
            // Term mit ln(x) nur, wenn der Exponent selbst von den Variablen abhängt (vermeidet 0·NaN)
            if n.d[k] != T::ZERO {
                d[k] += value * ln * n.d[k];
            }
        }
        Self { v: value, d }
    }
    fn powi(self, n: i32) -> Self {
        let df = if n == 0 {
            T::ZERO
        } else {
            T::from_f64(n as f64) * self.v.powi(n - 1)
        };
        self.chain(self.v.powi(n), df)
    }
    fn floor(self) -> Self {
        Self::constant(self.v.floor())
    }
    fn ceil(self) -> Self {
        Self::constant(self.v.ceil())
    }
    fn round(self) -> Self {
        Self::constant(self.v.round())
    }
    fn trunc(self) -> Self {
        Self::constant(self.v.trunc())
    }
    fn hypot(self, o: Self) -> Self {
        let h = self.v.hypot(o.v);
        let mut d = [T::ZERO; N];
        if h != T::ZERO {
            for k in 0..N {
                d[k] = (self.v * self.d[k] + o.v * o.d[k]) / h;
            }
        }
        Self { v: h, d }
    }
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
    }
}

// ===========================================================================
// Funktionsschnittstelle und Adapter
// ===========================================================================

/// Vektorwertige Funktion `f: ℝⁿ → ℝᵐ`, generisch über den Zahlentyp `S`.
///
/// Gleiches `eval` für `S = f64` (Funktionswert) und `S = Jet<f64, K>` (Funktionswert und Ableitungen).
pub trait VectorFn {
    /// Wertet `f(x)` aus und schreibt das Ergebnis nach `out`.
    fn eval<S: Real>(&self, x: &[S], out: &mut [S]);
}

/// Länge des benötigten `Jet`-Puffers für `jacobian_ad` (Eingänge `n` plus Ausgänge `m`).
pub const fn ad_scratch_len(m: usize, n: usize) -> usize {
    n + m
}

/// Exakte Jacobi-Matrix (zeilenweise, `m × n`) per Vorwärts-Autodiff.
///
/// Die `n` Unbekannten werden in **Blöcken von `K`** Richtungen verarbeitet („Strip-Mining“): `⌈n/K⌉`
/// Auswertungen mit `Jet<T, K>`. So funktioniert es auch, wenn `n` erst zur Laufzeit feststeht, ohne
/// Heap. `scratch` ist ein Puffer aus `Jet<T, K>` (nicht aus `T`!).
pub fn jacobian_ad<T: Real, const K: usize, F: VectorFn>(
    f: &F,
    x: &[T],
    m: usize,
    jac: &mut [T],
    scratch: &mut [Jet<T, K>],
) {
    let n = x.len();
    assert!(K >= 1 && scratch.len() >= ad_scratch_len(m, n) && jac.len() >= m * n);
    let (xj, rest) = scratch.split_at_mut(n);
    let outj = &mut rest[..m];
    let mut start = 0;
    while start < n {
        let width = K.min(n - start);
        for (i, xi) in xj.iter_mut().enumerate() {
            *xi = Jet::constant(x[i]);
        }
        for k in 0..width {
            xj[start + k] = Jet::variable(x[start + k], k);
        }
        f.eval::<Jet<T, K>>(xj, outj);
        for i in 0..m {
            for k in 0..width {
                jac[i * n + start + k] = outj[i].d[k];
            }
        }
        start += width;
    }
}

/// Hessematrix einer skalaren Funktion (`m = 1`) mit **geschachtelten** Jets: `K = n` Richtungen.
pub fn hessian_ad<T: Real, const K: usize, F: VectorFn>(f: &F, x: &[T; K]) -> [[T; K]; K] {
    type J2<T, const K: usize> = Jet<Jet<T, K>, K>;
    let mut xs = [J2::<T, K>::constant(Jet::constant(T::ZERO)); K];
    for i in 0..K {
        // Wert: innere Variable i (liefert den Gradienten); äußere Richtung i: konstante 1
        let mut outer = [Jet::<T, K>::constant(T::ZERO); K];
        outer[i] = Jet::constant(T::ONE);
        xs[i] = Jet {
            v: Jet::variable(x[i], i),
            d: outer,
        };
    }
    let mut out = [J2::<T, K>::constant(Jet::constant(T::ZERO)); 1];
    f.eval::<J2<T, K>>(&xs, &mut out);
    let mut h = [[T::ZERO; K]; K];
    for j in 0..K {
        for k in 0..K {
            h[j][k] = out[0].d[j].d[k];
        }
    }
    h
}

/// Komfortmakro: schreibt eine `VectorFn` wie eine Closure.
#[macro_export]
macro_rules! vector_fn {
    ($name:ident, |$x:ident, $out:ident| $body:block) => {
        struct $name;
        impl $crate::autodiff::VectorFn for $name {
            fn eval<S: $crate::Real>(&self, $x: &[S], $out: &mut [S]) $body
        }
    };
}
