//! Automatisches Differenzieren im Rückwärtsmodus (Tape), Feature `alloc`.
//!
//! Die Kosten eines Gradienten sind unabhängig von der Anzahl der Eingaben (ein Rückwärtsdurchlauf), im Gegensatz
//! zum Vorwärtsmodus mit [`Jet`](super::Jet). [`Var`] ist ein `Copy`-Handle auf einen Knoten des [`Tape`]s.
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::autodiff::gradient_ad;
//! // f(x, y) = x·y + sin(x)
//! let (value, grad) = gradient_ad(&[2.0_f64, 3.0], |v| v[0] * v[1] + v[0].sin());
//! assert!((value - (6.0 + 2.0_f64.sin())).abs() < 1e-14);
//! assert!((grad[0] - (3.0 + 2.0_f64.cos())).abs() < 1e-14);
//! assert!((grad[1] - 2.0).abs() < 1e-14);
//! # }
//! ```

use crate::Real;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::ops::{Add, Div, Mul, Neg, Sub};

const NONE: usize = usize::MAX;

#[derive(Clone, Copy)]
struct Node<T> {
    a: usize,
    da: T,
    b: usize,
    db: T,
}

/// Aufzeichnung der Rechenschritte.
pub struct Tape<T: Real> {
    nodes: RefCell<Vec<Node<T>>>,
}

impl<T: Real> Default for Tape<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Real> Tape<T> {
    /// Leeres Tape.
    pub fn new() -> Self {
        Self {
            nodes: RefCell::new(Vec::new()),
        }
    }

    fn push(&self, a: usize, da: T, b: usize, db: T) -> usize {
        let mut n = self.nodes.borrow_mut();
        n.push(Node { a, da, b, db });
        n.len() - 1
    }

    /// Neue Eingangsvariable (Blatt) mit Wert `v`.
    pub fn var(&self, v: T) -> Var<'_, T> {
        Var {
            tape: self,
            idx: self.push(NONE, T::ZERO, NONE, T::ZERO),
            v,
        }
    }

    /// Konstante (kein Beitrag zum Gradienten).
    pub fn constant(&self, v: T) -> Var<'_, T> {
        self.var(v)
    }

    /// Anzahl der Knoten.
    pub fn len(&self) -> usize {
        self.nodes.borrow().len()
    }

    /// `true`, wenn nichts aufgezeichnet wurde.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adjungierte (Ableitungen von `y` nach jedem Knoten) in einem Rückwärtsdurchlauf.
    pub fn adjoints(&self, y: &Var<'_, T>) -> Vec<T> {
        let nodes = self.nodes.borrow();
        let mut adj = vec![T::ZERO; nodes.len()];
        adj[y.idx] = T::ONE;
        for i in (0..=y.idx).rev() {
            let g = adj[i];
            if g == T::ZERO {
                continue;
            }
            let nd = nodes[i];
            if nd.a != NONE {
                adj[nd.a] += g * nd.da;
            }
            if nd.b != NONE {
                adj[nd.b] += g * nd.db;
            }
        }
        adj
    }
}

/// Wert samt Position im [`Tape`].
#[derive(Clone, Copy)]
pub struct Var<'t, T: Real> {
    tape: &'t Tape<T>,
    idx: usize,
    v: T,
}

impl<'t, T: Real> Var<'t, T> {
    /// Funktionswert.
    pub fn value(&self) -> T {
        self.v
    }

    /// Index im Tape (für [`Tape::adjoints`]).
    pub fn index(&self) -> usize {
        self.idx
    }

    fn unary(self, v: T, d: T) -> Self {
        Var {
            tape: self.tape,
            idx: self.tape.push(self.idx, d, NONE, T::ZERO),
            v,
        }
    }

    fn binary(self, o: Self, v: T, da: T, db: T) -> Self {
        Var {
            tape: self.tape,
            idx: self.tape.push(self.idx, da, o.idx, db),
            v,
        }
    }

    /// Quadratwurzel.
    pub fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.unary(s, T::from_f64(0.5) / s)
    }
    /// Sinus.
    pub fn sin(self) -> Self {
        self.unary(self.v.sin(), self.v.cos())
    }
    /// Kosinus.
    pub fn cos(self) -> Self {
        self.unary(self.v.cos(), -self.v.sin())
    }
    /// Tangens.
    pub fn tan(self) -> Self {
        let t = self.v.tan();
        self.unary(t, T::ONE + t * t)
    }
    /// Arkustangens.
    pub fn atan(self) -> Self {
        self.unary(self.v.atan(), T::ONE / (T::ONE + self.v * self.v))
    }
    /// Exponentialfunktion.
    pub fn exp(self) -> Self {
        let e = self.v.exp();
        self.unary(e, e)
    }
    /// Natürlicher Logarithmus.
    pub fn ln(self) -> Self {
        self.unary(self.v.ln(), T::ONE / self.v)
    }
    /// Hyperbeltangens.
    pub fn tanh(self) -> Self {
        let t = self.v.tanh();
        self.unary(t, T::ONE - t * t)
    }
    /// Betrag (Ableitung `signum`, bei 0 gleich +1).
    pub fn abs(self) -> Self {
        let d = if self.v < T::ZERO { -T::ONE } else { T::ONE };
        self.unary(self.v.abs(), d)
    }
    /// Ganzzahlige Potenz.
    pub fn powi(self, n: i32) -> Self {
        let d = if n == 0 {
            T::ZERO
        } else {
            T::from_f64(n as f64) * self.v.powi(n - 1)
        };
        self.unary(self.v.powi(n), d)
    }
    /// Reelle Potenz `self^p` mit konstantem Exponenten.
    pub fn powf(self, p: T) -> Self {
        self.unary(self.v.powf(p), p * self.v.powf(p - T::ONE))
    }
}

impl<'t, T: Real> Add for Var<'t, T> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        self.binary(o, self.v + o.v, T::ONE, T::ONE)
    }
}
impl<'t, T: Real> Sub for Var<'t, T> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        self.binary(o, self.v - o.v, T::ONE, -T::ONE)
    }
}
impl<'t, T: Real> Mul for Var<'t, T> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        self.binary(o, self.v * o.v, o.v, self.v)
    }
}
impl<'t, T: Real> Div for Var<'t, T> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let q = self.v / o.v;
        self.binary(o, q, T::ONE / o.v, -q / o.v)
    }
}
impl<'t, T: Real> Neg for Var<'t, T> {
    type Output = Self;
    fn neg(self) -> Self {
        self.unary(-self.v, -T::ONE)
    }
}
impl<'t, T: Real> Add<T> for Var<'t, T> {
    type Output = Self;
    fn add(self, c: T) -> Self {
        self.unary(self.v + c, T::ONE)
    }
}
impl<'t, T: Real> Sub<T> for Var<'t, T> {
    type Output = Self;
    fn sub(self, c: T) -> Self {
        self.unary(self.v - c, T::ONE)
    }
}
impl<'t, T: Real> Mul<T> for Var<'t, T> {
    type Output = Self;
    fn mul(self, c: T) -> Self {
        self.unary(self.v * c, c)
    }
}
impl<'t, T: Real> Div<T> for Var<'t, T> {
    type Output = Self;
    fn div(self, c: T) -> Self {
        self.unary(self.v / c, T::ONE / c)
    }
}

/// Wert und Gradient von `f` im Punkt `x` (ein Vorwärts- und ein Rückwärtsdurchlauf).
pub fn gradient_ad<T: Real, F>(x: &[T], f: F) -> (T, Vec<T>)
where
    F: for<'t> Fn(&[Var<'t, T>]) -> Var<'t, T>,
{
    let tape = Tape::new();
    let vars: Vec<Var<'_, T>> = x.iter().map(|&v| tape.var(v)).collect();
    let y = f(&vars);
    let adj = tape.adjoints(&y);
    (y.value(), vars.iter().map(|v| adj[v.index()]).collect())
}
