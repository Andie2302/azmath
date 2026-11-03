//! Polynome (Feature `alloc`): Arithmetik, Ausgleichspolynome, Tschebyscheff-Reihen und Kleinste-Quadrate-Varianten.
//!
//! | Typ / Funktion | Inhalt |
//! |---|---|
//! | [`Polynomial`] | Koeffizienten aufsteigend (`c[0] + c[1]·x + …`): Rechnen, Ableiten, Integrieren, Division, Einsetzen, ggT |
//! | [`polyfit`] | Ausgleichspolynom vom Grad `d` (kleinste Quadrate über QR) |
//! | [`ChebyshevSeries`] | Approximation einer Funktion auf `[a, b]` durch Tschebyscheff-Reihe, Auswertung nach Clenshaw |
//! | [`weighted_least_squares`], [`ridge`], [`nnls`] | gewichtete, regularisierte und nichtnegative kleinste Quadrate |
//!
//! ```
//! use azmath::poly::Polynomial;
//! let p = Polynomial::new(vec![-1.0f64, 0.0, 1.0]); // x² − 1
//! let q = Polynomial::new(vec![1.0, 1.0]); // x + 1
//! let (quot, rem) = p.div_rem(&q).unwrap();
//! assert_eq!(quot.coefficients(), &[-1.0, 1.0]); // x − 1
//! assert!(rem.is_zero());
//! ```

mod chebyshev;
mod lsq;

pub use chebyshev::{ChebyshevSeries, chebyshev_nodes};
pub use lsq::{NnlsResult, nnls, ridge, weighted_least_squares};

use crate::Real;
use crate::linalg::{DynMatrix, DynVector, LinalgError};
use alloc::vec;
use alloc::vec::Vec;
use core::ops::{Add, Mul, Neg, Sub};

/// Polynom mit aufsteigend geordneten Koeffizienten; führende Nullen werden entfernt (das Nullpolynom hat keine Koeffizienten).
#[derive(Debug, Clone, PartialEq)]
pub struct Polynomial<T> {
    c: Vec<T>,
}

impl<T: Real> Polynomial<T> {
    /// Aus Koeffizienten `c[0] + c[1]·x + …`.
    pub fn new(mut coeffs: Vec<T>) -> Self {
        while coeffs.last().is_some_and(|&v| v == T::ZERO) {
            coeffs.pop();
        }
        Self { c: coeffs }
    }

    /// Das Nullpolynom.
    pub fn zero() -> Self {
        Self { c: Vec::new() }
    }

    /// Das konstante Polynom `v`.
    pub fn constant(v: T) -> Self {
        Self::new(vec![v])
    }

    /// Das Polynom `∏ (x − rᵢ)` mit den gegebenen Nullstellen.
    pub fn from_roots(roots: &[T]) -> Self {
        let mut p = Self::constant(T::ONE);
        for &r in roots {
            p = &p * &Self::new(vec![-r, T::ONE]);
        }
        p
    }

    /// Koeffizienten (aufsteigend, ohne führende Nullen).
    pub fn coefficients(&self) -> &[T] {
        &self.c
    }

    /// `true` beim Nullpolynom.
    pub fn is_zero(&self) -> bool {
        self.c.is_empty()
    }

    /// Grad (das Nullpolynom liefert 0).
    pub fn degree(&self) -> usize {
        self.c.len().saturating_sub(1)
    }

    /// Wert an der Stelle `x` (Horner).
    pub fn eval(&self, x: T) -> T {
        self.c.iter().rev().fold(T::ZERO, |acc, &ck| acc * x + ck)
    }

    /// Ableitung.
    pub fn derivative(&self) -> Self {
        if self.c.len() <= 1 {
            return Self::zero();
        }
        Self::new(
            self.c
                .iter()
                .enumerate()
                .skip(1)
                .map(|(k, &v)| v * T::from_usize(k))
                .collect(),
        )
    }

    /// Stammfunktion mit Integrationskonstante 0.
    pub fn integral(&self) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let mut c = vec![T::ZERO];
        for (k, &v) in self.c.iter().enumerate() {
            c.push(v / T::from_usize(k + 1));
        }
        Self::new(c)
    }

    /// Bestimmtes Integral über `[a, b]`.
    pub fn integrate(&self, a: T, b: T) -> T {
        let f = self.integral();
        f.eval(b) - f.eval(a)
    }

    /// Mit einem Skalar multipliziert.
    pub fn scale(&self, s: T) -> Self {
        Self::new(self.c.iter().map(|&v| v * s).collect())
    }

    /// Division mit Rest: `self = q·d + r` mit `Grad(r) < Grad(d)`. Fehler [`LinalgError::Singular`] bei `d = 0`.
    pub fn div_rem(&self, d: &Self) -> Result<(Self, Self), LinalgError> {
        if d.is_zero() {
            return Err(LinalgError::Singular);
        }
        let dd = d.c.len() - 1;
        let lead = d.c[dd];
        let mut r = self.c.clone();
        if r.len() < d.c.len() {
            return Ok((Self::zero(), Self::new(r)));
        }
        let mut q = vec![T::ZERO; r.len() - dd];
        for k in (0..q.len()).rev() {
            let f = r[k + dd] / lead;
            q[k] = f;
            for j in 0..=dd {
                r[k + j] -= f * d.c[j];
            }
            r[k + dd] = T::ZERO;
        }
        r.truncate(dd);
        Ok((Self::new(q), Self::new(r)))
    }

    /// Einsetzen eines Polynoms: `self(inner(x))`.
    pub fn compose(&self, inner: &Self) -> Self {
        let mut acc = Self::zero();
        for &ck in self.c.iter().rev() {
            acc = &(&acc * inner) + &Self::constant(ck);
        }
        acc
    }

    /// Normierter größter gemeinsamer Teiler (Euklid). Reste mit Koeffizienten unter `tol·(größter Koeffizient)` gelten als 0.
    pub fn gcd(&self, other: &Self, tol: T) -> Self {
        let (mut a, mut b) = (self.clone(), other.clone());
        let cleanup = |p: &Self| {
            let m = p.c.iter().fold(T::ZERO, |m, v| m.max(v.abs()));
            Self::new(
                p.c.iter()
                    .map(|&v| if v.abs() <= tol * m { T::ZERO } else { v })
                    .collect(),
            )
        };
        while !b.is_zero() {
            let r = match a.div_rem(&b) {
                Ok((_, r)) => cleanup(&r),
                Err(_) => break,
            };
            a = b;
            b = r;
        }
        match a.c.last() {
            Some(&l) => a.scale(T::ONE / l),
            None => a,
        }
    }
}

impl<T: Real> Add for &Polynomial<T> {
    type Output = Polynomial<T>;
    fn add(self, o: Self) -> Polynomial<T> {
        let n = self.c.len().max(o.c.len());
        let c = (0..n)
            .map(|i| {
                self.c.get(i).copied().unwrap_or(T::ZERO) + o.c.get(i).copied().unwrap_or(T::ZERO)
            })
            .collect();
        Polynomial::new(c)
    }
}

impl<T: Real> Sub for &Polynomial<T> {
    type Output = Polynomial<T>;
    fn sub(self, o: Self) -> Polynomial<T> {
        let n = self.c.len().max(o.c.len());
        let c = (0..n)
            .map(|i| {
                self.c.get(i).copied().unwrap_or(T::ZERO) - o.c.get(i).copied().unwrap_or(T::ZERO)
            })
            .collect();
        Polynomial::new(c)
    }
}

impl<T: Real> Mul for &Polynomial<T> {
    type Output = Polynomial<T>;
    fn mul(self, o: Self) -> Polynomial<T> {
        if self.is_zero() || o.is_zero() {
            return Polynomial::zero();
        }
        let mut c = vec![T::ZERO; self.c.len() + o.c.len() - 1];
        for (i, &a) in self.c.iter().enumerate() {
            for (j, &b) in o.c.iter().enumerate() {
                c[i + j] += a * b;
            }
        }
        Polynomial::new(c)
    }
}

impl<T: Real> Neg for &Polynomial<T> {
    type Output = Polynomial<T>;
    fn neg(self) -> Polynomial<T> {
        self.scale(-T::ONE)
    }
}

/// Ausgleichspolynom vom Grad `degree` durch die Punkte `(x[i], y[i])` (kleinste Quadrate über QR der Vandermonde-Matrix).
///
/// Für hohe Grade ist die Vandermonde-Matrix schlecht konditioniert; dann besser [`ChebyshevSeries`]. Fehler:
/// [`LinalgError::DimensionMismatch`] (ungleiche Längen oder weniger Punkte als `degree + 1`), [`LinalgError::Singular`]
/// (zu wenige verschiedene Stützstellen), [`LinalgError::NotFinite`].
pub fn polyfit<T: Real>(x: &[T], y: &[T], degree: usize) -> Result<Polynomial<T>, LinalgError> {
    let m = x.len();
    let n = degree + 1;
    if y.len() != m || m < n {
        return Err(LinalgError::DimensionMismatch);
    }
    let mut a = DynMatrix::zeros(m, n);
    for i in 0..m {
        let mut p = T::ONE;
        for j in 0..n {
            a[(i, j)] = p;
            p *= x[i];
        }
    }
    let sol = a.least_squares(&DynVector::new(y.to_vec()))?;
    Ok(Polynomial::new(sol.as_slice().to_vec()))
}
