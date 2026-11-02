use core::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

use azmath_core::Real;

/// Vektor fester Größe `N` (liegt auf dem Stack).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector<T, const N: usize> {
    pub(crate) data: [T; N],
}

impl<T: Real, const N: usize> Vector<T, N> {
    /// Erzeugt einen Vektor aus einem Array.
    pub const fn new(data: [T; N]) -> Self {
        Self { data }
    }

    /// Der Nullvektor.
    pub fn zeros() -> Self {
        Self { data: [T::ZERO; N] }
    }

    /// Zugriff auf das zugrunde liegende Array.
    pub fn as_array(&self) -> &[T; N] {
        &self.data
    }

    /// Die Elemente als Slice (zum Beispiel für die Slice-API der iterativen Löser).
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Die Elemente als veränderlicher Slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Gibt das zugrunde liegende Array zurück.
    pub fn into_array(self) -> [T; N] {
        self.data
    }

    /// Anzahl der Elemente.
    pub const fn len(&self) -> usize {
        N
    }

    /// `true`, wenn der Vektor keine Elemente hat.
    pub const fn is_empty(&self) -> bool {
        N == 0
    }

    /// Skalarprodukt.
    pub fn dot(&self, other: &Self) -> T {
        let mut s = T::ZERO;
        for i in 0..N {
            s += self.data[i] * other.data[i];
        }
        s
    }

    /// Quadrat der euklidischen Länge.
    pub fn norm_squared(&self) -> T {
        self.dot(self)
    }

    /// Euklidische Länge.
    pub fn norm(&self) -> T {
        self.norm_squared().sqrt()
    }

    /// Größter Betrag aller Elemente (Maximumnorm).
    pub fn max_abs(&self) -> T {
        let mut m = T::ZERO;
        for &v in &self.data {
            m = m.max(v.abs());
        }
        m
    }

    /// `self += alpha · x` (in-place, ohne Zwischenvektor).
    pub fn axpy(&mut self, alpha: T, x: &Self) {
        for i in 0..N {
            self.data[i] += alpha * x.data[i];
        }
    }

    /// Multipliziert jedes Element mit `s`.
    pub fn scale(self, s: T) -> Self {
        let mut out = self;
        for v in out.data.iter_mut() {
            *v *= s;
        }
        out
    }
}

impl<T, const N: usize> Index<usize> for Vector<T, N> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T, const N: usize> IndexMut<usize> for Vector<T, N> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

impl<T: Real, const N: usize> Add for Vector<T, N> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self {
        for i in 0..N {
            self.data[i] += rhs.data[i];
        }
        self
    }
}

impl<T: Real, const N: usize> Sub for Vector<T, N> {
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self {
        for i in 0..N {
            self.data[i] -= rhs.data[i];
        }
        self
    }
}

impl<T: Real, const N: usize> Neg for Vector<T, N> {
    type Output = Self;
    fn neg(mut self) -> Self {
        for v in self.data.iter_mut() {
            *v = -*v;
        }
        self
    }
}

impl<T: Real, const N: usize> Mul<T> for Vector<T, N> {
    type Output = Self;
    fn mul(self, s: T) -> Self {
        self.scale(s)
    }
}
