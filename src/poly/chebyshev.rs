//! Tschebyscheff-Reihen.

use crate::Real;
use alloc::vec;
use alloc::vec::Vec;

/// Die `n` Tschebyscheff-Knoten (Nullstellen von `T_n`) auf `[a, b]`, aufsteigend.
pub fn chebyshev_nodes<T: Real>(n: usize, a: T, b: T) -> Vec<T> {
    let half = T::from_f64(0.5);
    let (c, r) = ((a + b) * half, (b - a) * half);
    (0..n)
        .rev()
        .map(|j| {
            let t = (T::PI * (T::from_usize(j) + half) / T::from_usize(n)).cos();
            c + r * t
        })
        .collect()
}

/// Reihe `Σ cₖ·Tₖ(t)` mit `t = (2x − a − b)/(b − a)` auf dem Intervall `[a, b]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ChebyshevSeries<T> {
    c: Vec<T>,
    a: T,
    b: T,
}

impl<T: Real> ChebyshevSeries<T> {
    /// Interpolation von `f` in den `n` Tschebyscheff-Knoten von `[a, b]` (Reihe mit `n` Koeffizienten, `n ≥ 1`).
    pub fn from_function(mut f: impl FnMut(T) -> T, a: T, b: T, n: usize) -> Self {
        assert!(n >= 1, "ChebyshevSeries: n muss ≥ 1 sein");
        let half = T::from_f64(0.5);
        let (cc, r) = ((a + b) * half, (b - a) * half);
        let fx: Vec<T> = (0..n)
            .map(|j| {
                let t = (T::PI * (T::from_usize(j) + half) / T::from_usize(n)).cos();
                f(cc + r * t)
            })
            .collect();
        let mut c = vec![T::ZERO; n];
        for (k, ck) in c.iter_mut().enumerate() {
            let mut s = T::ZERO;
            for (j, &v) in fx.iter().enumerate() {
                s += v
                    * (T::PI * T::from_usize(k) * (T::from_usize(j) + half) / T::from_usize(n))
                        .cos();
            }
            *ck = s * T::from_f64(2.0) / T::from_usize(n);
        }
        c[0] *= half;
        Self { c, a, b }
    }

    /// Aus Koeffizienten `cₖ` auf `[a, b]`.
    pub fn from_coefficients(c: Vec<T>, a: T, b: T) -> Self {
        Self { c, a, b }
    }

    /// Koeffizienten.
    pub fn coefficients(&self) -> &[T] {
        &self.c
    }

    /// Auswertung nach Clenshaw.
    pub fn eval(&self, x: T) -> T {
        let t = (T::from_f64(2.0) * x - self.a - self.b) / (self.b - self.a);
        let (mut b1, mut b2) = (T::ZERO, T::ZERO);
        let two_t = T::from_f64(2.0) * t;
        for &ck in self.c.iter().skip(1).rev() {
            let b0 = ck + two_t * b1 - b2;
            b2 = b1;
            b1 = b0;
        }
        self.c.first().copied().unwrap_or(T::ZERO) + t * b1 - b2
    }

    /// Ableitung nach `x` als Reihe mit einem Koeffizienten weniger.
    pub fn derivative(&self) -> Self {
        let n = self.c.len();
        if n <= 1 {
            return Self {
                c: vec![T::ZERO],
                a: self.a,
                b: self.b,
            };
        }
        let mut d = vec![T::ZERO; n + 1];
        for k in (1..n).rev() {
            d[k - 1] = d[k + 1] + T::from_f64(2.0) * T::from_usize(k) * self.c[k];
        }
        d[0] *= T::from_f64(0.5);
        d.truncate(n - 1);
        let scale = T::from_f64(2.0) / (self.b - self.a);
        Self {
            c: d.into_iter().map(|v| v * scale).collect(),
            a: self.a,
            b: self.b,
        }
    }
}
