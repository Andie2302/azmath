//! Pseudozufallszahlen ohne Allokator: [`Pcg32`], [`Xoshiro256pp`] und das Trait [`Rng`] mit Verteilungen.
//!
//! Beide Generatoren sind deterministisch bei gleichem Startwert (reproduzierbare Simulationen), aber **nicht**
//! kryptografisch sicher.
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::random::{Pcg32, Rng};
//! let mut rng = Pcg32::new(42, 54);
//! let u: f64 = rng.uniform();
//! assert!((0.0..1.0).contains(&u));
//! # }
//! ```

use crate::Real;

/// Quelle gleichverteilter Zufallsbits mit daraus abgeleiteten Verteilungen.
pub trait Rng {
    /// Nächste 64 Zufallsbits.
    fn next_u64(&mut self) -> u64;

    /// Nächste 32 Zufallsbits.
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Gleichverteilt in `[0, 1)` (53 Bit Auflösung).
    fn uniform<T: Real>(&mut self) -> T {
        T::from_f64((self.next_u64() >> 11) as f64 / 9007199254740992.0)
    }

    /// Gleichverteilt in `[lo, hi)`.
    fn uniform_range<T: Real>(&mut self, lo: T, hi: T) -> T {
        lo + (hi - lo) * self.uniform::<T>()
    }

    /// Gleichverteilte ganze Zahl in `0..n` (ohne Modulo-Verzerrung; `n > 0`).
    fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "below: n muss positiv sein");
        let zone = u64::MAX - (u64::MAX % n + 1) % n;
        loop {
            let r = self.next_u64();
            if r <= zone {
                return r % n;
            }
        }
    }

    /// Standardnormalverteilt (Box-Muller).
    fn normal<T: Real>(&mut self) -> T {
        let mut u1: T = self.uniform();
        while u1 <= T::ZERO {
            u1 = self.uniform();
        }
        let u2: T = self.uniform();
        (-T::from_f64(2.0) * u1.ln()).sqrt() * (T::from_f64(2.0) * T::PI * u2).cos()
    }

    /// Normalverteilt mit Mittelwert `mean` und Standardabweichung `sd`.
    fn normal_with<T: Real>(&mut self, mean: T, sd: T) -> T {
        mean + sd * self.normal::<T>()
    }

    /// Exponentialverteilt mit Rate `lambda > 0`.
    fn exponential<T: Real>(&mut self, lambda: T) -> T {
        let mut u: T = self.uniform();
        while u <= T::ZERO {
            u = self.uniform();
        }
        -u.ln() / lambda
    }

    /// Mischt `data` an Ort und Stelle (Fisher-Yates).
    fn shuffle<X>(&mut self, data: &mut [X]) {
        for i in (1..data.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            data.swap(i, j);
        }
    }
}

/// PCG-XSH-RR 64/32 (O'Neill): kleiner Zustand, gute Statistik.
#[derive(Debug, Clone)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Initialisierung mit Startwert `seed` und Strom `stream`.
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut g = Self {
            state: 0,
            inc: (stream << 1) | 1,
        };
        g.step();
        g.state = g.state.wrapping_add(seed);
        g.step();
        g
    }

    fn step(&mut self) -> u64 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        old
    }

    /// Nächste 32 Bit.
    pub fn next32(&mut self) -> u32 {
        let old = self.step();
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
}

impl Rng for Pcg32 {
    fn next_u32(&mut self) -> u32 {
        self.next32()
    }
    fn next_u64(&mut self) -> u64 {
        let hi = self.next32() as u64;
        let lo = self.next32() as u64;
        (hi << 32) | lo
    }
}

/// xoshiro256++ (Blackman/Vigna): schnell, 256 Bit Zustand.
#[derive(Debug, Clone)]
pub struct Xoshiro256pp {
    s: [u64; 4],
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

impl Xoshiro256pp {
    /// Initialisierung aus einem 64-Bit-Startwert (über SplitMix64 aufgefüllt).
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        Self {
            s: [
                splitmix64(&mut sm),
                splitmix64(&mut sm),
                splitmix64(&mut sm),
                splitmix64(&mut sm),
            ],
        }
    }
}

impl Rng for Xoshiro256pp {
    fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }
}
