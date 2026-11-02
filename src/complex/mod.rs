//! Komplexe Zahlen `re + i·im`, generisch über [`Real`].
//!
//! [`Complex`] ist ein einfaches `Copy`-Paar ohne Allokator. Es bringt die vier Grundrechenarten (auch mit
//! reellem Skalar), Betrag, Argument, Konjugation, Kehrwert, ganzzahlige und reelle Potenzen, Wurzel,
//! Exponentialfunktion, Logarithmus sowie `sin`/`cos`/`sinh`/`cosh` mit.
//!
//! Bewusst **nicht** `Real`: komplexe Zahlen haben keine Ordnung. `Complex` implementiert deshalb
//! `PartialEq`, aber kein `PartialOrd`.
//!
//! **Zweige:** `arg` liegt in `(−π, π]`; `sqrt`, `ln` und `powf` verwenden den Hauptzweig.
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::complex::Complex;
//!
//! // Wurzeln von z² + 1 = 0
//! let i = Complex::new(0.0_f64, 1.0);
//! assert_eq!(i * i, Complex::new(-1.0, 0.0));
//! let w = Complex::new(-4.0_f64, 0.0).sqrt();
//! assert!((w - Complex::new(0.0, 2.0)).abs() < 1e-14);
//! # }
//! ```

use crate::Real;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Komplexe Zahl `re + i·im`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Complex<T> {
    /// Realteil.
    pub re: T,
    /// Imaginärteil.
    pub im: T,
}

impl<T: Real> Complex<T> {
    /// Die Zahl 0.
    pub const ZERO: Self = Self {
        re: T::ZERO,
        im: T::ZERO,
    };
    /// Die Zahl 1.
    pub const ONE: Self = Self {
        re: T::ONE,
        im: T::ZERO,
    };
    /// Die imaginäre Einheit `i`.
    pub const I: Self = Self {
        re: T::ZERO,
        im: T::ONE,
    };

    /// Aus Real- und Imaginärteil.
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    /// Reelle Zahl als komplexe Zahl (Imaginärteil 0).
    pub const fn from_real(re: T) -> Self {
        Self { re, im: T::ZERO }
    }

    /// Aus Polarform `r·(cos φ + i·sin φ)`.
    pub fn from_polar(r: T, phi: T) -> Self {
        Self {
            re: r * phi.cos(),
            im: r * phi.sin(),
        }
    }

    /// Konjugiert komplexe Zahl `re − i·im`.
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Quadrat des Betrags `re² + im²` (ohne Wurzel, kein Überlaufschutz).
    pub fn norm_sqr(self) -> T {
        self.re * self.re + self.im * self.im
    }

    /// Betrag `√(re² + im²)`, überlaufsicher über `hypot`.
    pub fn abs(self) -> T {
        self.re.hypot(self.im)
    }

    /// Argument (Winkel) in `(−π, π]`.
    pub fn arg(self) -> T {
        self.im.atan2(self.re)
    }

    /// Polarform `(Betrag, Argument)`.
    pub fn to_polar(self) -> (T, T) {
        (self.abs(), self.arg())
    }

    /// `true`, wenn Real- oder Imaginärteil NaN ist.
    pub fn is_nan(self) -> bool {
        self.re.is_nan() || self.im.is_nan()
    }

    /// `true`, wenn beide Teile endlich sind.
    pub fn is_finite(self) -> bool {
        !(self.is_nan() || self.re.is_infinite() || self.im.is_infinite())
    }

    /// Kehrwert `1/z`. Für `z = 0` ergibt sich wie bei Gleitkommazahlen `∞`/NaN.
    pub fn recip(self) -> Self {
        Self::ONE / self
    }

    /// Multiplikation mit einem reellen Skalar.
    pub fn scale(self, k: T) -> Self {
        Self {
            re: self.re * k,
            im: self.im * k,
        }
    }

    /// Ganzzahlige Potenz (Binärpotenzierung; negative Exponenten über den Kehrwert).
    pub fn powi(self, n: i32) -> Self {
        let mut e = n.unsigned_abs();
        let mut base = self;
        let mut acc = Self::ONE;
        while e > 0 {
            if e & 1 == 1 {
                acc *= base;
            }
            base *= base;
            e >>= 1;
        }
        if n < 0 { acc.recip() } else { acc }
    }

    /// Hauptwert der Quadratwurzel (Realteil `≥ 0`).
    pub fn sqrt(self) -> Self {
        if self.re == T::ZERO && self.im == T::ZERO {
            return Self::ZERO;
        }
        let two = T::from_f64(2.0);
        let m = self.abs();
        let t = ((m + self.re.abs()) / two).sqrt();
        // numerisch stabil: keine Auslöschung bei re < 0
        if self.re >= T::ZERO {
            Self {
                re: t,
                im: self.im / (two * t),
            }
        } else {
            let im = if self.im < T::ZERO { -t } else { t };
            Self {
                re: self.im.abs() / (two * t),
                im,
            }
        }
    }

    /// Exponentialfunktion `e^z`.
    pub fn exp(self) -> Self {
        Self::from_polar(self.re.exp(), self.im)
    }

    /// Natürlicher Logarithmus, Hauptwert (Imaginärteil in `(−π, π]`).
    pub fn ln(self) -> Self {
        Self {
            re: self.abs().ln(),
            im: self.arg(),
        }
    }

    /// Potenz mit reellem Exponenten, Hauptwert; `0^x = 0` für `x > 0`.
    pub fn powf(self, x: T) -> Self {
        if self.re == T::ZERO && self.im == T::ZERO {
            return if x > T::ZERO {
                Self::ZERO
            } else {
                Self::from_real(T::NAN)
            };
        }
        let (r, phi) = self.to_polar();
        Self::from_polar(r.powf(x), phi * x)
    }

    /// Sinus.
    pub fn sin(self) -> Self {
        Self {
            re: self.re.sin() * self.im.cosh(),
            im: self.re.cos() * self.im.sinh(),
        }
    }

    /// Kosinus.
    pub fn cos(self) -> Self {
        Self {
            re: self.re.cos() * self.im.cosh(),
            im: -(self.re.sin() * self.im.sinh()),
        }
    }

    /// Sinus hyperbolicus.
    pub fn sinh(self) -> Self {
        Self {
            re: self.re.sinh() * self.im.cos(),
            im: self.re.cosh() * self.im.sin(),
        }
    }

    /// Kosinus hyperbolicus.
    pub fn cosh(self) -> Self {
        Self {
            re: self.re.cosh() * self.im.cos(),
            im: self.re.sinh() * self.im.sin(),
        }
    }
}

impl<T: Real> From<T> for Complex<T> {
    fn from(re: T) -> Self {
        Self::from_real(re)
    }
}

impl<T: Real> Add for Complex<T> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
}
impl<T: Real> Sub for Complex<T> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self {
            re: self.re - o.re,
            im: self.im - o.im,
        }
    }
}
impl<T: Real> Mul for Complex<T> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
}
/// Division nach Smith: vermeidet Überlauf/Unterlauf in `re² + im²` des Nenners.
impl<T: Real> Div for Complex<T> {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        if o.re.abs() >= o.im.abs() {
            let r = o.im / o.re;
            let d = o.re + o.im * r;
            Self {
                re: (self.re + self.im * r) / d,
                im: (self.im - self.re * r) / d,
            }
        } else {
            let r = o.re / o.im;
            let d = o.re * r + o.im;
            Self {
                re: (self.re * r + self.im) / d,
                im: (self.im * r - self.re) / d,
            }
        }
    }
}
impl<T: Real> Neg for Complex<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl<T: Real> Add<T> for Complex<T> {
    type Output = Self;
    fn add(self, k: T) -> Self {
        Self {
            re: self.re + k,
            im: self.im,
        }
    }
}
impl<T: Real> Sub<T> for Complex<T> {
    type Output = Self;
    fn sub(self, k: T) -> Self {
        Self {
            re: self.re - k,
            im: self.im,
        }
    }
}
impl<T: Real> Mul<T> for Complex<T> {
    type Output = Self;
    fn mul(self, k: T) -> Self {
        self.scale(k)
    }
}
impl<T: Real> Div<T> for Complex<T> {
    type Output = Self;
    fn div(self, k: T) -> Self {
        Self {
            re: self.re / k,
            im: self.im / k,
        }
    }
}

impl<T: Real> AddAssign for Complex<T> {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}
impl<T: Real> SubAssign for Complex<T> {
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}
impl<T: Real> MulAssign for Complex<T> {
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}
impl<T: Real> DivAssign for Complex<T> {
    fn div_assign(&mut self, o: Self) {
        *self = *self / o;
    }
}
