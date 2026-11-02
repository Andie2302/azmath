use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Universeller Trait für alle Fließkommatypen (`f32`, `f64`) in einer `no_std`-Umgebung.
///
/// Die Funktionen werden je nach aktivem Feature an ein Backend weitergeleitet
/// (Priorität: `std` → `libm` → `micromath`).
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::Real;
///
/// fn hypotenuse<T: Real>(a: T, b: T) -> T {
///     (a * a + b * b).sqrt()
/// }
///
/// assert_eq!(hypotenuse(3.0f32, 4.0f32), 5.0);
/// assert_eq!(hypotenuse(3.0f64, 4.0f64), 5.0);
/// # }
/// ```
pub trait Real:
    Copy
    + Clone
    + PartialEq
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + Sized
{
    // ---- Konstanten ----

    /// Die Zahl 0.
    const ZERO: Self;
    /// Die Zahl 1.
    const ONE: Self;
    /// Die Kreiszahl π.
    const PI: Self;
    /// Die Eulersche Zahl e.
    const E: Self;
    /// Der Abstand von 1.0 zur nächstgrößeren darstellbaren Zahl.
    const EPSILON: Self;

    // ---- Basis-Methoden (ohne Backend) ----

    /// Betrag (löscht das Vorzeichenbit).
    fn abs(self) -> Self;
    /// `true`, wenn der Wert NaN ist.
    fn is_nan(self) -> bool;
    /// `true`, wenn der Wert positiv oder negativ unendlich ist.
    fn is_infinite(self) -> bool;
    /// Das Minimum; ist ein Wert NaN, wird der andere zurückgegeben.
    fn min(self, other: Self) -> Self;
    /// Das Maximum; ist ein Wert NaN, wird der andere zurückgegeben.
    fn max(self, other: Self) -> Self;
    /// Vorzeichen: `1.0` für positive, `-1.0` für negative Werte (inkl. `±0.0`); NaN bleibt NaN.
    fn signum(self) -> Self;

    // ---- Transzendentale & höhere Methoden (Backend-abhängig) ----

    /// Quadratwurzel.
    fn sqrt(self) -> Self;
    /// Sinus (Bogenmaß).
    fn sin(self) -> Self;
    /// Kosinus (Bogenmaß).
    fn cos(self) -> Self;
    /// Tangens (Bogenmaß).
    fn tan(self) -> Self;
    /// Arkussinus.
    fn asin(self) -> Self;
    /// Arkuskosinus.
    fn acos(self) -> Self;
    /// Arkustangens.
    fn atan(self) -> Self;
    /// Vierquadranten-Arkustangens von `self / other`.
    fn atan2(self, other: Self) -> Self;
    /// e hoch `self`.
    fn exp(self) -> Self;
    /// Natürlicher Logarithmus.
    fn ln(self) -> Self;
    /// Logarithmus zur Basis 10.
    fn log10(self) -> Self;
    /// Logarithmus zur Basis 2.
    fn log2(self) -> Self;
    /// `self` hoch `n` (Fließkomma-Exponent).
    fn powf(self, n: Self) -> Self;
    /// `self` hoch `n` (ganzzahliger Exponent).
    fn powi(self, n: i32) -> Self;
    /// Größte ganze Zahl, die nicht größer als `self` ist.
    fn floor(self) -> Self;
    /// Kleinste ganze Zahl, die nicht kleiner als `self` ist.
    fn ceil(self) -> Self;
    /// Rundet zur nächsten ganzen Zahl; Hälften werden von 0 weg gerundet.
    fn round(self) -> Self;
    /// Schneidet die Nachkommastellen ab (Rundung Richtung 0).
    fn trunc(self) -> Self;
    /// Länge der Hypotenuse `sqrt(self² + other²)`.
    fn hypot(self, other: Self) -> Self;
    /// Fused multiply-add: `self * a + b`.
    fn mul_add(self, a: Self, b: Self) -> Self;

    // ---- Abgeleitete Methoden (mit Standardimplementierung) ----

    /// Hyperbelsinus.
    fn sinh(self) -> Self {
        (self.exp() - (-self).exp()) / Self::from_f64(2.0)
    }

    /// Hyperbelkosinus.
    fn cosh(self) -> Self {
        (self.exp() + (-self).exp()) / Self::from_f64(2.0)
    }

    /// Hyperbeltangens.
    fn tanh(self) -> Self {
        let a = (self + self).exp();
        (a - Self::ONE) / (a + Self::ONE)
    }

    /// Kubikwurzel (auch für negative Werte).
    fn cbrt(self) -> Self {
        if self == Self::ZERO || self.is_nan() || self.is_infinite() {
            return self;
        }
        let r = self.abs().powf(Self::ONE / Self::from_f64(3.0));
        if self < Self::ZERO { -r } else { r }
    }

    /// Nachkommaanteil: `self - self.trunc()`.
    fn fract(self) -> Self {
        self - self.trunc()
    }

    /// Kehrwert `1 / self`.
    fn recip(self) -> Self {
        Self::ONE / self
    }

    /// Begrenzt `self` auf den Bereich `[lo, hi]`.
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }

    /// Wandelt Grad in Bogenmaß um.
    fn to_radians(self) -> Self {
        self * Self::PI / Self::from_f64(180.0)
    }

    /// Wandelt Bogenmaß in Grad um.
    fn to_degrees(self) -> Self {
        self * Self::from_f64(180.0) / Self::PI
    }

    /// Konvertierung von f64-Literalen für generische Formeln.
    fn from_f64(v: f64) -> Self;
}

// ============================================================================
// Implementierung für f32
// ============================================================================
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const PI: Self = core::f32::consts::PI;
    const E: Self = core::f32::consts::E;
    const EPSILON: Self = f32::EPSILON;

    #[inline(always)]
    fn abs(self) -> Self {
        f32::from_bits(self.to_bits() & 0x7fff_ffff)
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }

    #[inline(always)]
    fn is_infinite(self) -> bool {
        f32::is_infinite(self)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f32::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f32::max(self, other)
    }

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }

    // --- Backend-Routing für f32 (Priorität: std -> libm -> micromath) ---

    #[inline(always)]
    fn sqrt(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::sqrt(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::sqrtf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::sqrt(self)
        }
    }

    #[inline(always)]
    fn sin(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::sin(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::sinf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::sin(self)
        }
    }

    #[inline(always)]
    fn cos(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::cos(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::cosf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::cos(self)
        }
    }

    #[inline(always)]
    fn tan(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::tan(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::tanf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::tan(self)
        }
    }

    #[inline(always)]
    fn asin(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::asin(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::asinf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::asin(self)
        }
    }

    #[inline(always)]
    fn acos(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::acos(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::acosf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::acos(self)
        }
    }

    #[inline(always)]
    fn atan(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::atan(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::atanf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::atan(self)
        }
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::atan2(self, other)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::atan2f(self, other)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::atan2(self, other)
        }
    }

    #[inline(always)]
    fn exp(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::exp(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::expf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::exp(self)
        }
    }

    #[inline(always)]
    fn ln(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::ln(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::logf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::ln(self)
        }
    }

    #[inline(always)]
    fn powf(self, n: Self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::powf(self, n)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::powf(self, n)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::powf(self, n)
        }
    }

    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::powi(self, n)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            exp_by_squaring(self, n)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::powi(self, n)
        }
    }

    #[inline(always)]
    fn floor(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::floor(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::floorf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::floor(self)
        }
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::ceil(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::ceilf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::ceil(self)
        }
    }

    #[inline(always)]
    fn signum(self) -> Self {
        if self.is_nan() {
            self
        } else {
            f32::from_bits(0x3f80_0000_u32 | (self.to_bits() & 0x8000_0000))
        }
    }

    #[inline(always)]
    fn round(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::round(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::roundf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::round(self)
        }
    }

    #[inline(always)]
    fn trunc(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::trunc(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::truncf(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::trunc(self)
        }
    }

    #[inline(always)]
    fn log10(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::log10(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::log10f(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::log10(self)
        }
    }

    #[inline(always)]
    fn log2(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::log2(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::log2f(self)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::log2(self)
        }
    }

    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::hypot(self, other)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::hypotf(self, other)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::hypot(self, other)
        }
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::mul_add(self, a, b)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::fmaf(self, a, b)
        }
        #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
        {
            micromath::F32Ext::mul_add(self, a, b)
        }
    }

    // Ohne std/libm greift die Standardimplementierung des Traits.
    #[cfg(any(feature = "std", feature = "libm"))]
    #[inline(always)]
    fn sinh(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::sinh(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::sinhf(self)
        }
    }

    // Ohne std/libm greift die Standardimplementierung des Traits.
    #[cfg(any(feature = "std", feature = "libm"))]
    #[inline(always)]
    fn cosh(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::cosh(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::coshf(self)
        }
    }

    // Ohne std/libm greift die Standardimplementierung des Traits.
    #[cfg(any(feature = "std", feature = "libm"))]
    #[inline(always)]
    fn tanh(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::tanh(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::tanhf(self)
        }
    }

    // Ohne std/libm greift die Standardimplementierung des Traits.
    #[cfg(any(feature = "std", feature = "libm"))]
    #[inline(always)]
    fn cbrt(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f32::cbrt(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::cbrtf(self)
        }
    }
}

// ============================================================================
// Implementierung für f64
// Hinweis: micromath unterstützt primär f32. Für f64 greifen std oder libm.
// ============================================================================
#[cfg(any(feature = "std", feature = "libm"))]
impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const PI: Self = core::f64::consts::PI;
    const E: Self = core::f64::consts::E;
    const EPSILON: Self = f64::EPSILON;

    #[inline(always)]
    fn abs(self) -> Self {
        f64::from_bits(self.to_bits() & 0x7fff_ffff_ffff_ffff)
    }

    #[inline(always)]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }

    #[inline(always)]
    fn is_infinite(self) -> bool {
        f64::is_infinite(self)
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        f64::min(self, other)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        f64::max(self, other)
    }

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::sqrt(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::sqrt(self)
        }
    }

    #[inline(always)]
    fn sin(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::sin(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::sin(self)
        }
    }

    #[inline(always)]
    fn cos(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::cos(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::cos(self)
        }
    }

    #[inline(always)]
    fn tan(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::tan(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::tan(self)
        }
    }

    #[inline(always)]
    fn asin(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::asin(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::asin(self)
        }
    }

    #[inline(always)]
    fn acos(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::acos(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::acos(self)
        }
    }

    #[inline(always)]
    fn atan(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::atan(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::atan(self)
        }
    }

    #[inline(always)]
    fn atan2(self, other: Self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::atan2(self, other)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::atan2(self, other)
        }
    }

    #[inline(always)]
    fn exp(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::exp(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::exp(self)
        }
    }

    #[inline(always)]
    fn ln(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::ln(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::log(self)
        }
    }

    #[inline(always)]
    fn powf(self, n: Self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::powf(self, n)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::pow(self, n)
        }
    }

    #[inline(always)]
    fn powi(self, n: i32) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::powi(self, n)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            exp_by_squaring(self, n)
        }
    }

    #[inline(always)]
    fn floor(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::floor(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::floor(self)
        }
    }

    #[inline(always)]
    fn ceil(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::ceil(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::ceil(self)
        }
    }

    #[inline(always)]
    fn signum(self) -> Self {
        if self.is_nan() {
            self
        } else {
            f64::from_bits(0x3ff0_0000_0000_0000_u64 | (self.to_bits() & 0x8000_0000_0000_0000))
        }
    }

    #[inline(always)]
    fn round(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::round(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::round(self)
        }
    }

    #[inline(always)]
    fn trunc(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::trunc(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::trunc(self)
        }
    }

    #[inline(always)]
    fn log10(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::log10(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::log10(self)
        }
    }

    #[inline(always)]
    fn log2(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::log2(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::log2(self)
        }
    }

    #[inline(always)]
    fn hypot(self, other: Self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::hypot(self, other)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::hypot(self, other)
        }
    }

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::mul_add(self, a, b)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::fma(self, a, b)
        }
    }

    // Ohne std/libm greift die Standardimplementierung des Traits.
    #[cfg(any(feature = "std", feature = "libm"))]
    #[inline(always)]
    fn sinh(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::sinh(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::sinh(self)
        }
    }

    // Ohne std/libm greift die Standardimplementierung des Traits.
    #[cfg(any(feature = "std", feature = "libm"))]
    #[inline(always)]
    fn cosh(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::cosh(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::cosh(self)
        }
    }

    // Ohne std/libm greift die Standardimplementierung des Traits.
    #[cfg(any(feature = "std", feature = "libm"))]
    #[inline(always)]
    fn tanh(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::tanh(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::tanh(self)
        }
    }

    // Ohne std/libm greift die Standardimplementierung des Traits.
    #[cfg(any(feature = "std", feature = "libm"))]
    #[inline(always)]
    fn cbrt(self) -> Self {
        #[cfg(feature = "std")]
        {
            std::primitive::f64::cbrt(self)
        }
        #[cfg(all(not(feature = "std"), feature = "libm"))]
        {
            libm::cbrt(self)
        }
    }
}

/// Schnelle ganzzahlige Potenzierung (Fallback für libm, da libm kein direktes powi hat)
#[cfg(all(not(feature = "std"), feature = "libm"))]
#[inline]
fn exp_by_squaring<T: Real>(mut base: T, mut exp: i32) -> T {
    if exp == 0 {
        return T::ONE;
    }
    if exp < 0 {
        base = T::ONE / base;
        exp = exp.wrapping_neg();
    }
    let mut acc = T::ONE;
    let mut uexp = exp as u32;
    while uexp > 1 {
        if (uexp & 1) == 1 {
            acc = acc * base;
        }
        base = base * base;
        uexp >>= 1;
    }
    acc * base
}
