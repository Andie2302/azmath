use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Universeller Trait für alle Fließkommatypen (f32, f64) in einer no_std Umgebung.
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
    // Konstanten
    const ZERO: Self;
    const ONE: Self;
    const PI: Self;
    const E: Self;
    const EPSILON: Self;

    // Basis-Methoden (direkt in core verfügbar oder bitweise lösbar)
    fn abs(self) -> Self;
    fn is_nan(self) -> bool;
    fn is_infinite(self) -> bool;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;

    // Transzendentale & höhere Methoden (Backend-abhängig)
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, other: Self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn powi(self, n: i32) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;

    /// Konvertierung von f64-Literalen für generische Formeln
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
