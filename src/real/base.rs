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
    /// Not a Number.
    const NAN: Self;
    /// Positiv unendlich.
    const INFINITY: Self;
    /// Negativ unendlich.
    const NEG_INFINITY: Self;
    /// Die kleinste positive normalisierte Zahl.
    const MIN_POSITIVE: Self;

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
// Hilfsmakros für das Backend-Routing
//
// Priorität: std -> libm -> micromath. `micromath` ist eine bewusste Wahl für
// Geschwindigkeit vor Genauigkeit (Bit-Tricks, Fehler im Prozentbereich);
// wer genaue Ergebnisse braucht, nimmt `std` oder `libm`.
// ============================================================================

/// Erzeugt Methoden für `f32`, die an `std` -> `libm` -> `micromath` delegieren.
/// Schreibweise: `name(arg, ...) => libm_name;` (`self` ist implizit).
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
macro_rules! route_f32 {
    ($($name:ident($($arg:ident),*) => $libm:ident;)*) => {$(
        #[inline(always)]
        fn $name(self $(, $arg: Self)*) -> Self {
            #[cfg(feature = "std")]
            { std::primitive::f32::$name(self $(, $arg)*) }
            #[cfg(all(not(feature = "std"), feature = "libm"))]
            { libm::$libm(self $(, $arg)*) }
            #[cfg(all(not(feature = "std"), not(feature = "libm"), feature = "micromath"))]
            { micromath::F32Ext::$name(self $(, $arg)*) }
        }
    )*};
}

/// Wie `route_f32!`, aber nur für `std`/`libm`. Ohne diese Backends (also nur
/// `micromath`) greift die Standardimplementierung aus dem Trait.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
macro_rules! route_f32_precise_only {
    ($($name:ident($($arg:ident),*) => $libm:ident;)*) => {$(
        #[cfg(any(feature = "std", feature = "libm"))]
        #[inline(always)]
        fn $name(self $(, $arg: Self)*) -> Self {
            #[cfg(feature = "std")]
            { std::primitive::f32::$name(self $(, $arg)*) }
            #[cfg(all(not(feature = "std"), feature = "libm"))]
            { libm::$libm(self $(, $arg)*) }
        }
    )*};
}

/// Erzeugt Methoden für `f64`, die an `std` -> `libm` delegieren
/// (`micromath` unterstützt kein `f64`).
#[cfg(any(feature = "std", feature = "libm"))]
macro_rules! route_f64 {
    ($($name:ident($($arg:ident),*) => $libm:ident;)*) => {$(
        #[inline(always)]
        fn $name(self $(, $arg: Self)*) -> Self {
            #[cfg(feature = "std")]
            { std::primitive::f64::$name(self $(, $arg)*) }
            #[cfg(all(not(feature = "std"), feature = "libm"))]
            { libm::$libm(self $(, $arg)*) }
        }
    )*};
}

/// Methoden ohne Backend, die direkt auf den Bits bzw. den Primitiven arbeiten.
#[cfg(any(feature = "std", feature = "libm", feature = "micromath"))]
macro_rules! basic_float {
    ($t:ident, $bits:ident, $abs_mask:expr, $sign_mask:expr, $one_bits:expr) => {
        #[inline(always)]
        fn abs(self) -> Self {
            $t::from_bits(self.to_bits() & $abs_mask)
        }

        #[inline(always)]
        fn is_nan(self) -> bool {
            $t::is_nan(self)
        }

        #[inline(always)]
        fn is_infinite(self) -> bool {
            $t::is_infinite(self)
        }

        #[inline(always)]
        fn min(self, other: Self) -> Self {
            $t::min(self, other)
        }

        #[inline(always)]
        fn max(self, other: Self) -> Self {
            $t::max(self, other)
        }

        #[inline(always)]
        fn signum(self) -> Self {
            if self.is_nan() {
                self
            } else {
                $t::from_bits($one_bits | (self.to_bits() & $sign_mask))
            }
        }
    };
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
    const NAN: Self = f32::NAN;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
    const MIN_POSITIVE: Self = f32::MIN_POSITIVE;

    basic_float!(f32, u32, 0x7fff_ffff, 0x8000_0000, 0x3f80_0000_u32);

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v as f32
    }

    route_f32! {
        sqrt() => sqrtf;
        sin() => sinf;
        cos() => cosf;
        tan() => tanf;
        asin() => asinf;
        acos() => acosf;
        atan() => atanf;
        atan2(other) => atan2f;
        exp() => expf;
        ln() => logf;
        log10() => log10f;
        log2() => log2f;
        powf(n) => powf;
        floor() => floorf;
        ceil() => ceilf;
        round() => roundf;
        trunc() => truncf;
        hypot(other) => hypotf;
        mul_add(a, b) => fmaf;
    }

    // libm hat kein powi; dort wird per Potenzierung durch Quadrieren gerechnet.
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

    // Ohne std/libm greifen die Standardimplementierungen des Traits.
    route_f32_precise_only! {
        sinh() => sinhf;
        cosh() => coshf;
        tanh() => tanhf;
        cbrt() => cbrtf;
    }
}

// ============================================================================
// Implementierung für f64
// Hinweis: micromath unterstützt nur f32. Für f64 greifen std oder libm.
// ============================================================================
#[cfg(any(feature = "std", feature = "libm"))]
impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const PI: Self = core::f64::consts::PI;
    const E: Self = core::f64::consts::E;
    const EPSILON: Self = f64::EPSILON;
    const NAN: Self = f64::NAN;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
    const MIN_POSITIVE: Self = f64::MIN_POSITIVE;

    basic_float!(
        f64,
        u64,
        0x7fff_ffff_ffff_ffff,
        0x8000_0000_0000_0000,
        0x3ff0_0000_0000_0000_u64
    );

    #[inline(always)]
    fn from_f64(v: f64) -> Self {
        v
    }

    route_f64! {
        sqrt() => sqrt;
        sin() => sin;
        cos() => cos;
        tan() => tan;
        asin() => asin;
        acos() => acos;
        atan() => atan;
        atan2(other) => atan2;
        exp() => exp;
        ln() => log;
        log10() => log10;
        log2() => log2;
        powf(n) => pow;
        floor() => floor;
        ceil() => ceil;
        round() => round;
        trunc() => trunc;
        hypot(other) => hypot;
        mul_add(a, b) => fma;
        sinh() => sinh;
        cosh() => cosh;
        tanh() => tanh;
        cbrt() => cbrt;
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
