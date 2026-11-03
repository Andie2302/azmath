//! Schnelle Fourier-Transformation (FFT) für beliebige Längen, ohne Allokator.
//!
//! | Funktion | Zweck |
//! |---|---|
//! | [`fft_radix2`] | in-place, Länge muss eine Zweierpotenz sein, **kein** Zusatzspeicher |
//! | [`fft`], [`ifft`] | beliebige Länge `n` (Radix-2 oder Bluestein-Algorithmus), Zusatzspeicher [`fft_work_len`]`(n)` |
//! | [`rfft`], [`irfft`] | reelle Eingabe: nur die `n/2 + 1` unabhängigen Spektralwerte |
//! | [`convolve`] | lineare Faltung zweier reeller Folgen (hinter `alloc`) |
//! | [`fft_freq`] | Frequenzen der Bins |
//!
//! **Konvention:** `X_k = Σ_j x_j·e^{−2πi·jk/n}` (Vorwärts, **unnormiert**); die Rücktransformation [`ifft`] verwendet das
//! Vorzeichen `+` und teilt durch `n`, sodass `ifft(fft(x)) = x` gilt. Bluestein-Transformationen kosten `O(n log n)` auch für
//! Primzahllängen, brauchen aber Zwischenspeicher der Größe etwa `4n`.
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::complex::Complex;
//! use azmath::fft::{fft, ifft, fft_work_len};
//!
//! // Ein Impuls hat ein konstantes Spektrum
//! let mut x = [Complex::new(0.0f64, 0.0); 6];
//! x[0] = Complex::new(1.0, 0.0);
//! let mut work = [Complex::new(0.0, 0.0); fft_work_len(6)];
//! fft(&mut x, &mut work).unwrap();
//! assert!(x.iter().all(|z| (z.re - 1.0).abs() < 1e-14 && z.im.abs() < 1e-14));
//! ifft(&mut x, &mut work).unwrap();
//! assert!((x[0].re - 1.0).abs() < 1e-14);
//! # }
//! ```

use crate::Real;
use crate::complex::Complex;
use core::fmt;

/// Fehler der FFT-Funktionen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FftError {
    /// [`fft_radix2`] verlangt eine Zweierpotenz als Länge.
    NotPowerOfTwo,
    /// Der Zwischenspeicher ist kürzer als gefordert, oder Ein-/Ausgabelängen passen nicht zusammen.
    InvalidLength,
}

impl fmt::Display for FftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            FftError::NotPowerOfTwo => "Länge ist keine Zweierpotenz",
            FftError::InvalidLength => "Längen passen nicht zusammen oder Zwischenspeicher zu kurz",
        })
    }
}

impl core::error::Error for FftError {}

const fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p *= 2;
    }
    p
}

/// Benötigter Zwischenspeicher (in `Complex`-Elementen) für [`fft`] und [`ifft`] bei Länge `n`: 0 für Zweierpotenzen,
/// sonst `2·m` mit der nächsten Zweierpotenz `m ≥ 2n − 1` (Bluestein).
pub const fn fft_work_len(n: usize) -> usize {
    if n <= 1 || n.is_power_of_two() {
        0
    } else {
        2 * next_pow2(2 * n - 1)
    }
}

/// Benötigter Zwischenspeicher für [`rfft`] und [`irfft`] bei Länge `n`: `n` plus [`fft_work_len`]`(n)` Elemente.
pub const fn rfft_work_len(n: usize) -> usize {
    n + fft_work_len(n)
}

/// In-place-FFT für Längen, die Zweierpotenzen sind (Cooley-Tukey, iterativ mit Bitumkehr). `inverse = true` rechnet mit
/// positivem Vorzeichen im Exponenten, **ohne** Division durch `n`.
///
/// Fehler: [`FftError::NotPowerOfTwo`]. Länge 0 und 1 sind erlaubt.
pub fn fft_radix2<T: Real>(data: &mut [Complex<T>], inverse: bool) -> Result<(), FftError> {
    let n = data.len();
    if n <= 1 {
        return Ok(());
    }
    if !n.is_power_of_two() {
        return Err(FftError::NotPowerOfTwo);
    }
    // Bitumkehr
    let bits = n.trailing_zeros();
    for i in 0..n {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j > i {
            data.swap(i, j);
        }
    }
    let two_pi = T::from_f64(2.0) * T::PI;
    let sign = if inverse { T::ONE } else { -T::ONE };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = sign * two_pi / T::from_usize(len);
        for k in 0..half {
            let w = Complex::from_polar(T::ONE, step * T::from_usize(k));
            let mut start = 0;
            while start < n {
                let (a, b) = (start + k, start + k + half);
                let t = data[b] * w;
                data[b] = data[a] - t;
                data[a] += t;
                start += len;
            }
        }
        len *= 2;
    }
    Ok(())
}

/// Kern für beliebige Länge (Vorwärtsrichtung, unnormiert).
fn forward<T: Real>(data: &mut [Complex<T>], work: &mut [Complex<T>]) -> Result<(), FftError> {
    let n = data.len();
    if n <= 1 {
        return Ok(());
    }
    if n.is_power_of_two() {
        return fft_radix2(data, false);
    }
    let m = next_pow2(2 * n - 1);
    if work.len() < 2 * m {
        return Err(FftError::InvalidLength);
    }
    let (a, rest) = work.split_at_mut(m);
    let b = &mut rest[..m];
    let pi = T::PI;
    let two_n = 2 * n;
    // Chirp w_j = e^{−iπ j²/n}; j² wird modulo 2n reduziert, damit das Argument klein bleibt
    let chirp = |j: usize| -> Complex<T> {
        let r = (j % two_n) * (j % two_n) % two_n;
        Complex::from_polar(T::ONE, -pi * T::from_usize(r) / T::from_usize(n))
    };
    a.fill(Complex::ZERO);
    b.fill(Complex::ZERO);
    for j in 0..n {
        a[j] = data[j] * chirp(j);
    }
    b[0] = chirp(0).conj();
    for j in 1..n {
        let c = chirp(j).conj();
        b[j] = c;
        b[m - j] = c;
    }
    fft_radix2(a, false)?;
    fft_radix2(b, false)?;
    for k in 0..m {
        a[k] *= b[k];
    }
    fft_radix2(a, true)?;
    let scale = T::ONE / T::from_usize(m);
    for k in 0..n {
        data[k] = (a[k] * chirp(k)).scale(scale);
    }
    Ok(())
}

/// Vorwärts-FFT beliebiger Länge `n = data.len()` (in-place, unnormiert). `work`: mindestens [`fft_work_len`]`(n)` Elemente.
///
/// Fehler: [`FftError::InvalidLength`] bei zu kurzem Zwischenspeicher.
pub fn fft<T: Real>(data: &mut [Complex<T>], work: &mut [Complex<T>]) -> Result<(), FftError> {
    forward(data, work)
}

/// Rücktransformation beliebiger Länge (in-place), **normiert** mit `1/n`: `ifft(fft(x)) = x`. Parameter wie bei [`fft`].
pub fn ifft<T: Real>(data: &mut [Complex<T>], work: &mut [Complex<T>]) -> Result<(), FftError> {
    // ifft(x) = conj(fft(conj(x)))/n
    for z in data.iter_mut() {
        *z = z.conj();
    }
    forward(data, work)?;
    let scale = T::ONE / T::from_usize(data.len().max(1));
    for z in data.iter_mut() {
        *z = z.conj().scale(scale);
    }
    Ok(())
}

/// FFT einer **reellen** Folge `x` (Länge `n`): schreibt die `n/2 + 1` unabhängigen Werte `X₀ … X_{n/2}` nach `spectrum`
/// (die übrigen folgen aus `X_{n−k} = conj(X_k)`). `work`: mindestens [`rfft_work_len`]`(n)` Elemente.
///
/// Fehler: [`FftError::InvalidLength`].
pub fn rfft<T: Real>(
    x: &[T],
    spectrum: &mut [Complex<T>],
    work: &mut [Complex<T>],
) -> Result<(), FftError> {
    let n = x.len();
    if spectrum.len() != n / 2 + 1 && n > 0 || work.len() < rfft_work_len(n) {
        return Err(FftError::InvalidLength);
    }
    if n == 0 {
        return Ok(());
    }
    let (full, rest) = work.split_at_mut(n);
    for (z, &v) in full.iter_mut().zip(x) {
        *z = Complex::from_real(v);
    }
    forward(full, rest)?;
    spectrum.copy_from_slice(&full[..n / 2 + 1]);
    Ok(())
}

/// Umkehrung von [`rfft`]: aus den `n/2 + 1` Spektralwerten die reelle Folge `out` (Länge `n`), normiert.
/// `work`: mindestens [`rfft_work_len`]`(n)` Elemente.
///
/// Fehler: [`FftError::InvalidLength`].
pub fn irfft<T: Real>(
    spectrum: &[Complex<T>],
    out: &mut [T],
    work: &mut [Complex<T>],
) -> Result<(), FftError> {
    let n = out.len();
    if n == 0 {
        return Ok(());
    }
    if spectrum.len() != n / 2 + 1 || work.len() < rfft_work_len(n) {
        return Err(FftError::InvalidLength);
    }
    let (full, rest) = work.split_at_mut(n);
    full[..=n / 2].copy_from_slice(&spectrum[..=n / 2]);
    for k in (n / 2 + 1)..n {
        full[k] = spectrum[n - k].conj();
    }
    ifft(full, rest)?;
    for (o, z) in out.iter_mut().zip(full.iter()) {
        *o = z.re;
    }
    Ok(())
}

/// Frequenz des Bins `k` einer Folge der Länge `n` mit Abtastabstand `d`: `k/(n·d)` für `k < n/2`-Hälfte, sonst negativ
/// (wie `numpy.fft.fftfreq`). Schreibt alle `n` Werte nach `out`.
pub fn fft_freq<T: Real>(n: usize, d: T, out: &mut [T]) -> Result<(), FftError> {
    if out.len() != n {
        return Err(FftError::InvalidLength);
    }
    let denom = T::from_usize(n) * d;
    for (k, o) in out.iter_mut().enumerate() {
        let signed = if k < n.div_ceil(2) {
            k as f64
        } else {
            k as f64 - n as f64
        };
        *o = T::from_f64(signed) / denom;
    }
    Ok(())
}

/// Lineare Faltung `out = a ∗ b` zweier reeller Folgen über die FFT (`out` hat Länge `a.len() + b.len() − 1`; leer, wenn
/// eine Eingabe leer ist). Aufwand `O(n log n)`; für kurze Folgen ist die direkte Summe schneller. Braucht `alloc`.
#[cfg(feature = "alloc")]
pub fn convolve<T: Real>(a: &[T], b: &[T]) -> alloc::vec::Vec<T> {
    use alloc::vec;
    if a.is_empty() || b.is_empty() {
        return alloc::vec::Vec::new();
    }
    let len = a.len() + b.len() - 1;
    let m = next_pow2(len);
    let mut fa = vec![Complex::ZERO; m];
    let mut fb = vec![Complex::ZERO; m];
    for (z, &v) in fa.iter_mut().zip(a) {
        *z = Complex::from_real(v);
    }
    for (z, &v) in fb.iter_mut().zip(b) {
        *z = Complex::from_real(v);
    }
    // Zweierpotenz-Längen brauchen keinen Zwischenspeicher
    let _ = fft_radix2(&mut fa, false);
    let _ = fft_radix2(&mut fb, false);
    for k in 0..m {
        fa[k] *= fb[k];
    }
    let _ = fft_radix2(&mut fa, true);
    let scale = T::ONE / T::from_usize(m);
    fa[..len].iter().map(|z| z.re * scale).collect()
}
