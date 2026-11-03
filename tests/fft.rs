//! Tests für die FFT.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::complex::Complex;
use azmath::fft::{
    FftError, fft, fft_freq, fft_radix2, fft_work_len, ifft, irfft, rfft, rfft_work_len,
};

type C = Complex<f64>;

/// Deterministische Pseudozufallszahlen in [-1, 1).
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

fn signal(n: usize, seed: u64) -> Vec<C> {
    let mut s = seed;
    (0..n).map(|_| C::new(lcg(&mut s), lcg(&mut s))).collect()
}

/// Direkte Summe `X_k = Σ x_j·e^{−2πi·jk/n}`.
fn naive_dft(x: &[C]) -> Vec<C> {
    let n = x.len();
    (0..n)
        .map(|k| {
            x.iter().enumerate().fold(C::ZERO, |acc, (j, &v)| {
                let angle = -2.0 * core::f64::consts::PI * ((j * k) % n) as f64 / n as f64;
                acc + v * C::from_polar(1.0, angle)
            })
        })
        .collect()
}

fn max_err(a: &[C], b: &[C]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn matches_the_naive_dft_for_all_kinds_of_lengths() {
    // Zweierpotenzen, Primzahlen, zusammengesetzte Zahlen
    for n in [
        1usize, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 20, 31, 32, 45, 64, 100,
    ] {
        let x = signal(n, n as u64 + 3);
        let want = naive_dft(&x);
        let mut got = x.clone();
        let mut work = vec![C::ZERO; fft_work_len(n)];
        fft(&mut got, &mut work).unwrap();
        let scale = want.iter().map(|z| z.abs()).fold(1.0, f64::max);
        assert!(
            max_err(&got, &want) < 1e-12 * scale * (n as f64).sqrt(),
            "n = {n}: Fehler {}",
            max_err(&got, &want)
        );
    }
}

#[test]
fn inverse_restores_the_signal() {
    for n in [1usize, 7, 8, 12, 30, 64, 97, 128] {
        let x = signal(n, 99 + n as u64);
        let mut y = x.clone();
        let mut work = vec![C::ZERO; fft_work_len(n)];
        fft(&mut y, &mut work).unwrap();
        ifft(&mut y, &mut work).unwrap();
        assert!(max_err(&x, &y) < 1e-13, "n = {n}");
    }
}

#[test]
fn known_transforms() {
    let n = 12;
    let mut work = vec![C::ZERO; fft_work_len(n)];
    // Impuls -> konstantes Spektrum
    let mut x = vec![C::ZERO; n];
    x[0] = C::ONE;
    fft(&mut x, &mut work).unwrap();
    assert!(x.iter().all(|z| (*z - C::ONE).abs() < 1e-14));
    // Konstante -> Impuls bei k = 0
    let mut c = vec![C::from_real(2.0); n];
    fft(&mut c, &mut work).unwrap();
    assert!((c[0] - C::from_real(24.0)).abs() < 1e-12);
    assert!(c[1..].iter().all(|z| z.abs() < 1e-12));
    // reiner Ton e^{2πi·3j/n} -> Impuls bei k = 3
    let mut t: Vec<C> = (0..n)
        .map(|j| C::from_polar(1.0, 2.0 * core::f64::consts::PI * 3.0 * j as f64 / n as f64))
        .collect();
    fft(&mut t, &mut work).unwrap();
    assert!((t[3] - C::from_real(n as f64)).abs() < 1e-11);
    for (k, z) in t.iter().enumerate() {
        if k != 3 {
            assert!(z.abs() < 1e-11, "k = {k}");
        }
    }
}

#[test]
fn parseval_and_linearity() {
    let n = 45;
    let (x, y) = (signal(n, 1), signal(n, 2));
    let mut work = vec![C::ZERO; fft_work_len(n)];
    let (mut fx, mut fy) = (x.clone(), y.clone());
    fft(&mut fx, &mut work).unwrap();
    fft(&mut fy, &mut work).unwrap();
    // Σ|x|² = (1/n)·Σ|X|²
    let e_time: f64 = x.iter().map(|z| z.norm_sqr()).sum();
    let e_freq: f64 = fx.iter().map(|z| z.norm_sqr()).sum::<f64>() / n as f64;
    assert!((e_time - e_freq).abs() < 1e-11 * e_time);
    // fft(a·x + b·y) = a·fft(x) + b·fft(y)
    let (a, b) = (C::new(0.7, -0.2), C::new(-1.3, 0.5));
    let mut combo: Vec<C> = x.iter().zip(&y).map(|(p, q)| a * *p + b * *q).collect();
    fft(&mut combo, &mut work).unwrap();
    let expected: Vec<C> = fx.iter().zip(&fy).map(|(p, q)| a * *p + b * *q).collect();
    assert!(max_err(&combo, &expected) < 1e-12);
}

#[test]
fn radix2_in_place_and_errors() {
    let x = signal(16, 5);
    let mut y = x.clone();
    fft_radix2(&mut y, false).unwrap();
    assert!(max_err(&y, &naive_dft(&x)) < 1e-13);
    let mut bad = signal(12, 1);
    assert_eq!(fft_radix2(&mut bad, false), Err(FftError::NotPowerOfTwo));
    let mut z = signal(12, 1);
    let mut short = vec![C::ZERO; fft_work_len(12) - 1];
    assert_eq!(fft(&mut z, &mut short), Err(FftError::InvalidLength));
    // Länge 0 und 1
    let mut empty: Vec<C> = Vec::new();
    assert_eq!(fft(&mut empty, &mut []), Ok(()));
    assert_eq!(fft_work_len(0) + fft_work_len(1) + fft_work_len(64), 0);
    assert!(fft_work_len(12) >= 2 * 23);
}

#[test]
fn real_transforms() {
    for n in [1usize, 2, 7, 8, 10, 33, 64] {
        let mut s = 7 + n as u64;
        let x: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
        let full: Vec<C> = x.iter().map(|&v| C::from_real(v)).collect();
        let want = naive_dft(&full);
        let mut spectrum = vec![C::ZERO; n / 2 + 1];
        let mut work = vec![C::ZERO; rfft_work_len(n)];
        rfft(&x, &mut spectrum, &mut work).unwrap();
        assert!(max_err(&spectrum, &want[..n / 2 + 1]) < 1e-12, "n = {n}");
        let mut back = vec![0.0; n];
        irfft(&spectrum, &mut back, &mut work).unwrap();
        for (a, b) in x.iter().zip(&back) {
            assert!((a - b).abs() < 1e-13, "n = {n}");
        }
    }
    let mut spec = vec![C::ZERO; 3];
    let mut work = vec![C::ZERO; rfft_work_len(8)];
    assert_eq!(
        rfft(&[0.0; 8], &mut spec, &mut work),
        Err(FftError::InvalidLength)
    );
}

#[test]
fn frequencies() {
    let mut f = [0.0; 8];
    fft_freq(8, 0.5, &mut f).unwrap();
    assert_eq!(f, [0.0, 0.25, 0.5, 0.75, -1.0, -0.75, -0.5, -0.25]);
    let mut g = [0.0f64; 5];
    fft_freq(5, 1.0, &mut g).unwrap();
    for (a, b) in g.iter().zip([0.0, 0.2, 0.4, -0.4, -0.2]) {
        assert!((a - b).abs() < 1e-15);
    }
    assert_eq!(
        fft_freq(4, 1.0, &mut [0.0; 3]),
        Err(FftError::InvalidLength)
    );
}

#[cfg(feature = "alloc")]
#[test]
fn convolution_matches_the_direct_sum() {
    use azmath::fft::convolve;
    let mut s = 4u64;
    for (la, lb) in [(1usize, 1usize), (3, 5), (8, 8), (17, 4), (30, 31)] {
        let a: Vec<f64> = (0..la).map(|_| lcg(&mut s)).collect();
        let b: Vec<f64> = (0..lb).map(|_| lcg(&mut s)).collect();
        let got = convolve(&a, &b);
        assert_eq!(got.len(), la + lb - 1);
        for k in 0..got.len() {
            let want: f64 = (0..la)
                .filter(|&i| k >= i && k - i < lb)
                .map(|i| a[i] * b[k - i])
                .sum();
            assert!((got[k] - want).abs() < 1e-12, "({la}, {lb}) Index {k}");
        }
    }
    assert!(convolve::<f64>(&[], &[1.0]).is_empty());
}

#[test]
fn works_for_f32() {
    type Cf = Complex<f32>;
    let n = 10;
    let x: Vec<Cf> = (0..n).map(|j| Cf::new((j as f32).sin(), 0.0)).collect();
    let mut y = x.clone();
    let mut work = vec![Cf::ZERO; fft_work_len(n)];
    fft(&mut y, &mut work).unwrap();
    ifft(&mut y, &mut work).unwrap();
    for (a, b) in x.iter().zip(&y) {
        assert!((*a - *b).abs() < 1e-5);
    }
}
