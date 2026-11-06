#![no_main]
#[path = "common.rs"]
mod common;

use azmath::complex::Complex;
use azmath::fft::{convolve, fft, fft_work_len, ifft, irfft, rfft, rfft_work_len};
use common::Bytes;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut r = Bytes(data);
    let n = r.range(0, 257); // Zweierpotenzen, Primzahlen und gemischte Längen
    let special = r.u8() % 8 == 0;
    let x: Vec<Complex<f64>> = (0..n)
        .map(|_| Complex::new(r.value(special), r.value(special)))
        .collect();

    // Beliebige (auch nichtendliche) Eingaben dürfen nicht paniken.
    let mut y = x.clone();
    let mut work = vec![Complex::ZERO; fft_work_len(n)];
    fft(&mut y, &mut work).unwrap();
    assert_eq!(y.len(), n);

    if x.iter().all(|z| z.re.is_finite() && z.im.is_finite()) && !special {
        // Roundtrip und Parseval für moderate Werte
        ifft(&mut y, &mut work).unwrap();
        let tol = 1e-10 * (n.max(2) as f64).log2();
        for (a, b) in x.iter().zip(&y) {
            assert!((*a - *b).abs() <= tol * 16.0, "roundtrip n={n}");
        }
        let mut f = x.clone();
        fft(&mut f, &mut work).unwrap();
        let e_t: f64 = x.iter().map(|z| z.abs() * z.abs()).sum();
        let e_f: f64 = f.iter().map(|z| z.abs() * z.abs()).sum::<f64>() / n.max(1) as f64;
        assert!((e_t - e_f).abs() <= 1e-9 * e_t.max(1.0), "parseval n={n}");
    }

    // zu kurzer Zwischenspeicher ist ein Fehler, kein Absturz
    if fft_work_len(n) > 0 {
        let mut short = vec![Complex::ZERO; fft_work_len(n) - 1];
        assert!(fft(&mut y.clone(), &mut short).is_err());
    }

    // reelle FFT
    let re: Vec<f64> = x.iter().map(|z| z.re).collect();
    if n > 0 {
        let mut spec = vec![Complex::ZERO; n / 2 + 1];
        let mut w = vec![Complex::ZERO; rfft_work_len(n)];
        rfft(&re, &mut spec, &mut w).unwrap();
        let mut back = vec![0.0; n];
        irfft(&spec, &mut back, &mut w).unwrap();
        if re.iter().all(|v| v.is_finite()) && !special {
            for (a, b) in re.iter().zip(&back) {
                assert!((a - b).abs() <= 1e-9 * 16.0, "rfft roundtrip n={n}");
            }
        }
        // falsche Spektrumlänge → Fehler
        let mut bad = vec![Complex::ZERO; n / 2 + 2];
        assert!(rfft(&re, &mut bad, &mut w).is_err());
    }

    // Faltung gegen direkte Summe
    let la = r.range(0, 40);
    let lb = r.range(0, 40);
    let a: Vec<f64> = (0..la).map(|_| r.moderate()).collect();
    let b: Vec<f64> = (0..lb).map(|_| r.moderate()).collect();
    let c = convolve(&a, &b);
    if la == 0 || lb == 0 {
        assert!(c.is_empty());
    } else {
        assert_eq!(c.len(), la + lb - 1);
        for (k, &v) in c.iter().enumerate() {
            let d: f64 = (0..la)
                .filter(|&i| k >= i && k - i < lb)
                .map(|i| a[i] * b[k - i])
                .sum();
            assert!(
                (v - d).abs() <= 1e-9 * (1.0 + d.abs()) * 64.0,
                "convolve k={k}"
            );
        }
    }
});
