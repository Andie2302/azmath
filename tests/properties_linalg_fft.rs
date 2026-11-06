//! Eigenschaftstests mit festem Seed für `linalg` und `fft`.
//!
//! Statt einzelner Beispiele werden viele erzeugte Eingaben gegen Invarianten geprüft
//! (Rückwärtsfehler von LU-Lösungen, `Q·R = A`, Parseval, Faltungssatz, FFT-Roundtrip).
//! Bei einem Fehler nennt die Meldung Seed-Index, Größe und Abweichung.
#![cfg(all(feature = "alloc", any(feature = "std", feature = "libm")))]

use azmath::complex::Complex;
use azmath::fft::{convolve, fft, fft_work_len, ifft, irfft, rfft, rfft_work_len};
use azmath::linalg::{DynLu, DynMatrix, DynQr, DynVector};

type C = Complex<f64>;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn size(&mut self, lo: usize, hi: usize) -> usize {
        lo + (((self.next() + 1.0) * 0.5 * (hi - lo + 1) as f64) as usize).min(hi - lo)
    }
}

fn random_matrix(rng: &mut Rng, m: usize, n: usize) -> DynMatrix<f64> {
    DynMatrix::new(m, n, (0..m * n).map(|_| rng.next()).collect()).unwrap()
}

fn max_abs_diff(a: &DynMatrix<f64>, b: &DynMatrix<f64>) -> f64 {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn lu_reproduces_permuted_matrix() {
    let mut rng = Rng(1);
    for case in 0..200 {
        let n = rng.size(1, 12);
        let a = random_matrix(&mut rng, n, n);
        let Ok(lu) = DynLu::new(&a) else { continue };
        // Rückwärtsfehler: A·solve(b) = b.
        let b = DynVector::new((0..n).map(|_| rng.next()).collect());
        let x = lu.solve(&b).unwrap();
        let r = a.mul_vec(&x).unwrap();
        let scale = a.max_abs() * x.max_abs() * n as f64;
        let err = r
            .as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(err <= 1e-13 * scale.max(1.0), "case {case} n={n}: {err}");
        // Permutation ist eine echte Permutation.
        let mut seen = vec![false; n];
        for &p in lu.permutation() {
            assert!(!seen[p], "case {case}: doppelter Index");
            seen[p] = true;
        }
        // A·A⁻¹ = I bis auf Konditionierung.
        let inv = lu.inverse();
        let prod = a.matmul(&inv).unwrap();
        let cond_scale = a.max_abs() * inv.max_abs() * n as f64;
        assert!(
            max_abs_diff(&prod, &DynMatrix::identity(n)) <= 1e-13 * cond_scale.max(1.0),
            "case {case} n={n}: Inverse"
        );
    }
}

#[test]
fn qr_reproduces_matrix_and_q_is_orthogonal() {
    let mut rng = Rng(2);
    for case in 0..200 {
        let n = rng.size(1, 8);
        let m = n + rng.size(0, 6);
        let a = random_matrix(&mut rng, m, n);
        let Ok(qr) = DynQr::new(&a) else { continue };
        let (q, r) = (qr.thin_q(), qr.r());
        let back = q.matmul(&r).unwrap();
        let scale = a.max_abs().max(1.0) * m as f64;
        assert!(
            max_abs_diff(&back, &a) <= 1e-13 * scale,
            "case {case} {m}x{n}: Q·R != A"
        );
        let qtq = q.transpose().matmul(&q).unwrap();
        assert!(
            max_abs_diff(&qtq, &DynMatrix::identity(n)) <= 1e-13 * m as f64,
            "case {case} {m}x{n}: Q nicht orthonormal"
        );
        // R ist obere Dreiecksmatrix.
        for i in 0..n {
            for j in 0..i {
                assert_eq!(r[(i, j)], 0.0, "case {case}: R nicht oberes Dreieck");
            }
        }
    }
}

#[test]
fn least_squares_residual_is_orthogonal_to_columns() {
    let mut rng = Rng(3);
    for case in 0..200 {
        let n = rng.size(1, 6);
        let m = n + rng.size(1, 8);
        let a = random_matrix(&mut rng, m, n);
        let b = DynVector::new((0..m).map(|_| rng.next()).collect());
        let Ok(x) = a.least_squares(&b) else { continue };
        let ax = a.mul_vec(&x).unwrap();
        let res: Vec<f64> = ax
            .as_slice()
            .iter()
            .zip(b.as_slice())
            .map(|(p, q)| p - q)
            .collect();
        let g = a.transpose().mul_vec(&DynVector::new(res)).unwrap();
        assert!(
            g.max_abs() <= 1e-11 * (m as f64) * a.max_abs().max(1.0),
            "case {case} {m}x{n}: Aᵀr = {}",
            g.max_abs()
        );
    }
}

fn signal(rng: &mut Rng, n: usize) -> Vec<C> {
    (0..n).map(|_| C::new(rng.next(), rng.next())).collect()
}

fn max_err(a: &[C], b: &[C]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn fft_roundtrip_and_parseval_for_arbitrary_lengths() {
    let mut rng = Rng(4);
    for case in 0..150 {
        let n = rng.size(1, 300); // auch Primzahlen und ungerade Längen (Bluestein)
        let x = signal(&mut rng, n);
        let mut y = x.clone();
        let mut work = vec![C::ZERO; fft_work_len(n)];
        fft(&mut y, &mut work).unwrap();
        let e_time: f64 = x.iter().map(|z| z.abs() * z.abs()).sum();
        let e_freq: f64 = y.iter().map(|z| z.abs() * z.abs()).sum::<f64>() / n as f64;
        assert!(
            (e_time - e_freq).abs() <= 1e-12 * e_time.max(1.0) * (n as f64).log2().max(1.0),
            "case {case} n={n}: Parseval {e_time} vs {e_freq}"
        );
        ifft(&mut y, &mut work).unwrap();
        assert!(
            max_err(&x, &y) <= 1e-12 * (n as f64).log2().max(1.0),
            "case {case} n={n}: Roundtrip"
        );
    }
}

#[test]
fn fft_is_linear() {
    let mut rng = Rng(5);
    for case in 0..60 {
        let n = rng.size(1, 100);
        let (a, b) = (signal(&mut rng, n), signal(&mut rng, n));
        let (alpha, beta) = (rng.next(), rng.next());
        let mut work = vec![C::ZERO; fft_work_len(n)];
        let mut fa = a.clone();
        let mut fb = b.clone();
        fft(&mut fa, &mut work).unwrap();
        fft(&mut fb, &mut work).unwrap();
        let mut mix: Vec<C> = a
            .iter()
            .zip(&b)
            .map(|(x, y)| x.scale(alpha) + y.scale(beta))
            .collect();
        fft(&mut mix, &mut work).unwrap();
        let expect: Vec<C> = fa
            .iter()
            .zip(&fb)
            .map(|(x, y)| x.scale(alpha) + y.scale(beta))
            .collect();
        assert!(
            max_err(&mix, &expect) <= 1e-12 * n as f64,
            "case {case} n={n}"
        );
    }
}

#[test]
fn real_fft_roundtrip_and_convolution_theorem() {
    let mut rng = Rng(6);
    for case in 0..100 {
        let n = rng.size(1, 200);
        let x: Vec<f64> = (0..n).map(|_| rng.next()).collect();
        let mut spec = vec![C::ZERO; n / 2 + 1];
        let mut work = vec![C::ZERO; rfft_work_len(n)];
        rfft(&x, &mut spec, &mut work).unwrap();
        let mut back = vec![0.0; n];
        irfft(&spec, &mut back, &mut work).unwrap();
        let err = x
            .iter()
            .zip(&back)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0, f64::max);
        assert!(err <= 1e-12, "case {case} n={n}: rfft-Roundtrip {err}");
    }
    for case in 0..100 {
        let (la, lb) = (rng.size(1, 40), rng.size(1, 40));
        let a: Vec<f64> = (0..la).map(|_| rng.next()).collect();
        let b: Vec<f64> = (0..lb).map(|_| rng.next()).collect();
        let fast = convolve(&a, &b);
        assert_eq!(fast.len(), la + lb - 1);
        for (k, &v) in fast.iter().enumerate() {
            let direct: f64 = (0..la)
                .filter(|&i| k >= i && k - i < lb)
                .map(|i| a[i] * b[k - i])
                .sum();
            assert!(
                (v - direct).abs() <= 1e-12 * (la.max(lb) as f64),
                "case {case} k={k}"
            );
        }
    }
}
