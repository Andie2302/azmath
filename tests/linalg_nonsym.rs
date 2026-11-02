//! Tests für die Eigenwerte nichtsymmetrischer Matrizen (Hessenberg-Form, QR-Algorithmus).
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::complex::Complex;
use azmath::linalg::{LinalgError, Matrix};

type C = Complex<f64>;

/// Deterministische Pseudozufallszahlen in [-1, 1).
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// Jede erwartete Zahl muss genau einem gefundenen Eigenwert (innerhalb `tol`) zugeordnet werden können.
fn assert_same_spectrum(found: &[C], expected: &[C], tol: f64) {
    assert_eq!(found.len(), expected.len());
    let mut used = vec![false; found.len()];
    for e in expected {
        let best = (0..found.len())
            .filter(|&i| !used[i])
            .min_by(|&a, &b| (found[a] - *e).abs().total_cmp(&(found[b] - *e).abs()))
            .unwrap();
        assert!(
            (found[best] - *e).abs() <= tol,
            "{e:?} nicht gefunden (nächster: {:?}); alle: {found:?}",
            found[best]
        );
        used[best] = true;
    }
}

fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}

#[test]
fn rotation_has_eigenvalues_plus_minus_i() {
    let r = Matrix::new([[0.0f64, -1.0], [1.0, 0.0]]);
    assert_same_spectrum(
        &r.eigenvalues().unwrap(),
        &[c(0.0, 1.0), c(0.0, -1.0)],
        1e-14,
    );
    // beliebiger Winkel: e^{±iθ}
    let t = 0.7f64;
    let r = Matrix::new([[t.cos(), -t.sin()], [t.sin(), t.cos()]]);
    assert_same_spectrum(
        &r.eigenvalues().unwrap(),
        &[c(t.cos(), t.sin()), c(t.cos(), -t.sin())],
        1e-14,
    );
}

#[test]
fn triangular_matrix_has_its_diagonal_as_spectrum() {
    let a = Matrix::new([
        [3.0f64, 1.0, 4.0, 1.0],
        [0.0, -2.0, 7.0, 1.0],
        [0.0, 0.0, 5.0, 9.0],
        [0.0, 0.0, 0.0, 0.5],
    ]);
    assert_same_spectrum(
        &a.eigenvalues().unwrap(),
        &[c(3.0, 0.0), c(-2.0, 0.0), c(5.0, 0.0), c(0.5, 0.0)],
        1e-12,
    );
}

#[test]
fn block_diagonal_with_known_complex_pairs() {
    // zwei 2×2-Blöcke [[a, −b], [b, a]] mit Eigenwerten a ± ib, plus reeller Eigenwert 7; danach ähnlich vermischt
    let mut a = Matrix::<f64, 5, 5>::zeros();
    for (k, (re, im)) in [(1.0, 2.0), (-3.0, 0.5)].into_iter().enumerate() {
        let o = 2 * k;
        a[(o, o)] = re;
        a[(o, o + 1)] = -im;
        a[(o + 1, o)] = im;
        a[(o + 1, o + 1)] = re;
    }
    a[(4, 4)] = 7.0;
    let expected = [
        c(1.0, 2.0),
        c(1.0, -2.0),
        c(-3.0, 0.5),
        c(-3.0, -0.5),
        c(7.0, 0.0),
    ];
    assert_same_spectrum(&a.eigenvalues().unwrap(), &expected, 1e-12);

    // Ähnlichkeitstransformation S·A·S⁻¹ ändert das Spektrum nicht
    let mut s = 11u64;
    let p = Matrix::<f64, 5, 5>::from_fn(|i, j| lcg(&mut s) + if i == j { 3.0 } else { 0.0 });
    let b = p * a * p.inverse().unwrap();
    assert_same_spectrum(&b.eigenvalues().unwrap(), &expected, 1e-9);
}

#[test]
fn trace_and_determinant_match_sum_and_product() {
    let mut s = 5u64;
    for n_case in 0..5 {
        let a = Matrix::<f64, 6, 6>::from_fn(|_, _| lcg(&mut s));
        let ev = a.eigenvalues().unwrap();
        let sum = ev.iter().fold(C::ZERO, |acc, &z| acc + z);
        let prod = ev.iter().fold(C::ONE, |acc, &z| acc * z);
        assert!(
            (sum.re - a.trace()).abs() < 1e-11 && sum.im.abs() < 1e-11,
            "Fall {n_case}"
        );
        assert!(
            (prod.re - a.determinant()).abs() < 1e-10 && prod.im.abs() < 1e-10,
            "Fall {n_case}"
        );
    }
}

#[test]
fn complex_eigenvalues_come_in_conjugate_pairs() {
    let mut s = 21u64;
    let a = Matrix::<f64, 7, 7>::from_fn(|_, _| lcg(&mut s));
    let ev = a.eigenvalues().unwrap();
    for z in &ev {
        if z.im.abs() > 1e-12 {
            assert!(
                ev.iter().any(|w| (*w - z.conj()).abs() < 1e-10),
                "kein Partner zu {z:?}"
            );
        }
    }
}

#[test]
fn symmetric_matrices_agree_with_the_jacobi_method() {
    let mut s = 3u64;
    let base = Matrix::<f64, 5, 5>::from_fn(|_, _| lcg(&mut s));
    let sym = base + base.transpose();
    let mut jac = sym.symmetric_eigen().unwrap().values.as_slice().to_vec();
    jac.sort_by(|a, b| a.total_cmp(b));
    let mut qr: Vec<f64> = sym.eigenvalues().unwrap().iter().map(|z| z.re).collect();
    qr.sort_by(|a, b| a.total_cmp(b));
    for (a, b) in jac.iter().zip(&qr) {
        assert!((a - b).abs() < 1e-11, "{a} gegen {b}");
    }
    assert!(
        sym.eigenvalues()
            .unwrap()
            .iter()
            .all(|z| z.im.abs() < 1e-12)
    );
}

#[test]
fn defective_and_degenerate_cases() {
    // Jordan-Block: dreifacher Eigenwert 2, Genauigkeit nur etwa ε^(1/3)
    let j = Matrix::new([[2.0f64, 1.0, 0.0], [0.0, 2.0, 1.0], [0.0, 0.0, 2.0]]);
    assert_same_spectrum(&j.eigenvalues().unwrap(), &[c(2.0, 0.0); 3], 1e-4);
    // Nullmatrix und Einheitsmatrix
    assert_same_spectrum(
        &Matrix::<f64, 4, 4>::zeros().eigenvalues().unwrap(),
        &[C::ZERO; 4],
        0.0,
    );
    assert_same_spectrum(
        &Matrix::<f64, 3, 3>::identity().eigenvalues().unwrap(),
        &[C::ONE; 3],
        1e-14,
    );
    // 1×1
    assert_eq!(
        Matrix::new([[4.5f64]]).eigenvalues().unwrap(),
        [c(4.5, 0.0)]
    );
    // Permutationsmatrix (zyklisch, 3): dritte Einheitswurzeln
    let p = Matrix::new([[0.0f64, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]);
    let w = 2.0 * core::f64::consts::PI / 3.0;
    assert_same_spectrum(
        &p.eigenvalues().unwrap(),
        &[c(1.0, 0.0), c(w.cos(), w.sin()), c(w.cos(), -w.sin())],
        1e-13,
    );
}

#[test]
fn badly_scaled_matrix_benefits_from_balancing() {
    // S⁻¹·B·S mit stark unterschiedlichen Skalen: Eigenwerte bleiben die von B
    let b = Matrix::new([[1.0f64, 2.0, 0.0], [-1.0, 3.0, 1.0], [0.0, 1.0, -2.0]]);
    let scale = [1e6, 1.0, 1e-6];
    let a = Matrix::<f64, 3, 3>::from_fn(|i, j| b[(i, j)] * scale[i] / scale[j]);
    let want = b.eigenvalues().unwrap();
    assert_same_spectrum(&a.eigenvalues().unwrap(), &want, 1e-8);
}

#[test]
fn hessenberg_form_has_zeros_below_the_subdiagonal_and_keeps_the_trace() {
    let mut s = 8u64;
    let a = Matrix::<f64, 6, 6>::from_fn(|_, _| lcg(&mut s));
    let h = a.hessenberg_form();
    for i in 2..6 {
        for j in 0..i - 1 {
            assert_eq!(h[(i, j)], 0.0);
        }
    }
    assert!((h.trace() - a.trace()).abs() < 1e-12);
    assert_same_spectrum(&h.eigenvalues().unwrap(), &a.eigenvalues().unwrap(), 1e-9);
}

#[test]
fn errors() {
    let nan = Matrix::new([[1.0f64, f64::NAN], [0.0, 1.0]]);
    assert_eq!(nan.eigenvalues(), Err(LinalgError::NotFinite));
    let inf = Matrix::new([[f64::INFINITY, 0.0], [0.0, 1.0]]);
    assert_eq!(inf.eigenvalues(), Err(LinalgError::NotFinite));
}

#[test]
fn works_for_f32() {
    let a = Matrix::new([[0.0f32, -2.0], [2.0, 0.0]]);
    let ev = a.eigenvalues().unwrap();
    assert!(
        ev.iter()
            .all(|z| z.re.abs() < 1e-5 && (z.im.abs() - 2.0).abs() < 1e-5)
    );
}

#[cfg(feature = "alloc")]
mod dynamic {
    use super::*;
    use azmath::linalg::DynMatrix;

    #[test]
    fn dynamic_matches_fixed_size() {
        let mut s = 4u64;
        let a = Matrix::<f64, 5, 5>::from_fn(|_, _| lcg(&mut s));
        let d = DynMatrix::new(5, 5, a.as_array().iter().flatten().copied().collect()).unwrap();
        assert_eq!(
            d.eigenvalues().unwrap().as_slice(),
            a.eigenvalues().unwrap().as_slice()
        );
        assert_eq!(
            d.hessenberg_form().unwrap().as_slice(),
            a.hessenberg_form().as_array().as_flattened()
        );
        let rect = DynMatrix::<f64>::zeros(2, 3);
        assert_eq!(rect.eigenvalues(), Err(LinalgError::DimensionMismatch));
        assert_eq!(rect.hessenberg_form(), Err(LinalgError::DimensionMismatch));
        assert_eq!(
            DynMatrix::<f64>::zeros(0, 0).eigenvalues().unwrap().len(),
            0
        );
    }
}
