//! Tests für komplexe QR, Cholesky, Eigenwerte und GMRES.
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::complex::Complex;
use azmath::random::{Pcg32, Rng};
use azmath::scalar::*;

type C = Complex<f64>;

fn rand_matrix(rng: &mut Pcg32, m: usize, n: usize) -> Vec<C> {
    (0..m * n)
        .map(|_| C::new(rng.uniform::<f64>() - 0.5, rng.uniform::<f64>() - 0.5))
        .collect()
}

fn matmul(a: &[C], b: &[C], m: usize, k: usize, n: usize) -> Vec<C> {
    let mut c = vec![C::ZERO; m * n];
    for i in 0..m {
        for j in 0..n {
            for t in 0..k {
                c[i * n + j] += a[i * k + t] * b[t * n + j];
            }
        }
    }
    c
}

fn maxdiff(a: &[C], b: &[C]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x - *y).abs())
        .fold(0.0, f64::max)
}

#[test]
fn qr_reconstructs_and_is_unitary() {
    let mut rng = Pcg32::new(1, 1);
    let (m, n) = (6, 4);
    let a = rand_matrix(&mut rng, m, n);
    let qr = ScalarQr::new(m, n, &a).unwrap();
    let q = qr.thin_q();
    assert!(maxdiff(&matmul(&q, qr.r(), m, n, n), &a) < 1e-12);
    // QᴴQ = I
    let qh: Vec<C> = (0..n * m)
        .map(|idx| q[(idx % m) * n + idx / m].conj())
        .collect();
    let id = matmul(&qh, &q, n, m, n);
    for i in 0..n {
        for j in 0..n {
            let e = if i == j { C::ONE } else { C::ZERO };
            assert!((id[i * n + j] - e).abs() < 1e-12);
        }
    }
    // R oberes Dreieck
    for i in 0..n {
        for j in 0..i {
            assert_eq!(qr.r()[i * n + j], C::ZERO);
        }
    }
}

#[test]
fn qr_least_squares_normal_equations() {
    let mut rng = Pcg32::new(2, 3);
    let (m, n) = (8, 3);
    let a = rand_matrix(&mut rng, m, n);
    let b: Vec<C> = (0..m)
        .map(|_| C::new(rng.uniform::<f64>(), rng.uniform::<f64>()))
        .collect();
    let x = ScalarQr::new(m, n, &a).unwrap().solve(&b).unwrap();
    // Residuum orthogonal zu Spalten von A: Aᴴ (b − A x) = 0
    let ax = matmul(&a, &x, m, n, 1);
    for j in 0..n {
        let mut s = C::ZERO;
        for i in 0..m {
            s += a[i * n + j].conj() * (b[i] - ax[i]);
        }
        assert!(s.abs() < 1e-12);
    }
    // reell funktioniert ebenfalls
    let ar = [1.0f64, 0.0, 0.0, 1.0, 1.0, 1.0];
    let xr = ScalarQr::new(3, 2, &ar)
        .unwrap()
        .solve(&[1.0, 2.0, 3.0])
        .unwrap();
    assert!((xr[0] - 1.0).abs() < 1e-12 && (xr[1] - 2.0).abs() < 1e-12);
    // Rangdefizit
    let rd = [C::ONE, C::ONE, C::ONE, C::ONE, C::ONE, C::ONE];
    assert!(
        ScalarQr::new(3, 2, &rd)
            .unwrap()
            .solve(&[C::ONE; 3])
            .is_err()
    );
    assert!(ScalarQr::new(2, 3, &[C::ONE; 6]).is_err());
}

fn hermitian_pd(rng: &mut Pcg32, n: usize) -> Vec<C> {
    let b = rand_matrix(rng, n, n);
    let bh: Vec<C> = (0..n * n)
        .map(|idx| b[(idx % n) * n + idx / n].conj())
        .collect();
    let mut a = matmul(&b, &bh, n, n, n);
    for i in 0..n {
        a[i * n + i] += C::from_real(1.0);
    }
    a
}

#[test]
fn cholesky_factor_and_solve() {
    let mut rng = Pcg32::new(5, 7);
    let n = 5;
    let a = hermitian_pd(&mut rng, n);
    let ch = ScalarCholesky::new(n, &a).unwrap();
    let l = ch.l();
    let lh: Vec<C> = (0..n * n)
        .map(|idx| l[(idx % n) * n + idx / n].conj())
        .collect();
    assert!(maxdiff(&matmul(l, &lh, n, n, n), &a) < 1e-12);
    let b: Vec<C> = (0..n).map(|i| C::new(i as f64, 1.0)).collect();
    let x = ch.solve(&b).unwrap();
    assert!(maxdiff(&matmul(&a, &x, n, n, 1), &b) < 1e-11);
    let lu_det = ScalarLu::new(n, &a).unwrap().determinant();
    assert!((ch.log_determinant() - lu_det.re.ln()).abs() < 1e-10);
}

#[test]
fn cholesky_errors() {
    let nonherm = [C::ONE, C::new(0.0, 1.0), C::new(0.0, 1.0), C::ONE];
    assert!(matches!(
        ScalarCholesky::new(2, &nonherm),
        Err(azmath::linalg::LinalgError::NotSymmetric)
    ));
    let indef = [C::ONE, C::new(2.0, 0.0), C::new(2.0, 0.0), C::ONE];
    assert!(matches!(
        ScalarCholesky::new(2, &indef),
        Err(azmath::linalg::LinalgError::NotPositiveDefinite)
    ));
    assert!(ScalarCholesky::new(2, &[C::ONE]).is_err());
}

#[test]
fn eigenvalues_known() {
    // Diagonalmatrix
    let d = [C::new(1.0, 2.0), C::ZERO, C::ZERO, C::new(-3.0, 0.5)];
    let mut ev = complex_eigenvalues(2, &d).unwrap();
    ev.sort_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
    assert!((ev[0] - C::new(-3.0, 0.5)).abs() < 1e-12 && (ev[1] - C::new(1.0, 2.0)).abs() < 1e-12);
    // Drehung: ±i
    let rot = [C::ZERO, C::new(-1.0, 0.0), C::ONE, C::ZERO];
    let ev = complex_eigenvalues(2, &rot).unwrap();
    assert!(
        ev.iter()
            .all(|z| z.re.abs() < 1e-12 && (z.im.abs() - 1.0).abs() < 1e-12)
    );
    // hermitesch 2×2 [[2, i], [−i, 2]]: Eigenwerte 1 und 3
    let h = [
        C::new(2.0, 0.0),
        C::new(0.0, 1.0),
        C::new(0.0, -1.0),
        C::new(2.0, 0.0),
    ];
    let r = hermitian_eigenvalues(2, &h).unwrap();
    assert!((r[0] - 1.0).abs() < 1e-12 && (r[1] - 3.0).abs() < 1e-12);
    assert!(hermitian_eigenvalues(2, &[C::ZERO, C::ONE, C::ZERO, C::ZERO]).is_err());
}

#[test]
fn eigenvalues_random_residuals() {
    let mut rng = Pcg32::new(9, 2);
    for n in [3usize, 6, 10] {
        let a = rand_matrix(&mut rng, n, n);
        let ev = complex_eigenvalues(n, &a).unwrap();
        // Spur = Summe der Eigenwerte
        let tr = (0..n).fold(C::ZERO, |s, i| s + a[i * n + i]);
        let sum = ev.iter().fold(C::ZERO, |s, z| s + *z);
        assert!((tr - sum).abs() < 1e-10, "n = {n}");
        // jeder Eigenwert hat einen Eigenvektor mit kleinem Residuum
        for &lam in &ev {
            let v = complex_eigenvector(n, &a, lam).unwrap();
            let av = matmul(&a, &v, n, n, 1);
            let res = (0..n)
                .map(|i| (av[i] - lam * v[i]).abs())
                .fold(0.0, f64::max);
            assert!(res < 1e-8, "n = {n}, res = {res}");
        }
    }
}

#[test]
fn csr_matvec_and_gmres() {
    let n = 30;
    let mut t = Vec::new();
    let mut rng = Pcg32::new(4, 4);
    for i in 0..n {
        t.push((i, i, C::new(4.0, 1.0)));
        if i + 1 < n {
            t.push((i, i + 1, C::new(-1.0, 0.5 * rng.uniform::<f64>())));
            t.push((i + 1, i, C::new(-1.0, -0.3)));
        }
    }
    t.push((0, 0, C::new(1.0, 0.0))); // Duplikat wird summiert
    let a = ScalarCsr::from_triplets(n, n, &t).unwrap();
    assert_eq!(a.nnz(), 3 * n - 2);
    let dense = a.to_dense();
    assert_eq!(dense[0], C::new(5.0, 1.0));
    let b: Vec<C> = (0..n).map(|i| C::new(1.0, i as f64 * 0.1)).collect();
    let r = gmres_scalar(&a, &b, 20, 1e-12, 500).unwrap();
    let mut ax = vec![C::ZERO; n];
    a.matvec(&r.x, &mut ax).unwrap();
    assert!(maxdiff(&ax, &b) < 1e-9);
    let direct = ScalarLu::new(n, &dense).unwrap().solve(&b).unwrap();
    assert!(maxdiff(&r.x, &direct) < 1e-8);
    // Adjungierte: ⟨Aᴴ u, v⟩ = ⟨u, A v⟩
    let u: Vec<C> = (0..n).map(|i| C::new(i as f64, 1.0)).collect();
    let (mut ahu, mut av) = (vec![C::ZERO; n], vec![C::ZERO; n]);
    a.adjoint_matvec(&u, &mut ahu).unwrap();
    a.matvec(&b, &mut av).unwrap();
    let lhs = (0..n).fold(C::ZERO, |s, i| s + ahu[i].conj() * b[i]);
    let rhs = (0..n).fold(C::ZERO, |s, i| s + u[i].conj() * av[i]);
    assert!((lhs - rhs).abs() < 1e-9);
    // Fehlerfälle
    assert!(ScalarCsr::from_triplets(2, 2, &[(2, 0, C::ONE)]).is_err());
    assert!(gmres_scalar(&a, &b[..5], 10, 1e-8, 10).is_err());
    let zero = gmres_scalar(&a, &vec![C::ZERO; n], 10, 1e-8, 10).unwrap();
    assert_eq!(zero.iterations, 0);
}

#[test]
fn schur_and_eigen_decomposition() {
    let mut rng = Pcg32::new(31, 7);
    for n in [1usize, 2, 5, 9] {
        let a = rand_matrix(&mut rng, n, n);
        let s = complex_schur(n, &a).unwrap();
        // T oberes Dreieck, Z unitär, A = Z T Zᴴ
        for i in 0..n {
            for j in 0..i {
                assert_eq!(s.t[i * n + j], C::ZERO);
            }
        }
        let zh: Vec<C> = (0..n * n)
            .map(|idx| s.z[(idx % n) * n + idx / n].conj())
            .collect();
        let back = matmul(&matmul(&s.z, &s.t, n, n, n), &zh, n, n, n);
        assert!(maxdiff(&back, &a) < 1e-10, "n = {n}");
        let zz = matmul(&s.z, &zh, n, n, n);
        for i in 0..n {
            for j in 0..n {
                let e = if i == j { C::ONE } else { C::ZERO };
                assert!((zz[i * n + j] - e).abs() < 1e-11);
            }
        }
        let eig = complex_eigen(n, &a).unwrap();
        for k in 0..n {
            let v: Vec<C> = (0..n).map(|i| eig.vectors[i * n + k]).collect();
            let av = matmul(&a, &v, n, n, 1);
            let res = (0..n)
                .map(|i| (av[i] - eig.values[k] * v[i]).abs())
                .fold(0.0, f64::max);
            assert!(res < 1e-9, "n = {n}, k = {k}, res = {res}");
        }
    }
}

#[test]
fn real_matrix_eigen() {
    use azmath::linalg::DynMatrix;
    let a = DynMatrix::new(3, 3, vec![0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0]).unwrap();
    let e = real_eigen(&a).unwrap();
    let mut re: Vec<f64> = e.values.iter().map(|z| z.re).collect();
    re.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert!(re[0].abs() < 1e-12 && re[1].abs() < 1e-12 && (re[2] - 2.0).abs() < 1e-12);
    assert!(real_eigen(&DynMatrix::<f64>::zeros(2, 3)).is_err());
}
