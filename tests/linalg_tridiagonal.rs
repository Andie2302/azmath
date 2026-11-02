//! Tests für den Thomas-Algorithmus und die zyklische Variante.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::linalg::{
    LinalgError, Matrix, Vector, cyclic_tridiagonal_work_len, solve_cyclic_tridiagonal,
    solve_tridiagonal, tridiagonal_work_len,
};

/// Diagonaldominantes Beispielsystem der Größe N.
fn system<const N: usize>() -> ([f64; N], [f64; N], [f64; N], [f64; N]) {
    let a = core::array::from_fn(|i| {
        if i == 0 {
            0.0
        } else {
            -1.0 + 0.1 * (i % 3) as f64
        }
    });
    let b = core::array::from_fn(|i| 4.0 + (i % 4) as f64);
    let c = core::array::from_fn(|i| {
        if i == N - 1 {
            0.0
        } else {
            -0.5 - 0.2 * (i % 2) as f64
        }
    });
    let d = core::array::from_fn(|i| 1.0 + i as f64 * 0.5 - (i % 3) as f64);
    (a, b, c, d)
}

/// Dichte Matrix zum tridiagonalen (bzw. zyklisch tridiagonalen) System. Einträge werden
/// **addiert**: bei `N = 2` fallen Ecken und Nebendiagonalen auf dieselben Positionen.
fn dense<const N: usize>(
    a: &[f64; N],
    b: &[f64; N],
    c: &[f64; N],
    cyclic: bool,
) -> Matrix<f64, N, N> {
    let mut m = Matrix::<f64, N, N>::zeros();
    for i in 0..N {
        m[(i, i)] += b[i];
        if i > 0 {
            m[(i, i - 1)] += a[i];
        }
        if i + 1 < N {
            m[(i, i + 1)] += c[i];
        }
    }
    if cyclic {
        m[(0, N - 1)] += a[0];
        m[(N - 1, 0)] += c[N - 1];
    }
    m
}

#[test]
fn matches_lu_on_a_diagonally_dominant_system() {
    const N: usize = 20;
    let (a, b, c, d) = system::<N>();
    let mut x = [0.0; N];
    let mut scratch = [0.0; tridiagonal_work_len(N)];
    solve_tridiagonal(&a, &b, &c, &d, &mut x, &mut scratch).unwrap();
    let reference = dense(&a, &b, &c, false).solve(&Vector::new(d)).unwrap();
    for i in 0..N {
        assert!((x[i] - reference[i]).abs() < 1e-12, "x[{i}]");
    }
    // Probe: A·x = d
    let r = dense(&a, &b, &c, false) * Vector::new(x) - Vector::new(d);
    assert!(r.max_abs() < 1e-12);
}

#[test]
fn poisson_1d_exact() {
    const N: usize = 9;
    let a = [-1.0; N];
    let b = [2.0; N];
    let c = [-1.0; N];
    // Lösung u_i = i(N+1−i)/2 zur rechten Seite 1
    let d = [1.0; N];
    let mut x = [0.0; N];
    let mut scratch = [0.0; N];
    solve_tridiagonal(&a, &b, &c, &d, &mut x, &mut scratch).unwrap();
    for (i, v) in x.iter().enumerate() {
        let k = (i + 1) as f64;
        assert!((v - k * (N as f64 + 1.0 - k) / 2.0).abs() < 1e-12);
    }
}

#[test]
fn tiny_systems() {
    let mut scratch = [0.0; 4];
    // n = 1
    let mut x = [0.0];
    solve_tridiagonal(&[0.0], &[4.0], &[0.0], &[8.0], &mut x, &mut scratch).unwrap();
    assert_eq!(x, [2.0]);
    // n = 0
    let mut empty: [f64; 0] = [];
    solve_tridiagonal(&[], &[], &[], &[], &mut empty, &mut scratch).unwrap();
    // n = 2: 2x + y = 5, x + 3y = 10
    let mut x2 = [0.0; 2];
    solve_tridiagonal(
        &[0.0, 1.0],
        &[2.0, 3.0],
        &[1.0, 0.0],
        &[5.0, 10.0],
        &mut x2,
        &mut scratch,
    )
    .unwrap();
    assert!((x2[0] - 1.0).abs() < 1e-14 && (x2[1] - 3.0).abs() < 1e-14);
}

#[test]
fn error_cases() {
    let mut x = [0.0; 3];
    let mut scratch = [0.0; 3];
    let ok = [1.0; 3];
    // Pivot 0 am Anfang
    assert_eq!(
        solve_tridiagonal(&ok, &[0.0, 1.0, 1.0], &ok, &ok, &mut x, &mut scratch),
        Err(LinalgError::Singular)
    );
    // Pivot 0 im Verlauf: b[1] − a[1]·c[0]/b[0] = 1 − 1·1/1 = 0
    assert_eq!(
        solve_tridiagonal(&ok, &[1.0, 1.0, 1.0], &ok, &ok, &mut x, &mut scratch),
        Err(LinalgError::Singular)
    );
    // Längen
    assert_eq!(
        solve_tridiagonal(&ok, &[1.0, 1.0], &ok, &ok, &mut x, &mut scratch),
        Err(LinalgError::DimensionMismatch)
    );
    let mut short = [0.0; 2];
    assert_eq!(
        solve_tridiagonal(&ok, &[4.0; 3], &ok, &ok, &mut x, &mut short),
        Err(LinalgError::DimensionMismatch)
    );
    // NaN
    assert_eq!(
        solve_tridiagonal(
            &ok,
            &[4.0; 3],
            &ok,
            &[1.0, f64::NAN, 1.0],
            &mut x,
            &mut scratch
        ),
        Err(LinalgError::NotFinite)
    );
    assert_eq!(
        solve_tridiagonal(&ok, &[f64::NAN, 4.0, 4.0], &ok, &ok, &mut x, &mut scratch),
        Err(LinalgError::NotFinite)
    );
}

#[test]
fn cyclic_matches_lu() {
    fn check<const N: usize>() {
        let (a, mut b, c, d) = system::<N>();
        // zyklisch: Eckeinträge a[0] und c[N−1] sind wirksam, die Matrix muss dominant bleiben
        let mut a = a;
        a[0] = -0.7;
        let mut c = c;
        c[N - 1] = -0.9;
        for v in b.iter_mut() {
            *v += 1.5;
        }
        let mut x = [0.0; N];
        // Arrayläge 4·N ist mit Const-Generics nicht ausdrückbar: Vec als Arbeitsspeicher
        let mut work = vec![0.0; cyclic_tridiagonal_work_len(N)];
        solve_cyclic_tridiagonal(&a, &b, &c, &d, &mut x, &mut work).unwrap();
        let reference = dense(&a, &b, &c, true).solve(&Vector::new(d)).unwrap();
        for i in 0..N {
            assert!(
                (x[i] - reference[i]).abs() < 1e-11,
                "N = {N}, x[{i}]: {} vs {}",
                x[i],
                reference[i]
            );
        }
    }
    check::<2>();
    check::<3>();
    check::<4>();
    check::<10>();
    check::<31>();
}

#[test]
fn cyclic_poisson_with_shift() {
    // periodisches Poisson mit Massenterm: 3·u_i − u_{i−1} − u_{i+1} = 1  ->  u ≡ 1
    const N: usize = 12;
    let a = [-1.0f64; N];
    let b = [3.0f64; N];
    let c = [-1.0f64; N];
    let d = [1.0f64; N];
    let mut x = [0.0f64; N];
    let mut work = [0.0f64; cyclic_tridiagonal_work_len(N)];
    solve_cyclic_tridiagonal(&a, &b, &c, &d, &mut x, &mut work).unwrap();
    assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-13));
}

#[test]
fn cyclic_error_cases() {
    let mut x = [0.0; 3];
    let mut work = [0.0; 12];
    let ok = [1.0; 3];
    assert_eq!(
        solve_cyclic_tridiagonal(&[1.0], &[4.0], &[1.0], &[1.0], &mut [0.0], &mut work),
        Err(LinalgError::DimensionMismatch)
    );
    let mut short = [0.0; 11];
    assert_eq!(
        solve_cyclic_tridiagonal(&ok, &[4.0; 3], &ok, &ok, &mut x, &mut short),
        Err(LinalgError::DimensionMismatch)
    );
    assert_eq!(
        solve_cyclic_tridiagonal(&ok, &[0.0, 4.0, 4.0], &ok, &ok, &mut x, &mut work),
        Err(LinalgError::Singular)
    );
    assert_eq!(
        solve_cyclic_tridiagonal(
            &ok,
            &[4.0; 3],
            &ok,
            &[f64::NAN, 1.0, 1.0],
            &mut x,
            &mut work
        ),
        Err(LinalgError::NotFinite)
    );
}

#[test]
fn f32_works() {
    let a = [0.0f32, -1.0, -1.0];
    let b = [2.0f32, 2.0, 2.0];
    let c = [-1.0f32, -1.0, 0.0];
    let mut x = [0.0f32; 3];
    let mut scratch = [0.0f32; 3];
    solve_tridiagonal(&a, &b, &c, &[1.0, 0.0, 1.0], &mut x, &mut scratch).unwrap();
    assert!(x.iter().all(|v| (v - 1.0).abs() < 1e-6));
}
