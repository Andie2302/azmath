//! Tests für `fdm::poisson` (AZM-040).
#![cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]

use azmath::fdm::poisson::*;
use azmath::linalg::iterative::IterOptions;
use core::f64::consts::PI;

const BCS: [Boundary; 3] = [Boundary::Dirichlet, Boundary::Neumann, Boundary::Periodic];

#[test]
fn symmetric_and_nullspace() {
    for &bx in &BCS {
        for &by in &BCS {
            let a = poisson_2d(6, 5, 0.2f64, 0.3, bx, by).unwrap();
            assert!(a.is_symmetric(1e-14), "{bx:?}/{by:?}");
            let ones = vec![1.0; 30];
            let mut y = vec![0.0; 30];
            a.matvec(&ones, &mut y).unwrap();
            let m = y.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if has_nullspace(bx, by) {
                assert!(m < 1e-12, "A*1 != 0 für {bx:?}/{by:?}");
                // Flussbilanz: alle Spaltensummen 0
                let mut yt = vec![0.0; 30];
                a.matvec_transposed(&ones, &mut yt).unwrap();
                assert!(yt.iter().all(|v| v.abs() < 1e-12));
            } else {
                assert!(m > 1e-3);
            }
        }
    }
}

fn run(bx: Boundary, by: Boundary, n: usize) -> f64 {
    let h = 1.0 / n as f64;
    let exact = |x: f64, y: f64| match (bx, by) {
        (Boundary::Dirichlet, _) => (PI * x).sin() * (PI * y).sin(),
        (Boundary::Neumann, _) => (PI * x).cos() * (PI * y).cos(),
        _ => (2.0 * PI * x).sin() * (2.0 * PI * y).sin(),
    };
    let lam = match (bx, by) {
        (Boundary::Periodic, _) => 8.0 * PI * PI,
        _ => 2.0 * PI * PI,
    };
    let a = poisson_2d(n, n, h, h, bx, by).unwrap();
    let (mut u, mut f) = (vec![], vec![]);
    for j in 0..n {
        for i in 0..n {
            let (x, y) = ((i as f64 + 0.5) * h, (j as f64 + 0.5) * h);
            u.push(exact(x, y));
            f.push(lam * exact(x, y));
        }
    }
    let ns = has_nullspace(bx, by);
    let (sol, _) = solve_poisson_2d(&a, &f, ns, 1e-12, IterOptions::new(1e-13, 5000)).unwrap();
    sol.iter()
        .zip(&u)
        .fold(0.0f64, |m, (s, e)| m.max((s - e).abs()))
}

#[test]
fn second_order_on_three_grids() {
    for (bx, by) in [
        (Boundary::Dirichlet, Boundary::Dirichlet),
        (Boundary::Neumann, Boundary::Neumann),
        (Boundary::Periodic, Boundary::Periodic),
    ] {
        let e: Vec<f64> = [16, 32, 64].iter().map(|&n| run(bx, by, n)).collect();
        for w in e.windows(2) {
            let p = (w[0] / w[1]).log2();
            assert!(p > 1.9 && p < 2.2, "{bx:?}: Ordnung {p}, Fehler {e:?}");
        }
    }
}

#[test]
fn rejects_incompatible_rhs_and_bad_grid() {
    let a = poisson_2d(4, 4, 0.25f64, 0.25, Boundary::Neumann, Boundary::Neumann).unwrap();
    let b = vec![1.0; 16];
    let r = solve_poisson_2d(&a, &b, true, 1e-10, IterOptions::new(1e-10, 100));
    assert_eq!(r.unwrap_err(), PoissonError::IncompatibleRhs);
    let mut c = b.clone();
    let mean = project_mean_zero(&mut c);
    assert!((mean - 1.0).abs() < 1e-15 && c.iter().all(|v| v.abs() < 1e-15));
    assert!(poisson_2d(2, 4, 0.1f64, 0.1, Boundary::Periodic, Boundary::Neumann).is_err());
    assert!(poisson_2d(4, 4, 0.0f64, 0.1, Boundary::Neumann, Boundary::Neumann).is_err());
    assert!(poisson_2d(4, 4, f64::NAN, 0.1, Boundary::Neumann, Boundary::Neumann).is_err());
    let nan = vec![f64::NAN; 16];
    assert_eq!(
        check_compatibility(&nan, 1e-10),
        Err(PoissonError::NotFinite)
    );
}
