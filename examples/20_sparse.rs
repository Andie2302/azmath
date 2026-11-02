// Beispiel 20 – Dünn besetzte Matrizen und Vorkonditionierer
//
// 2D-Poisson-Gleichung mit dem 5-Punkt-Stern als CSR-Matrix: CG ohne und mit Vorkonditionierer
// (IC(0), SSOR) im Vergleich der Iterationszahlen, dazu die größten und kleinsten Eigenwerte per Lanczos.
//
//   cargo run --example 20_sparse --features alloc

#[cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]
use azmath::{
    linalg::{
        eigen::{lanczos, lanczos_ritz, lanczos_ritz_work_len, lanczos_work_len},
        iterative::{IterOptions, cg, pcg},
    },
    sparse::{CooMatrix, CsrMatrix, Ic0, Ssor},
};

#[cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]
fn poisson_2d(n: usize) -> CsrMatrix<f64> {
    let mut coo = CooMatrix::new(n * n, n * n);
    for i in 0..n {
        for j in 0..n {
            let k = i * n + j;
            coo.push(k, k, 4.0).unwrap();
            if i > 0 {
                coo.push(k, k - n, -1.0).unwrap();
            }
            if i + 1 < n {
                coo.push(k, k + n, -1.0).unwrap();
            }
            if j > 0 {
                coo.push(k, k - 1, -1.0).unwrap();
            }
            if j + 1 < n {
                coo.push(k, k + 1, -1.0).unwrap();
            }
        }
    }
    CsrMatrix::from_coo(&coo)
}

#[cfg(all(any(feature = "std", feature = "libm"), feature = "alloc"))]
fn main() {
    let n = 40;
    let size = n * n;
    let a = poisson_2d(n);
    println!(
        "{} Unbekannte, {} Einträge ungleich 0 (dicht wären es {})",
        size,
        a.nnz(),
        size * size
    );

    let b = vec![1.0; size];
    let opts = IterOptions::new(1e-10, 5000);
    let mut work = vec![0.0; 4 * size];

    let mut x = vec![0.0; size];
    let plain = cg(&a, &b, &mut x, opts, &mut work).unwrap();
    println!(
        "CG ohne Vorkonditionierer: {:4} Iterationen",
        plain.iterations
    );

    let ic = Ic0::new(&a).unwrap();
    let mut x = vec![0.0; size];
    let r = pcg(&a, &ic, &b, &mut x, opts, &mut work).unwrap();
    println!("CG mit IC(0):              {:4} Iterationen", r.iterations);

    let ssor = Ssor::new(&a, 1.5).unwrap();
    let mut x = vec![0.0; size];
    let r = pcg(&a, &ssor, &b, &mut x, opts, &mut work).unwrap();
    println!("CG mit SSOR (ω = 1,5):     {:4} Iterationen", r.iterations);

    // Lanczos: extreme Eigenwerte ohne die Matrix zu zerlegen
    let k = 60;
    let start: Vec<f64> = (0..size).map(|i| 1.0 + (i % 7) as f64).collect();
    let (mut alpha, mut beta) = (vec![0.0; k], vec![0.0; k]);
    let mut w = vec![0.0; lanczos_work_len(size, k)];
    let m = lanczos(&a, &start, k, &mut alpha, &mut beta, &mut w).unwrap();
    let (mut values, mut res) = (vec![0.0; m], vec![0.0; m]);
    let mut rw = vec![0.0; lanczos_ritz_work_len(m)];
    lanczos_ritz(&alpha, &beta, m, &mut values, &mut res, &mut rw).unwrap();
    let h = 1.0 / (n + 1) as f64;
    let pi = core::f64::consts::PI;
    println!(
        "kleinster Eigenwert: Lanczos {:.6}, exakt {:.6}",
        values[0],
        4.0 - 4.0 * (pi * h).cos()
    );
    println!(
        "größter Eigenwert:   Lanczos {:.6}, exakt {:.6}",
        values[m - 1],
        4.0 + 4.0 * (pi * h).cos()
    );
}

#[cfg(not(all(any(feature = "std", feature = "libm"), feature = "alloc")))]
fn main() {
    println!("Dieses Beispiel braucht die Features `alloc` und `std` oder `libm`.");
}
