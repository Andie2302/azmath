// Beispiel 8 – Iterative Löser: CG matrixfrei und mit Jacobi-Vorkonditionierer
//
// 1. Das eindimensionale Poisson-Problem -u'' = f wird ohne explizite Matrix gelöst:
//    der Operator ist nur eine Closure, die das diskrete Laplace (Tridiagonal 2, -1) anwendet.
//    Der Speicherbedarf wächst dadurch nur linear mit der Gittergröße.
// 2. Eine schlecht skalierte SPD-Matrix zeigt, wie der Jacobi-Vorkonditionierer die Anzahl
//    der Iterationen senkt.
//
// Iterative Verfahren erreichen strenge Toleranzen nur mit genauem Backend (`std`, `libm`).
//
//   cargo run --example 08_conjugate_gradient
//   cargo run --example 08_conjugate_gradient --features alloc

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::linalg::{
    FnOperator, Matrix, Vector,
    iterative::{IterOptions, Jacobi, cg},
};

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    // --- 1. Matrixfrei: -u'' = 1 auf (0, 1), u(0) = u(1) = 0, Lösung u(x) = x(1-x)/2
    const N: usize = 200;
    let h = 1.0 / (N + 1) as f64;
    let laplace = FnOperator::new(N, N, |x: &[f64], y: &mut [f64]| {
        for i in 0..x.len() {
            let left = if i > 0 { x[i - 1] } else { 0.0 };
            let right = if i + 1 < x.len() { x[i + 1] } else { 0.0 };
            y[i] = (2.0 * x[i] - left - right) / (h * h);
        }
    });
    let b = [1.0f64; N];
    let mut u = [0.0f64; N];
    let mut work = [0.0f64; 4 * N];
    let res = cg(
        &laplace,
        &b,
        &mut u,
        IterOptions::new(1e-10, 2000),
        &mut work,
    )
    .unwrap();

    let mitte = u[N / 2];
    let x_mitte = (N / 2 + 1) as f64 * h;
    println!(
        "Poisson, n = {N}: {} Iterationen, Residuum {:e}",
        res.iterations, res.residual_norm
    );
    println!(
        "u(Mitte) = {mitte:.6}, exakt = {:.6}",
        x_mitte * (1.0 - x_mitte) / 2.0
    );

    // --- 2. Vorkonditionierung bei schlecht skalierter Matrix: A = S·B·S
    let s = [1.0f64, 10.0, 100.0, 1000.0, 10.0, 1.0];
    let a = Matrix::<f64, 6, 6>::from_fn(|i, j| {
        let basis = if i == j {
            4.0
        } else if i.abs_diff(j) == 1 {
            1.0
        } else {
            0.0
        };
        s[i] * s[j] * basis
    });
    let rhs = Vector::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let opts = IterOptions::new(1e-10, 5000);

    let (_, ohne) = a.solve_cg(&rhs, opts).unwrap();
    let (_, mit) = a.solve_pcg(&Jacobi::new(&a).unwrap(), &rhs, opts).unwrap();
    println!(
        "Skalierte Matrix: ohne Vorkonditionierer {} Iterationen, mit Jacobi {}",
        ohne.iterations, mit.iterations
    );

    #[cfg(feature = "alloc")]
    dynamisch();
}

#[cfg(all(feature = "alloc", any(feature = "std", feature = "libm")))]
fn dynamisch() {
    use azmath::linalg::{DynMatrix, DynVector};

    // Größe erst zur Laufzeit bekannt
    let n = 40;
    let mut a = DynMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        a[(i, i)] = 3.0;
        if i > 0 {
            a[(i, i - 1)] = -1.0;
        }
        if i + 1 < n {
            a[(i, i + 1)] = -1.0;
        }
    }
    let b = DynVector::new(vec![1.0; n]);
    let (x, res) = a.solve_cg(&b, IterOptions::default()).unwrap();
    println!(
        "Dynamisch {n}x{n}: {} Iterationen, x[0] = {:.6}",
        res.iterations, x[0]
    );
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    eprintln!("Bitte mit --features std oder --features libm ausführen.");
}
