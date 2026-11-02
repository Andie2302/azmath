// Beispiel 14 – GMRES(m): robust, auch wo CG und BiCGSTAB versagen
//
// 1. Eine schiefsymmetrische Matrix: CG meldet "nicht positiv definit", BiCGSTAB bricht
//    zusammen, GMRES löst sie.
// 2. Der Preis des Neustarts: 1D-Konvektions-Diffusion mit n = 100. Volles GMRES braucht n
//    Anwendungen von A, GMRES(m) mit kleinem m deutlich mehr, spart dafür Speicher.
// 3. Ein Rechts-Vorkonditionierer (Jacobi) gegen ungleichmäßig skalierte Spalten.
//
//   cargo run --example 14_gmres

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::linalg::{
    FnOperator, Matrix, Vector,
    iterative::{IterOptions, Jacobi, bicgstab, cg, gmres, gmres_work_len, pgmres},
};

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    // --- 1. Schiefsymmetrisch: A·x = (x₁, −x₀) = (1, 0)  ->  x = (0, 1)
    let a = Matrix::new([[0.0f64, 1.0], [-1.0, 0.0]]);
    let b = [1.0, 0.0];
    let opts = IterOptions::new(1e-12, 100);
    let (mut x, mut work) = ([0.0; 2], [0.0; gmres_work_len(2, 2)]);
    gmres(&a, &b, &mut x, 2, opts, &mut work).unwrap();
    println!("Schiefsymmetrisch: GMRES liefert x = {x:?}");
    println!(
        "  CG      : {:?}",
        cg(&a, &b, &mut [0.0; 2], opts, &mut [0.0; 8])
    );
    println!(
        "  BiCGSTAB: {:?}",
        bicgstab(&a, &b, &mut [0.0; 2], opts, &mut [0.0; 16])
    );

    // --- 2. Neustart-Länge: matrixfreie Konvektions-Diffusion (nichtsymmetrisch)
    const N: usize = 100;
    let c = 0.05;
    let op = FnOperator::new(N, N, move |x: &[f64], y: &mut [f64]| {
        for i in 0..x.len() {
            let links = if i > 0 { x[i - 1] } else { 0.0 };
            let rechts = if i + 1 < x.len() { x[i + 1] } else { 0.0 };
            y[i] = -(1.0 + c) * links + 2.0 * x[i] - (1.0 - c) * rechts;
        }
    });
    let rhs = [1.0f64; N];
    let opts = IterOptions::new(1e-10, 100_000);
    println!("Konvektion-Diffusion, n = {N} (Anwendungen von A, Speicher in Elementen):");
    for restart in [10, 20, 50, N] {
        let mut x = [0.0; N];
        let mut work = vec![0.0; gmres_work_len(N, restart)];
        let r = gmres(&op, &rhs, &mut x, restart, opts, &mut work).unwrap();
        println!(
            "  GMRES({restart:>3}): {:>4} Anwendungen, Speicher {:>6}",
            r.iterations,
            gmres_work_len(N, restart)
        );
    }
    let mut x = [0.0; N];
    let r = bicgstab(&op, &rhs, &mut x, opts, &mut [0.0; 8 * N]).unwrap();
    println!(
        "  BiCGSTAB   : {:>4} Anwendungen, Speicher {:>6}",
        2 * r.iterations,
        8 * N
    );

    // --- 3. Rechts-Vorkonditionierer gegen ungleichmäßige Spaltenskalierung
    let s = [1.0f64, 100.0, 1.0e4, 10.0, 1.0e3, 1.0];
    let m = Matrix::<f64, 6, 6>::from_fn(|i, j| {
        let basis = if i == j {
            5.0
        } else if j + 1 == i {
            2.0
        } else if i + 1 == j {
            -1.0
        } else {
            0.0
        };
        basis * s[j]
    });
    let rhs = Vector::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let opts = IterOptions::new(1e-12, 500);
    let mut ohne = [0.0; 6];
    let r_ohne = gmres(
        &m,
        rhs.as_slice(),
        &mut ohne,
        3,
        opts,
        &mut [0.0; gmres_work_len(6, 3)],
    );
    let mut mit = [0.0; 6];
    let jacobi = Jacobi::new(&m).unwrap();
    let r_mit = pgmres(
        &m,
        &jacobi,
        rhs.as_slice(),
        &mut mit,
        3,
        opts,
        &mut [0.0; gmres_work_len(6, 3)],
    );
    println!("Spaltenskalierte Matrix, GMRES(3):");
    println!(
        "  ohne Vorkonditionierer: {:?}",
        r_ohne.map(|r| r.iterations)
    );
    println!(
        "  mit Jacobi            : {:?}",
        r_mit.map(|r| r.iterations)
    );
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    eprintln!("Bitte mit --features std oder --features libm ausführen.");
}
