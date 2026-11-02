// Beispiel 17 – Multigrid für die Poisson-Gleichung
//
// 1. Gitterunabhängige Konvergenz: Die Zahl der Zyklen bleibt gleich, wenn das Gitter feiner wird.
//    Dasselbe Problem mit CG braucht immer mehr Iterationen.
// 2. Diskretisierungsfehler: −Δu = 2π²·sin(πx)·sin(πy) mit der bekannten Lösung sin(πx)·sin(πy);
//    der Fehler fällt mit h² (Faktor 4 beim Halbieren der Maschenweite).
// 3. Multigrid als Vorkonditionierer für CG: konstant wenige Iterationen auf jedem Gitter.
// 4. Einfluss von Zyklusform und Glätter.
//
//   cargo run --release --example 17_multigrid

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::{
    Real,
    linalg::iterative::{IterOptions, cg, pcg},
    multigrid::{
        CycleType, MultigridOptions, MultigridPreconditioner, Poisson2d, Smoother, SolveOptions,
        solve_2d, work_len_2d,
    },
};

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    use core::f64::consts::PI;

    println!(
        "{:>5} {:>9} {:>10} {:>10} {:>8}   Fehler zu sin·sin",
        "n", "Punkte", "MG-Zyklen", "CG (f=1)", "MG-PCG"
    );
    let mut vorher: Option<f64> = None;
    for k in 3..=7 {
        let n = (1usize << k) - 1;
        let h = 1.0 / (n as f64 + 1.0);
        let exakt: Vec<f64> = (0..n * n)
            .map(|idx| {
                Real::sin(PI * ((idx / n) + 1) as f64 * h)
                    * Real::sin(PI * ((idx % n) + 1) as f64 * h)
            })
            .collect();
        let f: Vec<f64> = exakt.iter().map(|v| 2.0 * PI * PI * v).collect();

        // --- Multigrid als Löser
        let mut u = vec![0.0; n * n];
        let mut work = vec![0.0; work_len_2d(n)];
        let r = solve_2d(
            &f,
            &mut u,
            n,
            MultigridOptions::default(),
            SolveOptions::default(),
            &mut work,
        )
        .unwrap();
        let fehler = u
            .iter()
            .zip(&exakt)
            .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));

        // --- CG ohne und mit Multigrid-Vorkonditionierer. Als rechte Seite dient f = 1: sin·sin ist auf
        // dem Gitter ein exakter Eigenvektor, damit wäre CG in einem einzigen Schritt fertig und der
        // Vergleich bedeutungslos.
        let a = Poisson2d::new(n);
        let eins = vec![1.0; n * n];
        let mut cg_work = vec![0.0; 4 * n * n];
        let mut x = vec![0.0; n * n];
        let c = cg(
            &a,
            &eins,
            &mut x,
            IterOptions::new(1e-10, 100_000),
            &mut cg_work,
        )
        .unwrap();
        let mut mg_work = vec![0.0; work_len_2d(n)];
        let mg =
            MultigridPreconditioner::new_2d(n, MultigridOptions::default(), &mut mg_work).unwrap();
        let mut y = vec![0.0; n * n];
        let p = pcg(
            &a,
            &mg,
            &eins,
            &mut y,
            IterOptions::new(1e-10, 1000),
            &mut cg_work,
        )
        .unwrap();

        let verhaeltnis = vorher.map_or(String::new(), |v| format!("  (÷{:.2})", v / fehler));
        println!(
            "{n:>5} {:>9} {:>10} {:>10} {:>8}   {fehler:.3e}{verhaeltnis}",
            n * n,
            r.cycles,
            c.iterations,
            p.iterations
        );
        vorher = Some(fehler);
    }

    // --- Zyklusform und Glätter auf dem 63×63-Gitter
    let n = 63;
    let f = vec![1.0; n * n];
    println!("Zyklen bis zur Toleranz 1e-12 auf dem {n}×{n}-Gitter, f = 1:");
    for (name, o) in [
        ("Gauß-Seidel V(2,2)", MultigridOptions::<f64>::default()),
        (
            "Gauß-Seidel V(1,1)",
            MultigridOptions {
                pre_smooth: 1,
                post_smooth: 1,
                ..Default::default()
            },
        ),
        (
            "Gauß-Seidel W(2,2)",
            MultigridOptions {
                cycle: CycleType::W,
                ..Default::default()
            },
        ),
        (
            "Jacobi(0.8) V(2,2)",
            MultigridOptions {
                smoother: Smoother::WeightedJacobi(0.8),
                ..Default::default()
            },
        ),
    ] {
        let mut u = vec![0.0; n * n];
        let mut work = vec![0.0; work_len_2d(n)];
        let r = solve_2d(&f, &mut u, n, o, SolveOptions::default(), &mut work).unwrap();
        println!("  {name}: {} Zyklen", r.cycles);
    }
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    eprintln!("Bitte mit --features std oder --features libm ausführen.");
}
