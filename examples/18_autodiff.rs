// Beispiel 18 – Automatisches Differenzieren
//
// 1. Die Modellfunktion y = a·exp(b·t) wird **einmal** generisch über den Zahlentyp geschrieben.
//    `jacobian_ad` liefert daraus die exakte Jacobi-Matrix für Levenberg-Marquardt.
// 2. Die Hessematrix der Rosenbrock-Funktion per geschachtelten Jets, ohne Differenzen.
//
//   cargo run --example 18_autodiff

#[cfg(any(feature = "std", feature = "libm"))]
use azmath::{
    Real,
    autodiff::{Jet, VectorFn, ad_scratch_len, hessian_ad, jacobian_ad},
    optimize::{OptimizeOptions, levenberg_marquardt, levenberg_marquardt_work_len},
    vector_fn,
};

#[cfg(any(feature = "std", feature = "libm"))]
struct ExpFit {
    t: [f64; 6],
    y: [f64; 6],
}

#[cfg(any(feature = "std", feature = "libm"))]
impl VectorFn for ExpFit {
    // Residuum r_i = a·exp(b·t_i) − y_i, für S = f64 und S = Jet<f64, K>
    fn eval<S: Real>(&self, p: &[S], out: &mut [S]) {
        for ((o, t), y) in out.iter_mut().zip(self.t).zip(self.y) {
            *o = p[0] * (p[1] * S::from_f64(t)).exp() - S::from_f64(y);
        }
    }
}

#[cfg(any(feature = "std", feature = "libm"))]
fn main() {
    // --- 1. Kurvenanpassung mit exakter Jacobi-Matrix
    let t = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
    let f = ExpFit {
        t,
        y: t.map(|t| 2.5 * Real::exp(-1.3 * t)),
    };
    let mut scratch = [Jet::<f64, 2>::constant(0.0); ad_scratch_len(6, 2)];
    let mut p = [1.0, -0.5];
    let mut work = [0.0; levenberg_marquardt_work_len(6, 2)];
    let r = levenberg_marquardt(
        |p, r| f.eval::<f64>(p, r),
        |p, j| jacobian_ad::<f64, 2, _>(&f, p, 6, j, &mut scratch),
        &mut p,
        6,
        OptimizeOptions::default(),
        &mut work,
    )
    .unwrap();
    println!(
        "Anpassung: a = {:.6}, b = {:.6} ({} Iterationen, ‖r‖ = {:.1e})",
        p[0], p[1], r.iterations, r.residual_norm
    );

    // --- 2. Hessematrix von f(x, y) = (1 − x)² + 100 (y − x²)² bei (−1.2, 1)
    vector_fn!(Rosenbrock, |x, out| {
        let c = |v: f64| S::from_f64(v);
        let a = c(1.0) - x[0];
        let b = x[1] - x[0] * x[0];
        out[0] = a * a + c(100.0) * b * b;
    });
    let h = hessian_ad::<f64, 2, _>(&Rosenbrock, &[-1.2, 1.0]);
    println!(
        "Hesse: [[{:.1}, {:.1}], [{:.1}, {:.1}]]",
        h[0][0], h[0][1], h[1][0], h[1][1]
    );
}

#[cfg(not(any(feature = "std", feature = "libm")))]
fn main() {
    println!("Dieses Beispiel braucht das Feature `std` oder `libm`.");
}
