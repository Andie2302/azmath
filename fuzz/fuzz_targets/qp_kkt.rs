use azmath::{opt::QpSolution, sparse::CsrMatrix};

/// Necessary KKT conditions. For the generated SPD problems these also
/// characterize the optimum. Arbitrary indefinite Q only permits this necessary check.
/// Normalize products before summation to avoid overflow in the extreme target.
#[allow(clippy::too_many_arguments)]
pub fn check(
    q: &CsrMatrix<f64>,
    c: &[f64],
    a: &CsrMatrix<f64>,
    g: &CsrMatrix<f64>,
    h: &[f64],
    s: &QpSolution<f64>,
    tol: f64,
) -> Result<(), String> {
    if !s.x.iter().chain(&s.y).chain(&s.z).all(|v| v.is_finite()) {
        return Err("non-finite primal/dual solution".into());
    }
    if s.z.iter().any(|&v| v < 0.0) {
        return Err("negative inequality multiplier".into());
    }
    let norm = |xs: &[f64]| xs.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    let coefficients = norm(q.values()).max(norm(a.values())).max(norm(g.values()));
    let variables = norm(&s.x).max(norm(&s.y)).max(norm(&s.z));
    let mut residual: Vec<f64> = c.iter().map(|v| v / coefficients / variables).collect();
    let mut magnitude: Vec<f64> = residual.iter().map(|v| v.abs()).collect();
    for i in 0..q.rows() {
        for (j, v) in q.row(i) {
            let term = (v / coefficients) * (s.x[j] / variables);
            residual[i] += term;
            magnitude[i] += term.abs();
        }
    }
    for (matrix, dual) in [(a, &s.y), (g, &s.z)] {
        for (i, &d) in dual.iter().enumerate() {
            for (j, v) in matrix.row(i) {
                let term = (v / coefficients) * (d / variables);
                residual[j] += term;
                magnitude[j] += term.abs();
            }
        }
    }
    // Same normwise scale as the solver, with tolerance margin and a separate
    // floating-point summation allowance for cancellation of large terms.
    let bound = 100.0 * tol * (1.0 + norm(c)) / coefficients / variables;
    for (i, (&r, &mag)) in residual.iter().zip(&magnitude).enumerate() {
        if r.abs() > bound + 512.0 * f64::EPSILON * mag {
            return Err(format!("stationarity row {i}: scaled residual={r:e}"));
        }
    }
    let xs = norm(&s.x);
    let zs = norm(&s.z);
    let gs = norm(g.values()).max(norm(h));
    for (i, &z) in s.z.iter().enumerate() {
        let mut slack = h[i] / gs / xs;
        let mut mag = slack.abs();
        for (j, v) in g.row(i) {
            let term = (v / gs) * (s.x[j] / xs);
            slack -= term;
            mag += term.abs();
        }
        let product = slack * (z / zs);
        // Internal slack differs from h-G*x by the primal stopping residual.
        // Nonnegative internal products sum to m*mu; allow that same budget.
        let bound = (g.rows() as f64 * 100.0 * tol / gs / xs) / zs
            + (100.0 * tol * (1.0 + norm(h)) / gs / xs) * (z / zs)
            + 512.0 * f64::EPSILON * mag * (z / zs);
        if product.abs() > bound {
            return Err(format!(
                "complementarity row {i}: scaled product={product:e}"
            ));
        }
    }
    Ok(())
}
