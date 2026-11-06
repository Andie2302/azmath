#![allow(clippy::needless_range_loop)]

use super::{OptError, QpOptions, QpStatus, interior_point_qp};
use crate::Real;
use alloc::vec;
use alloc::vec::Vec;

/// Einstellungen für [`sqp`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SqpOptions<T> {
    /// Toleranz für Stationarität, Zulässigkeit und Komplementarität.
    pub tol: T,
    /// Höchstzahl der SQP-Iterationen.
    pub max_iter: usize,
}

impl<T: Real> Default for SqpOptions<T> {
    /// `tol = √ε`, höchstens 200 Iterationen.
    fn default() -> Self {
        Self {
            tol: T::EPSILON.sqrt(),
            max_iter: 200,
        }
    }
}

/// Ausgang von [`sqp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqpStatus {
    /// KKT-Bedingungen innerhalb der Toleranz erfüllt.
    Optimal,
    /// Iterationslimit erreicht.
    MaxIterations,
    /// Das quadratische Teilproblem war nicht lösbar (zum Beispiel widersprüchlich linearisierte Nebenbedingungen).
    QpFailed,
    /// Die Schrittweitensuche auf der Merit-Funktion fand keinen Abstieg.
    LineSearchFailed,
}

/// Ergebnis von [`sqp`]; die Lösung steht im übergebenen `x`.
#[derive(Debug, Clone, PartialEq)]
pub struct SqpResult<T> {
    /// Ausgang.
    pub status: SqpStatus,
    /// Zielfunktionswert `f(x)`.
    pub f_value: T,
    /// Multiplikatoren der Gleichungen.
    pub lambda_eq: Vec<T>,
    /// Multiplikatoren der Ungleichungen (`≥ 0`).
    pub lambda_ineq: Vec<T>,
    /// Größte Verletzung der Nebenbedingungen.
    pub violation: T,
    /// Anzahl der Iterationen.
    pub iterations: usize,
}

fn finite_all<T: Real>(v: &[T]) -> bool {
    v.iter().all(|x| !x.is_nan() && !x.is_infinite())
}

fn inf_norm<T: Real>(v: &[T]) -> T {
    v.iter().fold(T::ZERO, |m, x| m.max(x.abs()))
}

/// Verletzung: `Σ|cₑ| + Σ max(0, cᵢ)` (Summe, für die ℓ₁-Merit-Funktion) und größte Einzelverletzung.
fn violation<T: Real>(c: &[T], n_eq: usize) -> (T, T) {
    let (mut sum, mut max) = (T::ZERO, T::ZERO);
    for (i, &ci) in c.iter().enumerate() {
        let v = if i < n_eq { ci.abs() } else { ci.max(T::ZERO) };
        sum += v;
        max = max.max(v);
    }
    (sum, max)
}

/// `Jᵀ·λ` für eine zeilenweise `m × n`-Matrix.
fn jt_lambda<T: Real>(j: &[T], m: usize, n: usize, lambda: &[T], out: &mut [T]) {
    out.fill(T::ZERO);
    for i in 0..m {
        for k in 0..n {
            out[k] += j[i * n + k] * lambda[i];
        }
    }
}

/// Löst das nichtlineare Programm
///
/// ```text
/// min f(x)   unter   cᵢ(x) = 0  (i < n_eq),   cⱼ(x) ≤ 0  (j ≥ n_eq)
/// ```
///
/// mit **sequentieller quadratischer Programmierung (SQP)**: In jeder Iteration wird das Problem um den aktuellen Punkt
/// linearisiert, ein konvexes quadratisches Teilproblem ([`interior_point_qp`]) mit einer BFGS-Näherung (Powell-gedämpft) der
/// Hessematrix der Lagrange-Funktion gelöst und die Schrittweite über eine ℓ₁-Merit-Funktion mit Armijo-Backtracking
/// gewählt. Superlinear konvergent nahe einer regulären Lösung, ohne zweite Ableitungen.
///
/// - `f(x)`, `grad(x, g)`: Ziel und Gradient. `constraints(x, c)` schreibt erst die `n_eq` Gleichungen, dann die `n_ineq`
///   Ungleichungen (Form `c ≤ 0`; `x ≥ 0` als `−x ≤ 0`). `jacobian(x, j)` schreibt die `(n_eq+n_ineq) × n`-Jacobi-Matrix
///   zeilenweise. Mit [`jacobian_ad`](crate::autodiff::jacobian_ad) lassen sich alle Ableitungen exakt bestimmen.
/// - `x`: Startwert (darf unzulässig sein) und Ergebnis.
///
/// **Grenzen:** lokale Lösung; dichte Matrizen; bei **unzulässigem Teilproblem** keine Wiederherstellungsphase
/// ([`SqpStatus::QpFailed`]); kein Maratos-Korrekturschritt (kann nahe der Lösung etwas bremsen).
///
/// Fehler: [`OptError::InvalidInput`], [`OptError::NotFinite`].
///
/// ```
/// use azmath::opt::{sqp, SqpOptions, SqpStatus};
///
/// // min x + y  unter  x² + y² = 2   ->  (−1, −1)
/// let mut x = [-0.5f64, -1.5];
/// let r = sqp(
///     |x: &[f64]| x[0] + x[1],
///     |_x: &[f64], g: &mut [f64]| { g[0] = 1.0; g[1] = 1.0; },
///     |x: &[f64], c: &mut [f64]| c[0] = x[0] * x[0] + x[1] * x[1] - 2.0,
///     |x: &[f64], j: &mut [f64]| { j[0] = 2.0 * x[0]; j[1] = 2.0 * x[1]; },
///     1, 0, &mut x, SqpOptions::default(),
/// ).unwrap();
/// assert_eq!(r.status, SqpStatus::Optimal);
/// assert!((x[0] + 1.0).abs() < 1e-6 && (x[1] + 1.0).abs() < 1e-6);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn sqp<T: Real>(
    mut f: impl FnMut(&[T]) -> T,
    mut grad: impl FnMut(&[T], &mut [T]),
    mut constraints: impl FnMut(&[T], &mut [T]),
    mut jacobian: impl FnMut(&[T], &mut [T]),
    n_eq: usize,
    n_ineq: usize,
    x: &mut [T],
    opts: SqpOptions<T>,
) -> Result<SqpResult<T>, OptError> {
    let n = x.len();
    let m = n_eq + n_ineq;
    if n == 0 {
        return Err(OptError::InvalidInput);
    }
    if !finite_all(x) {
        return Err(OptError::NotFinite);
    }
    let tol = opts.tol;

    let mut b = vec![T::ZERO; n * n];
    for i in 0..n {
        b[i * n + i] = T::ONE;
    }
    let mut g = vec![T::ZERO; n];
    let mut c = vec![T::ZERO; m];
    let mut jac = vec![T::ZERO; m * n];
    let mut fx = f(x);
    grad(x, &mut g);
    constraints(x, &mut c);
    jacobian(x, &mut jac);
    if fx.is_nan() || fx.is_infinite() || !finite_all(&g) || !finite_all(&c) || !finite_all(&jac) {
        return Err(OptError::NotFinite);
    }

    let qp_opts = QpOptions {
        tol: T::EPSILON
            .powf(T::from_f64(0.75))
            .max(tol * T::from_f64(1e-3)),
        max_iter: 100,
        regularization: QpOptions::<T>::default().regularization,
        kkt_backend: super::QpKktBackend::Auto,
    };
    let mut lambda = vec![T::ZERO; m];
    let mut mu = T::ONE;
    let mut tmp_n = vec![T::ZERO; n];
    let mut tmp_n2 = vec![T::ZERO; n];

    let finish = |status, fx: T, lambda: &[T], c: &[T], iterations| SqpResult {
        status,
        f_value: fx,
        lambda_eq: lambda[..n_eq].to_vec(),
        lambda_ineq: lambda[n_eq..].to_vec(),
        violation: violation(c, n_eq).1,
        iterations,
    };

    for iter in 0..opts.max_iter {
        // Teilproblem: min gᵀd + ½dᵀBd, J_e d = −c_e, J_i d ≤ −c_i
        let a_eq: Vec<T> = jac[..n_eq * n].to_vec();
        let b_eq: Vec<T> = c[..n_eq].iter().map(|v| -*v).collect();
        let g_in: Vec<T> = jac[n_eq * n..].to_vec();
        let h_in: Vec<T> = c[n_eq..].iter().map(|v| -*v).collect();
        let sol = match interior_point_qp(&b, &g, &a_eq, &b_eq, &g_in, &h_in, qp_opts) {
            Ok(s) if s.status == QpStatus::Optimal => s,
            Ok(_) | Err(_) => return Ok(finish(SqpStatus::QpFailed, fx, &lambda, &c, iter)),
        };
        let d = sol.x;
        lambda[..n_eq].copy_from_slice(&sol.y);
        lambda[n_eq..].copy_from_slice(&sol.z);

        // Optimalitätsprüfung am aktuellen Punkt
        let (viol_sum, viol_max) = violation(&c, n_eq);
        jt_lambda(&jac, m, n, &lambda, &mut tmp_n);
        let stationarity = (0..n).fold(T::ZERO, |mx, k| mx.max((g[k] + tmp_n[k]).abs()));
        let lam_max = inf_norm(&lambda);
        let comp = (n_eq..m).fold(T::ZERO, |mx, j| mx.max((lambda[j] * c[j]).abs()));
        let step_small = inf_norm(&d) <= tol * (T::ONE + inf_norm(x));
        if viol_max <= tol
            && comp <= tol * (T::ONE + lam_max)
            && (stationarity <= tol * (T::ONE + lam_max) || step_small)
        {
            return Ok(finish(SqpStatus::Optimal, fx, &lambda, &c, iter));
        }

        // Merit-Funktion und Schrittweite
        mu = mu.max(T::from_f64(1.5) * lam_max + T::ONE);
        let phi = fx + mu * viol_sum;
        let gd: T = (0..n).fold(T::ZERO, |s, k| s + g[k] * d[k]);
        let mut dir_deriv = gd - mu * viol_sum;
        if dir_deriv >= T::ZERO {
            dir_deriv = -tol;
        }
        let mut x_new = vec![T::ZERO; n];
        let mut c_new = vec![T::ZERO; m];
        let mut alpha = T::ONE;
        let mut accepted = false;
        let mut f_new = fx;
        for _ in 0..30 {
            for k in 0..n {
                x_new[k] = x[k] + alpha * d[k];
            }
            f_new = f(&x_new);
            constraints(&x_new, &mut c_new);
            let phi_new = f_new + mu * violation(&c_new, n_eq).0;
            if !phi_new.is_nan()
                && !phi_new.is_infinite()
                && phi_new <= phi + T::from_f64(1e-4) * alpha * dir_deriv
            {
                accepted = true;
                break;
            }
            alpha *= T::from_f64(0.5);
        }
        if !accepted {
            return Ok(finish(SqpStatus::LineSearchFailed, fx, &lambda, &c, iter));
        }

        // neue Ableitungen und BFGS-Update der Lagrange-Hessematrix (Powell-Dämpfung)
        let mut g_new = vec![T::ZERO; n];
        let mut jac_new = vec![T::ZERO; m * n];
        grad(&x_new, &mut g_new);
        jacobian(&x_new, &mut jac_new);
        if !(finite_all(&g_new) && finite_all(&jac_new)) {
            return Err(OptError::NotFinite);
        }
        let s_vec: Vec<T> = (0..n).map(|k| x_new[k] - x[k]).collect();
        jt_lambda(&jac_new, m, n, &lambda, &mut tmp_n2);
        let y_vec: Vec<T> = (0..n)
            .map(|k| (g_new[k] + tmp_n2[k]) - (g[k] + tmp_n[k]))
            .collect();
        let bs: Vec<T> = (0..n)
            .map(|i| (0..n).fold(T::ZERO, |sacc, j| sacc + b[i * n + j] * s_vec[j]))
            .collect();
        let s_bs = (0..n).fold(T::ZERO, |sacc, k| sacc + s_vec[k] * bs[k]);
        let s_y = (0..n).fold(T::ZERO, |sacc, k| sacc + s_vec[k] * y_vec[k]);
        if s_bs > T::EPSILON * T::EPSILON {
            let theta = if s_y >= T::from_f64(0.2) * s_bs {
                T::ONE
            } else {
                T::from_f64(0.8) * s_bs / (s_bs - s_y)
            };
            let r: Vec<T> = (0..n)
                .map(|k| theta * y_vec[k] + (T::ONE - theta) * bs[k])
                .collect();
            let s_r = (0..n).fold(T::ZERO, |sacc, k| sacc + s_vec[k] * r[k]);
            if s_r > T::EPSILON * T::EPSILON {
                for i in 0..n {
                    for j in 0..n {
                        b[i * n + j] += -bs[i] * bs[j] / s_bs + r[i] * r[j] / s_r;
                    }
                }
            }
        }

        x.copy_from_slice(&x_new);
        fx = f_new;
        g = g_new;
        c.copy_from_slice(&c_new);
        jac = jac_new;
        if fx.is_nan() || fx.is_infinite() || !finite_all(&c) {
            return Err(OptError::NotFinite);
        }
    }
    Ok(finish(
        SqpStatus::MaxIterations,
        fx,
        &lambda,
        &c,
        opts.max_iter,
    ))
}
