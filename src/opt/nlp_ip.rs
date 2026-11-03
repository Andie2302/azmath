#![allow(clippy::needless_range_loop)]

use super::OptError;
use crate::Real;
use crate::linalg::{DynMatrix, DynVector};
use alloc::vec;
use alloc::vec::Vec;

/// Einstellungen für [`interior_point_nlp`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NlpOptions<T> {
    /// Toleranz für Stationarität, Zulässigkeit und Komplementarität.
    pub tol: T,
    /// Höchstzahl der Newton-Iterationen insgesamt.
    pub max_iter: usize,
    /// Anfangswert des Barriereparameters `μ`.
    pub mu_init: T,
    /// Das Barriere-Teilproblem gilt als gelöst, wenn sein Fehler `≤ kappa_eps·μ` ist.
    pub kappa_eps: T,
}

impl<T: Real> Default for NlpOptions<T> {
    /// `tol = √ε`, 300 Iterationen, `μ₀ = 0,1`, `κ_ε = 10`.
    fn default() -> Self {
        Self {
            tol: T::EPSILON.sqrt(),
            max_iter: 300,
            mu_init: T::from_f64(0.1),
            kappa_eps: T::from_f64(10.0),
        }
    }
}

/// Ausgang von [`interior_point_nlp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NlpStatus {
    /// KKT-Bedingungen innerhalb der Toleranz erfüllt.
    Optimal,
    /// Iterationslimit erreicht.
    MaxIterations,
    /// Die Schrittweitensuche fand keinen Abstieg (typisch bei unzulässigen Problemen; es gibt keine Wiederherstellungsphase).
    LineSearchFailed,
    /// Das Newton-System war auch mit Regularisierung nicht lösbar oder lieferte keine Abstiegsrichtung.
    LinearSolveFailed,
}

/// Ergebnis von [`interior_point_nlp`]; die Lösung steht im übergebenen `x`.
#[derive(Debug, Clone, PartialEq)]
pub struct NlpResult<T> {
    /// Ausgang.
    pub status: NlpStatus,
    /// Zielfunktionswert `f(x)`.
    pub f_value: T,
    /// Multiplikatoren der Gleichungen.
    pub lambda_eq: Vec<T>,
    /// Multiplikatoren der Ungleichungen (`≥ 0`).
    pub lambda_ineq: Vec<T>,
    /// Größte Verletzung der Nebenbedingungen.
    pub violation: T,
    /// Letzter Wert des Barriereparameters.
    pub mu: T,
    /// Anzahl der Iterationen.
    pub iterations: usize,
}

fn finite_all<T: Real>(v: &[T]) -> bool {
    v.iter().all(|x| !x.is_nan() && !x.is_infinite())
}

fn inf_norm<T: Real>(v: &[T]) -> T {
    v.iter().fold(T::ZERO, |m, x| m.max(x.abs()))
}

/// `y = Mᵀ·x` für eine zeilenweise `rows × cols`-Matrix (`x`: `rows`, `y`: `cols` Einträge).
fn mat_t_vec<T: Real>(m: &[T], rows: usize, cols: usize, x: &[T], y: &mut [T]) {
    y.fill(T::ZERO);
    for i in 0..rows {
        for j in 0..cols {
            y[j] += m[i * cols + j] * x[i];
        }
    }
}

/// Größte Schrittweite `α ≤ 1` mit `v + α·dv ≥ (1 − τ)·v`.
fn boundary_step<T: Real>(v: &[T], dv: &[T], tau: T) -> T {
    let mut a = T::ONE;
    for (x, d) in v.iter().zip(dv) {
        if *d < -tau * *x {
            a = a.min(-tau * *x / *d);
        }
    }
    a
}

/// Löst das nichtlineare Programm
///
/// ```text
/// min f(x)   unter   cᵢ(x) = 0  (i < n_eq),   cⱼ(x) ≤ 0  (j ≥ n_eq)
/// ```
///
/// mit einem **primal-dualen Innere-Punkte-Verfahren** (Kern von IPOPT, vereinfacht): Die Ungleichungen werden durch
/// Schlupfvariablen `s > 0` mit `c_I(x) + s = 0` ersetzt, `μ·Σ ln s` dient als Barriere, und `μ` sinkt monoton gegen 0. Jeder
/// Schritt löst ein symmetrisches Newton-System mit der **exakten Hessematrix der Lagrange-Funktion** (dichte LU, mit
/// Trägheitskorrektur `δ_w·I` bei fehlender Abstiegsrichtung), die Schrittweite folgt der
/// Fraction-to-the-boundary-Regel und einer ℓ₁-Merit-Funktion mit Armijo-Backtracking.
///
/// - `f`, `grad`, `constraints`, `jacobian` wie bei [`sqp`](super::sqp). `hessian(x, lambda, h)` schreibt
///   `∇²f(x) + Σ λₖ·∇²cₖ(x)` zeilenweise nach `h` (`n × n`); `lambda` enthält erst die `n_eq` Gleichungs-, dann die
///   `n_ineq` Ungleichungsmultiplikatoren. Mit [`hessian_ad`](crate::autodiff::hessian_ad) lässt sie sich automatisch bestimmen.
/// - `x`: Startwert (darf unzulässig sein und die Ungleichungen verletzen) und Ergebnis.
///
/// **Gegenüber IPOPT vereinfacht:** ℓ₁-Merit-Funktion statt Filter-Linesearch; **keine Wiederherstellungsphase** (unzulässige
/// Probleme enden als [`NlpStatus::LineSearchFailed`]); keine Skalierung; dichte Matrizen; nur lokale Lösung. Bei vielen
/// Ungleichungen oder Schranken ist es dem [`sqp`](super::sqp) meist überlegen, weil keine aktive Menge erraten werden muss.
///
/// Fehler: [`OptError::InvalidInput`], [`OptError::NotFinite`].
///
/// ```
/// use azmath::opt::{interior_point_nlp, NlpOptions, NlpStatus};
///
/// // min (x − 2)² + (y − 1)²  unter  x + y ≤ 2   ->  (1,5; 0,5)
/// let mut x = [0.0f64, 0.0];
/// let r = interior_point_nlp(
///     |x: &[f64]| (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2),
///     |x: &[f64], g: &mut [f64]| { g[0] = 2.0 * (x[0] - 2.0); g[1] = 2.0 * (x[1] - 1.0); },
///     |x: &[f64], c: &mut [f64]| c[0] = x[0] + x[1] - 2.0,
///     |_x: &[f64], j: &mut [f64]| { j[0] = 1.0; j[1] = 1.0; },
///     |_x: &[f64], _l: &[f64], h: &mut [f64]| h.copy_from_slice(&[2.0, 0.0, 0.0, 2.0]),
///     0, 1, &mut x, NlpOptions::default(),
/// ).unwrap();
/// assert_eq!(r.status, NlpStatus::Optimal);
/// assert!((x[0] - 1.5).abs() < 1e-5 && (x[1] - 0.5).abs() < 1e-5);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn interior_point_nlp<T: Real>(
    mut f: impl FnMut(&[T]) -> T,
    mut grad: impl FnMut(&[T], &mut [T]),
    mut constraints: impl FnMut(&[T], &mut [T]),
    mut jacobian: impl FnMut(&[T], &mut [T]),
    mut hessian: impl FnMut(&[T], &[T], &mut [T]),
    n_eq: usize,
    n_ineq: usize,
    x: &mut [T],
    opts: NlpOptions<T>,
) -> Result<NlpResult<T>, OptError> {
    let n = x.len();
    let (me, mi) = (n_eq, n_ineq);
    let m = me + mi;
    if n == 0 {
        return Err(OptError::InvalidInput);
    }
    if !finite_all(x) {
        return Err(OptError::NotFinite);
    }
    let tol = opts.tol;
    let kappa = T::from_f64(1e10);
    let mu_min = tol / T::from_f64(10.0);
    let delta_c = T::EPSILON.sqrt() * T::from_f64(1e-3);

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

    let mut mu = opts.mu_init;
    let mut s: Vec<T> = (0..mi).map(|j| (-c[me + j]).max(T::ONE)).collect();
    let mut y = vec![T::ZERO; me];
    let mut z: Vec<T> = s.iter().map(|sj| mu / *sj).collect();
    let mut nu = T::ONE;
    let mut delta_w_last = T::ZERO;
    let mut w = vec![T::ZERO; n * n];
    let mut lambda = vec![T::ZERO; m];
    let mut jt_lam = vec![T::ZERO; n];

    let violation_of = |c: &[T], s: &[T]| -> T {
        let mut v = T::ZERO;
        for i in 0..me {
            v = v.max(c[i].abs());
        }
        for j in 0..mi {
            v = v.max((c[me + j] + s[j]).abs());
        }
        v
    };
    let true_violation = |c: &[T]| -> T {
        let mut v = T::ZERO;
        for i in 0..me {
            v = v.max(c[i].abs());
        }
        for j in 0..mi {
            v = v.max(c[me + j].max(T::ZERO));
        }
        v
    };

    let mut iter = 0usize;
    loop {
        // Residuen und Fehlermaße
        lambda[..me].copy_from_slice(&y);
        lambda[me..].copy_from_slice(&z);
        mat_t_vec(&jac, m, n, &lambda, &mut jt_lam);
        let rd: Vec<T> = (0..n).map(|k| g[k] + jt_lam[k]).collect();
        let lam_max = inf_norm(&lambda);
        let s_d = T::ONE.max(lam_max / T::from_f64(100.0));
        let primal_res = violation_of(&c, &s);
        let comp0 = (0..mi).fold(T::ZERO, |mx, j| mx.max((s[j] * z[j]).abs()));
        let comp_mu = (0..mi).fold(T::ZERO, |mx, j| mx.max((s[j] * z[j] - mu).abs()));
        let e0 = (inf_norm(&rd) / s_d).max(primal_res).max(comp0 / s_d);
        let result = |status, fx: T, y: &[T], z: &[T], c: &[T], mu: T, iterations| NlpResult {
            status,
            f_value: fx,
            lambda_eq: y.to_vec(),
            lambda_ineq: z.to_vec(),
            violation: true_violation(c),
            mu,
            iterations,
        };
        if e0 <= tol {
            return Ok(result(NlpStatus::Optimal, fx, &y, &z, &c, mu, iter));
        }
        if iter >= opts.max_iter {
            return Ok(result(NlpStatus::MaxIterations, fx, &y, &z, &c, mu, iter));
        }
        // Barriereparameter senken, solange das aktuelle Teilproblem gelöst ist
        let e_mu = (inf_norm(&rd) / s_d).max(primal_res).max(comp_mu / s_d);
        let mut guard = 0;
        while e_mu <= opts.kappa_eps * mu && mu > mu_min && guard < 20 {
            let next = (T::from_f64(0.2) * mu)
                .min(mu.powf(T::from_f64(1.5)))
                .max(mu_min);
            if next >= mu {
                break;
            }
            mu = next;
            guard += 1;
        }

        // Hessematrix der Lagrange-Funktion
        hessian(x, &lambda, &mut w);
        if !finite_all(&w) {
            return Err(OptError::NotFinite);
        }

        // rechte Seite: −rd + Jᵢᵀ(S⁻¹·rc − (Z/S)·ri) und −re, mit rc = s∘z − μ, ri = c_I + s
        let rc: Vec<T> = (0..mi).map(|j| s[j] * z[j] - mu).collect();
        let ri: Vec<T> = (0..mi).map(|j| c[me + j] + s[j]).collect();
        let t: Vec<T> = (0..mi)
            .map(|j| rc[j] / s[j] - (z[j] / s[j]) * ri[j])
            .collect();
        let mut gt = vec![T::ZERO; n];
        mat_t_vec(&jac[me * n..], mi, n, &t, &mut gt);
        let mut rhs = vec![T::ZERO; n + me];
        for k in 0..n {
            rhs[k] = -rd[k] + gt[k];
        }
        for i in 0..me {
            rhs[n + i] = -c[i];
        }

        // Newton-System mit Trägheitskorrektur
        let dim = n + me;
        let viol_sum: T = (0..me).fold(T::ZERO, |a, i| a + c[i].abs())
            + (0..mi).fold(T::ZERO, |a, j| a + ri[j].abs());
        let mut delta_w = T::ZERO;
        let mut attempt = 0;
        let step = loop {
            let mut k = vec![T::ZERO; dim * dim];
            for i in 0..n {
                for j in 0..n {
                    k[i * dim + j] = w[i * n + j];
                }
                k[i * dim + i] += delta_w;
            }
            for r in 0..mi {
                let wgt = z[r] / s[r];
                for i in 0..n {
                    let ji = jac[(me + r) * n + i] * wgt;
                    if ji != T::ZERO {
                        for j in 0..n {
                            k[i * dim + j] += ji * jac[(me + r) * n + j];
                        }
                    }
                }
            }
            for r in 0..me {
                for j in 0..n {
                    k[(n + r) * dim + j] = jac[r * n + j];
                    k[j * dim + n + r] = jac[r * n + j];
                }
                k[(n + r) * dim + n + r] = -delta_c;
            }
            let solved = DynMatrix::new(dim, dim, k)
                .ok()
                .and_then(|km| km.lu().ok())
                .and_then(|lu| lu.solve(&DynVector::new(rhs.clone())).ok());
            if let Some(sol) = solved {
                let dx: Vec<T> = sol.as_slice()[..n].to_vec();
                let dy: Vec<T> = sol.as_slice()[n..].to_vec();
                let mut jdx = vec![T::ZERO; mi];
                for r in 0..mi {
                    jdx[r] = (0..n).fold(T::ZERO, |a, k| a + jac[(me + r) * n + k] * dx[k]);
                }
                let ds: Vec<T> = (0..mi).map(|r| -ri[r] - jdx[r]).collect();
                let dz: Vec<T> = (0..mi).map(|r| -(rc[r] + z[r] * ds[r]) / s[r]).collect();
                if finite_all(&dx) && finite_all(&dy) && finite_all(&ds) && finite_all(&dz) {
                    // Abstieg der Merit-Funktion prüfen: ∇φ·d − ν·Verletzung < 0
                    let grad_phi_d = (0..n).fold(T::ZERO, |a, k| a + g[k] * dx[k])
                        - mu * (0..mi).fold(T::ZERO, |a, r| a + ds[r] / s[r]);
                    let lam_new = (0..me)
                        .map(|i| (y[i] + dy[i]).abs())
                        .chain((0..mi).map(|r| (z[r] + dz[r]).abs()))
                        .fold(T::ZERO, |a, v| a.max(v));
                    let nu_try = nu.max(T::from_f64(1.5) * lam_new + T::ONE);
                    let d_merit = grad_phi_d - nu_try * viol_sum;
                    if d_merit < T::ZERO || (viol_sum == T::ZERO && grad_phi_d <= T::ZERO) {
                        nu = nu_try;
                        break Some((dx, dy, ds, dz, d_merit));
                    }
                }
            }
            attempt += 1;
            if attempt > 12 {
                break None;
            }
            delta_w = if delta_w == T::ZERO {
                if delta_w_last == T::ZERO {
                    T::from_f64(1e-4)
                } else {
                    (delta_w_last / T::from_f64(3.0)).max(T::from_f64(1e-20))
                }
            } else {
                delta_w * T::from_f64(10.0)
            };
        };
        let Some((dx, dy, ds, dz, d_merit)) = step else {
            return Ok(result(
                NlpStatus::LinearSolveFailed,
                fx,
                &y,
                &z,
                &c,
                mu,
                iter,
            ));
        };
        delta_w_last = delta_w;

        // Schrittweite: Fraction-to-the-boundary, dann Backtracking auf der Merit-Funktion
        let tau = T::from_f64(0.99).max(T::ONE - mu);
        let alpha_max = boundary_step(&s, &ds, tau);
        let alpha_z = boundary_step(&z, &dz, tau);
        let barrier = |s: &[T]| s.iter().fold(T::ZERO, |a, v| a + v.ln());
        let phi = fx - mu * barrier(&s) + nu * viol_sum;
        let mut alpha = alpha_max;
        let mut x_new = vec![T::ZERO; n];
        let mut s_new = vec![T::ZERO; mi];
        let mut c_new = vec![T::ZERO; m];
        let mut f_new = fx;
        let mut accepted = false;
        for _ in 0..40 {
            for k in 0..n {
                x_new[k] = x[k] + alpha * dx[k];
            }
            for r in 0..mi {
                s_new[r] = s[r] + alpha * ds[r];
            }
            f_new = f(&x_new);
            constraints(&x_new, &mut c_new);
            if s_new.iter().all(|v| *v > T::ZERO)
                && !f_new.is_nan()
                && !f_new.is_infinite()
                && finite_all(&c_new)
            {
                let viol_new: T = (0..me).fold(T::ZERO, |a, i| a + c_new[i].abs())
                    + (0..mi).fold(T::ZERO, |a, r| a + (c_new[me + r] + s_new[r]).abs());
                let phi_new = f_new - mu * barrier(&s_new) + nu * viol_new;
                if phi_new <= phi + T::from_f64(1e-4) * alpha * d_merit {
                    accepted = true;
                    break;
                }
            }
            alpha *= T::from_f64(0.5);
        }
        if !accepted {
            return Ok(result(
                NlpStatus::LineSearchFailed,
                fx,
                &y,
                &z,
                &c,
                mu,
                iter,
            ));
        }

        // Aktualisierung (Dualvariablen: z mit eigener Schrittweite, danach in den Korridor um μ/s gezwungen)
        x.copy_from_slice(&x_new);
        s.copy_from_slice(&s_new);
        for i in 0..me {
            y[i] += alpha * dy[i];
        }
        for r in 0..mi {
            z[r] += alpha_z * dz[r];
            let lo = mu / (kappa * s[r]);
            let hi = kappa * mu / s[r];
            z[r] = z[r].max(lo).min(hi);
        }
        fx = f_new;
        grad(x, &mut g);
        constraints(x, &mut c);
        jacobian(x, &mut jac);
        if !(finite_all(&g) && finite_all(&c) && finite_all(&jac)) {
            return Err(OptError::NotFinite);
        }
        iter += 1;
    }
}
