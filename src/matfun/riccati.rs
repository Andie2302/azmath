//! Algebraische Riccati-Gleichungen (kontinuierlich und diskret).

use super::{add, mm, norm1, scale, sub};
use crate::Real;
use crate::linalg::{DynMatrix, DynVector, LinalgError};

fn check<T: Real>(
    a: &DynMatrix<T>,
    b: &DynMatrix<T>,
    q: &DynMatrix<T>,
    r: &DynMatrix<T>,
) -> Result<(usize, usize), LinalgError> {
    let n = a.rows();
    let m = b.cols();
    if a.cols() != n
        || b.rows() != n
        || q.rows() != n
        || q.cols() != n
        || r.rows() != m
        || r.cols() != m
    {
        return Err(LinalgError::DimensionMismatch);
    }
    for mat in [a, b, q, r] {
        if mat.as_slice().iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
    }
    Ok((n, m))
}

/// Löst die kontinuierliche algebraische Riccati-Gleichung `AᵀX + XA − X·B·R⁻¹·Bᵀ·X + Q = 0` (stabilisierende Lösung, `X`
/// symmetrisch positiv semidefinit) mit der Matrix-Vorzeichenfunktion der Hamilton-Matrix (Newton-Iteration mit Skalierung,
/// Roberts 1971/1980). `A` ist `n×n`, `B` `n×m`, `Q` `n×n` symmetrisch, `R` `m×m` symmetrisch positiv definit.
///
/// Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NotFinite`], [`LinalgError::Singular`] (`R` singulär oder
/// Eigenwerte der Hamilton-Matrix auf der imaginären Achse), [`LinalgError::NoConvergence`].
pub fn care<T: Real>(
    a: &DynMatrix<T>,
    b: &DynMatrix<T>,
    q: &DynMatrix<T>,
    r: &DynMatrix<T>,
) -> Result<DynMatrix<T>, LinalgError> {
    let (n, _) = check(a, b, q, r)?;
    let g = mm(&mm(b, &r.inverse()?), &b.transpose());
    // Hamilton-Matrix H = [[A, −G], [−Q, −Aᵀ]]
    let at = a.transpose();
    let mut s = DynMatrix::zeros(2 * n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            s[(i, j)] = a[(i, j)];
            s[(i, n + j)] = -g[(i, j)];
            s[(n + i, j)] = -q[(i, j)];
            s[(n + i, n + j)] = -at[(i, j)];
        }
    }
    let tol = T::EPSILON.powf(T::from_f64(0.75)) * T::from_usize(n.max(1));
    let half = T::from_f64(0.5);
    let mut converged = false;
    for _ in 0..100 {
        let inv = s.inverse()?;
        let c = (norm1(&inv) / norm1(&s)).sqrt();
        let next = scale(&add(&scale(&s, c), &scale(&inv, T::ONE / c)), half);
        let diff = norm1(&sub(&next, &s));
        let size = norm1(&next);
        s = next;
        if !diff.is_finite() {
            return Err(LinalgError::NoConvergence);
        }
        if diff <= tol * size {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(LinalgError::NoConvergence);
    }
    // [S12; S22 + I]·X = −[S11 + I; S21]
    let mut w = DynMatrix::zeros(2 * n, n);
    let mut v = DynMatrix::zeros(2 * n, n);
    for i in 0..n {
        for j in 0..n {
            w[(i, j)] = s[(i, n + j)];
            w[(n + i, j)] = s[(n + i, n + j)] + if i == j { T::ONE } else { T::ZERO };
            v[(i, j)] = s[(i, j)] + if i == j { T::ONE } else { T::ZERO };
            v[(n + i, j)] = s[(n + i, j)];
        }
    }
    let qr = w.qr()?;
    let mut x = DynMatrix::zeros(n, n);
    for j in 0..n {
        let col: alloc::vec::Vec<T> = (0..2 * n).map(|i| -v[(i, j)]).collect();
        let sol: DynVector<T> = qr.solve(&DynVector::new(col))?;
        for i in 0..n {
            x[(i, j)] = sol[i];
        }
    }
    // Symmetrisieren
    let xt = x.transpose();
    Ok(scale(&add(&x, &xt), half))
}

/// Löst die diskrete algebraische Riccati-Gleichung `X = AᵀXA − AᵀXB(R + BᵀXB)⁻¹BᵀXA + Q` mit dem Verdopplungsalgorithmus
/// (structure-preserving doubling, quadratische Konvergenz). `A` ist `n×n`, `B` `n×m`, `Q` `n×n` symmetrisch positiv semidefinit,
/// `R` `m×m` symmetrisch positiv definit. Fehler wie bei [`care`].
pub fn dare<T: Real>(
    a: &DynMatrix<T>,
    b: &DynMatrix<T>,
    q: &DynMatrix<T>,
    r: &DynMatrix<T>,
) -> Result<DynMatrix<T>, LinalgError> {
    let (n, _) = check(a, b, q, r)?;
    let mut ak = a.clone();
    let mut gk = mm(&mm(b, &r.inverse()?), &b.transpose());
    let mut hk = q.clone();
    let id = DynMatrix::identity(n);
    let tol = T::EPSILON.powf(T::from_f64(0.75)) * T::from_usize(n.max(1));
    for _ in 0..100 {
        let w = add(&id, &mm(&gk, &hk));
        let wi = w.inverse()?;
        let akt = ak.transpose();
        let a_next = mm(&mm(&ak, &wi), &ak);
        let g_next = add(&gk, &mm(&mm(&ak, &wi), &mm(&gk, &akt)));
        let h_next = add(&hk, &mm(&mm(&akt, &hk), &mm(&wi, &ak)));
        let diff = norm1(&sub(&h_next, &hk));
        let size = norm1(&h_next);
        ak = a_next;
        gk = g_next;
        hk = h_next;
        if !diff.is_finite() {
            return Err(LinalgError::NoConvergence);
        }
        if diff <= tol * (size + T::ONE) {
            let ht = hk.transpose();
            return Ok(scale(&add(&hk, &ht), T::from_f64(0.5)));
        }
    }
    Err(LinalgError::NoConvergence)
}
