//! QR-Zerlegung (Householder) und kleinste Quadrate für beliebige [`Scalar`] (reell oder komplex).

use super::Scalar;
use crate::Real;
use crate::linalg::LinalgError;
use alloc::vec;
use alloc::vec::Vec;

/// Euklidische Norm eines Vektors.
pub(super) fn norm2<T: Scalar>(v: &[T]) -> T::Real {
    let mut s = <T::Real as Real>::ZERO;
    for &x in v {
        s += x.modulus_sq();
    }
    s.sqrt()
}

/// Householder-Vektor für `x`: liefert `(v, beta, alpha)` mit `(I − beta·v·vᴴ)·x = alpha·e₁`; `None`, wenn `x = 0`.
pub(super) fn householder<T: Scalar>(x: &[T]) -> Option<(Vec<T>, T::Real, T)> {
    let nrm = norm2(x);
    if nrm == <T::Real as Real>::ZERO {
        return None;
    }
    let m0 = x[0].modulus();
    let phase = if m0 == <T::Real as Real>::ZERO {
        T::ONE
    } else {
        x[0] / T::from_real(m0)
    };
    let alpha = -(phase * T::from_real(nrm));
    let mut v = x.to_vec();
    v[0] = x[0] - alpha;
    let mut vv = <T::Real as Real>::ZERO;
    for &e in &v {
        vv += e.modulus_sq();
    }
    if vv == <T::Real as Real>::ZERO {
        return None;
    }
    Some((v, <T::Real as Real>::from_f64(2.0) / vv, alpha))
}

/// `A = Q·R` einer `m×n`-Matrix (`m ≥ n`, zeilenweise) mit unitärem `Q`.
#[derive(Debug, Clone)]
pub struct ScalarQr<T: Scalar> {
    m: usize,
    n: usize,
    /// Obere Dreiecksmatrix `R` (`n×n`, zeilenweise).
    r: Vec<T>,
    /// Householder-Vektoren (Länge `m − k` für Schritt `k`) und `beta`.
    reflectors: Vec<Option<(Vec<T>, T::Real)>>,
}

impl<T: Scalar> ScalarQr<T> {
    /// Zerlegt `a` (`m×n`, zeilenweise). Fehler: [`LinalgError::DimensionMismatch`] (`m < n` oder falsche Länge),
    /// [`LinalgError::NotFinite`].
    pub fn new(m: usize, n: usize, a: &[T]) -> Result<Self, LinalgError> {
        if a.len() != m * n || m < n {
            return Err(LinalgError::DimensionMismatch);
        }
        if a.iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        let mut w = a.to_vec();
        let mut refl = Vec::with_capacity(n);
        for k in 0..n {
            let col: Vec<T> = (k..m).map(|i| w[i * n + k]).collect();
            match householder(&col) {
                None => refl.push(None),
                Some((v, beta, alpha)) => {
                    for j in k..n {
                        let mut s = T::ZERO;
                        for (idx, i) in (k..m).enumerate() {
                            s += v[idx].conj() * w[i * n + j];
                        }
                        let f = s * T::from_real(beta);
                        for (idx, i) in (k..m).enumerate() {
                            w[i * n + j] -= f * v[idx];
                        }
                    }
                    w[k * n + k] = alpha;
                    for i in k + 1..m {
                        w[i * n + k] = T::ZERO;
                    }
                    refl.push(Some((v, beta)));
                }
            }
        }
        let mut r = vec![T::ZERO; n * n];
        for i in 0..n {
            for j in i..n {
                r[i * n + j] = w[i * n + j];
            }
        }
        Ok(Self {
            m,
            n,
            r,
            reflectors: refl,
        })
    }

    /// Die Matrix `R` (`n×n`, zeilenweise).
    pub fn r(&self) -> &[T] {
        &self.r
    }

    /// Wendet `Qᴴ` auf `b` (Länge `m`) an.
    pub fn apply_qh(&self, b: &[T]) -> Result<Vec<T>, LinalgError> {
        if b.len() != self.m {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut y = b.to_vec();
        for (k, rf) in self.reflectors.iter().enumerate() {
            if let Some((v, beta)) = rf {
                let mut s = T::ZERO;
                for (idx, i) in (k..self.m).enumerate() {
                    s += v[idx].conj() * y[i];
                }
                let f = s * T::from_real(*beta);
                for (idx, i) in (k..self.m).enumerate() {
                    y[i] -= f * v[idx];
                }
            }
        }
        Ok(y)
    }

    /// Wendet `Q` auf `y` (Länge `m`) an.
    pub fn apply_q(&self, y: &[T]) -> Result<Vec<T>, LinalgError> {
        if y.len() != self.m {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut b = y.to_vec();
        for (k, rf) in self.reflectors.iter().enumerate().rev() {
            if let Some((v, beta)) = rf {
                let mut s = T::ZERO;
                for (idx, i) in (k..self.m).enumerate() {
                    s += v[idx].conj() * b[i];
                }
                let f = s * T::from_real(*beta);
                for (idx, i) in (k..self.m).enumerate() {
                    b[i] -= f * v[idx];
                }
            }
        }
        Ok(b)
    }

    /// Die schmale Matrix `Q` (`m×n`, zeilenweise) mit `A = Q·R`.
    pub fn thin_q(&self) -> Vec<T> {
        let mut q = vec![T::ZERO; self.m * self.n];
        for j in 0..self.n {
            let mut e = vec![T::ZERO; self.m];
            e[j] = T::ONE;
            if let Ok(col) = self.apply_q(&e) {
                for i in 0..self.m {
                    q[i * self.n + j] = col[i];
                }
            }
        }
        q
    }

    /// Kleinste-Quadrate-Lösung `argmin ‖A·x − b‖₂` (Länge `n`). Fehler [`LinalgError::Singular`] bei Rangdefizit.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, LinalgError> {
        let y = self.apply_qh(b)?;
        let n = self.n;
        let mut scale = <T::Real as Real>::ZERO;
        for i in 0..n {
            scale = scale.max(self.r[i * n + i].modulus());
        }
        let tol = <T::Real as Real>::EPSILON * <T::Real as Real>::from_usize(self.m.max(n)) * scale;
        let mut x = vec![T::ZERO; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for j in i + 1..n {
                s -= self.r[i * n + j] * x[j];
            }
            let d = self.r[i * n + i];
            if !(d.modulus() > tol) {
                return Err(LinalgError::Singular);
            }
            x[i] = s / d;
        }
        Ok(x)
    }
}
