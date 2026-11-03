//! Numerisches Differenzieren: Differenzenquotienten, Richardson-Extrapolation (Ridders), Gradient, Hessematrix, Jacobi-Matrix.
//!
//! Differenzenquotienten verlieren Genauigkeit durch Auslöschung: Der Fehler eines zentralen Quotienten ist etwa
//! `h²·f‴/6 + ε·|f|/h`, am kleinsten bei `h ≈ ε^(1/3)` (bei `f64` etwa `6·10⁻⁶`), mit einem Fehler von ungefähr `ε^(2/3) ≈ 4·10⁻¹¹`.
//! [`derivative_ridders`] extrapoliert mehrere Schrittweiten und liefert zusätzlich eine Fehlerschätzung, meist mit deutlich
//! mehr gültigen Stellen. Wo es möglich ist, sind **exakte Ableitungen** durch [`autodiff`](crate::autodiff) vorzuziehen.
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::diff::{central_difference, derivative_ridders};
//!
//! let f = |x: f64| x.sin();
//! let d1 = central_difference(f, 1.0, 1e-5);
//! let (d2, err) = derivative_ridders(f, 1.0, 0.1);
//! assert!((d1 - 1.0f64.cos()).abs() < 1e-9);
//! assert!((d2 - 1.0f64.cos()).abs() < 1e-12 && err < 1e-10);
//! # }
//! ```

use crate::Real;

/// Vorwärtsdifferenz `(f(x + h) − f(x))/h` (Fehlerordnung `h`).
pub fn forward_difference<T: Real>(mut f: impl FnMut(T) -> T, x: T, h: T) -> T {
    (f(x + h) - f(x)) / h
}

/// Rückwärtsdifferenz `(f(x) − f(x − h))/h` (Fehlerordnung `h`).
pub fn backward_difference<T: Real>(mut f: impl FnMut(T) -> T, x: T, h: T) -> T {
    (f(x) - f(x - h)) / h
}

/// Zentrale Differenz `(f(x + h) − f(x − h))/(2h)` (Fehlerordnung `h²`).
pub fn central_difference<T: Real>(mut f: impl FnMut(T) -> T, x: T, h: T) -> T {
    (f(x + h) - f(x - h)) / (T::from_f64(2.0) * h)
}

/// Zentrale Differenz vierter Ordnung `(−f(x+2h) + 8f(x+h) − 8f(x−h) + f(x−2h))/(12h)`.
pub fn central_difference4<T: Real>(mut f: impl FnMut(T) -> T, x: T, h: T) -> T {
    let (two, eight) = (T::from_f64(2.0), T::from_f64(8.0));
    (-f(x + two * h) + eight * f(x + h) - eight * f(x - h) + f(x - two * h))
        / (T::from_f64(12.0) * h)
}

/// Zweite Ableitung `(f(x + h) − 2f(x) + f(x − h))/h²` (Fehlerordnung `h²`; optimal `h ≈ ε^(1/4)`).
pub fn second_derivative<T: Real>(mut f: impl FnMut(T) -> T, x: T, h: T) -> T {
    (f(x + h) - T::from_f64(2.0) * f(x) + f(x - h)) / (h * h)
}

/// Ableitung per **Ridders-Extrapolation**: zentrale Differenzen mit geometrisch fallender Schrittweite (Faktor 1,4,
/// Start `h`) werden im Neville-Schema extrapoliert. Rückgabe `(Ableitung, Fehlerschätzung)`. `h` sollte nicht zu klein sein
/// (eher ein Zehntel der Längenskala der Funktion als `10⁻⁸`).
pub fn derivative_ridders<T: Real>(mut f: impl FnMut(T) -> T, x: T, h: T) -> (T, T) {
    const NTAB: usize = 10;
    let con = T::from_f64(1.4);
    let con2 = con * con;
    let safe = T::from_f64(2.0);
    let two = T::from_f64(2.0);
    let mut a = [[T::ZERO; NTAB]; NTAB];
    let mut hh = h;
    a[0][0] = (f(x + hh) - f(x - hh)) / (two * hh);
    let mut err = T::from_f64(1e30);
    let mut ans = a[0][0];
    for i in 1..NTAB {
        hh /= con;
        a[0][i] = (f(x + hh) - f(x - hh)) / (two * hh);
        let mut fac = con2;
        for j in 1..=i {
            a[j][i] = (a[j - 1][i] * fac - a[j - 1][i - 1]) / (fac - T::ONE);
            fac *= con2;
            let errt = (a[j][i] - a[j - 1][i])
                .abs()
                .max((a[j][i] - a[j - 1][i - 1]).abs());
            if errt <= err {
                err = errt;
                ans = a[j][i];
            }
        }
        if (a[i][i] - a[i - 1][i - 1]).abs() >= safe * err {
            break;
        }
    }
    (ans, err)
}

/// Zwischenspeicher für [`gradient`] und [`hessian`] bei `n` Variablen.
pub const fn gradient_work_len(n: usize) -> usize {
    n
}

/// Gradient von `f: ℝⁿ → ℝ` per zentralen Differenzen mit Schrittweite `ε^(1/3)·max(1, |xᵢ|)` (`2n` Auswertungen).
/// `work`: mindestens [`gradient_work_len`]`(n)`. Panik bei zu kurzen Slices.
pub fn gradient<T: Real>(mut f: impl FnMut(&[T]) -> T, x: &[T], grad: &mut [T], work: &mut [T]) {
    let n = x.len();
    assert!(
        grad.len() >= n && work.len() >= gradient_work_len(n),
        "gradient: Slices zu kurz"
    );
    let xp = &mut work[..n];
    xp.copy_from_slice(x);
    let cbrt_eps = T::EPSILON.cbrt();
    for i in 0..n {
        let h = cbrt_eps * T::ONE.max(x[i].abs());
        xp[i] = x[i] + h;
        let fp = f(xp);
        xp[i] = x[i] - h;
        let fm = f(xp);
        xp[i] = x[i];
        grad[i] = (fp - fm) / ((x[i] + h) - (x[i] - h));
    }
}

/// Hessematrix (`n × n`, zeilenweise) von `f: ℝⁿ → ℝ` per zentralen Differenzen mit Schrittweite
/// `ε^(1/4)·max(1, |xᵢ|)`; symmetrisch. `work`: mindestens [`gradient_work_len`]`(n)`. Aufwand `O(n²)` Auswertungen.
/// Genauigkeit nur etwa `√ε`; mit [`hessian_ad`](crate::autodiff::hessian_ad) geht es exakt.
pub fn hessian<T: Real>(mut f: impl FnMut(&[T]) -> T, x: &[T], h_out: &mut [T], work: &mut [T]) {
    let n = x.len();
    assert!(
        h_out.len() >= n * n && work.len() >= gradient_work_len(n),
        "hessian: Slices zu kurz"
    );
    let xp = &mut work[..n];
    xp.copy_from_slice(x);
    let root = T::EPSILON.sqrt().sqrt();
    let f0 = f(xp);
    for i in 0..n {
        let hi = root * T::ONE.max(x[i].abs());
        xp[i] = x[i] + hi;
        let fp = f(xp);
        xp[i] = x[i] - hi;
        let fm = f(xp);
        xp[i] = x[i];
        h_out[i * n + i] = (fp - T::from_f64(2.0) * f0 + fm) / (hi * hi);
        for j in (i + 1)..n {
            let hj = root * T::ONE.max(x[j].abs());
            xp[i] = x[i] + hi;
            xp[j] = x[j] + hj;
            let fpp = f(xp);
            xp[j] = x[j] - hj;
            let fpm = f(xp);
            xp[i] = x[i] - hi;
            let fmm = f(xp);
            xp[j] = x[j] + hj;
            let fmp = f(xp);
            xp[i] = x[i];
            xp[j] = x[j];
            let v = (fpp - fpm - fmp + fmm) / (T::from_f64(4.0) * hi * hj);
            h_out[i * n + j] = v;
            h_out[j * n + i] = v;
        }
    }
}

/// Zwischenspeicher für [`jacobian`] bei `m` Funktionswerten und `n` Variablen.
pub const fn jacobian_work_len(m: usize, n: usize) -> usize {
    n + 2 * m
}

/// Jacobi-Matrix (`m × n`, zeilenweise) von `f: ℝⁿ → ℝᵐ` per zentralen Differenzen (`2n` Auswertungen). `f(x, out)` schreibt
/// die `m` Werte. `work`: mindestens [`jacobian_work_len`]`(m, n)`. Genauer als [`numerical_jacobian`](crate::optimize::numerical_jacobian)
/// (dort Vorwärtsdifferenzen), aber doppelt so teuer.
pub fn jacobian<T: Real>(
    mut f: impl FnMut(&[T], &mut [T]),
    x: &[T],
    m: usize,
    jac: &mut [T],
    work: &mut [T],
) {
    let n = x.len();
    assert!(
        jac.len() >= m * n && work.len() >= jacobian_work_len(m, n),
        "jacobian: Slices zu kurz"
    );
    let (xp, rest) = work.split_at_mut(n);
    let (fp, rest) = rest.split_at_mut(m);
    let fm = &mut rest[..m];
    xp.copy_from_slice(x);
    let cbrt_eps = T::EPSILON.cbrt();
    for k in 0..n {
        let h = cbrt_eps * T::ONE.max(x[k].abs());
        xp[k] = x[k] + h;
        f(xp, fp);
        xp[k] = x[k] - h;
        f(xp, fm);
        xp[k] = x[k];
        let width = (x[k] + h) - (x[k] - h);
        for i in 0..m {
            jac[i * n + k] = (fp[i] - fm[i]) / width;
        }
    }
}
