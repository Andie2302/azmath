use super::{PolyRootsResult, RootError};
use crate::Real;
use crate::complex::Complex;
use crate::linalg::LinalgError;
use crate::linalg::eigen::eigenvalues_in_place;

/// Benötigter Arbeitsspeicher für [`companion_roots`] bei einem Polynom vom Grad `degree`: die Begleitmatrix `degree × degree`.
pub const fn companion_work_len(degree: usize) -> usize {
    degree * degree
}

/// Alle Nullstellen eines Polynoms mit **reellen** Koeffizienten als Eigenwerte der **Begleitmatrix**.
///
/// Für `p(x) = c₀ + c₁x + … + cₙxⁿ` hat die Begleitmatrix Einsen auf der ersten Unterdiagonale und `−cᵢ/cₙ` in der letzten
/// Spalte; ihre Eigenwerte sind genau die Nullstellen. Sie werden mit dem QR-Algorithmus
/// ([`Matrix::eigenvalues`](crate::linalg::Matrix::eigenvalues)) berechnet. Gegenüber [`aberth`](super::aberth) braucht das
/// keine Startwerte und keine Toleranz und ist auch bei mehrfachen Nullstellen verlässlich (mit der dafür üblichen
/// Genauigkeit `ε^(1/m)`), kostet aber `O(n³)` Rechenzeit und `O(n²)` Speicher.
///
/// - `coeffs`: aufsteigend; führende Nullkoeffizienten werden ignoriert, Nullstellen bei 0 exakt als `0` ausgegeben.
/// - `roots`: Ausgabe, mindestens so lang wie der Grad. `work`: mindestens [`companion_work_len`]`(Grad)`.
///
/// Fehler: [`RootError::InvalidInput`] (Nullpolynom, Puffer zu kurz), [`RootError::NotFinite`], [`RootError::NoConvergence`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::complex::Complex;
/// use azmath::roots::{companion_roots, companion_work_len};
///
/// // x³ − 6x² + 11x − 6 = (x − 1)(x − 2)(x − 3)
/// let mut z = [Complex::new(0.0f64, 0.0); 3];
/// let mut work = [0.0; companion_work_len(3)];
/// companion_roots(&[-6.0, 11.0, -6.0, 1.0], &mut z, &mut work).unwrap();
/// let mut re: Vec<f64> = z.iter().map(|r| r.re).collect();
/// re.sort_by(|a, b| a.total_cmp(b));
/// assert!(re.iter().zip([1.0, 2.0, 3.0]).all(|(a, b)| (a - b).abs() < 1e-12));
/// # }
/// ```
pub fn companion_roots<T: Real>(
    coeffs: &[T],
    roots: &mut [Complex<T>],
    work: &mut [T],
) -> Result<PolyRootsResult<T>, RootError> {
    if coeffs.iter().any(|c| c.is_nan() || c.is_infinite()) {
        return Err(RootError::NotFinite);
    }
    let Some(hi) = coeffs.iter().rposition(|&c| c != T::ZERO) else {
        return Err(RootError::InvalidInput);
    };
    let lo = coeffs.iter().position(|&c| c != T::ZERO).unwrap_or(0);
    let d = hi - lo;
    if roots.len() < hi || work.len() < companion_work_len(d) {
        return Err(RootError::InvalidInput);
    }
    for r in roots[..lo].iter_mut() {
        *r = Complex::ZERO;
    }
    let result = PolyRootsResult {
        degree: hi,
        iterations: 0,
        last_correction: T::ZERO,
    };
    if d == 0 {
        return Ok(result);
    }
    if d == 1 {
        roots[lo] = Complex::from_real(-coeffs[lo] / coeffs[hi]);
        return Ok(result);
    }
    let a = &mut work[..d * d];
    a.fill(T::ZERO);
    for i in 1..d {
        a[i * d + i - 1] = T::ONE;
    }
    for i in 0..d {
        a[i * d + d - 1] = -coeffs[lo + i] / coeffs[hi];
    }
    eigenvalues_in_place(a, d, &mut roots[lo..hi]).map_err(|e| match e {
        LinalgError::NotFinite => RootError::NotFinite,
        LinalgError::NoConvergence => RootError::NoConvergence,
        _ => RootError::InvalidInput,
    })?;
    Ok(result)
}
