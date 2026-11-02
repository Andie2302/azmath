use super::super::iterative::{IterOptions, dot};
use super::super::{LinalgError, LinearOperator};
use crate::Real;

/// Ergebnis einer Potenz- oder inversen Iteration (der Eigenvektor steht im übergebenen `x`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EigenResult<T> {
    /// Der gefundene Eigenwert.
    pub eigenvalue: T,
    /// Anzahl der Iterationen.
    pub iterations: usize,
}

/// Benötigter Arbeitsspeicher für [`power_iteration`] und [`inverse_iteration`] bei Dimension `n`.
pub const fn power_iteration_work_len(n: usize) -> usize {
    n
}

fn bad<T: Real>(v: T) -> bool {
    v.is_nan() || v.is_infinite()
}

fn norm<T: Real>(v: &[T]) -> T {
    dot(v, v).sqrt()
}

/// Gemeinsamer Kern: Iteration `x ← M·x / ‖M·x‖` mit Rayleigh-Quotient `μ = xᵀ·M·x`.
/// Abbruch bei `‖M·x − μ·x‖ ≤ tol·|μ|`. Rückgabe: `(μ, Iterationen)`; `x` ist danach der normierte
/// Eigenvektor von `M` zum Eigenwert `μ`.
fn iterate<T: Real>(
    mut apply: impl FnMut(&[T], &mut [T]),
    x: &mut [T],
    opts: IterOptions<T>,
    y: &mut [T],
) -> Result<(T, usize), LinalgError> {
    if x.is_empty() || y.len() < x.len() {
        return Err(LinalgError::DimensionMismatch);
    }
    let n = x.len();
    let y = &mut y[..n];
    if x.iter().any(|v| bad(*v)) {
        return Err(LinalgError::NotFinite);
    }
    let nx = norm(x);
    if nx == T::ZERO {
        return Err(LinalgError::Breakdown);
    }
    for v in x.iter_mut() {
        *v /= nx;
    }
    for k in 0..opts.max_iter {
        apply(x, y);
        let mu = dot(x, y);
        if bad(mu) {
            return Err(LinalgError::NotFinite);
        }
        let mut r2 = T::ZERO;
        for (xi, yi) in x.iter().zip(y.iter()) {
            let r = *yi - mu * *xi;
            r2 += r * r;
        }
        if r2.sqrt() <= opts.tol * mu.abs() {
            return Ok((mu, k));
        }
        let ny = norm(y);
        if bad(ny) {
            return Err(LinalgError::NotFinite);
        }
        for (xi, yi) in x.iter_mut().zip(y.iter()) {
            *xi = *yi / ny;
        }
    }
    Err(LinalgError::NoConvergence)
}

/// Potenzmethode: bestimmt den betragsgrößten Eigenwert und den zugehörigen Eigenvektor von `A`.
///
/// Matrixfrei (braucht nur `A·x`); die Konvergenz ist linear mit dem Faktor `|λ₂/λ₁|`, also langsam,
/// wenn die beiden größten Eigenwerte dicht beieinander liegen. Konvergiert nicht, wenn der betragsgrößte
/// Eigenwert komplex ist oder mit einem gleich großen (`−λ`) zusammenfällt.
///
/// - `x`: Startvektor (nicht null, nicht senkrecht zum gesuchten Eigenvektor) und Ergebnis (normierter
///   Eigenvektor). Ein **glatter** Startvektor enthält den oft hochfrequenten (alternierenden) dominanten
///   Eigenvektor kaum; ein Vektor mit zufälligen Einträgen ist robuster (die Komfortmethoden
///   [`dominant_eigenpair`](crate::linalg::Matrix::dominant_eigenpair) nehmen einen deterministischen
///   Pseudozufallsvektor).
/// - `work`: mindestens [`power_iteration_work_len`]`(n)` Elemente.
/// - Abbruch bei `‖A·x − λ·x‖ ≤ tol·|λ|` (`IterOptions::tol`). Das Residuum garantiert, dass `λ` höchstens
///   `‖A·x − λ·x‖` von einem Eigenwert entfernt liegt (bei symmetrischen Matrizen sogar quadratisch
///   besser), nicht aber, dass es der **dominante** ist: Fehlt dem Startvektor der Anteil des gesuchten
///   Eigenvektors fast ganz, kann das Verfahren bei einem anderen Eigenwert anhalten.
///
/// Fehler: [`LinalgError::DimensionMismatch`], [`LinalgError::NotFinite`], [`LinalgError::Breakdown`]
/// (Startvektor null), [`LinalgError::NoConvergence`] (`x` enthält dann die letzte Näherung).
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::linalg::eigen::{power_iteration, power_iteration_work_len};
/// use azmath::linalg::iterative::IterOptions;
/// use azmath::linalg::Matrix;
///
/// // Eigenwerte 3 und 1
/// let a = Matrix::new([[2.0f64, 1.0], [1.0, 2.0]]);
/// let mut x = [1.0, 0.5];
/// let mut work = [0.0; power_iteration_work_len(2)];
/// let r = power_iteration(&a, &mut x, IterOptions::new(1e-12, 200), &mut work).unwrap();
/// assert!((r.eigenvalue - 3.0).abs() < 1e-10);
/// # }
/// ```
pub fn power_iteration<T: Real, A: LinearOperator<T> + ?Sized>(
    a: &A,
    x: &mut [T],
    opts: IterOptions<T>,
    work: &mut [T],
) -> Result<EigenResult<T>, LinalgError> {
    let n = x.len();
    if a.rows() != n || a.cols() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    let (eigenvalue, iterations) = iterate(|v, out| a.apply(v, out), x, opts, work)?;
    Ok(EigenResult {
        eigenvalue,
        iterations,
    })
}

/// Inverse Iteration mit Verschiebung: bestimmt den Eigenwert von `A`, der einem Wert `shift` am
/// **nächsten** liegt, und den Eigenvektor.
///
/// `solve(b, y)` muss `y = (A − shift·I)⁻¹·b` berechnen (etwa per LU-Zerlegung). Die Konvergenz ist
/// linear mit dem Faktor `|λ₁ − σ| / |λ₂ − σ|`: je näher `shift` am gesuchten Eigenwert, desto schneller
/// (oft 2 bis 3 Schritte). Liegt `shift` exakt auf einem Eigenwert, ist `A − σI` singulär; dann `shift`
/// minimal verschieben. Bequemer: [`Matrix::eigenpair_near`](crate::linalg::Matrix::eigenpair_near).
///
/// Parameter und Fehler wie bei [`power_iteration`].
pub fn inverse_iteration<T: Real>(
    solve: impl FnMut(&[T], &mut [T]),
    shift: T,
    x: &mut [T],
    opts: IterOptions<T>,
    work: &mut [T],
) -> Result<EigenResult<T>, LinalgError> {
    let (mu, iterations) = iterate(solve, x, opts, work)?;
    if mu == T::ZERO {
        return Err(LinalgError::Breakdown);
    }
    Ok(EigenResult {
        eigenvalue: shift + T::ONE / mu,
        iterations,
    })
}
