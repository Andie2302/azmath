use crate::Real;

/// Benötigter Arbeitsspeicher für [`numerical_jacobian`] bei `m` Residuen und `n` Parametern.
pub const fn numerical_jacobian_work_len(m: usize, n: usize) -> usize {
    m + n
}

/// Bestimmt die `m × n`-Jacobi-Matrix von `r(x)` per **Vorwärtsdifferenzen**.
///
/// - `residual(x, r)`: schreibt `r(x)` mit `m` Einträgen.
/// - `x`: Stelle (`n` Einträge), `r0`: bereits ausgewertetes `r(x)` (`m` Einträge).
/// - `jac`: Ergebnis, zeilenweise (`m·n` Einträge).
/// - `work`: Arbeitsspeicher mit mindestens [`numerical_jacobian_work_len`]`(m, n)` Einträgen.
///
/// Die Schrittweite ist `√ε · max(1, |xₖ|)`; der Fehler ist von der Ordnung dieser Schrittweite
/// (bei `f64` etwa `1e-8` relativ). Bei nichtendlicher Vorwärtsauswertung wird rückwärts
/// differenziert; sind beide Richtungen ungültig, enthält die betreffende Spalte `NaN`. Ist die analytische Ableitung bekannt, ist sie vorzuziehen.
///
/// # Panics
///
/// Wenn die Slices zu kurz sind.
pub fn numerical_jacobian<T: Real>(
    mut residual: impl FnMut(&[T], &mut [T]),
    x: &[T],
    r0: &[T],
    jac: &mut [T],
    work: &mut [T],
) {
    let (m, n) = (r0.len(), x.len());
    assert!(
        jac.len() >= m * n && work.len() >= numerical_jacobian_work_len(m, n),
        "numerical_jacobian: Slices zu kurz"
    );
    let (xp, rest) = work.split_at_mut(n);
    let r1 = &mut rest[..m];
    xp.copy_from_slice(x);
    let sqrt_eps = T::EPSILON.sqrt();
    for k in 0..n {
        let step = sqrt_eps * T::ONE.max(x[k].abs());
        let mut ok = false;
        for direction in [T::ONE, -T::ONE] {
            xp[k] = x[k] + direction * step;
            let h = xp[k] - x[k];
            if !super::finite(xp[k]) || !super::finite(h) || h == T::ZERO {
                continue;
            }
            r1.fill(T::NAN);
            residual(xp, r1);
            if r1.iter().chain(r0.iter()).all(|&v| super::finite(v)) {
                for i in 0..m {
                    jac[i * n + k] = crate::interp::slope(T::ZERO, h, r0[i], r1[i]);
                }
                ok = true;
                break;
            }
        }
        if !ok {
            for i in 0..m {
                jac[i * n + k] = T::NAN;
            }
        }
        xp[k] = x[k];
    }
}
