use azmath_core::Real;

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
/// (bei `f64` etwa `1e-8` relativ). Ist die analytische Ableitung bekannt, ist sie vorzuziehen.
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
        xp[k] = x[k] + step;
        // tatsächlich ausgeführter Schritt (vermeidet Rundungsfehler im Nenner)
        let h = xp[k] - x[k];
        residual(xp, r1);
        for i in 0..m {
            jac[i * n + k] = (r1[i] - r0[i]) / h;
        }
        xp[k] = x[k];
    }
}
