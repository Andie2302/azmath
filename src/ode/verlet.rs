//! Störmer-Verlet-Verfahren (Leapfrog) für separable Hamilton-Systeme.

use super::{OdeError, all_finite, finite, fixed_times};
use crate::Real;

/// Benötigter Arbeitsspeicher für [`verlet`] und [`verlet_separable`] bei `n` Freiheitsgraden
/// (`4·n` Elemente).
pub const fn verlet_work_len(n: usize) -> usize {
    4 * n
}

/// Symplektisches Störmer-Verlet-Verfahren (Ordnung 2) für das separable Hamilton-System
/// `H(q, p) = T(p) + V(q)` mit festen `steps` gleich großen Schritten von `t0` nach `t1`:
///
/// ```text
/// p½  = p + h/2 · F(t, q)              F = −∇V
/// q'  = q + h · ∇T(p½)
/// p'  = p½ + h/2 · F(t + h, q')
/// ```
///
/// - `force(t, q, out)` schreibt die Kraft `F = −∇V(q)` nach `out`.
/// - `drift(p, out)` schreibt `∇T(p)` nach `out` (also `dq/dt`; für `T = ½ pᵀM⁻¹p` ist das
///   `M⁻¹p`).
/// - `q`, `p`: Anfangszustand und Ergebnis (in-place), gleich lang (`n` Freiheitsgrade).
/// - `work`: mindestens [`verlet_work_len`]`(n)` Elemente.
/// - `observer(t, q, p)` wird mit dem Anfangszustand und nach jedem Schritt aufgerufen.
///
/// Eigenschaften: Pro Schritt eine Kraftauswertung (die Kraft am Ende eines Schritts wird im
/// nächsten Schritt wiederverwendet), zeitumkehrbar (ein Lauf mit vertauschten Zeitgrenzen und
/// gleichem `p` macht den Lauf bis auf Rundung rückgängig), keine Allokation, und die Energie
/// bleibt für glatte Systeme über lange Zeiten beschränkt (kein systematisches Wegdriften).
///
/// Fehler: [`OdeError::InvalidInput`] (leerer Zustand, `q` und `p` verschieden lang, `steps = 0`
/// oder zu wenig Arbeitsspeicher), [`OdeError::NotFinite`] (NaN/Unendlich in Zeiten, Zustand,
/// `force` oder `drift`), [`OdeError::StepSizeTooSmall`]. Bei einem Fehler behalten `q` und `p`
/// den zuletzt vollständig berechneten Schritt. Ein leeres Zeitintervall wertet weder `force`
/// noch `drift` aus.
#[allow(clippy::too_many_arguments)]
pub fn verlet_separable<T: Real>(
    mut force: impl FnMut(T, &[T], &mut [T]),
    mut drift: impl FnMut(&[T], &mut [T]),
    t0: T,
    t1: T,
    q: &mut [T],
    p: &mut [T],
    steps: usize,
    work: &mut [T],
    mut observer: impl FnMut(T, &[T], &[T]),
) -> Result<(), OdeError> {
    let n = q.len();
    if n == 0 || p.len() != n || steps == 0 || work.len() < verlet_work_len(n) {
        return Err(OdeError::InvalidInput);
    }
    if !finite(t0) || !finite(t1) || !all_finite(q) || !all_finite(p) {
        return Err(OdeError::NotFinite);
    }
    let (f, rest) = work.split_at_mut(n);
    let (qn, rest) = rest.split_at_mut(n);
    let (pn, rest) = rest.split_at_mut(n);
    let fnew = &mut rest[..n];

    observer(t0, q, p);
    if t0 == t1 {
        return Ok(());
    }
    force(t0, q, f);
    if !all_finite(f) {
        return Err(OdeError::NotFinite);
    }
    let two = T::from_f64(2.0);
    for i in 0..steps {
        let (_, next, h) = fixed_times(t0, t1, i, steps)?;
        let half = h / two;
        for j in 0..n {
            pn[j] = p[j] + half * f[j];
        }
        // `fnew` dient vorübergehend als Puffer für ∇T(p½).
        drift(pn, fnew);
        for j in 0..n {
            qn[j] = q[j] + h * fnew[j];
        }
        if !all_finite(pn) || !all_finite(qn) {
            return Err(OdeError::NotFinite);
        }
        force(next, qn, fnew);
        if !all_finite(fnew) {
            return Err(OdeError::NotFinite);
        }
        for j in 0..n {
            pn[j] += half * fnew[j];
        }
        if !all_finite(pn) {
            return Err(OdeError::NotFinite);
        }
        q.copy_from_slice(qn);
        p.copy_from_slice(pn);
        f.copy_from_slice(fnew);
        observer(next, q, p);
    }
    Ok(())
}

/// Geschwindigkeits-Verlet für die Bewegungsgleichung `q'' = a(t, q)` (Ordnung 2, symplektisch,
/// zeitumkehrbar) mit `steps` gleich großen Schritten von `t0` nach `t1`. Entspricht
/// [`verlet_separable`] mit `∇T(v) = v` und `F = a`; die Masse ist in der Beschleunigung
/// enthalten.
///
/// - `accel(t, q, out)` schreibt die Beschleunigung `a(t, q)` nach `out`.
/// - `q`, `v`: Anfangslage und Anfangsgeschwindigkeit, Ergebnis in-place.
/// - `work`: mindestens [`verlet_work_len`]`(n)` Elemente.
/// - `observer(t, q, v)` wird mit dem Anfangszustand und nach jedem Schritt aufgerufen.
///
/// Fehler und Verhalten bei Fehlern wie bei [`verlet_separable`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::ode::{verlet, verlet_work_len};
///
/// // Harmonischer Oszillator q'' = −q, q(0) = 1, v(0) = 0: nach einer Periode wieder am Start.
/// let accel = |_t: f64, q: &[f64], a: &mut [f64]| a[0] = -q[0];
/// let (mut q, mut v) = ([1.0], [0.0]);
/// let mut work = [0.0; verlet_work_len(1)];
/// let t_end = 2.0 * core::f64::consts::PI;
/// verlet(accel, 0.0, t_end, &mut q, &mut v, 1000, &mut work, |_, _, _| {}).unwrap();
/// assert!((q[0] - 1.0).abs() < 1e-3 && v[0].abs() < 1e-3);
/// # }
/// ```
#[allow(clippy::too_many_arguments)]
pub fn verlet<T: Real>(
    accel: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    q: &mut [T],
    v: &mut [T],
    steps: usize,
    work: &mut [T],
    observer: impl FnMut(T, &[T], &[T]),
) -> Result<(), OdeError> {
    verlet_separable(
        accel,
        |p: &[T], out: &mut [T]| out.copy_from_slice(p),
        t0,
        t1,
        q,
        v,
        steps,
        work,
        observer,
    )
}
