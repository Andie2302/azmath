use super::{OdeError, all_finite, finite};
use crate::Real;

/// Benötigter Arbeitsspeicher für [`rk45`] bei `n` Zustandsgrößen.
pub const fn rk45_work_len(n: usize) -> usize {
    // k1 … k7 (7n), Zwischenzustand und neuer Zustand (2n)
    9 * n
}

/// Einstellungen für das adaptive Verfahren [`rk45`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rk45Options<T> {
    /// Relative Toleranz pro Komponente.
    pub rtol: T,
    /// Absolute Toleranz pro Komponente (wichtig, wenn Komponenten nahe 0 kommen).
    pub atol: T,
    /// Anfangsschrittweite; `0` bedeutet: automatisch wählen.
    pub h_init: T,
    /// Größte erlaubte Schrittweite (Betrag); `0` bedeutet: unbegrenzt.
    pub h_max: T,
    /// Höchstzahl der Schrittversuche (akzeptierte und abgelehnte).
    pub max_steps: usize,
}

impl<T: Real> Default for Rk45Options<T> {
    /// `rtol = 1e-6`, `atol = 1e-9`, automatische Anfangsschrittweite, unbegrenzte Maximalschrittweite,
    /// 100 000 Schrittversuche.
    fn default() -> Self {
        Self {
            rtol: T::from_f64(1.0e-6),
            atol: T::from_f64(1.0e-9),
            h_init: T::ZERO,
            h_max: T::ZERO,
            max_steps: 100_000,
        }
    }
}

impl<T: Real> Rk45Options<T> {
    /// Setzt `rtol` und `atol` auf die gegebenen Werte, der Rest wie in `default()`.
    pub fn with_tolerance(rtol: T, atol: T) -> Self {
        Self {
            rtol,
            atol,
            ..Self::default()
        }
    }
}

/// Statistik eines erfolgreichen [`rk45`]-Laufs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rk45Result<T> {
    /// Anzahl der akzeptierten Schritte.
    pub accepted_steps: usize,
    /// Anzahl der abgelehnten Schritte (Fehler zu groß).
    pub rejected_steps: usize,
    /// Anzahl der Auswertungen von `f`.
    pub evaluations: usize,
    /// Die zuletzt verwendete (unbeschnittene) Schrittweite, brauchbar als `h_init` für eine
    /// Fortsetzung.
    pub last_step: T,
}

// Dormand-Prince-Koeffizienten (RK5(4)7M).
const C: [f64; 7] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];
const A: [[f64; 6]; 7] = [
    [0.0; 6],
    [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
    [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
    [
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
        0.0,
        0.0,
    ],
    [
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
        0.0,
    ],
    [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ],
];
/// Differenz der Gewichte 5. und 4. Ordnung (Fehlerschätzer).
const E: [f64; 7] = [
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
];

/// Gewichtete RMS-Norm von `v` bezüglich der Toleranzskala `atol + rtol·|y|`.
fn scaled_norm<T: Real>(v: &[T], scale_from: &[T], rtol: T, atol: T) -> T {
    let mut s = T::ZERO;
    for (vi, yi) in v.iter().zip(scale_from) {
        let q = *vi / (atol + rtol * yi.abs());
        s += q * q;
    }
    (s / T::from_f64(v.len() as f64)).sqrt()
}

/// Adaptives Runge-Kutta-Verfahren von Dormand und Prince (RK5(4), wie `ode45`/`dopri5`).
///
/// Die Schrittweite wird aus einer eingebetteten Fehlerschätzung automatisch so gewählt, dass der
/// lokale Fehler pro Komponente unter `atol + rtol·|y|` bleibt; es nutzt „First Same As Last“ (6
/// Funktionsauswertungen pro Schritt). Für nicht steife Probleme die erste Wahl; bei steifen Problemen
/// werden die Schritte winzig (dann [`implicit_euler`](super::implicit_euler)).
///
/// - `y`: Anfangszustand zur Zeit `t0` und Ergebnis zur Zeit `t1` (in-place); `t1 < t0` integriert
///   rückwärts.
/// - `observer(t, y)` wird mit dem Anfangszustand und nach jedem **akzeptierten** Schritt
///   aufgerufen (die Zeitpunkte sind nicht äquidistant). Für Werte zu bestimmten Zeiten ruft man
///   `rk45` einfach nacheinander für die Teilintervalle auf.
/// - `work`: mindestens [`rk45_work_len`]`(n)` Elemente.
///
/// Fehler: [`OdeError::InvalidInput`], [`OdeError::NotFinite`], [`OdeError::StepSizeTooSmall`],
/// [`OdeError::MaxStepsExceeded`].
pub fn rk45<T: Real>(
    mut f: impl FnMut(T, &[T], &mut [T]),
    t0: T,
    t1: T,
    y: &mut [T],
    opts: Rk45Options<T>,
    work: &mut [T],
    mut observer: impl FnMut(T, &[T]),
) -> Result<Rk45Result<T>, OdeError> {
    let n = y.len();
    if n == 0 || work.len() < rk45_work_len(n) {
        return Err(OdeError::InvalidInput);
    }
    if !finite(t0) || !finite(t1) || !all_finite(y) {
        return Err(OdeError::NotFinite);
    }
    observer(t0, y);
    if t0 == t1 {
        return Ok(Rk45Result {
            accepted_steps: 0,
            rejected_steps: 0,
            evaluations: 0,
            last_step: T::ZERO,
        });
    }

    let (k, rest) = work.split_at_mut(7 * n);
    let (ytmp, rest) = rest.split_at_mut(n);
    let ynew = &mut rest[..n];
    // k als sieben Zeilen k[i*n..(i+1)*n]
    let dir = if t1 > t0 { T::ONE } else { -T::ONE };
    let span = (t1 - t0).abs();
    let (rtol, atol) = (opts.rtol, opts.atol);
    let mut evals = 0usize;

    // k1 = f(t0, y)
    f(t0, y, &mut k[..n]);
    evals += 1;
    if !all_finite(&k[..n]) {
        return Err(OdeError::NotFinite);
    }

    // Anfangsschrittweite (Hairer, Nørsett, Wanner)
    let mut h = if opts.h_init > T::ZERO {
        opts.h_init
    } else {
        let d0 = scaled_norm(y, y, rtol, atol);
        let d1 = scaled_norm(&k[..n], y, rtol, atol);
        let h0 = if d0 < T::from_f64(1.0e-5) || d1 < T::from_f64(1.0e-5) {
            T::from_f64(1.0e-6)
        } else {
            T::from_f64(0.01) * d0 / d1
        };
        for i in 0..n {
            ytmp[i] = y[i] + dir * h0 * k[i];
        }
        f(t0 + dir * h0, ytmp, ynew);
        evals += 1;
        for i in 0..n {
            ynew[i] = (ynew[i] - k[i]) / h0;
        }
        let d2 = scaled_norm(ynew, y, rtol, atol);
        let h1 = if d1.max(d2) <= T::from_f64(1.0e-15) {
            T::from_f64(1.0e-6).max(h0 * T::from_f64(1.0e-3))
        } else {
            (T::from_f64(0.01) / d1.max(d2)).powf(T::from_f64(0.2))
        };
        (T::from_f64(100.0) * h0).min(h1)
    };
    let h_max = if opts.h_max > T::ZERO {
        opts.h_max
    } else {
        span
    };
    h = h.min(h_max).min(span);

    let mut t = t0;
    let (mut accepted, mut rejected) = (0usize, 0usize);
    let mut attempts = 0usize;
    let mut last_step = h;
    let safety = T::from_f64(0.9);
    let min_factor = T::from_f64(0.2);
    let max_factor = T::from_f64(5.0);
    let exponent = T::from_f64(-0.2);

    while dir * (t1 - t) > T::ZERO {
        if attempts >= opts.max_steps {
            return Err(OdeError::MaxStepsExceeded);
        }
        attempts += 1;

        // Letzten Schritt auf t1 beschneiden
        let remaining = dir * (t1 - t);
        let mut step = h;
        let mut last = false;
        // Auch dann auf t1 landen, wenn danach nur ein winziger Rest bliebe
        if step * T::from_f64(1.0001) >= remaining {
            step = remaining;
            last = true;
        }
        let hh = dir * step;
        if t + hh == t {
            return Err(OdeError::StepSizeTooSmall);
        }

        // Stufen k2 … k7
        for s in 1..7 {
            for i in 0..n {
                let mut acc = T::ZERO;
                for j in 0..s {
                    let a = A[s][j];
                    if a != 0.0 {
                        acc += T::from_f64(a) * k[j * n + i];
                    }
                }
                ytmp[i] = y[i] + hh * acc;
            }
            if s < 6 {
                let (done, tail) = k.split_at_mut(s * n);
                let _ = done;
                f(t + hh * T::from_f64(C[s]), ytmp, &mut tail[..n]);
                evals += 1;
            } else {
                // Stufe 7 liegt bei y_neu (FSAL): ytmp ist hier bereits die Lösung 5. Ordnung
                ynew.copy_from_slice(ytmp);
                let tail = &mut k[6 * n..7 * n];
                f(t + hh, ynew, tail);
                evals += 1;
            }
        }
        if !all_finite(ynew) || !all_finite(&k[6 * n..7 * n]) {
            return Err(OdeError::NotFinite);
        }

        // Fehlerschätzung: err = h · Σ eᵢ·kᵢ
        let mut err2 = T::ZERO;
        for i in 0..n {
            let mut e = T::ZERO;
            for (s, es) in E.iter().enumerate() {
                if *es != 0.0 {
                    e += T::from_f64(*es) * k[s * n + i];
                }
            }
            let e = hh * e;
            let sc = atol + rtol * y[i].abs().max(ynew[i].abs());
            let q = e / sc;
            err2 += q * q;
        }
        let err = (err2 / T::from_f64(n as f64)).sqrt();
        if !finite(err) {
            return Err(OdeError::NotFinite);
        }

        if err <= T::ONE {
            // akzeptieren
            accepted += 1;
            t = if last { t1 } else { t + hh };
            y.copy_from_slice(ynew);
            // FSAL: k7 wird zu k1
            let (first, tail) = k.split_at_mut(6 * n);
            first[..n].copy_from_slice(&tail[..n]);
            last_step = step;
            observer(t, y);
            let factor = if err == T::ZERO {
                max_factor
            } else {
                (safety * err.powf(exponent)).min(max_factor)
            };
            h = (step * factor.max(min_factor)).min(h_max);
        } else {
            rejected += 1;
            let factor = (safety * err.powf(exponent)).max(min_factor);
            h = step * factor.min(T::ONE);
        }
    }

    Ok(Rk45Result {
        accepted_steps: accepted,
        rejected_steps: rejected,
        evaluations: evals,
        last_step,
    })
}
