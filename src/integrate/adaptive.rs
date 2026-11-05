use super::{
    IntegralResult, IntegrateError, IntegrateOptions, WeightedSum, finite, integral_scale, point,
};
use crate::Real;

/// Gauß-Kronrod-Knoten (positive Hälfte, absteigend) der 15-Punkte-Kronrod-Regel auf `[−1, 1]`.
const XGK: [f64; 8] = [
    0.991_455_371_120_812_6,
    0.949_107_912_342_758_5,
    0.864_864_423_359_769_1,
    0.741_531_185_599_394_4,
    0.586_087_235_467_691_1,
    0.405_845_151_377_397_2,
    0.207_784_955_007_898_5,
    0.0,
];
/// Gewichte der 15-Punkte-Kronrod-Regel zu `XGK`.
const WGK: [f64; 8] = [
    0.022_935_322_010_529_22,
    0.063_092_092_629_978_55,
    0.104_790_010_322_250_2,
    0.140_653_259_715_525_9,
    0.169_004_726_639_267_9,
    0.190_350_578_064_785_4,
    0.204_432_940_075_298_9,
    0.209_482_141_084_727_8,
];
/// Gewichte der 7-Punkte-Gauß-Regel zu den Knoten `XGK[1]`, `XGK[3]`, `XGK[5]`, `XGK[7]`.
const WG: [f64; 4] = [
    0.129_484_966_168_869_7,
    0.279_705_391_489_276_7,
    0.381_830_050_505_118_9,
    0.417_959_183_673_469_4,
];

/// Maximale Zahl gleichzeitig gehaltener Teilintervalle (fester Speicher, kein Allokator nötig).
const MAX_SEGMENTS: usize = 256;

/// Ein Teilintervall mit seiner G7/K15-Auswertung.
#[derive(Clone, Copy)]
struct Segment<T> {
    a: T,
    b: T,
    value: T,
    error: T,
    /// Näherung von ∫|f| auf dem Teilintervall, für die Rundungsfehler-Grenze.
    resabs: T,
    /// `false`, wenn das Intervall in Maschinengenauigkeit nicht weiter teilbar ist.
    splittable: bool,
}

impl<T: Real> Segment<T> {
    /// Fehlerniveau, unterhalb dessen der Fehler nur noch Rundungsrauschen ist.
    fn noise(&self) -> T {
        T::from_f64(50.0) * T::EPSILON * self.resabs
    }
}

/// Wertet die G7/K15-Regel auf `[a, b]` aus (15 Funktionsauswertungen).
fn gk15<T: Real, F: FnMut(T) -> T>(
    f: &mut F,
    evals: &mut usize,
    a: T,
    b: T,
) -> Result<Segment<T>, IntegrateError> {
    let half = T::from_f64(0.5);
    let center = point(a, b, half);
    if center <= a || center >= b {
        return Err(IntegrateError::NoConvergence);
    }
    let fc = f(center);
    *evals += 1;
    let mut kronrod = WeightedSum::new();
    let mut gauss = WeightedSum::new();
    let mut resabs = WeightedSum::new();
    kronrod.add(fc, T::from_f64(WGK[7]) * half)?;
    gauss.add(fc, T::from_f64(WG[3]) * half)?;
    resabs.add(fc.abs(), T::from_f64(WGK[7]) * half)?;
    for j in 0..7 {
        let node = T::from_f64(XGK[j]);
        for t in [(T::ONE - node) * half, (T::ONE + node) * half] {
            let x = point(a, b, t);
            if x <= a || x >= b {
                return Err(IntegrateError::NoConvergence);
            }
            let value = f(x);
            *evals += 1;
            kronrod.add(value, T::from_f64(WGK[j]) * half)?;
            resabs.add(value.abs(), T::from_f64(WGK[j]) * half)?;
            if j % 2 == 1 {
                gauss.add(value, T::from_f64(WG[j / 2]) * half)?;
            }
        }
    }
    let value = integral_scale(kronrod.value(), a, b);
    let error = (value - integral_scale(gauss.value(), a, b)).abs();
    let resabs = integral_scale(resabs.value(), a, b);
    if !finite(value) || !finite(error) || !finite(resabs) {
        return Err(IntegrateError::NotFinite);
    }
    Ok(Segment {
        a,
        b,
        value,
        error,
        resabs,
        splittable: true,
    })
}

/// Adaptives Gauß-Kronrod-Verfahren (G7/K15): berechnet `∫ₐᵇ f(x) dx` bis die geschätzte
/// Gesamtabweichung `≤ max(abs_tol, rel_tol · |I|)` ist.
///
/// Es wird jeweils das Teilintervall mit dem **größten Fehler** halbiert (Strategie von
/// QUADPACK `qag`), sodass die Auswertungen dort landen, wo sie gebraucht werden. Die Funktion
/// wird **nicht** an den Intervallenden ausgewertet, daher sind Randsingularitäten wie `√x`,
/// `ln x` oder `1/√x` bei 0 erlaubt. Dafür sind aber viele Teilungen nötig: ein Randverhalten
/// `x^(−α)` braucht etwa `log₂(1/tol)/(1−α)` Halbierungen, `1/√x` bei der Standardtoleranz zum
/// Beispiel rund 80, stärkere Singularitäten schnell mehr als `max_intervals`; dann hilft eine
/// Substitution (bei `1/√x` zum Beispiel `x = t²`). Liegt eine solche Singularität im
/// **Inneren** des Intervalls, endet die Halbierung an der Auflösung der Gleitkommazahlen
/// (etwa `1e-16`); dort sind nur lockere Toleranzen erreichbar, und fällt ein Auswertungspunkt
/// exakt auf die Singularität, ergibt sich [`IntegrateError::NotFinite`]. Am besten teilt man
/// das Intervall an der Singularität und integriert die Teile getrennt.
///
/// **Grenzen:** Wie jede symmetrische Regel kann das Verfahren von Integranden getäuscht werden,
/// die um den Intervallmittelpunkt exakt (anti)symmetrisch sind, etwa `sin(50x)` über `[0, π]`:
/// die Knoten heben sich auf, beide Teilregeln liefern 0 und der Fehlerschätzer meldet
/// Konvergenz. Bei Verdacht das Intervall unsymmetrisch teilen und die Teile addieren.
///
/// Vertauschte Grenzen (`a > b`) liefern das negative Integral, `a = b` ergibt 0.
///
/// Fehler: [`IntegrateError::NoConvergence`] (Toleranz mit `max_intervals`/`max_evals` nicht
/// erreichbar, zum Beispiel bei `1/x` über 0), [`IntegrateError::NotFinite`] (NaN/Unendlich in
/// den Grenzen oder in `f`), [`IntegrateError::InvalidInput`] (negative/nichtendliche
/// Toleranzen oder `max_intervals = 0`). Ein Budget unter 15 erlaubt bei nichtleerem
/// Intervall keine erste Regel. Rundungsstillstand oberhalb der Toleranz ergibt `NoConvergence`.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::integrate::{adaptive, IntegrateOptions};
/// use azmath::Real;
///
/// // Randsingularität: ∫₀¹ ln(x) dx = −1
/// let r = adaptive(|x: f64| Real::ln(x), 0.0, 1.0, IntegrateOptions::default()).unwrap();
/// assert!((r.value + 1.0).abs() < 1e-10);
/// # }
/// ```
pub fn adaptive<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    opts: IntegrateOptions<T>,
) -> Result<IntegralResult<T>, IntegrateError> {
    if !finite(opts.abs_tol)
        || !finite(opts.rel_tol)
        || opts.abs_tol < T::ZERO
        || opts.rel_tol < T::ZERO
        || opts.max_intervals == 0
    {
        return Err(IntegrateError::InvalidInput);
    }
    if !finite(a) || !finite(b) {
        return Err(IntegrateError::NotFinite);
    }
    if a == b {
        return Ok(IntegralResult {
            value: T::ZERO,
            error_estimate: T::ZERO,
            evaluations: 0,
        });
    }
    if opts.max_evals < 15 {
        return Err(IntegrateError::NoConvergence);
    }
    // Intern immer aufsteigend rechnen und am Ende das Vorzeichen setzen.
    let (lo, hi, sign) = if a < b {
        (a, b, T::ONE)
    } else {
        (b, a, -T::ONE)
    };

    let capacity = opts.max_intervals.clamp(1, MAX_SEGMENTS);
    let empty = Segment {
        a: T::ZERO,
        b: T::ZERO,
        value: T::ZERO,
        error: T::ZERO,
        resabs: T::ZERO,
        splittable: false,
    };
    let mut segs = [empty; MAX_SEGMENTS];
    let mut n = 1;
    let mut evals = 0;
    segs[0] = gk15(&mut f, &mut evals, lo, hi)?;

    loop {
        let mut value = T::ZERO;
        let mut error = T::ZERO;
        for s in &segs[..n] {
            value += s.value;
            error += s.error;
        }
        let tol = opts.abs_tol.max(opts.rel_tol * value.abs());
        if error <= tol {
            return finish(sign * value, error, evals);
        }

        // Teilbares Intervall mit dem größten Fehler oberhalb des Rundungsrauschens suchen.
        let mut worst: Option<usize> = None;
        let mut worst_err = T::ZERO;
        for (i, s) in segs[..n].iter().enumerate() {
            if s.splittable && s.error > s.noise() && s.error > worst_err {
                worst = Some(i);
                worst_err = s.error;
            }
        }
        let Some(i) = worst else {
            // Rundungsrauschen oder nicht mehr teilbare Intervalle rechtfertigen keinen
            // Erfolg, wenn der Fehlerschätzer die geforderte Toleranz überschreitet.
            return Err(IntegrateError::NoConvergence);
        };
        if n >= capacity || evals + 30 > opts.max_evals {
            return Err(IntegrateError::NoConvergence);
        }

        let (sa, sb) = (segs[i].a, segs[i].b);
        let mid = point(sa, sb, T::from_f64(0.5));
        if !(sa < mid && mid < sb) {
            segs[i].splittable = false;
            continue;
        }
        segs[i] = gk15(&mut f, &mut evals, sa, mid)?;
        segs[n] = gk15(&mut f, &mut evals, mid, sb)?;
        n += 1;
    }
}

fn finish<T: Real>(
    value: T,
    error_estimate: T,
    evaluations: usize,
) -> Result<IntegralResult<T>, IntegrateError> {
    if finite(value) && finite(error_estimate) && error_estimate >= T::ZERO {
        Ok(IntegralResult {
            value,
            error_estimate,
            evaluations,
        })
    } else {
        Err(IntegrateError::NotFinite)
    }
}
