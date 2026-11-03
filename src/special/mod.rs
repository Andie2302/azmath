//! Spezialfunktionen: Gamma-, Beta-, Fehlerfunktion, unvollständige Gamma-/Beta-Funktion und Besselfunktionen `Jₙ`.
//!
//! | Funktion | Bedeutung |
//! |---|---|
//! | [`gamma`], [`ln_gamma`], [`digamma`] | Gammafunktion `Γ(x)`, `ln|Γ(x)|`, `ψ(x) = Γ′/Γ` |
//! | [`beta`], [`ln_beta`], [`beta_inc`] | Betafunktion `B(a, b)`, regularisierte unvollständige Betafunktion `I_x(a, b)` |
//! | [`gamma_p`], [`gamma_q`] | regularisierte unvollständige Gammafunktionen `P(a, x)` und `Q(a, x) = 1 − P(a, x)` |
//! | [`erf`], [`erfc`], [`erf_inv`] | Fehlerfunktion, komplementäre Fehlerfunktion, Umkehrfunktion |
//! | [`normal_cdf`], [`normal_quantile`] | Verteilungsfunktion und Quantilfunktion der Standardnormalverteilung |
//! | [`bessel_j`], [`bessel_i`] | Besselfunktion `Jₙ(x)` und modifizierte Besselfunktion `Iₙ(x)` für ganzzahlige Ordnung |
//! | [`elliptic_k`], [`elliptic_e`] | vollständige elliptische Integrale erster und zweiter Art |
//!
//! Verfahren: Lanczos-Näherung für `ln Γ` (`g = 7`, relativer Fehler etwa `10⁻¹⁵`), Reihe und Kettenbruch (Lentz) für die
//! unvollständigen Funktionen, `erf = P(½, x²)`, Miller-Rückwärtsrekursion für `Jₙ`. Alle Funktionen sind generisch über
//! [`Real`]; die Genauigkeit entspricht dem Typ nur bei `f64`, bei `f32` ist sie durch die Rundung begrenzt.
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::special::{erf, gamma};
//!
//! assert!((gamma(5.0f64) - 24.0).abs() < 1e-12);
//! assert!((erf(1.0f64) - 0.8427007929497149).abs() < 1e-14);
//! # }
//! ```

use crate::Real;

const LANCZOS_G: f64 = 7.0;
const LANCZOS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.520_368_121_885_1,
    -1_259.139_216_722_402_8,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_311_6e-7,
];
const HALF_LN_2PI: f64 = 0.918_938_533_204_672_7;
const MAX_ITER: usize = 1000;

fn fpmin<T: Real>() -> T {
    T::MIN_POSITIVE / T::EPSILON
}

fn is_nonpositive_integer<T: Real>(x: T) -> bool {
    x <= T::ZERO && x == x.floor()
}

/// Natürlicher Logarithmus des Betrags der Gammafunktion, `ln|Γ(x)|` (Lanczos-Näherung, Reflexionsformel für `x < ½`).
/// An den Polen `0, −1, −2, …` ergibt sich `+∞`.
pub fn ln_gamma<T: Real>(x: T) -> T {
    if x.is_nan() {
        return x;
    }
    if is_nonpositive_integer(x) {
        return T::INFINITY;
    }
    let half = T::from_f64(0.5);
    if x < half {
        // Γ(x)·Γ(1−x) = π/sin(πx)
        return (T::PI / (T::PI * x).sin().abs()).ln() - ln_gamma(T::ONE - x);
    }
    let x = x - T::ONE;
    let mut a = T::from_f64(LANCZOS[0]);
    for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
        a += T::from_f64(c) / (x + T::from_usize(i));
    }
    let t = x + T::from_f64(LANCZOS_G) + half;
    T::from_f64(HALF_LN_2PI) + (x + half) * t.ln() - t + a.ln()
}

/// Gammafunktion `Γ(x)`. Ganze Zahlen `1 ≤ x ≤ 171` liefern die exakte Fakultät `(x−1)!`; negative Werte über die
/// Reflexionsformel; an den Polen `NaN`; ab etwa `x > 171,6` überläuft das Ergebnis zu `+∞`.
pub fn gamma<T: Real>(x: T) -> T {
    if x.is_nan() || is_nonpositive_integer(x) {
        return T::NAN;
    }
    if x == x.floor() && x >= T::ONE && x <= T::from_f64(171.0) {
        let mut p = T::ONE;
        let mut k = T::from_f64(2.0);
        while k < x {
            p *= k;
            k += T::ONE;
        }
        return p;
    }
    if x < T::from_f64(0.5) {
        return T::PI / ((T::PI * x).sin() * gamma(T::ONE - x));
    }
    // Lanczos direkt (ohne Umweg über den Logarithmus), Potenz geteilt gegen Überlauf
    let xm = x - T::ONE;
    let half = T::from_f64(0.5);
    let mut a = T::from_f64(LANCZOS[0]);
    for (i, &c) in LANCZOS.iter().enumerate().skip(1) {
        a += T::from_f64(c) / (xm + T::from_usize(i));
    }
    let t = xm + T::from_f64(LANCZOS_G) + half;
    let p = t.powf((xm + half) * half);
    T::from_f64(HALF_LN_2PI).exp() * p * (-t).exp() * p * a
}

/// Digammafunktion `ψ(x) = d/dx ln Γ(x)` (Reflexion für `x < ½`, Rekursion bis `x ≥ 6`, danach asymptotische Reihe).
/// An den Polen `NaN`.
pub fn digamma<T: Real>(x: T) -> T {
    if x.is_nan() || is_nonpositive_integer(x) {
        return T::NAN;
    }
    if x < T::from_f64(0.5) {
        return digamma(T::ONE - x) - T::PI / (T::PI * x).tan();
    }
    let mut x = x;
    let mut shift = T::ZERO;
    while x < T::from_f64(6.0) {
        shift -= T::ONE / x;
        x += T::ONE;
    }
    let inv = T::ONE / x;
    let inv2 = inv * inv;
    // ln x − 1/(2x) − Σ B₂ₖ/(2k·x²ᵏ)
    let series = inv2
        * (T::from_f64(1.0 / 12.0)
            - inv2
                * (T::from_f64(1.0 / 120.0)
                    - inv2
                        * (T::from_f64(1.0 / 252.0)
                            - inv2
                                * (T::from_f64(1.0 / 240.0) - inv2 * T::from_f64(1.0 / 132.0)))));
    shift + x.ln() - T::from_f64(0.5) * inv - series
}

/// Logarithmus der Betafunktion, `ln B(a, b) = ln Γ(a) + ln Γ(b) − ln Γ(a + b)` (für `a, b > 0`).
pub fn ln_beta<T: Real>(a: T, b: T) -> T {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Betafunktion `B(a, b) = Γ(a)Γ(b)/Γ(a + b)` (für `a, b > 0`).
pub fn beta<T: Real>(a: T, b: T) -> T {
    ln_beta(a, b).exp()
}

/// Reihenentwicklung von `P(a, x)` (günstig für `x < a + 1`).
fn gamma_series<T: Real>(a: T, x: T) -> T {
    let mut ap = a;
    let mut del = T::ONE / a;
    let mut sum = del;
    for _ in 0..MAX_ITER {
        ap += T::ONE;
        del *= x / ap;
        sum += del;
        if del.abs() < sum.abs() * T::EPSILON {
            break;
        }
    }
    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Kettenbruch (Lentz) für `Q(a, x)` (günstig für `x ≥ a + 1`).
fn gamma_continued_fraction<T: Real>(a: T, x: T) -> T {
    let tiny = fpmin::<T>();
    let mut b = x + T::ONE - a;
    let mut c = T::ONE / tiny;
    let mut d = T::ONE / b;
    let mut h = d;
    for i in 1..=MAX_ITER {
        let fi = T::from_usize(i);
        let an = -fi * (fi - a);
        b += T::from_f64(2.0);
        d = an * d + b;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b + an / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = T::ONE / d;
        let del = d * c;
        h *= del;
        if (del - T::ONE).abs() < T::EPSILON {
            break;
        }
    }
    (-x + a * x.ln() - ln_gamma(a)).exp() * h
}

/// Regularisierte untere unvollständige Gammafunktion `P(a, x) = γ(a, x)/Γ(a)` (`a > 0`, `x ≥ 0`; sonst `NaN`).
pub fn gamma_p<T: Real>(a: T, x: T) -> T {
    if a.is_nan() || x.is_nan() || a <= T::ZERO || x < T::ZERO {
        return T::NAN;
    }
    if x == T::ZERO {
        return T::ZERO;
    }
    if x.is_infinite() {
        return T::ONE;
    }
    if x < a + T::ONE {
        gamma_series(a, x)
    } else {
        T::ONE - gamma_continued_fraction(a, x)
    }
}

/// Regularisierte obere unvollständige Gammafunktion `Q(a, x) = 1 − P(a, x)` (`a > 0`, `x ≥ 0`; sonst `NaN`); für große `x`
/// ohne Auslöschung berechnet.
pub fn gamma_q<T: Real>(a: T, x: T) -> T {
    if a.is_nan() || x.is_nan() || a <= T::ZERO || x < T::ZERO {
        return T::NAN;
    }
    if x == T::ZERO {
        return T::ONE;
    }
    if x.is_infinite() {
        return T::ZERO;
    }
    if x < a + T::ONE {
        T::ONE - gamma_series(a, x)
    } else {
        gamma_continued_fraction(a, x)
    }
}

/// Fehlerfunktion `erf(x) = 2/√π ∫₀ˣ e^{−t²} dt`.
pub fn erf<T: Real>(x: T) -> T {
    if x.is_nan() {
        return x;
    }
    let r = gamma_p(T::from_f64(0.5), x * x);
    if x < T::ZERO { -r } else { r }
}

/// Komplementäre Fehlerfunktion `erfc(x) = 1 − erf(x)`, auch für große `x` ohne Auslöschung.
pub fn erfc<T: Real>(x: T) -> T {
    if x.is_nan() {
        return x;
    }
    let q = gamma_q(T::from_f64(0.5), x * x);
    if x < T::ZERO { T::from_f64(2.0) - q } else { q }
}

/// Umkehrfunktion der Fehlerfunktion auf `(−1, 1)`: Startwert nach Winitzki, danach Halley-Schritte. `±1` ergibt `±∞`,
/// Werte außerhalb `NaN`.
pub fn erf_inv<T: Real>(y: T) -> T {
    if y.is_nan() || y.abs() > T::ONE {
        return T::NAN;
    }
    if y == T::ONE {
        return T::INFINITY;
    }
    if y == -T::ONE {
        return T::NEG_INFINITY;
    }
    if y == T::ZERO {
        return T::ZERO;
    }
    let a = T::from_f64(0.147);
    let ln1 = (T::ONE - y * y).ln();
    let t = T::from_f64(2.0) / (T::PI * a) + T::from_f64(0.5) * ln1;
    let mut x = ((t * t - ln1 / a).sqrt() - t).sqrt();
    if y < T::ZERO {
        x = -x;
    }
    let two_over_sqrt_pi = T::from_f64(2.0) / T::PI.sqrt();
    for _ in 0..4 {
        let err = erf(x) - y;
        let d = two_over_sqrt_pi * (-x * x).exp();
        if d == T::ZERO {
            break;
        }
        // Halley: x ← x − f/(f′ − f·f″/(2f′)) mit f″ = −2x·f′
        let step = err / (d + x * err);
        x -= step;
        if step.abs() <= T::EPSILON * x.abs() {
            break;
        }
    }
    x
}

/// Verteilungsfunktion der Standardnormalverteilung `Φ(x) = ½·erfc(−x/√2)`.
pub fn normal_cdf<T: Real>(x: T) -> T {
    T::from_f64(0.5) * erfc(-x / T::from_f64(core::f64::consts::SQRT_2))
}

/// Quantilfunktion der Standardnormalverteilung `Φ⁻¹(p)` für `p ∈ (0, 1)` (`0 → −∞`, `1 → +∞`, sonst `NaN`).
pub fn normal_quantile<T: Real>(p: T) -> T {
    if p.is_nan() || p < T::ZERO || p > T::ONE {
        return T::NAN;
    }
    T::from_f64(core::f64::consts::SQRT_2) * erf_inv(T::from_f64(2.0) * p - T::ONE)
}

/// Kettenbruch (Lentz) für die unvollständige Betafunktion.
fn beta_continued_fraction<T: Real>(a: T, b: T, x: T) -> T {
    let tiny = fpmin::<T>();
    let (qab, qap, qam) = (a + b, a + T::ONE, a - T::ONE);
    let mut c = T::ONE;
    let mut d = T::ONE - qab * x / qap;
    if d.abs() < tiny {
        d = tiny;
    }
    d = T::ONE / d;
    let mut h = d;
    for m in 1..=MAX_ITER {
        let fm = T::from_usize(m);
        let m2 = T::from_f64(2.0) * fm;
        let mut aa = fm * (b - fm) * x / ((qam + m2) * (a + m2));
        d = T::ONE + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = T::ONE + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = T::ONE / d;
        h *= d * c;
        aa = -(a + fm) * (qab + fm) * x / ((a + m2) * (qap + m2));
        d = T::ONE + aa * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = T::ONE + aa / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = T::ONE / d;
        let del = d * c;
        h *= del;
        if (del - T::ONE).abs() < T::EPSILON {
            break;
        }
    }
    h
}

/// Regularisierte unvollständige Betafunktion `I_x(a, b)` (`a, b > 0`, `0 ≤ x ≤ 1`; sonst `NaN`) – die Verteilungsfunktion
/// der Beta- und (über eine Variablensubstitution) der Student-t- und F-Verteilung.
pub fn beta_inc<T: Real>(a: T, b: T, x: T) -> T {
    if a.is_nan()
        || b.is_nan()
        || x.is_nan()
        || a <= T::ZERO
        || b <= T::ZERO
        || x < T::ZERO
        || x > T::ONE
    {
        return T::NAN;
    }
    if x == T::ZERO {
        return T::ZERO;
    }
    if x == T::ONE {
        return T::ONE;
    }
    let bt =
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (T::ONE - x).ln()).exp();
    if x < (a + T::ONE) / (a + b + T::from_f64(2.0)) {
        bt * beta_continued_fraction(a, b, x) / a
    } else {
        T::ONE - bt * beta_continued_fraction(b, a, T::ONE - x) / b
    }
}

/// Besselfunktion erster Art `Jₙ(x)` für ganzzahlige Ordnung `n ≥ 0` und beliebiges reelles `x` (Miller-Rückwärtsrekursion mit
/// Normierung `J₀ + 2ΣJ₂ₖ = 1`; stabil für alle `n`, Rechenzeit etwa `O(|x| + n)`). Für `|x|` über etwa `10⁶` unbrauchbar langsam.
pub fn bessel_j<T: Real>(n: u32, x: T) -> T {
    if x.is_nan() {
        return x;
    }
    if x == T::ZERO {
        return if n == 0 { T::ONE } else { T::ZERO };
    }
    if x.is_infinite() {
        return T::ZERO;
    }
    let ax = x.abs();
    let big = T::from_f64(1e10);
    let big_inv = T::from_f64(1e-10);
    let scale = T::from_f64(n as f64).max(ax);
    // gerade Startordnung oberhalb von max(n, |x|)
    let start = scale.to_f64() + (160.0 * scale.to_f64()).sqrt();
    let m = 2 * ((start as usize + 1) / 2) + 2;
    let tox = T::from_f64(2.0) / ax;
    let (mut bjp, mut bj) = (T::ZERO, T::ONE);
    let (mut sum, mut ans) = (T::ZERO, T::ZERO);
    let mut jsum = false;
    for j in (1..=m).rev() {
        let bjm = T::from_usize(j) * tox * bj - bjp;
        bjp = bj;
        bj = bjm;
        if bj.abs() > big {
            bj *= big_inv;
            bjp *= big_inv;
            ans *= big_inv;
            sum *= big_inv;
        }
        if jsum {
            sum += bj;
        }
        jsum = !jsum;
        if j == n as usize {
            ans = bjp;
        }
    }
    if n == 0 {
        ans = bj;
    }
    sum = T::from_f64(2.0) * sum - bj;
    let r = ans / sum;
    if x < T::ZERO && n % 2 == 1 { -r } else { r }
}

/// Vollständiges elliptisches Integral erster Art `K(m) = ∫₀^{π/2} dθ/√(1 − m·sin²θ)` mit Parameter `m = k² ∈ [0, 1]`
/// (arithmetisch-geometrisches Mittel). `m = 1` liefert `+∞`, Werte außerhalb von `[0, 1]` `NaN`.
pub fn elliptic_k<T: Real>(m: T) -> T {
    if m.is_nan() || m < T::ZERO || m > T::ONE {
        return T::NAN;
    }
    if m == T::ONE {
        return T::INFINITY;
    }
    let (mut a, mut b) = (T::ONE, (T::ONE - m).sqrt());
    for _ in 0..60 {
        if (a - b).abs() <= T::EPSILON * a {
            break;
        }
        let an = (a + b) * T::from_f64(0.5);
        b = (a * b).sqrt();
        a = an;
    }
    T::PI / (T::from_f64(2.0) * a)
}

/// Vollständiges elliptisches Integral zweiter Art `E(m) = ∫₀^{π/2} √(1 − m·sin²θ) dθ` mit Parameter `m ∈ [0, 1]`.
/// Werte außerhalb von `[0, 1]` liefern `NaN`.
pub fn elliptic_e<T: Real>(m: T) -> T {
    if m.is_nan() || m < T::ZERO || m > T::ONE {
        return T::NAN;
    }
    if m == T::ONE {
        return T::ONE;
    }
    let half = T::from_f64(0.5);
    let (mut a, mut b) = (T::ONE, (T::ONE - m).sqrt());
    // E = K·(1 − Σ 2^(n−1)·cₙ²), c₀² = m
    let mut sum = half * m;
    let mut pow2 = half;
    for _ in 0..60 {
        if (a - b).abs() <= T::EPSILON * a {
            break;
        }
        let c = (a - b) * half;
        let an = (a + b) * half;
        b = (a * b).sqrt();
        a = an;
        pow2 = pow2 * T::from_f64(2.0);
        sum += pow2 * c * c;
    }
    let k = T::PI / (T::from_f64(2.0) * a);
    k * (T::ONE - sum)
}

/// Modifizierte Besselfunktion erster Art `Iₙ(x)` für ganzzahlige Ordnung `n ≥ 0` (Potenzreihe mit lauter positiven Termen,
/// daher ohne Auslöschung; Rechenzeit etwa `O(|x|)`; läuft bei `|x|` über etwa 700 in den Überlauf).
pub fn bessel_i<T: Real>(n: u32, x: T) -> T {
    if x.is_nan() {
        return x;
    }
    let half_x = x.abs() * T::from_f64(0.5);
    let q = half_x * half_x;
    let mut term = T::ONE;
    for k in 1..=n {
        term = term * half_x / T::from_f64(k as f64);
    }
    let mut sum = term;
    for k in 1..100_000usize {
        term = term * q / (T::from_usize(k) * (T::from_usize(k) + T::from_f64(n as f64)));
        sum += term;
        if term <= T::EPSILON * sum {
            break;
        }
    }
    if x < T::ZERO && n % 2 == 1 { -sum } else { sum }
}
