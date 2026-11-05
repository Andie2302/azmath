use super::{IntegrateError, WeightedSum, checked, finite, from_usize, integral_scale, point};
use crate::Real;

/// Knoten (≥ 0) und Gewichte der Gauß-Legendre-Regeln der Ordnung 1 bis 5 auf `[−1, 1]`.
/// Pro Zeile: `(Knoten, Gewicht)`; Knoten ≠ 0 treten symmetrisch (±) auf.
const RULES: [&[(f64, f64)]; 5] = [
    &[(0.0, 2.0)],
    &[(0.577_350_269_189_625_8, 1.0)],
    &[(0.0, 8.0 / 9.0), (0.774_596_669_241_483_4, 5.0 / 9.0)],
    &[
        (0.339_981_043_584_856_3, 0.652_145_154_862_546_1),
        (0.861_136_311_594_052_6, 0.347_854_845_137_453_9),
    ],
    &[
        (0.0, 0.568_888_888_888_888_9),
        (0.538_469_310_105_683_1, 0.478_628_670_499_366_5),
        (0.906_179_845_938_664, 0.236_926_885_056_189_1),
    ],
];

/// Zusammengesetzte Gauß-Legendre-Regel: `panels` gleich breite Teilintervalle mit je
/// `order` Stützstellen (`order` von 1 bis 5).
///
/// Mit `order` Punkten pro Teilintervall ist die Regel exakt für Polynome bis Grad
/// `2·order − 1`. Bei glatten Funktionen konvergiert sie sehr schnell; die Funktion wird nie an
/// den Intervallenden ausgewertet.
///
/// Fehler: [`IntegrateError::InvalidInput`] bei `panels = 0` oder `order` außerhalb von 1..=5,
/// [`IntegrateError::NotFinite`].
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::integrate::gauss_legendre;
///
/// // 3 Punkte sind exakt bis Grad 5: ∫₀¹ x⁵ dx = 1/6
/// let v = gauss_legendre(|x: f64| x.powi(5), 0.0, 1.0, 3, 1).unwrap();
/// assert!((v - 1.0 / 6.0).abs() < 1e-15);
/// # }
/// ```
pub fn gauss_legendre<T: Real>(
    mut f: impl FnMut(T) -> T,
    a: T,
    b: T,
    order: usize,
    panels: usize,
) -> Result<T, IntegrateError> {
    if panels == 0 || !(1..=5).contains(&order) {
        return Err(IntegrateError::InvalidInput);
    }
    if !finite(a) || !finite(b) {
        return Err(IntegrateError::NotFinite);
    }
    if a == b {
        return Ok(T::ZERO);
    }
    let rule = RULES[order - 1];
    let count = from_usize::<T>(panels);
    let half = T::from_f64(0.5);
    let mut total = WeightedSum::new();
    for p in 0..panels {
        for &(node, weight) in rule {
            let x = T::from_f64(node);
            let w = T::from_f64(weight) * half / count;
            if node == 0.0 {
                total.add(f(point(a, b, (from_usize::<T>(p) + half) / count)), w)?;
            } else {
                for t in [(T::ONE - x) * half, (T::ONE + x) * half] {
                    total.add(f(point(a, b, (from_usize::<T>(p) + t) / count)), w)?;
                }
            }
        }
    }
    checked(integral_scale(total.value(), a, b))
}
