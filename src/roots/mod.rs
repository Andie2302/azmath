//! Nullstellen von Funktionen einer Veränderlichen: `f(x) = 0`.
//!
//! | Verfahren | Braucht | Eigenschaft |
//! |---|---|---|
//! | [`bisection`] | Intervall mit Vorzeichenwechsel | langsam, aber absolut sicher |
//! | [`brent`] | Intervall mit Vorzeichenwechsel | schnell **und** sicher; die erste Wahl |
//! | [`regula_falsi`] | Intervall mit Vorzeichenwechsel | Einschluss, aber oft einseitig langsam |
//! | [`illinois`] | Intervall mit Vorzeichenwechsel | Regula Falsi ohne Stagnation, Ordnung ≈ 1,44 |
//! | [`newton`] | Ableitung, Startwert | quadratisch konvergent, kann divergieren |
//! | [`secant`] | zwei Startwerte | ohne Ableitung, überlinear konvergent |
//! | [`aberth`], [`durand_kerner`] | Polynom-Koeffizienten | **alle** komplexen Nullstellen gleichzeitig |
//! | [`jenkins_traub`] | Polynom-Koeffizienten | alle Nullstellen nach Jenkins-Traub (CPOLY), Standard für hohen Grad |
//! | [`companion_roots`] | Polynom-Koeffizienten | alle Nullstellen als Eigenwerte der Begleitmatrix (QR-Algorithmus) |
//!
//! Alle Verfahren sind generisch über [`Real`] und brauchen keinen Allokator. Die Funktionen
//! werden als Closures `FnMut(T) -> T` übergeben.
//!
//! **Abbruch:** Ein Verfahren ist fertig, wenn Intervallbreite bzw. Schrittweite
//! `≤ tol · max(1, |x|)` ist (eine Mischung aus absoluter und relativer Toleranz) oder
//! `f(x)` exakt 0 ergibt. Sonst liefert es [`RootError::NoConvergence`].
//! Bei Bisektion, Brent, Newton und Sekante verlangt `tol = 0` einen exakten Nullwert;
//! Rundungsstillstand ohne erfülltes Kriterium ergibt `NoConvergence`. `max_iter = 0`
//! prüft nur die Startpunkte. Die Lagentoleranz garantiert keine kleine Funktionsabweichung:
//! insbesondere bei Polstellen oder schlecht skalierten Funktionen `f_value` mitbeurteilen.
//!
//! Hinweis zu `micromath`: Mit der bewusst ungenauen Näherung (z. B. bei `sin`, `exp`, `sqrt`
//! in `f`) lässt sich eine strenge Toleranz unter Umständen nie erreichen; für verlässliche
//! Ergebnisse `std` oder `libm` verwenden.
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::roots::{brent, RootOptions};
//! use azmath::Real;
//!
//! // Lösung von cos(x) = x
//! let r = brent(|x: f64| Real::cos(x) - x, 0.0, 1.0, RootOptions::default()).unwrap();
//! assert!((r.root - 0.7390851332151607).abs() < 1e-14);
//! # }
//! ```

mod bisection;
mod brent;
mod companion;
mod falsi;
mod jenkins_traub;
mod newton;
mod polynomial;

pub use bisection::bisection;
pub use brent::brent;
pub use companion::{companion_roots, companion_work_len};
pub use falsi::{illinois, regula_falsi};
pub use jenkins_traub::{jenkins_traub, jenkins_traub_complex, jenkins_traub_work_len};
pub use newton::{newton, secant};
pub use polynomial::{
    PolyRootsResult, aberth, aberth_complex, durand_kerner, durand_kerner_complex, polyval,
    polyval_complex,
};

use core::fmt;

use crate::Real;

/// Einstellungen für die Nullstellensuche.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootOptions<T> {
    /// Toleranz für die Lage der Nullstelle: Abbruch bei Intervallbreite bzw. Schritt
    /// `≤ tol · max(1, |x|)`. Muss endlich und nichtnegativ sein.
    pub tol: T,
    /// Maximale Anzahl an Iterationen.
    pub max_iter: usize,
}

impl<T: Real> RootOptions<T> {
    /// Erzeugt Einstellungen mit der gegebenen Toleranz und Iterationsgrenze.
    pub fn new(tol: T, max_iter: usize) -> Self {
        Self { tol, max_iter }
    }
}

impl<T: Real> Default for RootOptions<T> {
    /// Toleranz `8·ε` (etwa `1.8e-15` bei `f64`) und höchstens 500 Iterationen.
    ///
    /// Das großzügige Limit ist für mehrfache Nullstellen gedacht, bei denen Brent deutlich
    /// mehr Schritte braucht als bei einfachen (zum Beispiel etwa 150 bei `(x − 1)³`).
    fn default() -> Self {
        Self {
            tol: T::from_f64(8.0) * T::EPSILON,
            max_iter: 500,
        }
    }
}

/// Ergebnis einer erfolgreichen Nullstellensuche.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootResult<T> {
    /// Die gefundene Nullstelle.
    pub root: T,
    /// Funktionswert `f(root)` (Güte der Lösung).
    pub f_value: T,
    /// Anzahl der Iterationen (0, wenn ein Startwert bereits exakt eine Nullstelle ist).
    pub iterations: usize,
}

/// Fehler der Nullstellensuche.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RootError {
    /// Am Intervallrand hat `f` das gleiche Vorzeichen: es ist nicht sicher eingeschlossen.
    NotBracketed,
    /// Die Genauigkeit wurde in `max_iter` Schritten nicht erreicht.
    NoConvergence,
    /// Es ist ein NaN- oder unendlicher Wert aufgetreten (in `f`, `f'` oder den Startwerten).
    NotFinite,
    /// Die Ableitung (Newton) bzw. der Differenzenquotient (Sekante) ist 0.
    ZeroDerivative,
    /// Ungültige Eingabe (etwa negative/nichtendliche Toleranz, Nullpolynom oder zu kurzer Ausgabepuffer).
    InvalidInput,
}

impl fmt::Display for RootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            RootError::NotBracketed => "kein Vorzeichenwechsel im Intervall",
            RootError::NoConvergence => "Iteration ist nicht konvergiert",
            RootError::NotFinite => "NaN oder unendlicher Wert aufgetreten",
            RootError::ZeroDerivative => "Ableitung bzw. Differenzenquotient ist 0",
            RootError::InvalidInput => "ungültige Eingabe oder numerische Einstellung",
        })
    }
}

impl core::error::Error for RootError {}

/// `true`, wenn `x` weder NaN noch unendlich ist.
pub(crate) fn finite<T: Real>(x: T) -> bool {
    !(x.is_nan() || x.is_infinite())
}

/// `max(1, |x|)`: Skalierung für die gemischte absolute/relative Toleranz.
pub(crate) fn scale<T: Real>(x: T) -> T {
    T::ONE.max(x.abs())
}

/// Numerische Optionen vor Auswertung der Benutzerfunktion prüfen.
pub(crate) fn validate_options<T: Real>(opts: RootOptions<T>) -> Result<(), RootError> {
    if !finite(opts.tol) || opts.tol < T::ZERO {
        Err(RootError::InvalidInput)
    } else {
        Ok(())
    }
}

/// Mittelpunkt ohne überlaufende Differenz bei Endpunkten mit verschiedenem Vorzeichen.
pub(crate) fn midpoint<T: Real>(a: T, b: T) -> T {
    let half = T::from_f64(0.5);
    if (a < T::ZERO) == (b < T::ZERO) {
        a + half * (b - a)
    } else {
        half * a + half * b
    }
}

/// Gemischtes Abstandskriterium ohne Überlauf in tol*scale oder in a-b.
pub(crate) fn close_points<T: Real>(a: T, b: T, x: T, tol: T) -> bool {
    if tol == T::ZERO {
        return a == b;
    }
    let s = scale(x);
    let d = (a - b).abs();
    if finite(d) {
        d / s <= tol
    } else {
        (a / s - b / s).abs() <= tol
    }
}
