//! Besitzende Varianten der Interpolierenden (Feature `alloc`).

use alloc::vec;
use alloc::vec::Vec;

use super::{BoundaryCondition, CubicSpline, InterpError, Pchip, cubic_spline_work_len};
use azmath_core::Real;

/// Kubischer Spline, der seine Daten selbst besitzt (Feature `alloc`).
///
/// Bequemer als [`CubicSpline`], weil er sich nicht an geliehene Slices bindet.
#[derive(Debug, Clone, PartialEq)]
pub struct CubicSplineBuf<T> {
    x: Vec<T>,
    y: Vec<T>,
    m: Vec<T>,
}

impl<T: Real> CubicSplineBuf<T> {
    /// Berechnet den Spline; Parameter und Fehler wie bei [`CubicSpline::new`].
    pub fn new(x: &[T], y: &[T], boundary: BoundaryCondition<T>) -> Result<Self, InterpError> {
        let mut m = vec![T::ZERO; x.len()];
        let mut work = vec![T::ZERO; cubic_spline_work_len(x.len())];
        CubicSpline::new(x, y, boundary, &mut m, &mut work)?;
        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            m,
        })
    }

    /// Eine geliehene Sicht auf den Spline.
    pub fn as_spline(&self) -> CubicSpline<'_, T> {
        CubicSpline::from_parts(&self.x, &self.y, &self.m)
    }

    /// Wert `S(xq)`, siehe [`CubicSpline::value`].
    pub fn value(&self, xq: T) -> T {
        self.as_spline().value(xq)
    }

    /// Erste Ableitung, siehe [`CubicSpline::derivative`].
    pub fn derivative(&self, xq: T) -> T {
        self.as_spline().derivative(xq)
    }

    /// Zweite Ableitung, siehe [`CubicSpline::second_derivative`].
    pub fn second_derivative(&self, xq: T) -> T {
        self.as_spline().second_derivative(xq)
    }

    /// Integral über `[a, b]`, siehe [`CubicSpline::integral`].
    pub fn integral(&self, a: T, b: T) -> T {
        self.as_spline().integral(a, b)
    }

    /// Wertet den Spline an allen Stellen aus und gibt die Werte als `Vec` zurück.
    pub fn values_at(&self, xs: &[T]) -> Vec<T> {
        xs.iter().map(|&xq| self.value(xq)).collect()
    }

    /// Der Definitionsbereich `(x₀, xₙ₋₁)`.
    pub fn domain(&self) -> (T, T) {
        self.as_spline().domain()
    }

    /// `true`, wenn `xq` im Definitionsbereich liegt.
    pub fn contains(&self, xq: T) -> bool {
        self.as_spline().contains(xq)
    }
}

/// PCHIP, die ihre Daten selbst besitzt (Feature `alloc`), siehe [`Pchip`].
#[derive(Debug, Clone, PartialEq)]
pub struct PchipBuf<T> {
    x: Vec<T>,
    y: Vec<T>,
    d: Vec<T>,
}

impl<T: Real> PchipBuf<T> {
    /// Berechnet die Interpolierende; Parameter und Fehler wie bei [`Pchip::new`].
    pub fn new(x: &[T], y: &[T]) -> Result<Self, InterpError> {
        let mut d = vec![T::ZERO; x.len()];
        Pchip::new(x, y, &mut d)?;
        Ok(Self {
            x: x.to_vec(),
            y: y.to_vec(),
            d,
        })
    }

    /// Eine geliehene Sicht auf die Interpolierende.
    pub fn as_pchip(&self) -> Pchip<'_, T> {
        Pchip::from_parts(&self.x, &self.y, &self.d)
    }

    /// Wert `P(xq)`, siehe [`Pchip::value`].
    pub fn value(&self, xq: T) -> T {
        self.as_pchip().value(xq)
    }

    /// Erste Ableitung, siehe [`Pchip::derivative`].
    pub fn derivative(&self, xq: T) -> T {
        self.as_pchip().derivative(xq)
    }

    /// Integral über `[a, b]`, siehe [`Pchip::integral`].
    pub fn integral(&self, a: T, b: T) -> T {
        self.as_pchip().integral(a, b)
    }

    /// Wertet die Interpolierende an allen Stellen aus und gibt die Werte als `Vec` zurück.
    pub fn values_at(&self, xs: &[T]) -> Vec<T> {
        xs.iter().map(|&xq| self.value(xq)).collect()
    }
}
