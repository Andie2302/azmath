//! Dichte Bunch–Kaufman-Zerlegung mit symmetrischer Pivotierung.
use super::{LinalgError, Matrix, Vector};
use crate::Real;

/// Numerische Trägheit der Blockdiagonalen D (Sylvesters Trägheitssatz bei exakter Arithmetik).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Inertia {
    /// Positive Richtungen.
    pub positive: usize,
    /// Negative Richtungen.
    pub negative: usize,
    /// Numerisch verschwindende Richtungen.
    pub zero: usize,
}

fn entry<T: Real>(a: &[T], n: usize, i: usize, j: usize) -> T {
    if i >= j { a[i * n + j] } else { a[j * n + i] }
}
fn determinant2<T: Real>(a: T, b: T, c: T) -> T {
    let bb = b * b;
    a.mul_add(c, -bb) - b.mul_add(b, -bb)
}
fn swap<T: Real>(a: &mut [T], n: usize, perm: &mut [usize], p: usize, q: usize) {
    if p == q {
        return;
    }
    for j in 0..n {
        a.swap(p * n + j, q * n + j);
    }
    for i in 0..n {
        a.swap(i * n + p, i * n + q);
    }
    perm.swap(p, q);
}
fn decompose<T: Real>(
    a: &mut [T],
    n: usize,
    perm: &mut [usize],
    blocks: &mut [u8],
    work: &mut [T],
) -> Result<T, LinalgError> {
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    let scale = a.iter().fold(T::ZERO, |s, v| s.max(v.abs()));
    let scale = if scale == T::ZERO { T::ONE } else { scale };
    for v in a.iter_mut() {
        *v /= scale;
    }
    let tol = T::from_f64(8.0) * T::from_usize(n) * T::EPSILON;
    for i in 0..n {
        for j in 0..i {
            if (a[i * n + j] - a[j * n + i]).abs() > tol {
                return Err(LinalgError::NotSymmetric);
            }
            a[j * n + i] = a[i * n + j];
        }
    }
    for (i, p) in perm.iter_mut().enumerate() {
        *p = i;
    }
    blocks.fill(1);
    let alpha = T::from_f64(0.6403882032022076);
    let mut k = 0;
    while k < n {
        let mut imax = k;
        let mut col = T::ZERO;
        for i in k + 1..n {
            if a[i * n + k].abs() > col {
                col = a[i * n + k].abs();
                imax = i;
            }
        }
        let absakk = a[k * n + k].abs();
        let mut two = false;
        let mut pivot = k;
        if col > T::ZERO && absakk < alpha * col {
            let mut row = T::ZERO;
            for j in k..n {
                if j != imax {
                    row = row.max(entry(a, n, imax, j).abs());
                }
            }
            if absakk >= alpha * col * (col / row) {
                // scalar pivot k
            } else if a[imax * n + imax].abs() >= alpha * row {
                pivot = imax;
            } else {
                two = true;
                pivot = imax;
            }
        }
        if !two {
            swap(a, n, perm, k, pivot);
            let d = a[k * n + k];
            if d != T::ZERO {
                for i in k + 1..n {
                    a[i * n + k] /= d;
                }
                for i in k + 1..n {
                    for j in k + 1..=i {
                        let v = a[i * n + j] - a[i * n + k] * (d * a[j * n + k]);
                        if !v.is_finite() {
                            return Err(LinalgError::NotFinite);
                        }
                        a[i * n + j] = v;
                        a[j * n + i] = v;
                    }
                }
            }
            k += 1;
        } else {
            swap(a, n, perm, k + 1, pivot);
            blocks[k] = 2;
            blocks[k + 1] = 0;
            let aa = a[k * n + k];
            let bb = a[(k + 1) * n + k];
            let cc = a[(k + 1) * n + k + 1];
            let s = aa.abs().max(bb.abs()).max(cc.abs());
            let (u, v, w) = (aa / s, bb / s, cc / s);
            let det = determinant2(u, v, w);
            if det == T::ZERO {
                return Err(LinalgError::Singular);
            }
            for i in k + 2..n {
                work[i] = a[i * n + k];
                work[n + i] = a[i * n + k + 1];
                let r = work[i] / s;
                let t = work[n + i] / s;
                a[i * n + k] = (w * r - v * t) / det;
                a[i * n + k + 1] = (u * t - v * r) / det;
            }
            for i in k + 2..n {
                for j in k + 2..=i {
                    let value =
                        a[i * n + j] - a[i * n + k] * work[j] - a[i * n + k + 1] * work[n + j];
                    if !value.is_finite() {
                        return Err(LinalgError::NotFinite);
                    }
                    a[i * n + j] = value;
                    a[j * n + i] = value;
                }
            }
            k += 2;
        }
    }
    if a.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    Ok(scale)
}
fn inertia<T: Real>(a: &[T], n: usize, blocks: &[u8], tol: T) -> Inertia {
    let mut result = Inertia {
        positive: 0,
        negative: 0,
        zero: 0,
    };
    let mut add = |v: T| {
        if v.abs() <= tol {
            result.zero += 1;
        } else if v > T::ZERO {
            result.positive += 1;
        } else {
            result.negative += 1;
        }
    };
    let mut k = 0;
    while k < n {
        if blocks[k] == 1 {
            add(a[k * n + k]);
            k += 1;
        } else {
            let aa = a[k * n + k];
            let bb = a[(k + 1) * n + k];
            let cc = a[(k + 1) * n + k + 1];
            let s = aa.abs().max(bb.abs()).max(cc.abs());
            let (u, v, w) = (aa / s, bb / s, cc / s);
            let h = T::from_f64(0.5);
            let mid = u * h + w * h;
            let radius = (u * h - w * h).hypot(v);
            let big = if mid >= T::ZERO {
                mid + radius
            } else {
                mid - radius
            };
            let small = determinant2(u, v, w) / big;
            add(big * s);
            add(small * s);
            k += 2;
        }
    }
    result
}
fn lower<T: Real>(a: &[T], n: usize, blocks: &[u8], i: usize, j: usize) -> T {
    if i == j {
        T::ONE
    } else if i < j || (blocks[j] == 2 && i == j + 1) {
        T::ZERO
    } else {
        a[i * n + j]
    }
}
fn solve<T: Real>(
    a: &[T],
    n: usize,
    perm: &[usize],
    blocks: &[u8],
    scale: T,
    b: &[T],
    x: &mut [T],
    tmp: &mut [T],
) -> Result<(), LinalgError> {
    if b.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    if inertia(a, n, blocks, (T::from_usize(n) * T::EPSILON).min(T::ONE)).zero > 0 {
        return Err(LinalgError::Singular);
    }
    for i in 0..n {
        let mut value = b[perm[i]] / scale;
        for j in 0..i {
            value -= lower(a, n, blocks, i, j) * tmp[j];
        }
        tmp[i] = value;
    }
    let mut k = 0;
    while k < n {
        if blocks[k] == 1 {
            tmp[k] /= a[k * n + k];
            k += 1;
        } else {
            let aa = a[k * n + k];
            let bb = a[(k + 1) * n + k];
            let cc = a[(k + 1) * n + k + 1];
            let s = aa.abs().max(bb.abs()).max(cc.abs());
            let (u, v, w) = (aa / s, bb / s, cc / s);
            let det = determinant2(u, v, w);
            let r = tmp[k] / s;
            let t = tmp[k + 1] / s;
            tmp[k] = (w * r - v * t) / det;
            tmp[k + 1] = (u * t - v * r) / det;
            k += 2;
        }
    }
    for i in (0..n).rev() {
        let mut v = tmp[i];
        for j in i + 1..n {
            v -= lower(a, n, blocks, j, i) * tmp[j];
        }
        tmp[i] = v;
    }
    if tmp.iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    for i in 0..n {
        x[perm[i]] = tmp[i];
    }
    Ok(())
}

/// Symmetrisch pivotierte Bunch–Kaufman-Zerlegung `A[perm,perm] = L·D·Lᵀ`.
/// `D` enthält 1×1- und 2×2-Blöcke. Auch singuläre Matrizen können zerlegt werden;
/// `solve` lehnt numerisch singuläre Blöcke ab. Die Matrix wird intern global skaliert.
/// Symmetrieprüfung: `8·N·ε·max|Aᵢⱼ|`, die untere Hälfte ist maßgeblich.
/// Für genaue Trägheitsdiagnosen `std` oder `libm` verwenden.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ldlt<T, const N: usize> {
    factor: Matrix<T, N, N>,
    perm: [usize; N],
    blocks: [u8; N],
    scale: T,
}
impl<T: Real, const N: usize> Ldlt<T, N> {
    /// Zerlegt eine symmetrische Matrix. Fehler: `NotSymmetric`, `NotFinite`, bei degeneriertem 2×2-Pivot `Singular`.
    pub fn new(a: &Matrix<T, N, N>) -> Result<Self, LinalgError> {
        let mut factor = *a;
        let mut perm = [0; N];
        let mut blocks = [0; N];
        let scale = decompose(
            factor.data.as_flattened_mut(),
            N,
            &mut perm,
            &mut blocks,
            &mut [[T::ZERO; N]; 2].as_flattened_mut()[..],
        )?;
        Ok(Self {
            factor,
            perm,
            blocks,
            scale,
        })
    }
    /// Ursprüngliche Indizes in pivotierter Reihenfolge (für Zeilen und Spalten).
    pub fn permutation(&self) -> &[usize; N] {
        &self.perm
    }
    /// Blockmarker: 1 = skalar, 2 = Start eines 2×2-Blocks, 0 = dessen zweiter Eintrag.
    pub fn block_sizes(&self) -> &[u8; N] {
        &self.blocks
    }
    /// Numerische Trägheit von D mit Schwelle `min(N·ε,1)·max|Aᵢⱼ|`.
    /// Nahe null ist dies eine Diagnose der Pivotblöcke, keine Eigenwert-Rangschätzung für A.
    pub fn inertia(&self) -> Inertia {
        inertia(
            self.factor.data.as_flattened(),
            N,
            &self.blocks,
            (T::from_usize(N) * T::EPSILON).min(T::ONE),
        )
    }
    /// Trägheit mit eigener absoluter Schwelle für die Eigenwerte der D-Blöcke.
    /// Schwelle 0 zählt exakte Vorzeichen; negative/nichtendliche Schwellen sind ungültig.
    pub fn inertia_with_tolerance(&self, tol: T) -> Result<Inertia, LinalgError> {
        if !tol.is_finite() || tol < T::ZERO {
            return Err(LinalgError::InvalidInput);
        }
        Ok(inertia(
            self.factor.data.as_flattened(),
            N,
            &self.blocks,
            tol / self.scale,
        ))
    }
    /// Unterer Einheitsdreiecksfaktor L.
    pub fn l(&self) -> Matrix<T, N, N> {
        Matrix::from_fn(|i, j| lower(self.factor.data.as_flattened(), N, &self.blocks, i, j))
    }
    /// Blockdiagonale D in Originaleinheiten; nicht darstellbare Faktoren ergeben `NotFinite`.
    pub fn d(&self) -> Result<Matrix<T, N, N>, LinalgError> {
        let d = Matrix::from_fn(|i, j| {
            if i == j || (i == j + 1 && self.blocks[j] == 2) || (j == i + 1 && self.blocks[i] == 2)
            {
                entry(self.factor.data.as_flattened(), N, i, j) * self.scale
            } else {
                T::ZERO
            }
        });
        if d.data.as_flattened().iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        Ok(d)
    }
    /// Löst Ax=b mit Prüfung auf numerischen Rangmangel und nichtendliche Daten/Ergebnisse.
    /// Nicht darstellbare Zwischenwerte (auch bei globaler Skalierung) ergeben `NotFinite`.
    pub fn solve(&self, b: &Vector<T, N>) -> Result<Vector<T, N>, LinalgError> {
        let mut x = Vector::zeros();
        solve(
            self.factor.data.as_flattened(),
            N,
            &self.perm,
            &self.blocks,
            self.scale,
            &b.data,
            &mut x.data,
            &mut [T::ZERO; N],
        )?;
        Ok(x)
    }
}
impl<T: Real, const N: usize> Matrix<T, N, N> {
    /// Symmetrische indefinite Zerlegung mit Bunch–Kaufman-Pivotierung.
    pub fn ldlt(&self) -> Result<Ldlt<T, N>, LinalgError> {
        Ldlt::new(self)
    }
}

#[cfg(feature = "alloc")]
mod dynamic {
    use super::*;
    use crate::linalg::{DynMatrix, DynVector};
    use alloc::{vec, vec::Vec};
    /// Dynamische Bunch–Kaufman-Zerlegung; siehe [`Ldlt`].
    #[derive(Debug, Clone, PartialEq)]
    pub struct DynLdlt<T> {
        factor: Vec<T>,
        n: usize,
        perm: Vec<usize>,
        blocks: Vec<u8>,
        scale: T,
    }
    impl<T: Real> DynLdlt<T> {
        /// Zerlegt eine quadratische symmetrische Matrix; Fehler wie bei [`Ldlt::new`].
        pub fn new(a: &DynMatrix<T>) -> Result<Self, LinalgError> {
            let n = a.rows();
            if n != a.cols() {
                return Err(LinalgError::DimensionMismatch);
            }
            let mut factor = a.as_slice().to_vec();
            let mut perm = vec![0; n];
            let mut blocks = vec![0; n];
            let scale = decompose(
                &mut factor,
                n,
                &mut perm,
                &mut blocks,
                &mut vec![T::ZERO; 2 * n],
            )?;
            Ok(Self {
                factor,
                n,
                perm,
                blocks,
                scale,
            })
        }
        /// Zeilen- und Spaltenpermutation.
        pub fn permutation(&self) -> &[usize] {
            &self.perm
        }
        /// Blockmarker 1, 2 oder 0; siehe [`Ldlt::block_sizes`].
        pub fn block_sizes(&self) -> &[u8] {
            &self.blocks
        }
        /// Numerische Trägheit; siehe [`Ldlt::inertia`].
        pub fn inertia(&self) -> Inertia {
            inertia(
                &self.factor,
                self.n,
                &self.blocks,
                (T::from_usize(self.n) * T::EPSILON).min(T::ONE),
            )
        }
        /// Trägheit mit eigener absoluter Schwelle der D-Blöcke.
        pub fn inertia_with_tolerance(&self, tol: T) -> Result<Inertia, LinalgError> {
            if !tol.is_finite() || tol < T::ZERO {
                return Err(LinalgError::InvalidInput);
            }
            Ok(inertia(
                &self.factor,
                self.n,
                &self.blocks,
                tol / self.scale,
            ))
        }
        /// Unterer Einheitsdreiecksfaktor L.
        pub fn l(&self) -> DynMatrix<T> {
            let mut m = DynMatrix::zeros(self.n, self.n);
            for i in 0..self.n {
                for j in 0..=i {
                    m[(i, j)] = lower(&self.factor, self.n, &self.blocks, i, j);
                }
            }
            m
        }
        /// Blockdiagonale D; Fehler bei nicht darstellbaren Einträgen.
        pub fn d(&self) -> Result<DynMatrix<T>, LinalgError> {
            let mut d = DynMatrix::zeros(self.n, self.n);
            for i in 0..self.n {
                d[(i, i)] = self.factor[i * self.n + i] * self.scale;
                if self.blocks[i] == 2 {
                    let v = self.factor[(i + 1) * self.n + i] * self.scale;
                    d[(i + 1, i)] = v;
                    d[(i, i + 1)] = v;
                }
            }
            if d.as_slice().iter().any(|v| !v.is_finite()) {
                return Err(LinalgError::NotFinite);
            }
            Ok(d)
        }
        /// Löst Ax=b; Fehler bei falscher Länge, Rangmangel oder nichtendlichen Daten/Ergebnissen.
        pub fn solve(&self, b: &DynVector<T>) -> Result<DynVector<T>, LinalgError> {
            if b.len() != self.n {
                return Err(LinalgError::DimensionMismatch);
            }
            let mut x = vec![T::ZERO; self.n];
            solve(
                &self.factor,
                self.n,
                &self.perm,
                &self.blocks,
                self.scale,
                b.as_slice(),
                &mut x,
                &mut vec![T::ZERO; self.n],
            )?;
            Ok(DynVector::new(x))
        }
    }
    impl<T: Real> DynMatrix<T> {
        /// Bunch–Kaufman-Zerlegung einer symmetrischen quadratischen Matrix.
        pub fn ldlt(&self) -> Result<DynLdlt<T>, LinalgError> {
            DynLdlt::new(self)
        }
    }
}
#[cfg(feature = "alloc")]
pub use dynamic::DynLdlt;
