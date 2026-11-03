//! Matrixfunktionen auf [`DynMatrix`]: `expm`, `sqrtm`, `logm`, `powm`, Sylvester- und Lyapunov-Gleichung.
//!
//! | Funktion | Verfahren |
//! |---|---|
//! | [`expm`] | Padé-Approximation Grad 13 mit Skalieren und Quadrieren (Higham 2005) |
//! | [`sqrtm`] | Denman-Beavers-Iteration (Hauptwurzel; Eigenwerte nicht auf der negativen reellen Achse) |
//! | [`logm`] | inverses Skalieren und Quadrieren (wiederholte Wurzeln) mit Reihe für `log(I + X)` |
//! | [`powm`] | ganzzahlige Potenz durch binäres Potenzieren |
//! | [`sylvester`], [`lyapunov`] | `A·X + X·B = C` über die Kronecker-Form (für kleine Dimensionen) |
//! | [`care`], [`dare`] | algebraische Riccati-Gleichungen (Vorzeichenfunktion bzw. Verdopplungsalgorithmus) |
//!
//! Alle Funktionen verlangen quadratische Matrizen und geben sonst [`LinalgError::DimensionMismatch`] zurück.

mod riccati;

pub use riccati::{care, dare};

use crate::Real;
use crate::linalg::{DynMatrix, DynVector, LinalgError};
use alloc::vec::Vec;

fn check_square<T: Real>(a: &DynMatrix<T>) -> Result<usize, LinalgError> {
    if a.rows() != a.cols() {
        return Err(LinalgError::DimensionMismatch);
    }
    if a.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    Ok(a.rows())
}

fn map2<T: Real>(a: &DynMatrix<T>, b: &DynMatrix<T>, f: impl Fn(T, T) -> T) -> DynMatrix<T> {
    let data: Vec<T> = a
        .as_slice()
        .iter()
        .zip(b.as_slice())
        .map(|(&x, &y)| f(x, y))
        .collect();
    DynMatrix::new(a.rows(), a.cols(), data).expect("gleiche Größe")
}

pub(crate) fn add<T: Real>(a: &DynMatrix<T>, b: &DynMatrix<T>) -> DynMatrix<T> {
    map2(a, b, |x, y| x + y)
}

pub(crate) fn sub<T: Real>(a: &DynMatrix<T>, b: &DynMatrix<T>) -> DynMatrix<T> {
    map2(a, b, |x, y| x - y)
}

pub(crate) fn scale<T: Real>(a: &DynMatrix<T>, s: T) -> DynMatrix<T> {
    let data: Vec<T> = a.as_slice().iter().map(|&x| x * s).collect();
    DynMatrix::new(a.rows(), a.cols(), data).expect("gleiche Größe")
}

/// Spaltensummennorm ‖A‖₁.
pub(crate) fn norm1<T: Real>(a: &DynMatrix<T>) -> T {
    let mut best = T::ZERO;
    for j in 0..a.cols() {
        let mut s = T::ZERO;
        for i in 0..a.rows() {
            s += a[(i, j)].abs();
        }
        best = best.max(s);
    }
    best
}

pub(crate) fn mm<T: Real>(a: &DynMatrix<T>, b: &DynMatrix<T>) -> DynMatrix<T> {
    a.matmul(b).expect("gleiche Größe")
}

/// Matrixexponential `exp(A)`.
pub fn expm<T: Real>(a: &DynMatrix<T>) -> Result<DynMatrix<T>, LinalgError> {
    let n = check_square(a)?;
    if n == 0 {
        return Ok(DynMatrix::zeros(0, 0));
    }
    const B: [f64; 14] = [
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    ];
    let b = |k: usize| T::from_f64(B[k]);
    let theta = T::from_f64(5.371920351148152);
    let nrm = norm1(a);
    let mut s = 0u32;
    let mut scaled = nrm;
    while scaled > theta && s < 1000 {
        scaled *= T::from_f64(0.5);
        s += 1;
    }
    let mut half = T::ONE;
    for _ in 0..s {
        half *= T::from_f64(0.5);
    }
    let a1 = scale(a, half);
    let id = DynMatrix::identity(n);
    let a2 = mm(&a1, &a1);
    let a4 = mm(&a2, &a2);
    let a6 = mm(&a4, &a2);
    // U = A1 * (A6 (b13 A6 + b11 A4 + b9 A2) + b7 A6 + b5 A4 + b3 A2 + b1 I)
    let t = add(
        &add(&scale(&a6, b(13)), &scale(&a4, b(11))),
        &scale(&a2, b(9)),
    );
    let mut inner = mm(&a6, &t);
    inner = add(&inner, &scale(&a6, b(7)));
    inner = add(&inner, &scale(&a4, b(5)));
    inner = add(&inner, &scale(&a2, b(3)));
    inner = add(&inner, &scale(&id, b(1)));
    let u = mm(&a1, &inner);
    // V = A6 (b12 A6 + b10 A4 + b8 A2) + b6 A6 + b4 A4 + b2 A2 + b0 I
    let t = add(
        &add(&scale(&a6, b(12)), &scale(&a4, b(10))),
        &scale(&a2, b(8)),
    );
    let mut v = mm(&a6, &t);
    v = add(&v, &scale(&a6, b(6)));
    v = add(&v, &scale(&a4, b(4)));
    v = add(&v, &scale(&a2, b(2)));
    v = add(&v, &scale(&id, b(0)));
    // (V − U) R = V + U
    let denom = sub(&v, &u);
    let numer = add(&v, &u);
    let lu = denom.lu()?;
    let mut r = DynMatrix::zeros(n, n);
    for j in 0..n {
        let col: Vec<T> = (0..n).map(|i| numer[(i, j)]).collect();
        let x = lu.solve(&DynVector::new(col))?;
        for i in 0..n {
            r[(i, j)] = x[i];
        }
    }
    for _ in 0..s {
        r = mm(&r, &r);
    }
    if r.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(LinalgError::NotFinite);
    }
    Ok(r)
}

/// Hauptwurzel `X` mit `X·X = A` (Denman-Beavers). Fehler `NoConvergence`, falls die Iteration nicht konvergiert
/// (zum Beispiel bei Eigenwerten auf der negativen reellen Achse), `Singular` bei singulären Zwischenmatrizen.
pub fn sqrtm<T: Real>(a: &DynMatrix<T>) -> Result<DynMatrix<T>, LinalgError> {
    let n = check_square(a)?;
    let mut y = a.clone();
    let mut z = DynMatrix::identity(n);
    let tol = T::EPSILON * T::from_f64(64.0) * (T::ONE + norm1(a));
    let half = T::from_f64(0.5);
    for _ in 0..100 {
        let yi = y.inverse()?;
        let zi = z.inverse()?;
        let y_next = scale(&add(&y, &zi), half);
        let z_next = scale(&add(&z, &yi), half);
        let diff = norm1(&sub(&y_next, &y));
        y = y_next;
        z = z_next;
        if !diff.is_finite() {
            return Err(LinalgError::NoConvergence);
        }
        if diff <= tol {
            return Ok(y);
        }
    }
    Err(LinalgError::NoConvergence)
}

/// Hauptlogarithmus `X` mit `exp(X) = A` über inverses Skalieren und Quadrieren.
/// Fehler `NoConvergence`, falls die Wurzeliteration scheitert (Eigenwerte auf der negativen Achse oder 0).
pub fn logm<T: Real>(a: &DynMatrix<T>) -> Result<DynMatrix<T>, LinalgError> {
    let n = check_square(a)?;
    let id = DynMatrix::identity(n);
    let mut cur = a.clone();
    let mut k = 0u32;
    let limit = T::from_f64(0.25);
    while norm1(&sub(&cur, &id)) > limit {
        if k >= 60 {
            return Err(LinalgError::NoConvergence);
        }
        cur = sqrtm(&cur)?;
        k += 1;
    }
    // log(I + X) = X − X²/2 + X³/3 − …
    let x = sub(&cur, &id);
    let mut term = x.clone();
    let mut sum = x.clone();
    for m in 2..200usize {
        term = mm(&term, &x);
        let c = T::ONE / T::from_usize(m);
        let contrib = scale(&term, c);
        sum = if m % 2 == 0 {
            sub(&sum, &contrib)
        } else {
            add(&sum, &contrib)
        };
        if norm1(&contrib) <= T::EPSILON * T::from_f64(0.01) * (T::ONE + norm1(&sum)) {
            break;
        }
    }
    let mut factor = T::ONE;
    for _ in 0..k {
        factor *= T::from_f64(2.0);
    }
    Ok(scale(&sum, factor))
}

/// Ganzzahlige Potenz `A^p` (`p = 0` liefert die Einheitsmatrix).
pub fn powm<T: Real>(a: &DynMatrix<T>, mut p: u32) -> Result<DynMatrix<T>, LinalgError> {
    let n = check_square(a)?;
    let mut result = DynMatrix::identity(n);
    let mut base = a.clone();
    while p > 0 {
        if p & 1 == 1 {
            result = mm(&result, &base);
        }
        p >>= 1;
        if p > 0 {
            base = mm(&base, &base);
        }
    }
    Ok(result)
}

/// Löst die Sylvester-Gleichung `A·X + X·B = C` mit `A` (n×n), `B` (m×m), `C` (n×m) über die Kronecker-Form
/// (lineares System der Größe `n·m`; gedacht für kleine Dimensionen). Fehler `Singular`, wenn `A` und `−B` gemeinsame
/// Eigenwerte haben.
pub fn sylvester<T: Real>(
    a: &DynMatrix<T>,
    b: &DynMatrix<T>,
    c: &DynMatrix<T>,
) -> Result<DynMatrix<T>, LinalgError> {
    let n = check_square(a)?;
    let m = check_square(b)?;
    if c.rows() != n || c.cols() != m {
        return Err(LinalgError::DimensionMismatch);
    }
    let size = n * m;
    let mut k = DynMatrix::zeros(size, size);
    // Unbekannte X[i, j] liegt an Index j·n + i
    for j in 0..m {
        for i in 0..n {
            let row = j * n + i;
            for l in 0..n {
                k[(row, j * n + l)] += a[(i, l)];
            }
            for l in 0..m {
                k[(row, l * n + i)] += b[(l, j)];
            }
        }
    }
    let rhs: Vec<T> = (0..size).map(|idx| c[(idx % n, idx / n)]).collect();
    let x = k.solve(&DynVector::new(rhs))?;
    let mut out = DynMatrix::zeros(n, m);
    for j in 0..m {
        for i in 0..n {
            out[(i, j)] = x[j * n + i];
        }
    }
    Ok(out)
}

/// Löst die Lyapunov-Gleichung `A·X + X·Aᵀ = −Q` (`X` ist für symmetrisches `Q` symmetrisch).
pub fn lyapunov<T: Real>(a: &DynMatrix<T>, q: &DynMatrix<T>) -> Result<DynMatrix<T>, LinalgError> {
    sylvester(a, &a.transpose(), &scale(q, -T::ONE))
}
