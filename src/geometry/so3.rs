use crate::Real;
use crate::linalg::{Matrix, Vector};

/// Kreuzprodukt `a × b`.
pub fn cross<T: Real>(a: Vector<T, 3>, b: Vector<T, 3>) -> Vector<T, 3> {
    Vector::new([
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

/// Schiefsymmetrische Matrix `[w]×` mit `[w]× · v = w × v`.
fn hat<T: Real>(w: Vector<T, 3>) -> Matrix<T, 3, 3> {
    Matrix::new([
        [T::ZERO, -w[2], w[1]],
        [w[2], T::ZERO, -w[0]],
        [-w[1], w[0], T::ZERO],
    ])
}

/// Exponentialabbildung `so(3) → SO(3)`: Rotationsvektor `w` (Achse mal Winkel in Bogenmaß) → Drehmatrix
/// (Rodrigues-Formel `R = I + sin θ/θ·[w]× + (1 − cos θ)/θ²·[w]×²`). Für kleine Winkel wird eine
/// Reihenentwicklung benutzt, damit nichts durch `θ` geteilt wird, das gegen 0 geht.
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::geometry::exp_so3;
/// use azmath::linalg::Vector;
///
/// let r = exp_so3(Vector::new([0.0f64, 0.0, core::f64::consts::FRAC_PI_2]));
/// assert!((r[(1, 0)] - 1.0).abs() < 1e-15); // x → y
/// # }
/// ```
pub fn exp_so3<T: Real>(w: Vector<T, 3>) -> Matrix<T, 3, 3> {
    let k = hat(w);
    let theta2 = w.norm_squared();
    let theta = theta2.sqrt();
    let (a, b) = if theta < T::EPSILON.sqrt().sqrt() {
        // sinθ/θ ≈ 1 − θ²/6,  (1 − cosθ)/θ² ≈ 1/2 − θ²/24
        (
            T::ONE - theta2 / T::from_f64(6.0),
            T::from_f64(0.5) - theta2 / T::from_f64(24.0),
        )
    } else {
        (theta.sin() / theta, (T::ONE - theta.cos()) / theta2)
    };
    Matrix::<T, 3, 3>::identity() + k.scale(a) + (k * k).scale(b)
}

/// Rodrigues-Formel: Drehmatrix aus Achse und Winkel. Die Achse muss nicht normiert sein; `None` bei Nullachse
/// oder nicht endlichen Werten.
pub fn rotation_matrix_from_axis_angle<T: Real>(
    axis: Vector<T, 3>,
    angle: T,
) -> Option<Matrix<T, 3, 3>> {
    let n = axis.norm();
    if n.is_nan() || n <= T::ZERO || n.is_infinite() || angle.is_nan() || angle.is_infinite() {
        return None;
    }
    Some(exp_so3(axis.scale(angle / n)))
}

/// Logarithmus-Abbildung `SO(3) → so(3)`: Rotationsvektor mit Winkel `θ ∈ [0, π]`. Behandelt `θ ≈ 0` (Reihe)
/// und `θ ≈ π` (Achse aus `R + I`) gesondert. Setzt eine Drehmatrix voraus (bei Bedarf vorher
/// [`nearest_rotation`](crate::linalg::Matrix::nearest_rotation) anwenden).
///
/// ```
/// # #[cfg(any(feature = "std", feature = "libm"))] {
/// use azmath::geometry::{exp_so3, log_so3};
/// use azmath::linalg::Vector;
///
/// let w = Vector::new([0.3f64, -0.2, 0.5]);
/// let back = log_so3(&exp_so3(w));
/// assert!((back - w).max_abs() < 1e-14);
/// # }
/// ```
pub fn log_so3<T: Real>(r: &Matrix<T, 3, 3>) -> Vector<T, 3> {
    let half = T::from_f64(0.5);
    let tr = r[(0, 0)] + r[(1, 1)] + r[(2, 2)];
    let c = ((tr - T::ONE) * half).min(T::ONE).max(-T::ONE);
    // vee(R − Rᵀ)/2 = sinθ · n
    let v = Vector::new([
        (r[(2, 1)] - r[(1, 2)]) * half,
        (r[(0, 2)] - r[(2, 0)]) * half,
        (r[(1, 0)] - r[(0, 1)]) * half,
    ]);
    let s = v.norm();
    let theta = s.atan2(c); // stabil für alle Winkel in [0, π]
    let pi_margin = T::from_f64(1e-3);
    if theta < T::EPSILON.sqrt().sqrt() {
        // θ/sinθ ≈ 1 + θ²/6
        return v.scale(T::ONE + theta * theta / T::from_f64(6.0));
    }
    if theta > T::PI - pi_margin {
        // Nahe π ist sinθ ≈ 0: die Achse steckt in den Spalten von (R + I)/2 = n·nᵀ (bei θ = π exakt)
        let mut best = 0;
        for i in 1..3 {
            if r[(i, i)] > r[(best, best)] {
                best = i;
            }
        }
        let col = Vector::new([
            r[(0, best)] + if best == 0 { T::ONE } else { T::ZERO },
            r[(1, best)] + if best == 1 { T::ONE } else { T::ZERO },
            r[(2, best)] + if best == 2 { T::ONE } else { T::ZERO },
        ]);
        let mut n = col.scale(T::ONE / col.norm());
        if n.dot(&v) < T::ZERO {
            n = -n; // Vorzeichen so wählen, dass es zur (kleinen) Antisymmetrie passt
        }
        return n.scale(theta);
    }
    v.scale(theta / s)
}
