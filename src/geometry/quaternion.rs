use super::euler::EulerZyx;
use super::so3::cross;
use crate::Real;
use crate::linalg::{LinalgError, Matrix, Vector};
use core::ops::{Add, Div, Mul, Neg, Sub};

/// Quaternion `w + x·i + y·j + z·k` (Skalarteil zuerst). **Einheitsquaternionen** (`‖q‖ = 1`) beschreiben
/// Drehungen; `q` und `−q` stehen für dieselbe Drehung.
///
/// Funktionen, die eine Einheitsquaternion voraussetzen (`rotate`, `to_rotation_matrix`, `slerp`, `nlerp`),
/// normieren nicht selbst; mit [`normalize`](Self::normalize) lässt sich Rundungsdrift korrigieren.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion<T> {
    /// Skalarteil.
    pub w: T,
    /// Koeffizient von `i`.
    pub x: T,
    /// Koeffizient von `j`.
    pub y: T,
    /// Koeffizient von `k`.
    pub z: T,
}

impl<T: Real> Quaternion<T> {
    /// Das Einselement (keine Drehung).
    pub const IDENTITY: Self = Self {
        w: T::ONE,
        x: T::ZERO,
        y: T::ZERO,
        z: T::ZERO,
    };

    /// Aus den vier Komponenten.
    pub const fn new(w: T, x: T, y: T, z: T) -> Self {
        Self { w, x, y, z }
    }

    /// Reines (Vektor-)Quaternion `0 + v`.
    pub fn from_vector(v: Vector<T, 3>) -> Self {
        Self::new(T::ZERO, v[0], v[1], v[2])
    }

    /// Der Vektoranteil `(x, y, z)`.
    pub fn vector(&self) -> Vector<T, 3> {
        Vector::new([self.x, self.y, self.z])
    }

    /// Konjugierte `w − x·i − y·j − z·k` (bei Einheitsquaternionen die Inverse).
    pub fn conj(self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Skalarprodukt der vier Komponenten.
    pub fn dot(self, o: Self) -> T {
        self.w * o.w + self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// Quadrat der Norm.
    pub fn norm_sqr(self) -> T {
        self.dot(self)
    }

    /// Norm `√(w² + x² + y² + z²)`.
    pub fn norm(self) -> T {
        self.w.hypot(self.x).hypot(self.y.hypot(self.z))
    }

    /// Auf Länge 1 normiert; `None` bei Nullquaternion oder nicht endlicher Norm.
    pub fn normalize(self) -> Option<Self> {
        let n = self.norm();
        if n > T::ZERO && !n.is_infinite() {
            Some(self / n)
        } else {
            None
        }
    }

    /// Inverse `q̄/‖q‖²`; `None` bei Nullquaternion.
    pub fn inverse(self) -> Option<Self> {
        let n2 = self.norm_sqr();
        if n2 > T::ZERO && !n2.is_infinite() {
            Some(self.conj() / n2)
        } else {
            None
        }
    }

    /// Dreht den Vektor `v` um die **Einheits**quaternion `self`: `q·v·q̄`.
    pub fn rotate(self, v: Vector<T, 3>) -> Vector<T, 3> {
        let u = self.vector();
        let two = T::from_f64(2.0);
        let t = cross(u, v).scale(two);
        v + t.scale(self.w) + cross(u, t)
    }

    /// Drehung um `axis` (muss nicht normiert sein) mit dem Winkel `angle` in Bogenmaß. `None` bei
    /// Nullachse oder nicht endlichen Werten.
    pub fn from_axis_angle(axis: Vector<T, 3>, angle: T) -> Option<Self> {
        let n = axis.norm();
        if n.is_nan() || n <= T::ZERO || n.is_infinite() || angle.is_nan() || angle.is_infinite() {
            return None;
        }
        let h = angle * T::from_f64(0.5);
        let s = h.sin() / n;
        Some(Self::new(h.cos(), axis[0] * s, axis[1] * s, axis[2] * s))
    }

    /// Achse (Länge 1) und Winkel `∈ [0, π]` der Drehung; normiert `self` zuvor. Bei Winkel 0 ist die Achse
    /// `(1, 0, 0)`. `None` bei Nullquaternion.
    pub fn to_axis_angle(self) -> Option<(Vector<T, 3>, T)> {
        let q = self.normalize()?;
        let q = if q.w < T::ZERO { -q } else { q }; // kürzere der beiden Darstellungen
        let s = q.vector().norm();
        let angle = T::from_f64(2.0) * s.atan2(q.w);
        if s <= T::MIN_POSITIVE {
            return Some((Vector::new([T::ONE, T::ZERO, T::ZERO]), T::ZERO));
        }
        Some((q.vector().scale(T::ONE / s), angle))
    }

    /// Drehung aus einem Rotationsvektor (Achse mal Winkel), siehe [`exp_so3`](crate::geometry::exp_so3).
    pub fn from_rotation_vector(w: Vector<T, 3>) -> Self {
        let theta = w.norm();
        let h = theta * T::from_f64(0.5);
        let k = if theta < T::EPSILON.sqrt() {
            // sin(h)/θ ≈ 1/2 − θ²/48
            T::from_f64(0.5) - theta * theta / T::from_f64(48.0)
        } else {
            h.sin() / theta
        };
        Self::new(h.cos(), w[0] * k, w[1] * k, w[2] * k)
    }

    /// Rotationsvektor (Achse mal Winkel `∈ [0, π]`); `None` bei Nullquaternion.
    pub fn to_rotation_vector(self) -> Option<Vector<T, 3>> {
        let (axis, angle) = self.to_axis_angle()?;
        Some(axis.scale(angle))
    }

    /// Exponentialfunktion `e^q = e^w·(cos|u| + u/|u|·sin|u|)`.
    pub fn exp(self) -> Self {
        let u = self.vector();
        let theta = u.norm();
        let e = self.w.exp();
        let k = if theta < T::EPSILON.sqrt() {
            T::ONE - theta * theta / T::from_f64(6.0)
        } else {
            theta.sin() / theta
        };
        Self::new(
            e * theta.cos(),
            e * k * self.x,
            e * k * self.y,
            e * k * self.z,
        )
    }

    /// Natürlicher Logarithmus (Hauptwert); `None` bei Nullquaternion.
    pub fn ln(self) -> Option<Self> {
        let n = self.norm();
        if n.is_nan() || n <= T::ZERO || n.is_infinite() {
            return None;
        }
        let u = self.vector();
        let theta = u.norm();
        let w = n.ln();
        if theta <= T::MIN_POSITIVE {
            // reell: positiv → ln(w); negativ → Richtung unbestimmt, Achse x gewählt
            return Some(if self.w >= T::ZERO {
                Self::new(w, T::ZERO, T::ZERO, T::ZERO)
            } else {
                Self::new(w, T::PI, T::ZERO, T::ZERO)
            });
        }
        let k = (self.w / n).min(T::ONE).max(-T::ONE).acos() / theta;
        Some(Self::new(w, self.x * k, self.y * k, self.z * k))
    }

    /// Drehmatrix der **Einheits**quaternion.
    pub fn to_rotation_matrix(self) -> Matrix<T, 3, 3> {
        let two = T::from_f64(2.0);
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        Matrix::new([
            [
                T::ONE - two * (y * y + z * z),
                two * (x * y - w * z),
                two * (x * z + w * y),
            ],
            [
                two * (x * y + w * z),
                T::ONE - two * (x * x + z * z),
                two * (y * z - w * x),
            ],
            [
                two * (x * z - w * y),
                two * (y * z + w * x),
                T::ONE - two * (x * x + y * y),
            ],
        ])
    }

    /// Einheitsquaternion (mit `w ≥ 0`) zu einer Drehmatrix nach Shepperd: Es wird immer der numerisch
    /// größte der vier Zweige benutzt, damit nie durch eine kleine Zahl geteilt wird.
    ///
    /// Weicht die Matrix merklich von einer Drehmatrix ab (`‖RᵀR − I‖_F > √ε`), wird sie vorher mit
    /// [`nearest_rotation`](crate::linalg::Matrix::nearest_rotation) bereinigt (SVD). Fehler dann wie dort,
    /// zum Beispiel [`LinalgError::Singular`] für eine rangdefizite Matrix.
    pub fn from_rotation_matrix(m: &Matrix<T, 3, 3>) -> Result<Self, LinalgError> {
        let cleaned;
        let m = if m.orthogonality_error() > T::EPSILON.sqrt() {
            cleaned = m.nearest_rotation()?;
            &cleaned
        } else {
            m
        };
        let four = T::from_f64(4.0);
        let tr = m[(0, 0)] + m[(1, 1)] + m[(2, 2)];
        let q = if tr > T::ZERO {
            let s = (tr + T::ONE).sqrt() * T::from_f64(2.0);
            Self::new(
                s / four,
                (m[(2, 1)] - m[(1, 2)]) / s,
                (m[(0, 2)] - m[(2, 0)]) / s,
                (m[(1, 0)] - m[(0, 1)]) / s,
            )
        } else if m[(0, 0)] > m[(1, 1)] && m[(0, 0)] > m[(2, 2)] {
            let s = (T::ONE + m[(0, 0)] - m[(1, 1)] - m[(2, 2)]).sqrt() * T::from_f64(2.0);
            Self::new(
                (m[(2, 1)] - m[(1, 2)]) / s,
                s / four,
                (m[(0, 1)] + m[(1, 0)]) / s,
                (m[(0, 2)] + m[(2, 0)]) / s,
            )
        } else if m[(1, 1)] > m[(2, 2)] {
            let s = (T::ONE + m[(1, 1)] - m[(0, 0)] - m[(2, 2)]).sqrt() * T::from_f64(2.0);
            Self::new(
                (m[(0, 2)] - m[(2, 0)]) / s,
                (m[(0, 1)] + m[(1, 0)]) / s,
                s / four,
                (m[(1, 2)] + m[(2, 1)]) / s,
            )
        } else {
            let s = (T::ONE + m[(2, 2)] - m[(0, 0)] - m[(1, 1)]).sqrt() * T::from_f64(2.0);
            Self::new(
                (m[(1, 0)] - m[(0, 1)]) / s,
                (m[(0, 2)] + m[(2, 0)]) / s,
                (m[(1, 2)] + m[(2, 1)]) / s,
                s / four,
            )
        };
        let q = q.normalize().ok_or(LinalgError::NotFinite)?;
        Ok(if q.w < T::ZERO { -q } else { q })
    }

    /// Normalisierte lineare Interpolation zwischen **Einheits**quaternionen (kürzester Weg). Schnell, aber
    /// die Winkelgeschwindigkeit ist nicht konstant.
    pub fn nlerp(self, other: Self, t: T) -> Self {
        let b = if self.dot(other) < T::ZERO {
            -other
        } else {
            other
        };
        let q = self * (T::ONE - t) + b * t;
        q.normalize().unwrap_or(self)
    }

    /// Sphärische Interpolation (konstante Winkelgeschwindigkeit) zwischen **Einheits**quaternionen auf dem
    /// kürzesten Weg; bei fast parallelen Quaternionen wird zu [`nlerp`](Self::nlerp) gewechselt.
    pub fn slerp(self, other: Self, t: T) -> Self {
        let mut d = self.dot(other);
        let b = if d < T::ZERO {
            d = -d;
            -other
        } else {
            other
        };
        if d > T::from_f64(0.9995) {
            return self.nlerp(b, t);
        }
        let theta = d.min(T::ONE).acos();
        let s = theta.sin();
        let (wa, wb) = (((T::ONE - t) * theta).sin() / s, (t * theta).sin() / s);
        self * wa + b * wb
    }

    /// Einheitsquaternion zu Euler-Winkeln (intrinsisch Z-Y-X): `q = qz(yaw)·qy(pitch)·qx(roll)`.
    pub fn from_euler_zyx(e: EulerZyx<T>) -> Self {
        let h = T::from_f64(0.5);
        let (sy, cy) = ((e.yaw * h).sin(), (e.yaw * h).cos());
        let (sp, cp) = ((e.pitch * h).sin(), (e.pitch * h).cos());
        let (sr, cr) = ((e.roll * h).sin(), (e.roll * h).cos());
        Self::new(
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    }

    /// Euler-Winkel (intrinsisch Z-Y-X) der Drehung; normiert `self` zuvor, `None` bei Nullquaternion.
    ///
    /// Bei Gimbal Lock (Nickwinkel `±90°` bis auf einige Einheiten `ε`) wird der Rollwinkel 0 und die ganze
    /// Drehung steckt im Gierwinkel; `gimbal_lock` ist dann `true`. Das Argument von `asin` wird auf `[−1, 1]`
    /// begrenzt, es entsteht also kein NaN durch Rundung.
    pub fn to_euler_zyx(self) -> Option<EulerZyx<T>> {
        let q = self.normalize()?;
        let two = T::from_f64(2.0);
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);
        let sp = (two * (w * y - x * z)).min(T::ONE).max(-T::ONE);
        let r10 = two * (x * y + w * z);
        let r00 = T::ONE - two * (y * y + z * z);
        if sp.abs() >= T::ONE - T::from_f64(4.0) * T::EPSILON {
            let r01 = two * (x * y - w * z);
            let r11 = T::ONE - two * (x * x + z * z);
            return Some(EulerZyx {
                yaw: (-r01).atan2(r11),
                pitch: sp.signum() * T::PI * T::from_f64(0.5),
                roll: T::ZERO,
                gimbal_lock: true,
            });
        }
        Some(EulerZyx {
            yaw: r10.atan2(r00),
            pitch: sp.asin(),
            roll: (two * (y * z + w * x)).atan2(T::ONE - two * (x * x + y * y)),
            gimbal_lock: false,
        })
    }
}

impl<T: Real> Add for Quaternion<T> {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.w + o.w, self.x + o.x, self.y + o.y, self.z + o.z)
    }
}
impl<T: Real> Sub for Quaternion<T> {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.w - o.w, self.x - o.x, self.y - o.y, self.z - o.z)
    }
}
impl<T: Real> Neg for Quaternion<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.w, -self.x, -self.y, -self.z)
    }
}
/// Hamilton-Produkt (nicht kommutativ). `a * b` wirkt als Drehung „erst `b`, dann `a`“.
impl<T: Real> Mul for Quaternion<T> {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        )
    }
}
impl<T: Real> Mul<T> for Quaternion<T> {
    type Output = Self;
    fn mul(self, k: T) -> Self {
        Self::new(self.w * k, self.x * k, self.y * k, self.z * k)
    }
}
impl<T: Real> Div<T> for Quaternion<T> {
    type Output = Self;
    fn div(self, k: T) -> Self {
        Self::new(self.w / k, self.x / k, self.y / k, self.z / k)
    }
}
