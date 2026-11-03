use super::{Quaternion, cross};
use crate::Real;
use crate::linalg::Vector;
use core::ops::Mul;

/// Starre Bewegung im Raum (SE(3)): erst drehen, dann verschieben, `p' = R·p + t`. `rotation` muss eine Einheitsquaternion sein.
///
/// Verkettung `a * b` wendet erst `b`, dann `a` an (wie Matrixprodukt `T_a·T_b`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se3<T> {
    /// Drehung (Einheitsquaternion).
    pub rotation: Quaternion<T>,
    /// Verschiebung.
    pub translation: Vector<T, 3>,
}

impl<T: Real> Se3<T> {
    /// Die identische Bewegung.
    pub fn identity() -> Self {
        Self {
            rotation: Quaternion::IDENTITY,
            translation: Vector::zeros(),
        }
    }

    /// Aus Drehung und Verschiebung.
    pub fn new(rotation: Quaternion<T>, translation: Vector<T, 3>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Wendet die Bewegung auf einen Punkt an.
    pub fn transform_point(&self, p: Vector<T, 3>) -> Vector<T, 3> {
        let r = self.rotation.rotate(p);
        Vector::new([
            r[0] + self.translation[0],
            r[1] + self.translation[1],
            r[2] + self.translation[2],
        ])
    }

    /// Wendet nur die Drehung an (für Richtungsvektoren).
    pub fn transform_vector(&self, v: Vector<T, 3>) -> Vector<T, 3> {
        self.rotation.rotate(v)
    }

    /// Inverse Bewegung.
    pub fn inverse(&self) -> Self {
        let ri = self.rotation.conj();
        let t = ri.rotate(self.translation);
        Self {
            rotation: ri,
            translation: Vector::new([-t[0], -t[1], -t[2]]),
        }
    }

    /// Verkettung `self ∘ other` (erst `other`, dann `self`); die Quaternion wird dabei normiert.
    pub fn compose(&self, other: &Self) -> Self {
        let rot = self.rotation * other.rotation;
        let rot = rot.normalize().unwrap_or(rot);
        let t = self.rotation.rotate(other.translation);
        Self {
            rotation: rot,
            translation: Vector::new([
                t[0] + self.translation[0],
                t[1] + self.translation[1],
                t[2] + self.translation[2],
            ]),
        }
    }

    /// Homogene `4×4`-Matrix (zeilenweise).
    pub fn to_matrix(&self) -> [[T; 4]; 4] {
        let r = self.rotation.to_rotation_matrix();
        let mut m = [[T::ZERO; 4]; 4];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = r[(i, j)];
            }
            m[i][3] = self.translation[i];
        }
        m[3][3] = T::ONE;
        m
    }

    /// Exponentialabbildung eines Twists `(v, ω)` (Verschiebungs- und Drehanteil) der Lie-Algebra se(3).
    pub fn exp(v: Vector<T, 3>, w: Vector<T, 3>) -> Self {
        let theta = w.norm();
        let (a, b) = if theta < T::EPSILON.sqrt() {
            (T::from_f64(0.5), T::from_f64(1.0 / 6.0))
        } else {
            (
                (T::ONE - theta.cos()) / (theta * theta),
                (theta - theta.sin()) / (theta * theta * theta),
            )
        };
        let wv = cross(w, v);
        let wwv = cross(w, wv);
        let t = Vector::new([
            v[0] + a * wv[0] + b * wwv[0],
            v[1] + a * wv[1] + b * wwv[1],
            v[2] + a * wv[2] + b * wwv[2],
        ]);
        Self {
            rotation: Quaternion::from_rotation_vector(w),
            translation: t,
        }
    }

    /// Logarithmus: Twist `(v, ω)` mit `exp(v, ω) = self` (Drehwinkel in `[0, π]`).
    pub fn log(&self) -> Option<(Vector<T, 3>, Vector<T, 3>)> {
        let w = self.rotation.to_rotation_vector()?;
        let theta = w.norm();
        let c = if theta < T::EPSILON.sqrt() {
            T::from_f64(1.0 / 12.0)
        } else {
            let half = theta * T::from_f64(0.5);
            (T::ONE - half * half.cos() / half.sin()) / (theta * theta)
        };
        let wt = cross(w, self.translation);
        let wwt = cross(w, wt);
        let half = T::from_f64(0.5);
        let v = Vector::new([
            self.translation[0] - half * wt[0] + c * wwt[0],
            self.translation[1] - half * wt[1] + c * wwt[1],
            self.translation[2] - half * wt[2] + c * wwt[2],
        ]);
        Some((v, w))
    }

    /// Interpolation auf der Geodäte: `self · exp(t·log(self⁻¹·other))`.
    pub fn interpolate(&self, other: &Self, t: T) -> Option<Self> {
        let rel = self.inverse().compose(other);
        let (v, w) = rel.log()?;
        Some(self.compose(&Self::exp(v.scale(t), w.scale(t))))
    }
}

impl<T: Real> Mul for Se3<T> {
    type Output = Se3<T>;
    fn mul(self, o: Se3<T>) -> Se3<T> {
        self.compose(&o)
    }
}

/// Verkettet eine Folge von Bewegungen `T₁·T₂·…·Tₙ` (leer: identisch).
pub fn compose_chain<T: Real>(chain: &[Se3<T>]) -> Se3<T> {
    chain.iter().fold(Se3::identity(), |acc, t| acc.compose(t))
}

/// Duales Quaternion `r + ε·d` für starre Bewegungen: `r` Drehung, `d = ½·t·r` mit `t` als reinem Quaternion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualQuaternion<T> {
    /// Realteil (Drehung).
    pub real: Quaternion<T>,
    /// Dualteil (kodiert die Verschiebung).
    pub dual: Quaternion<T>,
}

impl<T: Real> DualQuaternion<T> {
    /// Das Einselement.
    pub fn identity() -> Self {
        Self {
            real: Quaternion::IDENTITY,
            dual: Quaternion::new(T::ZERO, T::ZERO, T::ZERO, T::ZERO),
        }
    }

    /// Aus einer Bewegung.
    pub fn from_se3(s: &Se3<T>) -> Self {
        let t = Quaternion::new(
            T::ZERO,
            s.translation[0],
            s.translation[1],
            s.translation[2],
        );
        Self {
            real: s.rotation,
            dual: (t * s.rotation) * T::from_f64(0.5),
        }
    }

    /// Zurück zur Bewegung (für Einheits-Dualquaternionen).
    pub fn to_se3(&self) -> Se3<T> {
        let t = (self.dual * self.real.conj()) * T::from_f64(2.0);
        Se3 {
            rotation: self.real,
            translation: Vector::new([t.x, t.y, t.z]),
        }
    }

    /// Quaternionen-Konjugierte beider Teile (Inverse für Einheits-Dualquaternionen).
    pub fn conj(&self) -> Self {
        Self {
            real: self.real.conj(),
            dual: self.dual.conj(),
        }
    }

    /// Normiert so, dass `|real| = 1` und `real·dual` orthogonal sind; `None` bei Nullrealteil.
    pub fn normalize(&self) -> Option<Self> {
        let n = self.real.norm();
        if !(n > T::ZERO) {
            return None;
        }
        let r = self.real / n;
        let d = self.dual / n;
        // Dualteil orthogonal zum Realteil machen
        let k = r.dot(d);
        Some(Self {
            real: r,
            dual: d - r * k,
        })
    }

    /// Wendet die Bewegung auf einen Punkt an.
    pub fn transform_point(&self, p: Vector<T, 3>) -> Vector<T, 3> {
        self.to_se3().transform_point(p)
    }

    /// Lineare Mischung mit Normierung (DLB): `(1−t)·self + t·other`, bei kürzestem Weg.
    pub fn blend(&self, other: &Self, t: T) -> Option<Self> {
        let s = if self.real.dot(other.real) < T::ZERO {
            -T::ONE
        } else {
            T::ONE
        };
        let a = T::ONE - t;
        Self {
            real: self.real * a + other.real * (t * s),
            dual: self.dual * a + other.dual * (t * s),
        }
        .normalize()
    }
}

impl<T: Real> Mul for DualQuaternion<T> {
    type Output = DualQuaternion<T>;
    fn mul(self, o: Self) -> Self {
        Self {
            real: self.real * o.real,
            dual: self.real * o.dual + self.dual * o.real,
        }
    }
}
