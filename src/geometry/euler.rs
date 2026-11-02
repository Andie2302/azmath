use crate::Real;

/// Euler-Winkel in der Konvention **intrinsisch Z-Y-X** (Gieren, Nicken, Rollen): `R = Rz(yaw)·Ry(pitch)·Rx(roll)`.
///
/// Nur für Ein- und Ausgabe gedacht. Bei einem Nickwinkel von `±90°` fallen Gier- und Rollachse zusammen
/// (**Gimbal Lock**): Die Zerlegung ist dann nicht eindeutig. [`Quaternion::to_euler_zyx`](crate::geometry::Quaternion::to_euler_zyx)
/// setzt in diesem Fall den Rollwinkel auf 0, legt die ganze Drehung in den Gierwinkel und meldet
/// `gimbal_lock = true`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EulerZyx<T> {
    /// Gieren (Drehung um z), Bogenmaß, in `(−π, π]`.
    pub yaw: T,
    /// Nicken (Drehung um y), Bogenmaß, in `[−π/2, π/2]`.
    pub pitch: T,
    /// Rollen (Drehung um x), Bogenmaß, in `(−π, π]`.
    pub roll: T,
    /// `true`, wenn der Nickwinkel (numerisch) `±90°` ist und der Rollwinkel auf 0 gesetzt wurde.
    pub gimbal_lock: bool,
}

impl<T: Real> EulerZyx<T> {
    /// Neue Euler-Winkel (ohne Gimbal-Lock-Markierung).
    pub fn new(yaw: T, pitch: T, roll: T) -> Self {
        Self {
            yaw,
            pitch,
            roll,
            gimbal_lock: false,
        }
    }
}
