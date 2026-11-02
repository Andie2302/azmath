//! Rotationen im Raum: Quaternionen, Rotationsmatrizen, Achse-Winkel, Euler-Winkel.
//!
//! | Baustein | Inhalt |
//! |---|---|
//! | [`Quaternion`] | Hamilton-Produkt, Konjugierte, Inverse, Drehen von Vektoren, `slerp`/`nlerp`, `exp`/`ln` |
//! | [`rotation_matrix_from_axis_angle`], [`exp_so3`], [`log_so3`] | Rodrigues-Formel, Exponential- und Logarithmus-Abbildung der Drehgruppe |
//! | [`EulerZyx`] | Euler-Winkel (Gieren, Nicken, Rollen) **nur für Ein- und Ausgabe**, mit Gimbal-Lock-Erkennung |
//!
//! **Empfehlung:** Intern mit Quaternionen oder Rotationsmatrizen rechnen. Euler-Winkel haben bei Nickwinkel
//! `±90°` einen Gimbal Lock (ein Freiheitsgrad geht verloren); das ist keine Rechenungenauigkeit, sondern
//! eine Eigenschaft der Darstellung.
//!
//! **Konventionen:** Rechtssystem; Quaternion `w + x·i + y·j + z·k` mit Skalarteil zuerst; Drehwinkel positiv
//! gegen den Uhrzeigersinn, wenn man entlang der Achse auf den Ursprung blickt; Rotationsmatrizen wirken auf
//! Spaltenvektoren (`v' = R·v`). Funktionen, die eine *Einheits*quaternion voraussetzen, sagen es in der Doku.
//!
//! ```
//! # #[cfg(any(feature = "std", feature = "libm"))] {
//! use azmath::geometry::Quaternion;
//! use azmath::linalg::Vector;
//!
//! // 90° um die z-Achse: x-Achse wird zur y-Achse
//! let q = Quaternion::from_axis_angle(Vector::new([0.0f64, 0.0, 1.0]), core::f64::consts::FRAC_PI_2).unwrap();
//! let v = q.rotate(Vector::new([1.0, 0.0, 0.0]));
//! assert!((v[0]).abs() < 1e-15 && (v[1] - 1.0).abs() < 1e-15);
//! # }
//! ```

mod euler;
mod quaternion;
mod so3;

pub use euler::EulerZyx;
pub use quaternion::Quaternion;
pub use so3::{cross, exp_so3, log_so3, rotation_matrix_from_axis_angle};
