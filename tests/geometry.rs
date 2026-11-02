//! Tests für Quaternionen, Rotationsmatrizen, Rodrigues/so(3) und Euler-Winkel.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::geometry::{
    EulerZyx, Quaternion, cross, exp_so3, log_so3, rotation_matrix_from_axis_angle,
};
use azmath::linalg::{LinalgError, Matrix, Vector};
use core::f64::consts::{FRAC_PI_2, PI};

type Q = Quaternion<f64>;
type V3 = Vector<f64, 3>;

fn v(x: f64, y: f64, z: f64) -> V3 {
    Vector::new([x, y, z])
}

fn rz(t: f64) -> Matrix<f64, 3, 3> {
    Matrix::new([
        [t.cos(), -t.sin(), 0.0],
        [t.sin(), t.cos(), 0.0],
        [0.0, 0.0, 1.0],
    ])
}
fn ry(t: f64) -> Matrix<f64, 3, 3> {
    Matrix::new([
        [t.cos(), 0.0, t.sin()],
        [0.0, 1.0, 0.0],
        [-t.sin(), 0.0, t.cos()],
    ])
}
fn rx(t: f64) -> Matrix<f64, 3, 3> {
    Matrix::new([
        [1.0, 0.0, 0.0],
        [0.0, t.cos(), -t.sin()],
        [0.0, t.sin(), t.cos()],
    ])
}

fn close_q(a: Q, b: Q, tol: f64) {
    // q und −q sind dieselbe Drehung
    let d = (a - b).norm().min((a + b).norm());
    assert!(d <= tol, "{a:?} gegen {b:?} (Δ = {d:e})");
}

fn close_m(a: &Matrix<f64, 3, 3>, b: &Matrix<f64, 3, 3>, tol: f64) {
    assert!((*a - *b).max_abs() <= tol, "{a:?} gegen {b:?}");
}

fn close_v(a: V3, b: V3, tol: f64) {
    assert!((a - b).max_abs() <= tol, "{a:?} gegen {b:?}");
}

#[test]
fn quaternion_algebra() {
    let a = Q::new(1.0, 2.0, -1.0, 0.5);
    let b = Q::new(-0.5, 1.0, 3.0, 2.0);
    let c = Q::new(0.3, -2.0, 1.0, 1.0);
    // i·j = k, j·i = −k
    let (i, j) = (Q::new(0.0, 1.0, 0.0, 0.0), Q::new(0.0, 0.0, 1.0, 0.0));
    assert_eq!(i * j, Q::new(0.0, 0.0, 0.0, 1.0));
    assert_eq!(j * i, Q::new(0.0, 0.0, 0.0, -1.0));
    // Assoziativität, Normmultiplikativität, Inverse
    close_q((a * b) * c, a * (b * c), 1e-13);
    assert!(((a * b).norm() - a.norm() * b.norm()).abs() < 1e-13);
    close_q(a * a.inverse().unwrap(), Q::IDENTITY, 1e-14);
    close_q((a * b).conj(), b.conj() * a.conj(), 1e-14);
    assert_eq!(Q::new(0.0, 0.0, 0.0, 0.0).inverse(), None);
    assert_eq!(Q::new(0.0, 0.0, 0.0, 0.0).normalize(), None);
    assert!((a.normalize().unwrap().norm() - 1.0).abs() < 1e-15);
}

#[test]
fn rotate_matches_the_rotation_matrix_and_known_cases() {
    let q = Q::from_axis_angle(v(0.0, 0.0, 1.0), FRAC_PI_2).unwrap();
    close_v(q.rotate(v(1.0, 0.0, 0.0)), v(0.0, 1.0, 0.0), 1e-15);
    let q = Q::from_axis_angle(v(1.0, 2.0, -0.5), 1.1).unwrap();
    let p = v(0.3, -1.2, 2.0);
    close_v(q.rotate(p), q.to_rotation_matrix() * p, 1e-14);
    // Länge bleibt erhalten, Komposition: (a*b) dreht erst b, dann a
    let (a, b) = (
        Q::from_axis_angle(v(0.0, 1.0, 0.0), 0.4).unwrap(),
        Q::from_axis_angle(v(1.0, 0.0, 0.0), -0.9).unwrap(),
    );
    close_v((a * b).rotate(p), a.rotate(b.rotate(p)), 1e-14);
    assert!((q.rotate(p).norm() - p.norm()).abs() < 1e-14);
    // reines Quaternion q·(0,p)·q̄ ergibt dasselbe wie rotate
    let pq = q * Q::from_vector(p) * q.conj();
    close_v(pq.vector(), q.rotate(p), 1e-14);
}

#[test]
fn axis_angle_roundtrip() {
    for (axis, angle) in [
        (v(0.0, 0.0, 1.0), 0.0),
        (v(1.0, 0.0, 0.0), 1e-9),
        (v(1.0, 1.0, 0.0), 1.0),
        (v(-0.3, 0.2, 0.9), 2.5),
        (v(0.0, 1.0, 0.0), PI - 1e-9),
    ] {
        let q = Q::from_axis_angle(axis, angle).unwrap();
        let (a, t) = q.to_axis_angle().unwrap();
        assert!((t - angle).abs() < 1e-8, "Winkel {t} statt {angle}");
        if angle > 1e-6 {
            close_v(a, axis.scale(1.0 / axis.norm()), 1e-8);
        }
    }
    assert_eq!(Q::from_axis_angle(v(0.0, 0.0, 0.0), 1.0), None);
    assert_eq!(Q::from_axis_angle(v(1.0, 0.0, 0.0), f64::NAN), None);
    // Winkel > π wird zur kürzeren Darstellung
    let (a, t) = Q::from_axis_angle(v(0.0, 0.0, 1.0), 1.5 * PI)
        .unwrap()
        .to_axis_angle()
        .unwrap();
    assert!((t - FRAC_PI_2).abs() < 1e-14);
    close_v(a, v(0.0, 0.0, -1.0), 1e-14);
}

#[test]
fn matrix_roundtrip_in_all_shepperd_branches() {
    // Drehwinkel nahe 0 (Spur-Zweig) und nahe π um jede Achse (drei Diagonalzweige)
    let cases = [
        (v(1.0, 0.0, 0.0), 1e-6),
        (v(1.0, 0.0, 0.0), PI),
        (v(0.0, 1.0, 0.0), PI),
        (v(0.0, 0.0, 1.0), PI),
        (v(1.0, 2.0, 3.0), PI - 1e-7),
        (v(-1.0, 0.5, 0.2), 2.9),
        (v(0.2, -1.0, 0.1), 3.0),
        (v(0.1, 0.2, -1.0), 3.1),
        (v(0.3, 0.4, 0.5), 0.8),
    ];
    for (axis, angle) in cases {
        let q = Q::from_axis_angle(axis, angle).unwrap();
        let m = q.to_rotation_matrix();
        let back = Q::from_rotation_matrix(&m).unwrap();
        close_q(back, q, 1e-7);
        assert!(back.w >= 0.0);
        close_m(&back.to_rotation_matrix(), &m, 1e-12);
    }
}

#[test]
fn from_rotation_matrix_cleans_up_a_noisy_matrix() {
    let q = Q::from_axis_angle(v(0.3, -0.5, 0.8), 1.2).unwrap();
    let noisy = q.to_rotation_matrix()
        + Matrix::new([
            [1e-3, -2e-3, 5e-4],
            [-1e-3, 1e-3, 2e-3],
            [3e-4, -1e-3, 1e-3],
        ]);
    let back = Q::from_rotation_matrix(&noisy).unwrap();
    assert!((back.norm() - 1.0).abs() < 1e-14);
    assert!((back - q).norm().min((back + q).norm()) < 5e-3);
    assert_eq!(
        Q::from_rotation_matrix(&Matrix::<f64, 3, 3>::zeros()),
        Err(LinalgError::Singular)
    );
}

#[test]
fn slerp_properties() {
    let a = Q::from_axis_angle(v(0.0, 0.0, 1.0), 0.2).unwrap();
    let b = Q::from_axis_angle(v(0.0, 0.0, 1.0), 2.2).unwrap();
    close_q(a.slerp(b, 0.0), a, 1e-14);
    close_q(a.slerp(b, 1.0), b, 1e-14);
    close_q(
        a.slerp(b, 0.5),
        Q::from_axis_angle(v(0.0, 0.0, 1.0), 1.2).unwrap(),
        1e-14,
    );
    // konstante Winkelgeschwindigkeit: gleiche Winkelschritte
    let step = |s: f64, t: f64| {
        let d = a.slerp(b, s).conj() * a.slerp(b, t);
        d.to_axis_angle().unwrap().1
    };
    assert!((step(0.1, 0.3) - step(0.6, 0.8)).abs() < 1e-13);
    // nlerp stimmt an den Enden und liegt auf dem Einheitskugel
    close_q(a.nlerp(b, 0.0), a, 1e-14);
    assert!((a.nlerp(b, 0.37).norm() - 1.0).abs() < 1e-15);
    // −b ist dieselbe Drehung: kurzer Weg
    close_q(a.slerp(-b, 0.5), a.slerp(b, 0.5), 1e-14);
    // fast parallel → nlerp-Zweig bleibt normiert
    let c = Q::from_axis_angle(v(0.0, 0.0, 1.0), 0.2 + 1e-5).unwrap();
    assert!((a.slerp(c, 0.5).norm() - 1.0).abs() < 1e-14);
}

#[test]
fn quaternion_exp_ln_and_rotation_vector() {
    let q = Q::new(0.5, 0.2, -0.3, 0.9);
    close_q(q.ln().unwrap().exp(), q, 1e-14);
    close_q(Q::new(0.0, 0.0, 0.0, 0.0).exp(), Q::IDENTITY, 0.0);
    assert_eq!(Q::new(0.0, 0.0, 0.0, 0.0).ln(), None);
    assert_eq!(
        Q::new(2.0, 0.0, 0.0, 0.0).ln().unwrap(),
        Q::new(2f64.ln(), 0.0, 0.0, 0.0)
    );
    let w = v(0.2, -0.4, 0.7);
    let r = Q::from_rotation_vector(w);
    close_v(r.to_rotation_vector().unwrap(), w, 1e-14);
    close_v(
        Q::from_rotation_vector(v(1e-12, 0.0, 0.0)).vector(),
        v(5e-13, 0.0, 0.0),
        1e-24,
    );
    close_m(&r.to_rotation_matrix(), &exp_so3(w), 1e-14);
    // exp eines reinen Quaternions (0, w/2) ist die Drehung zum Rotationsvektor w
    close_q(Q::from_vector(w.scale(0.5)).exp(), r, 1e-14);
}

#[test]
fn so3_exp_log_roundtrip_including_small_and_pi() {
    for w in [
        v(0.0, 0.0, 0.0),
        v(1e-9, -2e-9, 3e-9),
        v(0.3, -0.2, 0.5),
        v(1.0, 2.0, -1.5),
        v(0.0, 0.0, PI - 1e-4),
        v(PI - 1e-9, 0.0, 0.0),
    ] {
        let r = exp_so3(w);
        assert!(r.orthogonality_error() < 1e-14);
        assert!((r.determinant() - 1.0).abs() < 1e-14);
        close_v(log_so3(&r), w, 1e-7);
    }
    // genau π: Achse nur bis auf das Vorzeichen bestimmt, Winkel stimmt
    let n = v(1.0, 2.0, 2.0).scale(1.0 / 3.0);
    let r = exp_so3(n.scale(PI));
    let w = log_so3(&r);
    assert!((w.norm() - PI).abs() < 1e-7);
    close_m(&exp_so3(w), &r, 1e-7);
    // Rodrigues mit nicht normierter Achse
    let m = rotation_matrix_from_axis_angle(v(0.0, 0.0, 5.0), 0.7).unwrap();
    close_m(&m, &rz(0.7), 1e-15);
    assert_eq!(rotation_matrix_from_axis_angle(v(0.0, 0.0, 0.0), 1.0), None);
    assert_eq!(cross(v(1.0, 0.0, 0.0), v(0.0, 1.0, 0.0)), v(0.0, 0.0, 1.0));
}

#[test]
fn euler_zyx_matches_matrix_composition() {
    let (yaw, pitch, roll) = (0.7, -0.4, 1.9);
    let q = Q::from_euler_zyx(EulerZyx::new(yaw, pitch, roll));
    close_m(
        &q.to_rotation_matrix(),
        &(rz(yaw) * ry(pitch) * rx(roll)),
        1e-14,
    );
    assert!((q.norm() - 1.0).abs() < 1e-15);
}

#[test]
fn euler_roundtrip_on_a_grid_away_from_gimbal_lock() {
    let n = 9;
    for iy in 0..n {
        for ip in 1..n - 1 {
            for ir in 0..n {
                let yaw = -PI + 0.05 + (2.0 * PI - 0.1) * iy as f64 / (n - 1) as f64;
                let pitch = -FRAC_PI_2 + 0.05 + (PI - 0.1) * ip as f64 / (n - 1) as f64;
                let roll = -PI + 0.05 + (2.0 * PI - 0.1) * ir as f64 / (n - 1) as f64;
                let e = Q::from_euler_zyx(EulerZyx::new(yaw, pitch, roll))
                    .to_euler_zyx()
                    .unwrap();
                assert!(!e.gimbal_lock);
                assert!((e.yaw - yaw).abs() < 1e-12, "yaw {} gegen {yaw}", e.yaw);
                assert!((e.pitch - pitch).abs() < 1e-12);
                assert!((e.roll - roll).abs() < 1e-12);
            }
        }
    }
}

#[test]
fn gimbal_lock_is_detected_and_keeps_the_rotation() {
    for pitch in [FRAC_PI_2, -FRAC_PI_2] {
        for (yaw, roll) in [(0.3, 0.9), (-2.0, 0.4), (1.0, -1.0)] {
            let q = Q::from_euler_zyx(EulerZyx::new(yaw, pitch, roll));
            let e = q.to_euler_zyx().unwrap();
            assert!(e.gimbal_lock);
            assert_eq!(e.roll, 0.0);
            assert!((e.pitch - pitch).abs() < 1e-12);
            // andere Winkel, dieselbe Drehung
            let again = Q::from_euler_zyx(e);
            close_m(&again.to_rotation_matrix(), &q.to_rotation_matrix(), 1e-7);
        }
    }
    // Rundungsdrift: |sin(pitch)| ein Hauch über 1 darf kein NaN ergeben
    let q = Q::new(0.5, 0.5, 0.5, 0.5).normalize().unwrap();
    let e = q.to_euler_zyx().unwrap();
    assert!(e.yaw.is_finite() && e.pitch.is_finite() && e.roll.is_finite());
    assert_eq!(Q::new(0.0, 0.0, 0.0, 0.0).to_euler_zyx(), None);
}

#[test]
fn works_for_f32() {
    type Qf = Quaternion<f32>;
    let q = Qf::from_axis_angle(
        Vector::new([0.0f32, 0.0, 1.0]),
        core::f32::consts::FRAC_PI_2,
    )
    .unwrap();
    let p = q.rotate(Vector::new([1.0, 0.0, 0.0]));
    assert!(p[0].abs() < 1e-6 && (p[1] - 1.0).abs() < 1e-6);
    let back = Qf::from_rotation_matrix(&q.to_rotation_matrix()).unwrap();
    assert!((back - q).norm() < 1e-6);
}
