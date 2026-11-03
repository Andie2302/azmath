//! Tests für SE(3) und Dualquaternionen.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::geometry::{DualQuaternion, Quaternion, Se3, compose_chain};
use azmath::linalg::Vector;

fn v3(a: f64, b: f64, c: f64) -> Vector<f64, 3> {
    Vector::new([a, b, c])
}

fn close(a: Vector<f64, 3>, b: Vector<f64, 3>, tol: f64) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() < tol)
}

fn sample(angle: f64, axis: Vector<f64, 3>, t: Vector<f64, 3>) -> Se3<f64> {
    Se3::new(Quaternion::from_axis_angle(axis, angle).unwrap(), t)
}

#[test]
fn transform_inverse_compose() {
    let a = sample(0.7, v3(0.0, 0.0, 1.0), v3(1.0, 2.0, 3.0));
    let b = sample(-0.4, v3(1.0, 1.0, 0.0), v3(-1.0, 0.5, 0.0));
    let p = v3(0.3, -0.2, 1.1);
    assert!(close(
        a.inverse().transform_point(a.transform_point(p)),
        p,
        1e-13
    ));
    let ab = a * b;
    assert!(close(
        ab.transform_point(p),
        a.transform_point(b.transform_point(p)),
        1e-13
    ));
    let id = a * a.inverse();
    assert!(close(id.transform_point(p), p, 1e-13));
    // Kette
    let c = compose_chain(&[a, b, a]);
    assert!(close(
        c.transform_point(p),
        a.transform_point(b.transform_point(a.transform_point(p))),
        1e-12
    ));
    assert!(close(
        compose_chain::<f64>(&[]).transform_point(p),
        p,
        1e-15
    ));
    // homogene Matrix
    let m = a.to_matrix();
    let r = a.transform_point(p);
    for i in 0..3 {
        let s = m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + m[i][3];
        assert!((s - r[i]).abs() < 1e-13);
    }
    assert_eq!(m[3], [0.0, 0.0, 0.0, 1.0]);
}

#[test]
fn exp_log_roundtrip() {
    for (v, w) in [
        (v3(1.0, -2.0, 0.5), v3(0.3, 0.2, -0.4)),
        (v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 0.0)),
        (v3(1.0, 1.0, 1.0), v3(1e-9, 0.0, 0.0)),
        (v3(0.2, 0.1, -0.3), v3(1.0, 1.5, -1.2)),
    ] {
        let s = Se3::exp(v, w);
        let (v2, w2) = s.log().unwrap();
        assert!(close(v2, v, 1e-8) && close(w2, w, 1e-8), "{v:?} {w:?}");
    }
    // reine Verschiebung
    let s = Se3::exp(v3(1.0, 2.0, 3.0), v3(0.0, 0.0, 0.0));
    assert!(close(s.translation, v3(1.0, 2.0, 3.0), 1e-14));
}

#[test]
fn interpolation_endpoints_and_midpoint() {
    let a = sample(0.2, v3(0.0, 1.0, 0.0), v3(0.0, 0.0, 0.0));
    let b = sample(1.2, v3(0.0, 1.0, 0.0), v3(2.0, 0.0, 0.0));
    let p = v3(1.0, 0.5, -0.3);
    assert!(close(
        a.interpolate(&b, 0.0).unwrap().transform_point(p),
        a.transform_point(p),
        1e-12
    ));
    assert!(close(
        a.interpolate(&b, 1.0).unwrap().transform_point(p),
        b.transform_point(p),
        1e-12
    ));
    let m = a.interpolate(&b, 0.5).unwrap();
    let (angle_axis, _) = m.rotation.to_axis_angle().map(|(x, y)| (y, x)).unwrap();
    assert!((angle_axis - 0.7).abs() < 1e-12);
}

#[test]
fn dual_quaternions() {
    let a = sample(0.9, v3(1.0, 0.0, 1.0), v3(1.0, -1.0, 2.0));
    let b = sample(-0.5, v3(0.0, 1.0, 0.0), v3(0.0, 3.0, -1.0));
    let (da, db) = (DualQuaternion::from_se3(&a), DualQuaternion::from_se3(&b));
    let p = v3(0.4, 0.2, -0.7);
    assert!(close(da.transform_point(p), a.transform_point(p), 1e-12));
    // Produkt entspricht Verkettung
    assert!(close(
        (da * db).transform_point(p),
        (a * b).transform_point(p),
        1e-12
    ));
    // Rückumwandlung
    let back = da.to_se3();
    assert!(close(back.translation, a.translation, 1e-12));
    // Konjugierte als Inverse
    assert!(close((da * da.conj()).transform_point(p), p, 1e-12));
    // Blend bei t = 0 und 1
    assert!(close(
        da.blend(&db, 0.0).unwrap().transform_point(p),
        a.transform_point(p),
        1e-12
    ));
    assert!(close(
        da.blend(&db, 1.0).unwrap().transform_point(p),
        b.transform_point(p),
        1e-12
    ));
    let zero = DualQuaternion {
        real: Quaternion::new(0.0, 0.0, 0.0, 0.0),
        dual: Quaternion::new(0.0, 0.0, 0.0, 0.0),
    };
    assert!(zero.normalize().is_none());
    assert!(close(
        DualQuaternion::<f64>::identity().transform_point(p),
        p,
        1e-15
    ));
}
