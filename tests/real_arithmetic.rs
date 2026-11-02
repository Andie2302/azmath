//! Tests für die Operatoren, die das `Real`-Trait voraussetzt.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::Real;

fn add<T: Real>(a: T, b: T) -> T {
    a + b
}
fn sub<T: Real>(a: T, b: T) -> T {
    a - b
}
fn mul<T: Real>(a: T, b: T) -> T {
    a * b
}
fn div<T: Real>(a: T, b: T) -> T {
    a / b
}
fn neg<T: Real>(a: T) -> T {
    -a
}

#[test]
fn operators_f32() {
    assert_eq!(add(2.0f32, 3.0), 5.0);
    assert_eq!(sub(2.0f32, 3.0), -1.0);
    assert_eq!(mul(2.0f32, 3.0), 6.0);
    assert_eq!(div(3.0f32, 2.0), 1.5);
    assert_eq!(neg(2.0f32), -2.0);
}

#[test]
fn assign_operators_f32() {
    fn run<T: Real>(start: T, other: T) -> (T, T, T, T) {
        let (mut a, mut b, mut c, mut d) = (start, start, start, start);
        a += other;
        b -= other;
        c *= other;
        d /= other;
        (a, b, c, d)
    }
    assert_eq!(run(8.0f32, 2.0), (10.0, 6.0, 16.0, 4.0));
}

#[test]
fn division_by_zero_f32() {
    assert!(div(1.0f32, 0.0).is_infinite());
    assert!(div(0.0f32, 0.0).is_nan());
    assert_eq!(div(-1.0f32, 0.0), f32::NEG_INFINITY);
}

#[test]
#[allow(clippy::assertions_on_constants)]
fn comparison_f32() {
    fn less<T: Real>(a: T, b: T) -> bool {
        a < b
    }
    assert!(less(1.0f32, 2.0));
    assert!(!less(2.0f32, 1.0));
    assert!(!less(f32::NAN, 1.0));
    assert!(<f32 as Real>::ZERO < <f32 as Real>::ONE);
    assert_eq!(<f32 as Real>::ONE, 1.0f32);
}

#[test]
fn operators_f64() {
    assert_eq!(add(2.0f64, 3.0), 5.0);
    assert_eq!(sub(2.0f64, 3.0), -1.0);
    assert_eq!(mul(2.0f64, 3.0), 6.0);
    assert_eq!(div(3.0f64, 2.0), 1.5);
    assert_eq!(neg(2.0f64), -2.0);
    assert!(div(1.0f64, 0.0).is_infinite());
}
