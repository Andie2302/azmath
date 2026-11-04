//! Tests für Edge Cases des Sparse-Formats (0x0, Null-Matrix, rektangulär, Vektor-ähnlich).
//! Diese Tests ergänzen die grundlegende Strukturschätzung in tests/sparse.rs.
//!
//! Die Tests laufen nur mit den genauen Backends `std` und `libm`.
#![cfg(any(feature = "std", feature = "libm"))]

use azmath::sparse::{CsrView, SparseError};

/// Test case: Empty matrix (0x0)
#[test]
fn empty_matrix_is_valid() {
    assert!(CsrView::<f64>::new(0, 0, &[0], &[], &[]).is_ok());
    let a = CsrView::<f64>::new(0, 0, &[0], &[], &[]).unwrap();
    assert_eq!((a.rows(), a.cols(), a.nnz()), (0, 0, 0));
    let mut y = [0.0; 0];
    a.matvec(&[], &mut y).unwrap();
}

/// Test case: Transpose on empty matrix (0x0)
#[test]
fn transpose_on_empty_matrix() {
    // 0 rows, 0 columns: row_ptr must have len rows + 1 = 1
    let (rp, ci, v): (&[usize], &[usize], &[f64]) = (&[0], &[], &[]);
    let a = CsrView::new(0, 0, rp, ci, v).unwrap();
    let mut yt = [0.0; 0];
    a.matvec_transposed(&[], &mut yt).unwrap();
    assert_eq!(yt, []);
}

#[test]
fn zero_matrix_is_valid() {
    // 2 rows, 3 cols: explizite Nullen gespeichert
    // Zeile 0: Spalte 0 (0.0), Spalte 2 (0.0)
    // Zeile 1: Spalte 1 (0.0)
    let (rp, ci, v) = ([0usize, 2, 3], [0usize, 2, 1], [0.0, 0.0, 0.0]);
    let a = CsrView::new(2, 3, &rp, &ci, &v).unwrap();
    let mut y = [0.0; 2];
    a.matvec(&[1.0, 2.0, 3.0], &mut y).unwrap();
    assert_eq!(y, [0.0, 0.0]);
}

/// Test case: Non-square rectangular matrix (2x3)
#[test]
fn rectangular_matrix_2x3() {
    // 2 rows, 3 cols:
    // Zeile 0: col 0 = 2.0, col 2 = 1.0  => [2.0, 0.0, 1.0]
    // Zeile 1: col 1 = 3.0              => [0.0, 3.0, 0.0]
    let (rp, ci, v) = ([0usize, 2, 3], [0usize, 2, 1], [2.0, 1.0, 3.0]);
    let a = CsrView::new(2, 3, &rp, &ci, &v).unwrap();
    assert_eq!(a.rows(), 2);
    assert_eq!(a.cols(), 3);

    // y = A * x:
    // y[0] = 2.0*1.0 + 1.0*3.0 = 5.0
    // y[1] = 3.0*2.0 = 6.0
    let mut y = [0.0; 2];
    a.matvec(&[1.0, 2.0, 3.0], &mut y).unwrap();
    assert_eq!(y, [5.0, 6.0]);

    // Test transpose (3x2):
    // A^T Zeile 0: [2.0, 0.0] * [1.0, 1.0] = 2.0
    // A^T Zeile 1: [0.0, 3.0] * [1.0, 1.0] = 3.0
    // A^T Zeile 2: [1.0, 0.0] * [1.0, 1.0] = 1.0
    let mut yt = [0.0; 3];
    a.matvec_transposed(&[1.0, 1.0], &mut yt).unwrap();
    assert_eq!(yt, [2.0, 3.0, 1.0]);
}

/// Test case: Non-square rectangular matrix (3x2)
#[test]
fn rectangular_matrix_3x2() {
    // 3 rows, 2 cols:
    // row_ptr muss 4 Elemente haben (rows + 1)
    // Zeile 0: col 0 = 1.0
    // Zeile 1: col 1 = 2.0
    // Zeile 2: col 0 = 3.0
    let (rp, ci, v) = ([0usize, 1, 2, 3], [0usize, 1, 0], [1.0, 2.0, 3.0]);
    let a = CsrView::new(3, 2, &rp, &ci, &v).unwrap();
    assert_eq!(a.rows(), 3);
    assert_eq!(a.cols(), 2);

    // y = A * [1.0, 2.0]:
    // y[0] = 1.0 * 1.0 = 1.0
    // y[1] = 2.0 * 2.0 = 4.0
    // y[2] = 3.0 * 1.0 = 3.0
    let mut y = [0.0; 3];
    a.matvec(&[1.0, 2.0], &mut y).unwrap();
    assert_eq!(y, [1.0, 4.0, 3.0]);

    // yt = A^T * [1.0, 2.0, 3.0]:
    // A^T Spalte 0 (aus A Zeile 0 und 2): 1.0*1.0 + 3.0*3.0 = 10.0
    // A^T Spalte 1 (aus A Zeile 1):       2.0*2.0 = 4.0
    let mut yt = [0.0; 2];
    a.matvec_transposed(&[1.0, 2.0, 3.0], &mut yt).unwrap();
    assert_eq!(yt, [10.0, 4.0]);
}

#[test]
fn vector_like_sparse_1x3() {
    // 1 row, 3 columns: row_ptr muss Länge 2 haben ([0, 3])
    let (rp, ci, v) = ([0usize, 3], [0usize, 1, 2], [1.0, 2.0, 3.0]);
    let a = CsrView::new(1, 3, &rp, &ci, &v).unwrap();
    assert_eq!(a.rows(), 1);
    assert_eq!(a.cols(), 3);

    // 1*5 + 2*6 + 3*7 = 5 + 12 + 21 = 38
    let mut y = [0.0; 1];
    a.matvec(&[5.0, 6.0, 7.0], &mut y).unwrap();
    assert_eq!(y, [38.0]);
}

/// Test case: Diagonal retrieval for Vector-like (1x3)
#[test]
fn diagonal_vector_like_1x3() {
    let (rp, ci, v) = ([0usize, 3], [0usize, 1, 2], [1.0, 2.0, 3.0]);
    let a = CsrView::new(1, 3, &rp, &ci, &v).unwrap();
    let mut d = [0.0; 1];
    a.diagonal_into(&mut d).unwrap();
    assert_eq!(d, [1.0]);
}

#[test]
fn column_vector_like_3x1() {
    // 3 rows, 1 column: row_ptr hat Länge 4 ([0, 1, 2, 3])
    // Alle Werte liegen in Spalte 0
    let (rp, ci, v) = ([0usize, 1, 2, 3], [0usize, 0, 0], [1.0, 2.0, 3.0]);
    let a = CsrView::new(3, 1, &rp, &ci, &v).unwrap();
    assert_eq!(a.rows(), 3);
    assert_eq!(a.cols(), 1);

    let mut y = [0.0; 3];
    a.matvec(&[5.0], &mut y).unwrap();
    assert_eq!(y, [5.0, 10.0, 15.0]);

    // Transponiert ist 1x3: Eingabe 3, Ausgabe 1
    // 1.0*1.0 + 2.0*2.0 + 3.0*3.0 = 14.0
    let mut yt = [0.0; 1];
    a.matvec_transposed(&[1.0, 2.0, 3.0], &mut yt).unwrap();
    assert_eq!(yt, [14.0]);
}

/// Test case: Diagonal retrieval for Column-vector-like (3x1)
#[test]
fn diagonal_column_vector_like_3x1() {
    let (rp, ci, v) = ([0usize, 1, 2, 3], [0usize, 0, 0], [1.0, 2.0, 3.0]);
    let a = CsrView::new(3, 1, &rp, &ci, &v).unwrap();
    let mut d = [0.0; 1];
    a.diagonal_into(&mut d).unwrap();
    assert_eq!(d, [1.0]);
}

#[test]
fn matvec_with_zero_input() {
    let (rp, ci, v) = ([0usize, 2, 3], [0usize, 2, 1], [2.0, 1.0, 3.0]);
    let a = CsrView::new(2, 3, &rp, &ci, &v).unwrap();
    let mut y = [0.0; 2];
    a.matvec(&[0.0, 0.0, 0.0], &mut y).unwrap();
    assert_eq!(y, [0.0, 0.0]);
}

/// Test case: Product with zero matrix
#[test]
fn matvec_with_zero_matrix() {
    let (rp, ci, v) = ([0usize, 2, 3], [0usize, 2, 1], [0.0, 0.0, 0.0]);
    let a = CsrView::new(2, 3, &rp, &ci, &v).unwrap();
    let mut y = [0.0; 2];
    a.matvec(&[1.0, 2.0, 3.0], &mut y).unwrap();
    assert_eq!(y, [0.0, 0.0]);
}

/// Test case: Diagonal retrieval on non-square matrix (safe retrieval)
#[test]
fn diagonal_on_non_square() {
    // 2x3 Matrix:
    // Zeile 0: col 0 = 2.0, col 2 = 1.0 (Diagonalelement (0,0) ist 2.0)
    // Zeile 1: col 1 = 3.0              (Diagonalelement (1,1) ist 3.0)
    let (rp, ci, v) = ([0usize, 2, 3], [0usize, 2, 1], [2.0, 1.0, 3.0]);
    let a = CsrView::new(2, 3, &rp, &ci, &v).unwrap();
    let mut d = [0.0; 2];
    a.diagonal_into(&mut d).unwrap();
    assert_eq!(d, [2.0, 3.0]);
}

/// Test case: Dimension mismatch for 1x3 -> 2 output
#[test]
fn dimension_mismatch_1x3_to_2() {
    let (rp, ci, v) = ([0usize, 3], [0usize, 1, 2], [1.0, 2.0, 3.0]);
    let a = CsrView::new(1, 3, &rp, &ci, &v).unwrap();
    let mut y = [0.0; 2]; // Erwartet 1, bereitgestellt 2
    assert_eq!(a.matvec(&[5.0, 6.0, 7.0], &mut y), Err(SparseError::DimensionMismatch));
}

/// Test case: Transpose dimensions check (1x3 -> 3x1)
#[test]
fn transpose_dimension_check_1x3() {
    let (rp, ci, v) = ([0usize, 3], [0usize, 1, 2], [1.0, 2.0, 3.0]);
    let a = CsrView::new(1, 3, &rp, &ci, &v).unwrap();
    let mut yt = [0.0; 3];
    a.matvec_transposed(&[5.0], &mut yt).unwrap();
    assert_eq!(yt, [5.0, 10.0, 15.0]);

    let mut yt_bad = [0.0; 2];
    assert_eq!(a.matvec_transposed(&[5.0], &mut yt_bad), Err(SparseError::DimensionMismatch));
}