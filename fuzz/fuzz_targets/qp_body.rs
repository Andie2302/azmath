#[path = "qp_kkt.rs"]
mod qp_kkt;
use crate::common::Bytes;
use azmath::opt::{QpKktBackend, QpOptions, QpStatus, QpWorkspace, interior_point_qp_sparse};
use azmath::sparse::CsrMatrix;

fn matrix(r: &mut Bytes, rows: usize, cols: usize, special: bool, extreme: bool) -> CsrMatrix<f64> {
    let d: Vec<f64> = (0..rows * cols)
        .map(|_| {
            if r.u8().is_multiple_of(3) {
                r.value_with(special, extreme)
            } else {
                0.0
            }
        })
        .collect();
    // nichtendliche Einträge werden von from_dense_slice ggf. abgelehnt → dann leere Matrix
    CsrMatrix::from_dense_slice(rows, cols, &d, 0.0).unwrap_or_else(|_| {
        CsrMatrix::from_dense_slice(rows, cols, &vec![0.0; rows * cols], 0.0).unwrap()
    })
}

#[allow(clippy::too_many_arguments)]
fn details(
    n: usize,
    q: &CsrMatrix<f64>,
    a: &CsrMatrix<f64>,
    g: &CsrMatrix<f64>,
    c: &[f64],
    b: &[f64],
    h: &[f64],
    s: &azmath::opt::QpSolution<f64>,
) -> String {
    format!(
        "n={n} status={:?} iterations={}\nQ={:?}\nA={:?}\nG={:?}\nc={c:?}\nb={b:?}\nh={h:?}\nx={:?}\ny={:?}\nz={:?}",
        s.status,
        s.iterations,
        q.to_dense(),
        a.to_dense(),
        g.to_dense(),
        s.x,
        s.y,
        s.z
    )
}

/// `extreme = false`: moderate Größenordnungen (nur NaN/Inf/±0 als Sonderwerte), zeilenweise Zulässigkeitsprüfung.
/// `extreme = true`: Werte bis ±1e300 und 5e-324, normweise Prüfung (wie das Abbruchkriterium des Lösers).
pub fn run(data: &[u8], extreme: bool) {
    let mut r = Bytes(data);
    let n = r.range(1, 4);
    let p = r.range(0, 3);
    let m = r.range(0, 5);
    let special = r.u8().is_multiple_of(6);
    let backend = [
        QpKktBackend::Auto,
        QpKktBackend::Sparse,
        QpKktBackend::Dense,
    ][r.range(0, 2)];
    let opts = QpOptions {
        kkt_backend: backend,
        max_iter: 60,
        ..QpOptions::default()
    };

    // Q = Mᵀ·M + Diagonale (symmetrisch); bei `special` darf Q beliebig sein
    let q = if special {
        matrix(&mut r, n, n, true, extreme)
    } else {
        let mm: Vec<f64> = (0..n * n).map(|_| r.moderate()).collect();
        let mut qd = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                qd[i * n + j] = (0..n).map(|k| mm[k * n + i] * mm[k * n + j]).sum::<f64>();
            }
            qd[i * n + i] += 0.1;
        }
        CsrMatrix::from_dense_slice(n, n, &qd, 0.0).unwrap()
    };
    let a = matrix(&mut r, p, n, special, extreme);
    let g = matrix(&mut r, m, n, special, extreme);
    let vec_of = |len: usize, r: &mut Bytes| -> Vec<f64> {
        (0..len).map(|_| r.value_with(special, extreme)).collect()
    };
    let mut ws = QpWorkspace::new(q.clone(), a.clone(), g.clone(), opts);

    // Folge verwandter Aufgaben: Arbeitsraum und Kaltstart dürfen nicht paniken
    for round in 0..3 {
        let c = vec_of(n, &mut r);
        let b = vec_of(p, &mut r);
        let h = vec_of(m, &mut r);
        let warm = ws.solve(&c, &b, &h);
        let cold = interior_point_qp_sparse(&q, &c, &a, &b, &g, &h, opts);
        assert_eq!(
            warm.is_ok(),
            cold.is_ok(),
            "Eingabevalidierung muss übereinstimmen"
        );
        if let (Ok(w), Ok(k)) = (warm, cold) {
            for (label, s) in [("workspace", &w), ("cold", &k)] {
                assert_eq!(s.x.len(), n);
                assert_eq!(s.y.len(), p);
                assert_eq!(s.z.len(), m);
                if s.status == QpStatus::Optimal {
                    // Bei optimaler Meldung muss die Lösung endlich und zulässig (mit Toleranz) sein.
                    if let Err(error) = qp_kkt::check(&q, &c, &a, &g, &h, s, opts.tol) {
                        panic!(
                            "KKT violation ({label}, round {round}): {error}\n{}",
                            details(n, &q, &a, &g, &c, &b, &h, s)
                        );
                    }
                    // Das Abbruchkriterium des Lösers ist normweise (relativ zur größten Datengröße),
                    // nicht zeilenweise: Residuen werden gegen ‖b‖∞ bzw. ‖h‖∞ gemessen.
                    let b_inf = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    let h_inf = h.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    for i in 0..m {
                        let gx: f64 = g.row(i).map(|(j, v)| v * s.x[j]).sum();
                        let gt: f64 = g.row(i).map(|(j, v)| (v * s.x[j]).abs()).sum();
                        assert!(
                            gx <= h[i]
                                + 1e-3 * (1.0 + if extreme { h_inf } else { h[i].abs() } + gt),
                            "G·x ≤ h verletzt ({label}, Runde {round}, Zeile {i}): G·x={gx}, h={}\n{}",
                            h[i],
                            details(n, &q, &a, &g, &c, &b, &h, s)
                        );
                    }
                    for i in 0..p {
                        let ax: f64 = a.row(i).map(|(j, v)| v * s.x[j]).sum();
                        let at: f64 = a.row(i).map(|(j, v)| (v * s.x[j]).abs()).sum();
                        assert!(
                            (ax - b[i]).abs()
                                <= 1e-3 * (1.0 + if extreme { b_inf } else { b[i].abs() } + at),
                            "A·x = b verletzt ({label}, Runde {round}, Zeile {i}): A·x={ax}, b={}\n{}",
                            b[i],
                            details(n, &q, &a, &g, &c, &b, &h, s)
                        );
                    }
                }
            }
        }
    }
}
