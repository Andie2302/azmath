//! Geometrisches Multigrid für den zellzentrierten Operator aus [`fdm::poisson`](crate::fdm::poisson).
//!
//! Anders als [`solve_2d`](super::solve_2d) (Dirichlet, Knotengitter mit `2ᵏ − 1` Punkten) arbeitet dieses
//! Verfahren auf `nx × ny` **Zellen**, die sich durch Zusammenfassen von 2×2 Zellen vergröbern lassen, und
//! unterstützt je Achse Dirichlet, Neumann und periodisch.
//!
//! **Nullraum:** Restriktion (Mittel über 2×2 Zellen) und Prolongation (bilinear, Gewichte 9/16, 3/16, 3/16,
//! 1/16) haben Gewichtssumme 1. Konstanten bleiben daher bei Neumann und periodisch unverändert; der Zyklus
//! hält zusätzlich Residuum und Korrektur auf jeder Ebene mittelwertfrei, sodass der Mittelwert nicht
//! wegläuft. Der Zyklus ist symmetrisch (Vorglättung vorwärts, Nachglättung rückwärts gleich oft) und damit
//! als PCG-Vorkonditionierer geeignet.

use crate::Real;
use crate::fdm::poisson::{Boundary, PoissonError, has_nullspace, project_mean_zero};
use crate::linalg::iterative::Preconditioner;
use alloc::vec;
use alloc::vec::Vec;

#[derive(Debug, Clone)]
struct Level<T> {
    nx: usize,
    ny: usize,
    wx: T, // 1/hx²
    wy: T, // 1/hy²
}

/// Mehrgitter-Hierarchie für `−Δ` auf Zellgittern.
#[derive(Debug, Clone)]
pub struct CellMultigrid2d<T> {
    levels: Vec<Level<T>>,
    bcx: Boundary,
    bcy: Boundary,
    sweeps: usize,
}

fn min_cells(bc: Boundary) -> usize {
    if bc == Boundary::Periodic { 3 } else { 2 }
}

/// Wert des Nachbarn von Zelle `k` in Richtung `dir` (±1) entlang einer Achse, mit Geisterwert am Rand.
/// Liefert `(Wert, echter_Nachbar)`; am Rand ist der Wert `c·u[k]` mit `c = ±1` (Neumann/Dirichlet).
#[inline]
fn nb<T: Real>(
    u: &[T],
    k: usize,
    pos: usize,
    len: usize,
    stride: usize,
    bc: Boundary,
    up: bool,
) -> (T, bool) {
    let idx = if up {
        if pos + 1 < len {
            Some(k + stride)
        } else if bc == Boundary::Periodic {
            Some(k - (len - 1) * stride)
        } else {
            None
        }
    } else if pos > 0 {
        Some(k - stride)
    } else if bc == Boundary::Periodic {
        Some(k + (len - 1) * stride)
    } else {
        None
    };
    match idx {
        Some(m) => (u[m], true),
        None if bc == Boundary::Neumann => (u[k], false),
        None => (-u[k], false),
    }
}

impl<T: Real> CellMultigrid2d<T> {
    /// Baut die Hierarchie; vergröbert, solange beide Zellzahlen gerade bleiben und die Mindestgröße
    /// (periodisch 3, sonst 2) erhalten bleibt. `sweeps`: Glättungsdurchläufe je Seite (mindestens 1).
    pub fn new(
        nx: usize,
        ny: usize,
        hx: T,
        hy: T,
        bcx: Boundary,
        bcy: Boundary,
        sweeps: usize,
    ) -> Result<Self, PoissonError> {
        if nx < min_cells(bcx)
            || ny < min_cells(bcy)
            || !(hx.is_finite() && hx > T::ZERO)
            || !(hy.is_finite() && hy > T::ZERO)
            || sweeps == 0
        {
            return Err(PoissonError::InvalidGrid);
        }
        let (mut cx, mut cy, mut chx, mut chy) = (nx, ny, hx, hy);
        let two = T::from_f64(2.0);
        let mut levels = vec![];
        loop {
            levels.push(Level {
                nx: cx,
                ny: cy,
                wx: T::ONE / (chx * chx),
                wy: T::ONE / (chy * chy),
            });
            if cx % 2 != 0 || cy % 2 != 0 || cx / 2 < min_cells(bcx) || cy / 2 < min_cells(bcy) {
                break;
            }
            cx /= 2;
            cy /= 2;
            chx *= two;
            chy *= two;
        }
        Ok(Self {
            levels,
            bcx,
            bcy,
            sweeps,
        })
    }

    /// Anzahl der Ebenen (1 = nur das feinste Gitter, direkt geglättet).
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Anzahl der Unbekannten auf dem feinsten Gitter.
    pub fn len(&self) -> usize {
        self.levels[0].nx * self.levels[0].ny
    }

    /// `true`, wenn keine Unbekannten existieren (nie der Fall nach erfolgreichem `new`).
    pub fn is_empty(&self) -> bool {
        false
    }

    fn residual(&self, l: &Level<T>, u: &[T], b: &[T], r: &mut [T]) {
        let two = T::from_f64(2.0);
        for j in 0..l.ny {
            for i in 0..l.nx {
                let k = j * l.nx + i;
                let (xl, _) = nb(u, k, i, l.nx, 1, self.bcx, false);
                let (xh, _) = nb(u, k, i, l.nx, 1, self.bcx, true);
                let (yl, _) = nb(u, k, j, l.ny, l.nx, self.bcy, false);
                let (yh, _) = nb(u, k, j, l.ny, l.nx, self.bcy, true);
                r[k] = b[k] - (l.wx * (two * u[k] - xl - xh) + l.wy * (two * u[k] - yl - yh));
            }
        }
    }

    fn gs_point(&self, l: &Level<T>, u: &mut [T], b: &[T], i: usize, j: usize) {
        let k = j * l.nx + i;
        let (mut diag, mut sum) = (T::ZERO, b[k]);
        for (pos, len, stride, bc, w) in [
            (i, l.nx, 1, self.bcx, l.wx),
            (j, l.ny, l.nx, self.bcy, l.wy),
        ] {
            for up in [false, true] {
                let (v, real) = nb(u, k, pos, len, stride, bc, up);
                if real {
                    diag += w;
                    sum += w * v;
                } else {
                    // Geisterwert c·u[k]: Neumann c = 1 (Beitrag 0), Dirichlet c = −1 (Beitrag 2w)
                    diag += if bc == Boundary::Neumann {
                        T::ZERO
                    } else {
                        w + w
                    };
                }
            }
        }
        u[k] = sum / diag;
    }

    fn smooth(&self, l: &Level<T>, u: &mut [T], b: &[T], forward: bool) {
        for _ in 0..self.sweeps {
            if forward {
                for j in 0..l.ny {
                    for i in 0..l.nx {
                        self.gs_point(l, u, b, i, j);
                    }
                }
            } else {
                for j in (0..l.ny).rev() {
                    for i in (0..l.nx).rev() {
                        self.gs_point(l, u, b, i, j);
                    }
                }
            }
        }
    }

    /// Restriktion: Mittel über 2×2 Zellen.
    fn restrict(fine: &Level<T>, r: &[T], out: &mut [T]) {
        let (cx, cy) = (fine.nx / 2, fine.ny / 2);
        let q = T::from_f64(0.25);
        for j in 0..cy {
            for i in 0..cx {
                let k = 2 * j * fine.nx + 2 * i;
                out[j * cx + i] = q * (r[k] + r[k + 1] + r[k + fine.nx] + r[k + fine.nx + 1]);
            }
        }
    }

    /// Wert einer Grobzelle in Nachbarschaft `(di, dj)` mit Randbehandlung des Grobgitters.
    fn coarse_at(&self, c: &[T], cx: usize, cy: usize, i: isize, j: isize) -> T {
        let wrap = |p: isize, n: usize, bc: Boundary| -> Option<(usize, T)> {
            if p >= 0 && (p as usize) < n {
                Some((p as usize, T::ONE))
            } else {
                match bc {
                    Boundary::Periodic => Some((p.rem_euclid(n as isize) as usize, T::ONE)),
                    Boundary::Neumann => Some((p.clamp(0, n as isize - 1) as usize, T::ONE)),
                    Boundary::Dirichlet => Some((p.clamp(0, n as isize - 1) as usize, -T::ONE)),
                }
            }
        };
        match (wrap(i, cx, self.bcx), wrap(j, cy, self.bcy)) {
            (Some((a, sa)), Some((b, sb))) => sa * sb * c[b * cx + a],
            _ => T::ZERO,
        }
    }

    /// Prolongation (bilinear) addiert auf `u`.
    fn prolong_add(&self, fine: &Level<T>, c: &[T], u: &mut [T]) {
        let (cx, cy) = (fine.nx / 2, fine.ny / 2);
        let (a, bq, cq) = (
            T::from_f64(9.0 / 16.0),
            T::from_f64(3.0 / 16.0),
            T::from_f64(1.0 / 16.0),
        );
        for j in 0..fine.ny {
            for i in 0..fine.nx {
                let (ci, cj) = ((i / 2) as isize, (j / 2) as isize);
                let di = if i % 2 == 0 { -1 } else { 1 };
                let dj = if j % 2 == 0 { -1 } else { 1 };
                let v = a * self.coarse_at(c, cx, cy, ci, cj)
                    + bq * self.coarse_at(c, cx, cy, ci + di, cj)
                    + bq * self.coarse_at(c, cx, cy, ci, cj + dj)
                    + cq * self.coarse_at(c, cx, cy, ci + di, cj + dj);
                u[j * fine.nx + i] += v;
            }
        }
    }

    fn cycle_level(&self, lev: usize, u: &mut [T], b: &mut [T]) {
        let l = &self.levels[lev];
        let null = has_nullspace(self.bcx, self.bcy);
        if lev + 1 == self.levels.len() {
            if null {
                project_mean_zero(b);
            }
            for _ in 0..20 {
                self.smooth(l, u, b, true);
                self.smooth(l, u, b, false);
            }
            if null {
                project_mean_zero(u);
            }
            return;
        }
        self.smooth(l, u, b, true);
        let mut r = vec![T::ZERO; u.len()];
        self.residual(l, u, b, &mut r);
        let nc = (l.nx / 2) * (l.ny / 2);
        let mut rc = vec![T::ZERO; nc];
        Self::restrict(l, &r, &mut rc);
        if null {
            project_mean_zero(&mut rc);
        }
        let mut ec = vec![T::ZERO; nc];
        self.cycle_level(lev + 1, &mut ec, &mut rc);
        self.prolong_add(l, &ec, u);
        self.smooth(l, u, b, false);
    }

    /// Ein symmetrischer V-Zyklus auf `u` für `A·u = b` (feinste Ebene). Bei Nullraum sollte `b`
    /// verträglich sein; die Korrektur bleibt mittelwertfrei.
    pub fn v_cycle(&self, u: &mut [T], b: &[T]) {
        assert!(
            u.len() == self.len() && b.len() == self.len(),
            "v_cycle: Länge"
        );
        let mut bb = b.to_vec();
        self.cycle_level(0, u, &mut bb);
    }
}

impl<T: Real> Preconditioner<T> for CellMultigrid2d<T> {
    /// `z = V-Zyklus(0, r)`; bei Nullraum wird `r` vorher auf Mittelwert 0 projiziert und `z` ebenso.
    fn apply(&self, r: &[T], z: &mut [T]) {
        z.fill(T::ZERO);
        let mut rr = r.to_vec();
        if has_nullspace(self.bcx, self.bcy) {
            project_mean_zero(&mut rr);
        }
        self.v_cycle(z, &rr);
        if has_nullspace(self.bcx, self.bcy) {
            project_mean_zero(z);
        }
    }
}
