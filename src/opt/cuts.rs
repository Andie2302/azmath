#![allow(clippy::needless_range_loop)]

use super::dual::LpWarmStart;
use super::milp::{MilpOptions, MilpSolution, MilpStatus, NodeSelection};
use super::{LpStatus, OptError, Relation, SimplexOptions};
use crate::Real;
use alloc::vec;
use alloc::vec::Vec;

/// Einstellungen für [`branch_and_cut`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutOptions<T> {
    /// Einstellungen des Branch-and-Bound-Teils (Toleranzen, Knotenlimit, Auswahl, Simplex).
    pub milp: MilpOptions<T>,
    /// Anzahl der Schnittrunden an der Wurzel (0 = reines Branch-and-Bound mit Warmstart).
    pub cut_rounds: usize,
    /// Höchstzahl der Gomory-Schnitte je Runde.
    pub max_cuts_per_round: usize,
}

impl<T: Real> Default for CutOptions<T> {
    /// 10 Runden mit höchstens 10 Schnitten, sonst wie [`MilpOptions::default`].
    fn default() -> Self {
        Self {
            milp: MilpOptions::default(),
            cut_rounds: 10,
            max_cuts_per_round: 10,
        }
    }
}

struct Node<T> {
    lp: LpWarmStart<T>,
    bound: T,
}

/// Gomory-Mischganzzahlschnitte (GMI) aus den Tableauzeilen der Basisvariablen mit gebrochenem Wert; liefert
/// `(Zeile über alle Spalten, rechte Seite)` für [`LpWarmStart::append_row`], sortiert nach Bruchnähe zu ½.
fn gomory_rows<T: Real>(
    lp: &LpWarmStart<T>,
    integer: &[bool],
    int_tol: T,
    limit: usize,
) -> Vec<(Vec<T>, T)> {
    let tab = &lp.tab;
    let cols = tab.cols;
    let w = cols + 1;
    let mut is_basic = vec![false; cols];
    for &b in &tab.basis {
        is_basic[b] = true;
    }
    let min_frac = T::from_f64(0.01).max(int_tol);
    let mut cand: Vec<(T, Vec<T>, T)> = Vec::new();
    for i in 0..tab.rows {
        let k = tab.basis[i];
        if k >= lp.n || !integer[k] {
            continue;
        }
        let beta = tab.t[i * w + cols];
        let f0 = beta - beta.floor();
        if f0 < min_frac || f0 > T::ONE - min_frac {
            continue;
        }
        let mut row = vec![T::ZERO; cols];
        let (mut gmax, mut gmin) = (T::ZERO, T::INFINITY);
        for j in 0..cols {
            if is_basic[j] || !lp.allowed[j] {
                continue;
            }
            let a = tab.t[i * w + j];
            let g = if j < lp.n && integer[j] {
                let fj = a - a.floor();
                if fj <= f0 {
                    fj / f0
                } else {
                    (T::ONE - fj) / (T::ONE - f0)
                }
            } else if a >= T::ZERO {
                a / f0
            } else {
                -a / (T::ONE - f0)
            };
            if g > T::EPSILON.sqrt() {
                row[j] = -g;
                gmax = gmax.max(g);
                gmin = gmin.min(g);
            }
        }
        // numerisch unsichere Schnitte überspringen
        if gmax == T::ZERO || gmax / gmin > T::from_f64(1.0e7) {
            continue;
        }
        cand.push(((f0 - T::from_f64(0.5)).abs(), row, -T::ONE));
    }
    cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(core::cmp::Ordering::Equal));
    cand.into_iter()
        .take(limit)
        .map(|(_, r, rhs)| (r, rhs))
        .collect()
}

fn fractional_branch<T: Real>(lp: &LpWarmStart<T>, integer: &[bool], tol: T) -> Option<(usize, T)> {
    let sol = lp.solution();
    let mut best: Option<(usize, T)> = None;
    let mut worst = tol;
    for (k, &is_int) in integer.iter().enumerate() {
        if is_int {
            let frac = (sol.x[k] - sol.x[k].round()).abs();
            if frac > worst {
                worst = frac;
                best = Some((k, sol.x[k]));
            }
        }
    }
    best
}

/// Gemischt-ganzzahliges lineares Programm `min cᵀx`, `A·x ≤ / = / ≥ b`, `x ≥ 0`, `xᵢ ∈ ℤ` für `integer[i]` durch
/// **Branch-and-Cut**: an der Wurzel werden Gomory-Mischganzzahlschnitte (global gültig) eingefügt, danach folgt
/// Branch-and-Bound, dessen Knoten mit dem Dual-Simplex-Verfahren aus dem Tableau des Elternknotens neu gelöst werden
/// (Warmstart statt Neustart).
///
/// Variablen sind nichtnegativ; Obergrenzen gehören als Zeilen in `A`. Für allgemeine Schranken: [`branch_and_bound`](super::branch_and_bound).
/// Jeder Knoten hält ein eigenes Tableau (Speicher `O(Knoten·m·(n+m))`); für sehr große Bäume `NodeSelection::DepthFirst` wählen.
///
/// Fehler: [`OptError::InvalidInput`], [`OptError::NotFinite`].
///
/// ```
/// use azmath::opt::{branch_and_cut, CutOptions, MilpStatus, Relation};
///
/// // max 5x + 4y  unter  6x + 4y ≤ 24,  x + 2y ≤ 6,  x, y ganzzahlig  (Optimum x = 4, y = 0 → 20)
/// let sol = branch_and_cut(
///     &[-5.0f64, -4.0], &[6.0, 4.0, 1.0, 2.0], &[24.0, 6.0], &[Relation::Le; 2],
///     &[true, true], CutOptions::default(),
/// ).unwrap();
/// assert_eq!(sol.status, MilpStatus::Optimal);
/// assert!((sol.objective + 20.0).abs() < 1e-9);
/// ```
pub fn branch_and_cut<T: Real>(
    c: &[T],
    a: &[T],
    b: &[T],
    relations: &[Relation],
    integer: &[bool],
    opts: CutOptions<T>,
) -> Result<MilpSolution<T>, OptError> {
    let n = c.len();
    if integer.len() != n {
        return Err(OptError::InvalidInput);
    }
    let lp_opts: SimplexOptions<T> = opts.milp.lp;
    let mut root = LpWarmStart::new(c, a, b, relations, lp_opts)?;
    let mut nodes = 0usize;
    let mut lp_iterations = root.iterations();
    let fail = |status, nodes, its| MilpSolution {
        status,
        x: vec![T::ZERO; n],
        objective: if status == MilpStatus::Unbounded {
            T::NEG_INFINITY
        } else {
            T::INFINITY
        },
        best_bound: if status == MilpStatus::Unbounded {
            T::NEG_INFINITY
        } else {
            T::INFINITY
        },
        nodes,
        lp_iterations: its,
    };
    match root.status() {
        LpStatus::Infeasible => return Ok(fail(MilpStatus::Infeasible, 1, lp_iterations)),
        LpStatus::Unbounded => return Ok(fail(MilpStatus::Unbounded, 1, lp_iterations)),
        LpStatus::IterationLimit => return Ok(fail(MilpStatus::NodeLimit, 1, lp_iterations)),
        LpStatus::Optimal => {}
    }
    // Schnittrunden an der Wurzel
    let mut last_obj = root.solution().objective;
    let mut stagnation = 0;
    for _ in 0..opts.cut_rounds {
        let rows = gomory_rows(&root, integer, opts.milp.int_tol, opts.max_cuts_per_round);
        if rows.is_empty() {
            break;
        }
        for (row, rhs) in rows {
            root.append_row(row, rhs);
        }
        let before = root.iterations();
        root.reoptimize();
        lp_iterations += root.iterations() - before;
        match root.status() {
            LpStatus::Optimal => {}
            LpStatus::Infeasible => return Ok(fail(MilpStatus::Infeasible, 1, lp_iterations)),
            _ => return Ok(fail(MilpStatus::NodeLimit, 1, lp_iterations)),
        }
        let obj = root.solution().objective;
        if obj - last_obj < T::from_f64(1.0e-9) * T::ONE.max(obj.abs()) {
            stagnation += 1;
            if stagnation >= 2 {
                break;
            }
        } else {
            stagnation = 0;
        }
        last_obj = obj;
    }

    let mut incumbent: Option<Vec<T>> = None;
    let mut inc_obj = T::INFINITY;
    let mut open = vec![Node {
        bound: root.solution().objective,
        lp: root,
    }];
    let mut limit_hit = false;
    while !open.is_empty() {
        let idx = match opts.milp.node_selection {
            NodeSelection::DepthFirst => open.len() - 1,
            NodeSelection::BestFirst => {
                let mut best = 0;
                for (i, nd) in open.iter().enumerate() {
                    if nd.bound < open[best].bound {
                        best = i;
                    }
                }
                best
            }
        };
        let node = open.swap_remove(idx);
        let prune = inc_obj - opts.milp.gap_tol * T::ONE.max(inc_obj.abs());
        if incumbent.is_some() && node.bound >= prune {
            continue;
        }
        if nodes >= opts.milp.max_nodes {
            open.push(node);
            limit_hit = true;
            break;
        }
        nodes += 1;
        let lp = node.lp;
        let sol = lp.solution();
        if incumbent.is_some() && sol.objective >= prune {
            continue;
        }
        match fractional_branch(&lp, integer, opts.milp.int_tol) {
            None => {
                let mut x = sol.x.clone();
                for k in 0..n {
                    if integer[k] {
                        x[k] = x[k].round();
                    }
                }
                let obj = c.iter().zip(&x).fold(T::ZERO, |s, (ci, xi)| s + *ci * *xi);
                if obj < inc_obj {
                    inc_obj = obj;
                    incumbent = Some(x);
                }
            }
            Some((k, value)) => {
                let mut unit = vec![T::ZERO; n];
                unit[k] = T::ONE;
                for (rel, rhs) in [(Relation::Le, value.floor()), (Relation::Ge, value.ceil())] {
                    let mut child = lp.clone();
                    let before = child.iterations();
                    let st = child.add_constraint(&unit, rel, rhs)?;
                    lp_iterations += child.iterations() - before;
                    match st {
                        LpStatus::Optimal => {
                            let bound = child.solution().objective;
                            open.push(Node { lp: child, bound });
                        }
                        LpStatus::Infeasible => {}
                        _ => {
                            limit_hit = true;
                        }
                    }
                }
            }
        }
    }
    let lowest_open = open.iter().fold(T::INFINITY, |m, nd| m.min(nd.bound));
    Ok(match incumbent {
        Some(x) => MilpSolution {
            status: if limit_hit {
                MilpStatus::NodeLimit
            } else {
                MilpStatus::Optimal
            },
            x,
            objective: inc_obj,
            best_bound: if limit_hit {
                lowest_open.min(inc_obj)
            } else {
                inc_obj
            },
            nodes,
            lp_iterations,
        },
        None => MilpSolution {
            status: if limit_hit {
                MilpStatus::NodeLimit
            } else {
                MilpStatus::Infeasible
            },
            x: vec![T::ZERO; n],
            objective: T::INFINITY,
            best_bound: if limit_hit { lowest_open } else { T::INFINITY },
            nodes,
            lp_iterations,
        },
    })
}
