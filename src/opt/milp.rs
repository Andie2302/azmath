use super::{LpStatus, OptError, Relation, SimplexOptions, simplex_bounded};
use crate::Real;
use alloc::vec;
use alloc::vec::Vec;

/// Reihenfolge, in der offene Knoten abgearbeitet werden.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeSelection {
    /// Knoten mit dem kleinsten LP-Wert zuerst: beweist die Optimalität mit den wenigsten Knoten.
    BestFirst,
    /// Zuletzt erzeugter Knoten zuerst: findet schnell eine erste zulässige Lösung, braucht wenig Speicher.
    DepthFirst,
}

/// Einstellungen für [`branch_and_bound`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MilpOptions<T> {
    /// Eine Variable gilt als ganzzahlig, wenn sie höchstens so weit von der nächsten ganzen Zahl entfernt ist.
    pub int_tol: T,
    /// Höchstzahl der untersuchten Knoten (LP-Relaxationen).
    pub max_nodes: usize,
    /// Knotenauswahl.
    pub node_selection: NodeSelection,
    /// Knoten, deren LP-Wert um weniger als `gap_tol·max(1, |Incumbent|)` unter dem Incumbent liegt, werden verworfen.
    pub gap_tol: T,
    /// Einstellungen des Simplex-Verfahrens für die Relaxationen.
    pub lp: SimplexOptions<T>,
}

impl<T: Real> Default for MilpOptions<T> {
    /// `int_tol = 1e-6`, 100 000 Knoten, `BestFirst`, `gap_tol = 1e-9`.
    fn default() -> Self {
        Self {
            int_tol: T::from_f64(1e-6),
            max_nodes: 100_000,
            node_selection: NodeSelection::BestFirst,
            gap_tol: T::from_f64(1e-9),
            lp: SimplexOptions::default(),
        }
    }
}

/// Ausgang von [`branch_and_bound`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MilpStatus {
    /// Optimale ganzzahlige Lösung gefunden und bewiesen.
    Optimal,
    /// Es gibt keine zulässige ganzzahlige Lösung.
    Infeasible,
    /// Die LP-Relaxation ist unbeschränkt (das gemischt-ganzzahlige Problem dann in der Regel auch).
    Unbounded,
    /// Knotenlimit (oder Iterationslimit einer Relaxation) erreicht; `x` enthält die beste bisher gefundene Lösung,
    /// falls es eine gibt (`objective` ist dann endlich, sonst `∞`).
    NodeLimit,
}

/// Ergebnis von [`branch_and_bound`].
#[derive(Debug, Clone, PartialEq)]
pub struct MilpSolution<T> {
    /// Ausgang.
    pub status: MilpStatus,
    /// Beste gefundene Lösung (ganzzahlige Variablen exakt gerundet).
    pub x: Vec<T>,
    /// Zielfunktionswert der besten Lösung (`∞`, wenn keine gefunden wurde).
    pub objective: T,
    /// Untere Schranke für den optimalen Wert (bei `Optimal` gleich `objective`).
    pub best_bound: T,
    /// Zahl der untersuchten Knoten.
    pub nodes: usize,
    /// Summe der Simplex-Pivotschritte.
    pub lp_iterations: usize,
}

struct Node<T> {
    bounds: Vec<(T, T)>,
    /// LP-Wert des Elternknotens: untere Schranke für alles unterhalb dieses Knotens.
    bound: T,
}

/// Löst das gemischt-ganzzahlige lineare Programm
///
/// ```text
/// min cᵀx   unter   A·x ≤ / = / ≥ b,   lᵢ ≤ xᵢ ≤ uᵢ,   xᵢ ∈ ℤ  (integer[i] = true)
/// ```
///
/// mit **Branch-and-Bound**: Die LP-Relaxation ([`simplex_bounded`]) liefert eine untere Schranke; ist eine ganzzahlige Variable
/// bruchzahlig, wird auf der am stärksten bruchzahligen Variable in `xᵢ ≤ ⌊x̂ᵢ⌋` und `xᵢ ≥ ⌈x̂ᵢ⌉` verzweigt. Knoten, deren Schranke
/// nicht unter dem besten bekannten Wert liegt, werden verworfen. Binärvariablen: Schranken `(0, 1)` und `integer = true`.
///
/// **Grenzen:** Jeder Knoten löst die Relaxation neu (kein Warmstart); keine Schnittebenen und keine Heuristiken – für
/// größere Probleme ist das langsam. Aufwand wächst im schlechtesten Fall exponentiell.
///
/// Parameter wie bei [`simplex_bounded`], dazu `integer` (Länge `n`). Fehler: [`OptError::InvalidInput`], [`OptError::NotFinite`].
///
/// ```
/// use azmath::opt::{branch_and_bound, MilpOptions, MilpStatus, Relation};
///
/// // maximiere 5x + 4y  unter  6x + 4y ≤ 24,  x + 2y ≤ 6,  x, y ganzzahlig ≥ 0   (LP: 21, ganzzahlig: 20)
/// let sol = branch_and_bound(
///     &[-5.0f64, -4.0], &[6.0, 4.0, 1.0, 2.0], &[24.0, 6.0], &[Relation::Le; 2],
///     &[(0.0, f64::INFINITY); 2], &[true, true], MilpOptions::default(),
/// ).unwrap();
/// assert_eq!(sol.status, MilpStatus::Optimal);
/// assert!((sol.objective + 20.0).abs() < 1e-9);
/// ```
pub fn branch_and_bound<T: Real>(
    c: &[T],
    a: &[T],
    b: &[T],
    relations: &[Relation],
    bounds: &[(T, T)],
    integer: &[bool],
    opts: MilpOptions<T>,
) -> Result<MilpSolution<T>, OptError> {
    let n = c.len();
    if integer.len() != n || bounds.len() != n {
        return Err(OptError::InvalidInput);
    }
    // Ganzzahlige Variablen: Schranken auf ganze Zahlen verschärfen
    let mut root_bounds = bounds.to_vec();
    for (k, bd) in root_bounds.iter_mut().enumerate() {
        if integer[k] {
            if !bd.0.is_infinite() {
                bd.0 = bd.0.ceil();
            }
            if !bd.1.is_infinite() {
                bd.1 = bd.1.floor();
            }
        }
    }
    let mut incumbent: Option<Vec<T>> = None;
    let mut inc_obj = T::INFINITY;
    let mut open = vec![Node {
        bounds: root_bounds,
        bound: T::NEG_INFINITY,
    }];
    let (mut nodes, mut lp_iterations) = (0usize, 0usize);
    let mut limit_hit = false;
    let mut root = true;

    while !open.is_empty() {
        let idx = match opts.node_selection {
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
        let prune_level = inc_obj - opts.gap_tol * T::ONE.max(inc_obj.abs());
        if incumbent.is_some() && node.bound >= prune_level {
            continue;
        }
        if nodes >= opts.max_nodes {
            open.push(node);
            limit_hit = true;
            break;
        }
        nodes += 1;
        let lp = simplex_bounded(c, a, b, relations, &node.bounds, opts.lp)?;
        lp_iterations += lp.iterations;
        match lp.status {
            LpStatus::Infeasible => {
                if root {
                    return Ok(MilpSolution {
                        status: MilpStatus::Infeasible,
                        x: vec![T::ZERO; n],
                        objective: T::INFINITY,
                        best_bound: T::INFINITY,
                        nodes,
                        lp_iterations,
                    });
                }
                continue;
            }
            LpStatus::Unbounded => {
                return Ok(MilpSolution {
                    status: MilpStatus::Unbounded,
                    x: vec![T::ZERO; n],
                    objective: T::NEG_INFINITY,
                    best_bound: T::NEG_INFINITY,
                    nodes,
                    lp_iterations,
                });
            }
            LpStatus::IterationLimit => {
                open.push(node);
                limit_hit = true;
                break;
            }
            LpStatus::Optimal => {}
        }
        root = false;
        if incumbent.is_some() && lp.objective >= prune_level {
            continue;
        }
        // am stärksten bruchzahlige ganzzahlige Variable suchen
        let mut branch: Option<(usize, T)> = None;
        let mut worst = opts.int_tol;
        for k in 0..n {
            if integer[k] {
                let frac = (lp.x[k] - lp.x[k].round()).abs();
                if frac > worst {
                    worst = frac;
                    branch = Some((k, lp.x[k]));
                }
            }
        }
        match branch {
            None => {
                // ganzzahlig: runden und als neuen Incumbent übernehmen
                let mut x = lp.x.clone();
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
                let mut left = node.bounds.clone();
                left[k].1 = value.floor();
                let mut right = node.bounds;
                right[k].0 = value.ceil();
                open.push(Node {
                    bounds: right,
                    bound: lp.objective,
                });
                open.push(Node {
                    bounds: left,
                    bound: lp.objective,
                });
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
