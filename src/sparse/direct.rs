#![allow(clippy::needless_range_loop)]

//! Direkte Löser für dünn besetzte symmetrische Matrizen: LDLᵀ und Cholesky mit Umordnung (Reverse Cuthill-McKee).

use super::{CooMatrix, CsrMatrix};
use crate::Real;
use crate::linalg::LinalgError;
use alloc::rc::Rc;
use alloc::vec;
use alloc::vec::Vec;

/// Umordnung der Zeilen/Spalten vor der Faktorisierung (verringert das Auffüllen, „Fill-in“).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ordering {
    /// Keine Umordnung.
    Natural,
    /// Reverse Cuthill-McKee: verkleinert die Bandbreite; gut für gitterartige Matrizen.
    ReverseCuthillMcKee,
    /// Minimum Degree (exakter Grad auf dem Eliminationsgraphen): wenig Auffüllung auch bei unregelmäßigen Strukturen.
    MinimumDegree,
}

/// Reverse-Cuthill-McKee-Ordnung der (strukturell symmetrischen) Matrix `a`: `perm[neu] = alt`. Jede Zusammenhangskomponente
/// beginnt beim Knoten kleinsten Grades; die Nachbarn werden nach wachsendem Grad besucht, am Ende wird die Reihenfolge
/// umgekehrt.
pub fn rcm_ordering<T: Real>(a: &CsrMatrix<T>) -> Vec<usize> {
    let n = a.rows();
    let degree: Vec<usize> = (0..n)
        .map(|i| a.row(i).filter(|&(j, _)| j != i).count())
        .collect();
    let mut visited = vec![false; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let mut queue: Vec<usize> = Vec::new();
    let mut next_start = 0;
    while order.len() < n {
        // unbesuchten Knoten kleinsten Grades als Start
        let mut start = None;
        for v in next_start..n {
            if !visited[v] && start.is_none_or(|s: usize| degree[v] < degree[s]) {
                start = Some(v);
            }
        }
        let Some(s) = start else { break };
        while next_start < n && visited[next_start] {
            next_start += 1;
        }
        visited[s] = true;
        queue.clear();
        queue.push(s);
        let mut head = 0;
        while head < queue.len() {
            let v = queue[head];
            head += 1;
            order.push(v);
            let mut nbrs: Vec<usize> = a
                .row(v)
                .map(|(j, _)| j)
                .filter(|&j| j != v && !visited[j])
                .collect();
            nbrs.sort_by_key(|&j| degree[j]);
            for j in nbrs {
                visited[j] = true;
                queue.push(j);
            }
        }
    }
    order.reverse();
    order
}

/// Minimum-Degree-Ordnung der Matrix `a` (Muster von `A + Aᵀ`): in jedem Schritt wird der Knoten mit dem kleinsten Grad im
/// Eliminationsgraphen eliminiert und seine Nachbarn zu einer Clique verbunden. `perm[neu] = alt`. Aufwand etwa `O(n² + Fill)`,
/// gedacht für Matrizen bis zu einigen zehntausend Unbekannten.
pub fn minimum_degree_ordering<T: Real>(a: &CsrMatrix<T>) -> Vec<usize> {
    use alloc::collections::BTreeSet;
    let n = a.rows();
    let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for i in 0..n {
        for (j, _) in a.row(i) {
            if i != j {
                adj[i].insert(j);
                adj[j].insert(i);
            }
        }
    }
    let mut done = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for _ in 0..n {
        let mut best = usize::MAX;
        for v in 0..n {
            if !done[v] && (best == usize::MAX || adj[v].len() < adj[best].len()) {
                best = v;
            }
        }
        let nbrs: Vec<usize> = adj[best].iter().copied().collect();
        for &u in &nbrs {
            adj[u].remove(&best);
        }
        for (x, &u) in nbrs.iter().enumerate() {
            for &w in &nbrs[x + 1..] {
                adj[u].insert(w);
                adj[w].insert(u);
            }
        }
        adj[best].clear();
        done[best] = true;
        order.push(best);
    }
    order
}

/// Symmetrisch umgeordnete Matrix `C = P·A·Pᵀ` mit `C[i][j] = A[perm[i]][perm[j]]`.
pub fn permute_symmetric<T: Real>(a: &CsrMatrix<T>, perm: &[usize]) -> CsrMatrix<T> {
    let n = a.rows();
    let mut inv = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        inv[old] = new;
    }
    let mut coo = CooMatrix::with_capacity(n, n, a.nnz());
    for r in 0..n {
        for (c, v) in a.row(r) {
            // Indizes liegen in Reichweite, Werte sind endlich (Invarianten von CsrMatrix)
            let _ = coo.push(inv[r], inv[c], v);
        }
    }
    CsrMatrix::from_coo(&coo)
}

/// Eliminationsbaum (Liu) einer strukturell symmetrischen Matrix; `usize::MAX` für Wurzeln.
fn elimination_tree<T: Real>(c: &CsrMatrix<T>) -> Vec<usize> {
    let n = c.rows();
    let none = usize::MAX;
    let mut parent = vec![none; n];
    let mut ancestor = vec![none; n];
    for k in 0..n {
        for (j, _) in c.row(k) {
            let mut i = j;
            while i != none && i < k {
                let next = ancestor[i];
                ancestor[i] = k;
                if next == none {
                    parent[i] = k;
                }
                i = next;
            }
        }
    }
    parent
}

/// Muster der Zeile `k` von `L` (ohne Diagonale) in topologischer Reihenfolge (Nachkommen vor Vorfahren) nach `stack`.
fn ereach<T: Real>(
    c: &CsrMatrix<T>,
    k: usize,
    parent: &[usize],
    mark: &mut [bool],
    stack: &mut Vec<usize>,
    path: &mut Vec<usize>,
) {
    stack.clear();
    mark[k] = true;
    for (j, _) in c.row(k) {
        if j >= k {
            continue;
        }
        let mut i = j;
        path.clear();
        while !mark[i] {
            path.push(i);
            mark[i] = true;
            i = parent[i];
        }
        stack.extend(path.iter().rev());
    }
    stack.reverse();
    for &i in stack.iter() {
        mark[i] = false;
    }
    mark[k] = false;
}

/// Wertunabhängige Analyse für Sparse-LDLᵀ und Cholesky bei exakt gleichem CSR-Muster.
/// Speichert Umordnung, Eliminationsbaum-Erreichbarkeiten und vollständiges Füllmuster.
/// Die untere Hälfte des Eingangsmusters ist maßgeblich, auch nach Umordnung.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseLdlSymbolic {
    n: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    perm: Vec<usize>,
    lp: Vec<usize>,
    li: Vec<usize>,
    entries: Vec<Vec<(usize, usize)>>,
    operations: Vec<Vec<(usize, usize)>>,
}
impl SparseLdlSymbolic {
    /// Analysiert nur Struktur und Dimensionen; Zahlenwerte werden erst bei factorize geprüft.
    pub fn analyze<T: Real>(a: &CsrMatrix<T>, ordering: Ordering) -> Result<Self, LinalgError> {
        let n = a.rows();
        if a.cols() != n {
            return Err(LinalgError::DimensionMismatch);
        }
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            for (j, _) in a.row(i) {
                if j <= i {
                    let _ = coo.push(i, j, T::ONE);
                    if i != j {
                        let _ = coo.push(j, i, T::ONE);
                    }
                }
            }
        }
        let pattern = CsrMatrix::from_coo(&coo);
        let perm = match ordering {
            Ordering::Natural => (0..n).collect(),
            Ordering::ReverseCuthillMcKee => rcm_ordering(&pattern),
            Ordering::MinimumDegree => minimum_degree_ordering(&pattern),
        };
        let c = permute_symmetric(&pattern, &perm);
        let parent = elimination_tree(&c);
        let mut mark = vec![false; n];
        let (mut stack, mut path) = (Vec::new(), Vec::new());
        let mut reaches = vec![Vec::new(); n];
        let mut count = vec![0; n];
        for (k, reach) in reaches.iter_mut().enumerate() {
            ereach(&c, k, &parent, &mut mark, &mut stack, &mut path);
            reach.extend_from_slice(&stack);
            for &i in &stack {
                count[i] += 1;
            }
        }
        let mut lp = vec![0; n + 1];
        for i in 0..n {
            lp[i + 1] = lp[i] + count[i];
        }
        let mut li = vec![0; lp[n]];
        let mut operations = vec![Vec::new(); n];
        count.fill(0);
        for k in 0..n {
            for &i in &reaches[k] {
                let slot = lp[i] + count[i];
                count[i] += 1;
                li[slot] = k;
                operations[k].push((i, slot));
            }
        }
        let mut inv = vec![0; n];
        for (i, &old) in perm.iter().enumerate() {
            inv[old] = i;
        }
        let mut entries = vec![Vec::new(); n];
        for i in 0..n {
            for src in a.row_ptr()[i]..a.row_ptr()[i + 1] {
                let j = a.col_idx()[src];
                if j <= i {
                    let (r, c) = (inv[i].max(inv[j]), inv[i].min(inv[j]));
                    entries[r].push((c, src));
                }
            }
        }
        Ok(Self {
            n,
            row_ptr: a.row_ptr().to_vec(),
            col_idx: a.col_idx().to_vec(),
            perm,
            lp,
            li,
            entries,
            operations,
        })
    }
    /// Dimension.
    pub fn n(&self) -> usize {
        self.n
    }
    /// Vorhergesagte Einträge unterhalb der Diagonalen von L.
    pub fn nnz_l(&self) -> usize {
        self.li.len()
    }
    /// Umordnung, neuer Index → ursprünglicher Index.
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }
    fn validate<T: Real>(&self, a: &CsrMatrix<T>) -> Result<(), LinalgError> {
        if a.rows() != self.n || a.cols() != self.n {
            return Err(LinalgError::DimensionMismatch);
        }
        if a.row_ptr() != self.row_ptr || a.col_idx() != self.col_idx {
            return Err(LinalgError::InvalidInput);
        }
        if a.values().iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        Ok(())
    }
    /// Erzeugt numerische Faktoren und einmalig deren Arbeitspuffer.
    pub fn factorize<T: Real>(&self, a: &CsrMatrix<T>) -> Result<SparseLdl<T>, LinalgError> {
        let mut factor = SparseLdl {
            symbolic: Rc::new(self.clone()),
            lx: vec![T::ZERO; self.li.len()],
            d: vec![T::ZERO; self.n],
            y: vec![T::ZERO; self.n],
            valid: false,
        };
        factor.refactor(a)?;
        Ok(factor)
    }
    /// Cholesky-Faktorisierung desselben Musters mit Prüfung positiver Pivots.
    pub fn factorize_cholesky<T: Real>(
        &self,
        a: &CsrMatrix<T>,
    ) -> Result<SparseCholesky<T>, LinalgError> {
        let mut factor = SparseCholesky {
            ldl: SparseLdl {
                symbolic: Rc::new(self.clone()),
                lx: vec![T::ZERO; self.li.len()],
                d: vec![T::ZERO; self.n],
                y: vec![T::ZERO; self.n],
                valid: false,
            },
        };
        factor.refactor(a)?;
        Ok(factor)
    }
}

/// LDLᵀ-Zerlegung `P·A·Pᵀ = L·D·Lᵀ` einer symmetrischen dünn besetzten Matrix (`L` untere Einheitsdreiecksmatrix, `D` diagonal)
/// **ohne Pivotisierung**. Das funktioniert für symmetrisch positiv definite und für „quasi-definite“ Matrizen (zum Beispiel
/// KKT-Matrizen `[[H, Aᵀ], [A, −δI]]`); bei einer Null als Pivot meldet sie [`LinalgError::Singular`]. Von `A` wird nur die untere Hälfte
/// (`Spalte ≤ Zeile`) gelesen; die Symmetrie wird nicht geprüft.
///
/// ```
/// use azmath::sparse::{CooMatrix, CsrMatrix, Ordering, SparseLdl};
///
/// // Tridiagonal: 2 auf der Diagonale, −1 daneben
/// let n = 6;
/// let mut coo = CooMatrix::new(n, n);
/// for i in 0..n {
///     coo.push(i, i, 2.0f64).unwrap();
///     if i + 1 < n {
///         coo.push(i, i + 1, -1.0).unwrap();
///         coo.push(i + 1, i, -1.0).unwrap();
///     }
/// }
/// let ldl = SparseLdl::new(&CsrMatrix::from_coo(&coo), Ordering::Natural).unwrap();
/// let x = ldl.solve(&[1.0; 6]).unwrap();
/// assert!((x[2] - 6.0).abs() < 1e-12); // x_i = i(n+1−i)/2 für i = 3
/// assert_eq!(ldl.inertia(), (6, 0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SparseLdl<T> {
    symbolic: Rc<SparseLdlSymbolic>,
    lx: Vec<T>,
    d: Vec<T>,
    y: Vec<T>,
    valid: bool,
}
impl<T: Real> SparseLdl<T> {
    /// Analysiert und faktorisiert a; für Wiederholungen Analyse oder Faktor aufbewahren.
    pub fn new(a: &CsrMatrix<T>, ordering: Ordering) -> Result<Self, LinalgError> {
        SparseLdlSymbolic::analyze(a, ordering)?.factorize(a)
    }
    /// Referenz auf die wertunabhängige Analyse.
    pub fn symbolic(&self) -> &SparseLdlSymbolic {
        &self.symbolic
    }
    /// Ob die letzte numerische Faktorisierung erfolgreich war.
    pub fn is_valid(&self) -> bool {
        self.valid
    }
    /// Neue Zahlenwerte bei exakt gleichem CSR-Muster, ohne Allokationen.
    /// Struktur-/Eingabefehler erhalten den alten Faktor. Numerische Fehler invalidieren
    /// ihn bis zur nächsten erfolgreichen Refaktorisierung; solve meldet dann InvalidInput.
    pub fn refactor(&mut self, a: &CsrMatrix<T>) -> Result<(), LinalgError> {
        self.symbolic.validate(a)?;
        self.valid = false;
        self.y.fill(T::ZERO);
        self.d.fill(T::ZERO);
        self.lx.fill(T::ZERO);
        for k in 0..self.symbolic.n {
            for &(j, src) in &self.symbolic.entries[k] {
                self.y[j] = a.values()[src];
            }
            let mut dk = self.y[k];
            self.y[k] = T::ZERO;
            for &(i, slot) in &self.symbolic.operations[k] {
                let yi = self.y[i];
                self.y[i] = T::ZERO;
                for p in self.symbolic.lp[i]..slot {
                    self.y[self.symbolic.li[p]] -= self.lx[p] * yi;
                }
                let lki = yi / self.d[i];
                dk -= lki * yi;
                if !lki.is_finite() || !dk.is_finite() {
                    return Err(LinalgError::NotFinite);
                }
                self.lx[slot] = lki;
            }
            if !dk.is_finite() {
                return Err(LinalgError::NotFinite);
            }
            if dk == T::ZERO {
                return Err(LinalgError::Singular);
            }
            self.d[k] = dk;
        }
        self.valid = true;
        Ok(())
    }

    /// Größe der Matrix.
    pub fn n(&self) -> usize {
        self.symbolic.n
    }

    /// Anzahl der gespeicherten Einträge von `L` unterhalb der Diagonale (Maß für das Auffüllen).
    pub fn nnz_l(&self) -> usize {
        self.lx.len()
    }

    /// Die Diagonale von `D`.
    pub fn d(&self) -> &[T] {
        &self.d
    }

    /// Die verwendete Umordnung (`perm[neu] = alt`).
    pub fn permutation(&self) -> &[usize] {
        &self.symbolic.perm
    }

    /// Trägheit: Anzahl positiver und negativer Einträge von `D` (nach Sylvester die Zahl positiver und negativer
    /// Eigenwerte von `A`). Bei KKT-Matrizen zeigt sie, ob die Matrix die richtige Trägheit hat.
    pub fn inertia(&self) -> (usize, usize) {
        let pos = self.d.iter().filter(|v| **v > T::ZERO).count();
        (pos, self.symbolic.n - pos)
    }

    /// Löst `A·x = b`. Fehler bei falscher Länge von `b`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, LinalgError> {
        if !self.valid {
            return Err(LinalgError::InvalidInput);
        }
        if b.iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        if b.len() != self.symbolic.n {
            return Err(LinalgError::DimensionMismatch);
        }
        let n = self.symbolic.n;
        let mut z: Vec<T> = (0..n).map(|i| b[self.symbolic.perm[i]]).collect();
        // L·w = z (spaltenweise)
        for j in 0..n {
            let zj = z[j];
            for p in self.symbolic.lp[j]..self.symbolic.lp[j + 1] {
                z[self.symbolic.li[p]] -= self.lx[p] * zj;
            }
        }
        for j in 0..n {
            z[j] /= self.d[j];
        }
        // Lᵀ·v = w
        for j in (0..n).rev() {
            let mut s = z[j];
            for p in self.symbolic.lp[j]..self.symbolic.lp[j + 1] {
                s -= self.lx[p] * z[self.symbolic.li[p]];
            }
            z[j] = s;
        }
        let mut x = vec![T::ZERO; n];
        for i in 0..n {
            x[self.symbolic.perm[i]] = z[i];
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(LinalgError::NotFinite);
        }
        Ok(x)
    }
}

/// Cholesky-Zerlegung einer symmetrisch positiv definiten dünn besetzten Matrix (über [`SparseLdl`] mit Prüfung `D > 0`).
/// Für KKT-Matrizen mit negativen Pivots ist [`SparseLdl`] zu verwenden.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseCholesky<T> {
    ldl: SparseLdl<T>,
}

impl<T: Real> SparseCholesky<T> {
    /// Faktorisiert `a`. Fehler wie bei [`SparseLdl::new`], zusätzlich [`LinalgError::NotPositiveDefinite`], wenn ein Pivot `≤ 0` ist.
    pub fn new(a: &CsrMatrix<T>, ordering: Ordering) -> Result<Self, LinalgError> {
        SparseLdlSymbolic::analyze(a, ordering)?.factorize_cholesky(a)
    }
    /// Erneuert Zahlenwerte ohne neue Analyse oder Allokationen; verlangt positive Pivots.
    /// Bei numerischem Fehlschlag ist der Faktor ungültig; Strukturfehler erhalten ihn.
    pub fn refactor(&mut self, a: &CsrMatrix<T>) -> Result<(), LinalgError> {
        match self.ldl.refactor(a) {
            Err(LinalgError::Singular) => return Err(LinalgError::NotPositiveDefinite),
            Err(e) => return Err(e),
            Ok(()) => {}
        }
        if self.ldl.d.iter().any(|v| *v <= T::ZERO) {
            self.ldl.valid = false;
            return Err(LinalgError::NotPositiveDefinite);
        }
        Ok(())
    }
    /// Ob die Faktoren gültig sind.
    pub fn is_valid(&self) -> bool {
        self.ldl.is_valid()
    }
    /// Wertunabhängige Analyse.
    pub fn symbolic(&self) -> &SparseLdlSymbolic {
        self.ldl.symbolic()
    }

    /// Anzahl der gespeicherten Einträge von `L` unterhalb der Diagonale.
    pub fn nnz_l(&self) -> usize {
        self.ldl.nnz_l()
    }

    /// Logarithmus der Determinante, `Σ ln dᵢ`.
    pub fn log_determinant(&self) -> T {
        self.ldl.d().iter().fold(T::ZERO, |s, v| s + v.ln())
    }

    /// Löst `A·x = b`.
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, LinalgError> {
        self.ldl.solve(b)
    }
}
