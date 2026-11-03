# Roadmap und ToDo-Liste

Dieses Dokument hält fest, was **fertig** ist, was **noch fehlt**, was wovon **abhängt** und wie die Bibliothek
langfristig in **Modulen** gegliedert wird. Es ist eine lebende Liste: Ideen kommen jederzeit dazu.

Legende: ✅ fertig · ⚠️ teilweise · ⬜ offen · 🔶 Entscheidung nötig
Aufwand (grob, relativ zu den bisherigen Schritten): **S** = klein (wie die Nullstellensuche), **M** = mittel (wie
GMRES oder Interpolation), **L** = groß (wie ODE oder Multigrid), **XL** = Projekt für sich.

---

## 1. Stand heute (Version 0.8.0, ein Crate mit Modulen)

| Bereich | Fertig |
|---|---|
| Grundlage | `Real` für `f32`/`f64` mit den Backends `std`, `libm`, `micromath`; `no_std`, optional `alloc` |
| Lineare Algebra, direkt | LU (Pivot), Gauß-Elimination (`gauss_solve`, `rank`, `row_echelon`), Cholesky, QR (Householder), Dreieckslöser, Tridiagonal (Thomas, zyklisch), SVD, symmetrische Eigenwerte (Jacobi) |
| Lineare Algebra, iterativ | CG, BiCGSTAB, GMRES(m), Jacobi-Vorkonditionierer, Potenzmethode, inverse Iteration |
| Nullstellen (1D) | Bisektion, Brent, Newton, Sekante, Regula Falsi, Illinois; Polynome: Aberth-Ehrlich, Durand-Kerner, Jenkins-Traub, Begleitmatrix (komplex) |
| Nichtlineare Systeme (Vertrauensbereich) | Powell-Dogleg (Newton-/Cauchy-Schritt), robust bei schlechtem Startwert und singulärer Jacobi-Matrix |
| Nichtlineare Systeme (quasi-Newton) | Broyden (inverse Rang-1-Updates, Jacobi-Matrix nur am Start) |
| Nichtsymmetrische Eigenwerte | `Matrix::eigenvalues` (Balancieren, Hessenberg, QR mit Doppelshift), `hessenberg_form`, Polynom-Nullstellen über die Begleitmatrix |
| Ganzzahlige Programme | `opt::branch_and_bound`: MILP über Simplex-Relaxationen, Best-First oder Depth-First, Binär-/Ganzzahlvariablen (hinter `alloc`) |
| Nichtlineare Programme (Innere Punkte) | `opt::interior_point_nlp`: Schlupfvariablen, Barriere mit monoton fallendem μ, exakte Hessematrix (auch per `hessian_ad`), Trägheitskorrektur (hinter `alloc`) |
| Nichtlineare Programme | `opt::sqp`: SQP mit gedämpftem BFGS, ℓ₁-Merit-Funktion und Armijo-Backtracking; Ableitungen per Closure oder `jacobian_ad` |
| Quadratische Programme | `opt::interior_point_qp` (konvexes QP mit Gleichungen und Ungleichungen), `interior_point_lp` (Mehrotra-Prädiktor-Korrektor, hinter `alloc`) |
| Lineare Programme | `opt::simplex`: Zwei-Phasen-Simplex mit `≤`, `=`, `≥`, Bland-Schutz gegen Zyklen (hinter `alloc`) |
| Minimierung | BFGS, L-BFGS (Zweischleifen-Rekursion), starke Wolfe-Schrittweitensuche |
| Nichtlinear (mehrdimensional) | Newton-System mit Schrittweitensteuerung (gedämpft), Gauß-Newton, Levenberg-Marquardt |
| Integration | Trapez, Simpson, Gauß-Legendre, adaptives Gauß-Kronrod, Messdaten |
| Interpolation | linear, kubische Splines (4 Randbedingungen), PCHIP, Lagrange |
| ODE | Euler, RK4, Dormand-Prince (adaptiv), implizites Euler, implizite Trapezregel; Adams-Bashforth-Moulton, BDF 1–5, Radau IIA (Ordnung 5), ROS2, Hermite-Ausgabe + Ereignissuche (RK4) |
| Orthonormalisierung | modifiziertes Gram-Schmidt (optional mit Reorthogonalisierung), `thin_q` aus Householder-QR, Polarzerlegung `A = U·P`, `nearest_orthogonal`/`nearest_rotation`, `orthogonality_error`; fest und dynamisch |
| Äquilibrierung | Ruiz (∞-Norm) und Sinkhorn-Knopp, Zweierpotenzen als exakte Faktoren, `scale_rhs`/`recover_solution`; fest und dynamisch |
| Matrixnormen | Frobenius (überlaufsicher), 1, ∞, 2-Norm über SVD; fest und dynamisch |
| Geometrie | `Quaternion<T>` (Hamilton-Produkt, `rotate`, `slerp`/`nlerp`, `exp`/`ln`, Matrix-Umrechnung nach Shepperd), `exp_so3`/`log_so3`, Rodrigues, Euler Z-Y-X mit Gimbal-Lock-Erkennung |
| Komplexe Zahlen | `Complex<T>`: Arithmetik (Smith-Division), Betrag/Argument, `sqrt`, `exp`, `ln`, `powi`/`powf`, `sin`/`cos`/`sinh`/`cosh` |
| Sparse | `CooMatrix` (Assemblierung, Duplikate summiert), `CsrMatrix` (Operator für CG/GMRES/BiCGSTAB, Transponierte, Summe, Skalierung), `CsrView` ohne Allokator |
| Autodiff | `Jet<T, N>` (Vorwärtsmodus, Stack), `VectorFn`, Jacobi-Adapter mit Strip-Mining, Hessematrix per geschachtelten Jets |
| PDE | geometrisches Multigrid für Poisson (1D/2D), als Löser und als CG-Vorkonditionierer |

---

## 2. Gesamtliste (deine Brainstorming-Liste plus Ergänzungen)

### A. Grundbausteine
| | Eintrag | Aufwand | Hängt ab von | Ermöglicht |
|---|---|---|---|---|
| ✅ | **`Complex<T>`** (komplexe Zahlen, `azmath::complex`) | S–M | `Real` | Polynom-Nullstellen, nichtsymmetrische Eigenwerte, FFT |
| ✅ | **Automatisches Differenzieren** (`azmath::autodiff`; `Jet<T,N>`, `VectorFn`, `jacobian_ad`, `hessian_ad`; `gradient_ad` noch offen): `Jet<T, N>` = Wert plus Gradient `[T; N]` auf dem Stack (Vorwärtsmodus, ohne Heap), `Dual<T>` als Sonderfall `N = 1`, geschachtelt für Hessians (Entwurf siehe Abschnitt 7) | M–L | `Real` | exakte Jacobi-/Gradientenmatrizen für `newton_system` und `levenberg_marquardt`, SQP/IPOPT, IAPWS-Ableitungen, Taylor-ODE |
| ✅ | ~~**Matrixnormen** (Frobenius, 1, ∞, 2-Norm über SVD)~~ ✅ und ~~**Äquilibrierung**~~ ✅ (Zeilen-/Spaltenskalierung nach Ruiz bzw. Sinkhorn) | S–M | `linalg`, SVD ✅ | bessere Konditionierung, Fehlerabschätzung |
| ✅ | **Orthonormalisierung** von Spalten (modifiziertes Gram-Schmidt, Householder-QR ✅) und Polarzerlegung `R = U·Vᵀ` über SVD | S | QR ✅, SVD ✅ | stabile Rotationsmatrizen |
| ⬜ | `Real` ergänzen (`copysign`, `from_usize`, `fma`-Varianten …) | S | – | bei Bedarf |
| ⚠️ | Sparse-Formate (COO ✅, CSR ✅, `CsrView` ohne Allokator ✅), Bandmatrizen ✅ | M | `alloc` | große Systeme, FEM, Innere-Punkte |

### B. Lineare Algebra – Lücken
| | Eintrag | Aufwand | Hängt ab von |
|---|---|---|---|
| ✅ | **Gauß-Elimination** (`gauss_solve`, `rank`, `row_echelon`) | S | LU ✅ |
| ✅ | Banded-LU/-Cholesky (`BandLu`, `BandCholesky`) | M | Bandformat |
| ✅ | Links-Vorkonditionierer (`pgmres_left`), ILU(0), IC(0), SSOR (ungetestet) | M | Sparse |
| ⬜ | Iterative Verfeinerung, Konditionsschätzer, QR mit Spaltenpivot, LDLᵀ | S–M | – |
| ⚠️ | Eigenwerte **nichtsymmetrisch**: Balancieren, Hessenberg, QR mit Doppelshift ✅ (nur Eigenwerte); Schur-Vektoren, Eigenvektoren, verallgemeinert ⬜ | L | `Complex` |
| ⚠️ | Lanczos ✅ (Ritz-Werte), Arnoldi-Zerlegung ✅ (ungetestet); Ritz-Werte aus Arnoldi folgen mit dem nichtsymmetrischen Eigenlöser | M | Sparse/`LinearOperator` ✅ |
| ⬜ | Algebraisches Multigrid (AMG) | XL | Sparse |

### C. Nullstellen und Nichtlineares
| | Eintrag | Aufwand | Hängt ab von |
|---|---|---|---|
| ✅ | **Regula Falsi** (+ Illinois-Variante) | S | – |
| ✅ | **Quasi-Newton: BFGS**, **L-BFGS**, **Broyden** (Jacobi-Update) | M | Linesearch ✅, optional Autodiff |
| ⚠️ | **Gedämpftes Newton** – grundlegend vorhanden (Schrittweiten-Halbierung); starke Wolfe-Linesearch ✅ (`line_search_wolfe`, für Minimierung); Trust-Region ✅ (`dogleg`, Powell, für Gleichungssysteme); fehlen: Trust-Region für Minimierung, Jacobi-Wiederverwendung | M | Autodiff (für exakte Jacobi) |
| ⚠️ | **Polynom-Nullstellen**: Durand-Kerner ✅, Aberth-Ehrlich ✅, Begleitmatrix ✅, Jenkins-Traub ✅ (vereinfachte CPOLY-Fassung) | M–L | `Complex`; Begleitmatrix zusätzlich nichtsymmetrische Eigenwerte |
| ⬜ | Minimierer ohne Ableitung (Golden Section, Nelder-Mead) | S–M | – |

### D. Differentialgleichungen
| | Eintrag | Aufwand | Hängt ab von |
|---|---|---|---|
| ✅ | **Adams-Bashforth-Moulton** (Ordnung 1–4, PECE, feste Schrittweite) | M | RK-Anlaufrechnung ✅ |
| ⚠️ | **BDF** Ordnung 1–5 mit Radau-Anlauf, feste Schrittweite ✅; variable Ordnung/Schrittweite ⬜ | L | Newton ✅ (ode → optimize), Autodiff nützlich |
| ⚠️ | **Radau IIA** (3 Stufen, Ordnung 5, feste Schrittweite) ✅; Schrittweitensteuerung ⬜ | L | Newton, lineare Systeme mit Blockstruktur |
| ⚠️ | **Rosenbrock**: ROS2 (Ordnung 2) ✅; RODAS (Ordnung 4) ⬜ | L | Jacobi (Autodiff), LU |
| ⚠️ | Hermite-Dichte Ausgabe und Ereignissuche für RK4 ✅ (`rk4_event`); für `rk45` ⬜ | M | `rk45` ✅, Nullstellen ✅ |
| ⬜ | Taylor-Reihen-Verfahren hoher Ordnung | M–L | `Jet` |
| ⬜ | DAE-Löser, Rand­wertprobleme (Schießverfahren) | L | BDF/Radau, Newton |

### E. PDE und Strömung
| | Eintrag | Aufwand | Hängt ab von |
|---|---|---|---|
| ⚠️ | **FDM** – bisher nur der 5-Punkte-Stern für Poisson; fehlt: Gitter-/Stencil-Gerüst, Zeitschritte (Wärmeleitung, Wellen, Advektion), Randbedingungen | L | Multigrid ✅, ODE ✅, `alloc` |
| ⬜ | Multigrid erweitern: **Neumann/periodische Ränder**, variable Koeffizienten, anisotrop, 3D | M | Multigrid ✅ |
| ⬜ | **FEM** (1D/2D, Dreiecke, Assemblierung, Dirichlet/Neumann) | XL | Sparse, Cholesky/CG ✅, `alloc` |
| ⬜ | **FVM** (Erhaltungsgleichungen, Flussfunktionen, Riemann-Löser) | XL | Gitter, ODE-Zeitschritte |
| ⬜ | **SPH** (netzfrei: Nachbarsuche, Kernel, Zeitintegration) | XL | `alloc`, Geometrie |
| ⬜ | **Navier-Stokes** (inkompressibel 2D, Projektionsverfahren, MAC-Gitter, Lid-driven Cavity als Testfall) | XL | FDM-Gerüst, **Multigrid mit Neumann-Rand** (Druck), ODE-Zeitschritte |

### F. Optimierung
| | Eintrag | Aufwand | Hängt ab von |
|---|---|---|---|
| ⚠️ | **LP: Simplex** (Zwei-Phasen, dichtes Tableau, Bland-Schutz) ✅; Variablenschranken ✅ (`simplex_bounded`, über Umformung); Dual-Simplex, Warmstart ⬜ | M–L | LU ✅ |
| ✅ | **LP: Innere-Punkte (Primal-Dual)** (`interior_point_lp`, ohne zertifizierte Status) | L | Cholesky/LDLᵀ, Sparse |
| ✅ | **QP** konvex, dicht (`interior_point_qp`, Mehrotra) – Baustein für SQP | L | Simplex/Innere-Punkte, KKT-Löser |
| ⚠️ | **NLP: SQP** ✅ (BFGS, ℓ₁-Merit, QP über Innere Punkte; ohne Wiederherstellungsphase und Maratos-Korrektur) | XL | QP, Linesearch, **Autodiff (Hessians)**, BFGS |
| ⚠️ | **NLP: Innere-Punkte** ✅ (`interior_point_nlp`, exakte Hessematrix, ℓ₁-Merit statt Filter, ohne Wiederherstellungsphase) | XL | Sparse LDLᵀ, Filter-Linesearch, Autodiff |
| ⚠️ | **MILP: Branch-and-Bound** ✅ (`branch_and_bound`, ohne Warmstart); Branch-and-Cut ⬜ | L–XL | Simplex (Warmstart), `alloc` |

### G. Geometrie und Rotationen
| | Eintrag | Aufwand | Hängt ab von |
|---|---|---|---|
| ✅ | `Vec3`, `Mat3` (`Vector<T,3>`, `Matrix<T,3,3>`), **Rotationsmatrizen (SO(3))**, **Quaternionen** (Einheitsquaternion, SLERP/NLERP), Achse-Winkel, Rodrigues, Exponential-/Logarithmus-Abbildung | M | `Real`, kleine feste Matrizen ✅ |
| ✅ | **Euler-Winkel-Umrechnung mit Gimbal-Lock-Behandlung** (Konvention Z-Y-X) (die Winkel selbst vermeiden; nur Ein-/Ausgabe) | S | Quaternion |
| ⬜ | Dual-Quaternionen, SE(3), Transformationsketten | M | Quaternion |
| ✅ | Re-Orthonormalisierung von Rotationsmatrizen (`Matrix::nearest_rotation`, `nearest_orthogonal`) | S | Polarzerlegung (SVD ✅) |

*Hinweis zum Gimbal Lock:* Er ist kein Rechenfehler, sondern eine Eigenschaft der Euler-Winkel-Darstellung (bei
Neigung ±90° geht ein Freiheitsgrad verloren). Die Abhilfe ist, intern mit **Quaternionen oder Rotationsmatrizen** zu
rechnen und Euler-Winkel nur für Ein-/Ausgabe zu verwenden.

### H. Fachgebiet Thermodynamik
| | Eintrag | Aufwand | Hängt ab von |
|---|---|---|---|
| ⬜ | **IAPWS-IF97** (Wasser/Dampf, Gebiete 1–5, Rückwärtsgleichungen) | L | Newton ✅; Prüfwerte der IAPWS-Tabellen als Tests |
| ⬜ | **IAPWS-95** (Helmholtz-Energie `φ(δ, τ)`; alle Größen folgen aus ihren Ableitungen) | L | **Autodiff (`Jet`)** – `φ` einmal programmieren, alle Eigenschaften automatisch; Newton/Brent ✅ |
| ⬜ | Sättigungslinie (Maxwell-Konstruktion), Zustandsgrößen aus (p,h), (p,s) usw. | M | Newton ✅, Autodiff |

### I. SAT / SMT
| | Eintrag | Aufwand | Hängt ab von |
|---|---|---|---|
| ⬜ | **DPLL**, dann **CDCL** (Konfliktlernen, Watched Literals, Restarts) | L | – (benötigt kein `Real`) |
| ⬜ | **SMT / Simplex-Kombination (Z3-artig)**: Theorielöser für lineare Arithmetik | XL | CDCL, Simplex über **exakten Brüchen** (`Rational`) |

---

## 3. Abhängigkeitsgraph

```
                     Real  (✅)
        ┌──────────┬─┴──────────┬───────────────┐
     Complex     Dual/Jet     linalg (✅)      roots (✅), integrate (✅)
        │        (Autodiff)   ┌──┴───────────┬────────────┐
        │            │        │              │            │
   Polynom-      exakte    Sparse/Band    eigen (✅)   interp (✅)
   Nullstellen   Jacobi/   (CSR)             │
   Eig. nichtsym.  Hessian    │         Rotationen
        │            │        │         (Polar via SVD)
        │            ▼        ▼
        │        optimize (✅ Newton/GN/LM) ──► BFGS/Broyden ──► QP ──► SQP / IPOPT
        │            │                                  ▲        ▲
        │            ▼                                  │        │
        │        ode (✅ RK, DP5, impl. Euler) ──► BDF / Radau / Rodas      LP (Simplex, IPM) ──► MILP
        │            │
        │            ▼
        │        multigrid (✅ Poisson) ──► Neumann/variable Koeff. ──► FDM-Gerüst ──► Navier-Stokes
        │                                                  Sparse ──► FEM, FVM
        └─► IAPWS (Newton ✅ + Jet)                         SPH (eigenständig, alloc)

   SAT/SMT:  DPLL ──► CDCL ──► SMT (Simplex über Brüche)      (unabhängig von allem oben)
```

### Die wichtigsten Abhängigkeiten in Worten

1. **Autodiff ist der zentrale Hebel.** BFGS, SQP, IPOPT, BDF/Rodas (Jacobi), Newton und die IAPWS-Eigenschaften
   brauchen alle Ableitungen. Mit `Jet` bekommen sie diese exakt statt über Differenzen (`_fd`-Varianten, Genauigkeit ~`√ε`).
   Der größte Gewinn liegt bei **zweiten Ableitungen** (Hessematrix) und bei Verfahren, die Ableitungen als Ergebnis brauchen; bei
   gut konditionierten Problemen ändert eine exakte Jacobi-Matrix die Iterationszahl kaum (gemessen in
   [Entwurf 0001](docs/design/0001-autodiff-und-arbeitsspeicher.md)).
2. **Ein API-Entwurf muss vorher entschieden werden (🔶).** Unsere Löser nehmen Closures `FnMut(&[T], &mut [T])`, die für
   *einen* Zahlentyp `T` geschrieben sind. Autodiff ruft die Funktion aber mit `Dual<T>` auf. Dafür braucht es eine
   Funktion, die über den Zahlentyp generisch ist, etwa
   `trait VectorFn { fn eval<S: Real>(&self, x: &[S], out: &mut [S]); }`. Ohne Autodiff bleibt die Closure-Schnittstelle
   bestehen. Beides parallel anzubieten ist möglich, sollte aber **vor** den nächsten großen Löser-Familien festgelegt
   werden, damit nichts doppelt gebaut wird.
3. **`Complex` blockiert zwei Gruppen:** Polynom-Nullstellen und nichtsymmetrische Eigenwerte (die auch die
   Begleitmatrix brauchen). Durand-Kerner und Aberth brauchen *nur* `Complex`, nicht den Eigenwertlöser.
4. **Sparse-Formate** öffnen FEM, Innere-Punkte, AMG und große Systeme. Da `LinearOperator` schon existiert, laufen CG,
   GMRES und BiCGSTAB mit einem CSR-Typ sofort.
5. **Navier-Stokes** hängt an *drei* Dingen, die einzeln nützlich sind: FDM-Gerüst, Multigrid mit Neumann-Rand (für den
   Druck) und ODE-Zeitschritten. Reihenfolge: erst Multigrid-Erweiterung, dann FDM, dann Navier-Stokes.
6. **MILP und SMT hängen am Simplex.** Für SMT muss der Simplex über exakten Brüchen laufen, nicht über `f64`
   (Kollision mit der `Real`-Philosophie → eigener Typ `Rational`).
7. **Rotationen sind fast frei verfügbar:** `Real`, feste Matrizen und die SVD (für Polar-Re-Orthonormalisierung)
   existieren bereits. Ein schneller, sichtbarer Gewinn.

---

## 4. Struktur: ein Crate, Module in Unterordnern

Seit Phase 0a war das Projekt ein Cargo-Workspace mit acht Crates; auf Wunsch ist es wieder **ein** Crate `azmath`, gegliedert
in Unterordner von `src/`. Die öffentlichen Pfade (`azmath::linalg`, `azmath::roots`, …) sind unverändert; der Umbau hat
**kein Verhalten geändert** (420 von 420 Tests wie zuvor grün).

```
src/
  real/ (real.rs)   Real, Backends                    ← später: Complex
  linalg/           Vektoren/Matrizen, Zerlegungen, iterativ, eigen, Tridiagonal
  roots/            Nullstellen (1D)
  integrate/        Integration
  interp/           Interpolation, Splines
  optimize/         Newton, Gauß-Newton, Levenberg-Marquardt
  ode/              Anfangswertprobleme
  multigrid/        Multigrid (Poisson)
  complex/          Complex<T>
  autodiff/         Jet<T, N>, VectorFn, jacobian_ad, hessian_ad
  geometry/         Quaternion, exp_so3/log_so3, Euler-Winkel (Z-Y-X)
tests/              Integrationstests, Präfix = Modul
examples/
```

Geplante weitere **Module** (Namen vorläufig): `opt` (LP, QP,
MILP, NLP), `pde` (FDM, FEM, FVM, Navier-Stokes), `particles` (SPH), `iapws` (Wasser/Dampf), `sat` (DPLL, CDCL, SMT).

**Konventionen:** Die Features (`std`, `libm`, `micromath`, `alloc`) gelten crateweit. Mathematische Kernmodule bleiben ohne
Allokator nutzbar; Module, die `alloc` brauchen (FEM, SPH, MILP, SAT), werden mit `#[cfg(feature = "alloc")]` geschaltet.
Interne Rechenkerne (z. B. `linalg::kernel`) sind `pub(crate)`.

## 5. Reihenfolge

| Phase | Inhalt | Stand |
|---|---|---|
| **0a** | Aufteilung in Module (zuerst als Workspace, dann auf Wunsch wieder in **ein** Crate gefaltet; ohne API-Änderung) | ✅ |
| **0b** | **API-Entwurf** (Arbeitsspeicher-Muster, für Autodiff generische Funktionsschnittstelle): Entwurf und Prototyp liegen vor, [`docs/design/0001`](docs/design/0001-autodiff-und-arbeitsspeicher.md); **Entscheidung steht aus** | ⚠️ |
| **0c** | `CHANGELOG`, Crate-Metadaten, `cargo publish --dry-run` | ⬜ |
| **1** | ~~`Complex`~~ ✅, ~~Modul `autodiff` (**`Jet<T, N>`**)~~ ✅, ~~Matrixnormen~~ ✅, ~~Äquilibrierung (Ruiz/Sinkhorn)~~ ✅, ~~Orthonormalisierung (MGS/Householder)~~ ✅ | ✅ |
| **2** | Schnelle Gewinne: ~~Regula Falsi~~ ✅, ~~Rotationen/Quaternionen~~ ✅, ~~benannte Gauß-Elimination~~ ✅ | ✅ |
| **3** | Nichtlinear ausbauen: ~~BFGS/L-BFGS~~ ✅, ~~Broyden~~ ✅, ~~Linesearch~~ ✅, ~~Trust-Region (Dogleg)~~ ✅, ~~**Polynom-Nullstellen** (Durand-Kerner, Aberth)~~ ✅ | ✅ |
| **4** | Sparse/Band ✅, Links-Vorkonditionierer/ILU ✅, Lanczos/Arnoldi ✅ (Teile ohne Tests) | ✅ |
| **5** | Steife ODE: Adams-Bashforth ✅, **BDF** ⚠️ (fest), **Radau IIA** ⚠️ (fest), **Rodas** ⚠️ (ROS2), Ereignissuche ✅ (RK4) | ⚠️ |
| **6** | Nichtsymmetrische Eigenwerte (QR-Algorithmus) ✅ (nur Eigenwerte), Begleitmatrix ✅, Jenkins-Traub ✅ (vereinfacht) | ✅ |
| **7** | Optimierung: Simplex ✅ → Innere-Punkte ✅ → QP ✅ → SQP ✅ → IPOPT-artig ✅ → Branch-and-Bound ✅ (Branch-and-Cut, Warmstart, Dual-Simplex ⬜) | ✅ |
| **8** | **IAPWS-IF97** (Regionen 1–5), danach **IAPWS-95** mit `Jet` | ⬜ |
| **9** | PDE: Multigrid mit Neumann-Rand → FDM-Gerüst → **Navier-Stokes 2D inkompressibel** (Projektionsverfahren/SIMPLE, Druck per Multigrid); FEM; FVM; SPH | ⬜ |
| **A** | SAT/SMT (eigenes Modul `sat`, jederzeit unabhängig möglich) | ⬜ |

---

## 6. Entscheidungen

| # | Frage | Entscheidung |
|---|---|---|
| 1 | „Normieren von Matrizen“ | **Alle drei:** (a) Normen berechnen (Frobenius, 1, ∞, 2-Norm über SVD); (b) Zeilen-/Spaltenäquilibrierung zur Konditionierung (Ruiz, Sinkhorn); (c) Orthonormalisierung von Spalten (modifiziertes Gram-Schmidt, Householder) |
| 2 | „Jet“ | Mehrdimensionales Vorwärts-AD wie `Jet<T, N>` in Ceres: Wert `T` plus Gradient `[T; N]` fester Größe auf dem Stack, keine Heap-Allokation; für exakte Jacobi-Matrizen in `newton_system` und `levenberg_marquardt` statt Finite Differenzen |
| 3 | IAPWS | **IF97 zuerst** (Industriestandard, Regionen 1–5), **IAPWS-95** danach (Helmholtz-Energie, Ableitungen per Autodiff) |
| 4 | Navier-Stokes | **2D, inkompressibel, kartesisch** (Chorin-Projektion oder SIMPLE), Druck-Poisson per Multigrid; 3D und kompressibel später und optional |
| 5 | Struktur | **Ein Crate mit Unterordnern/Modulen**, keine Vielzahl von Crates ✅ |
| 6 | `alloc`-Politik | Mathematische Kernmodule (`real`, `linalg`, `roots`, `integrate`, `interp`, `optimize`, `ode`, `autodiff`) bleiben strikt `#![no_std]` **ohne** Allokator nutzbar; große Toolkits (FEM, SPH, MILP, SAT) dürfen `alloc` voraussetzen |
| – | Autodiff-API | Löser-Schnittstellen müssen über `T: Real` generisch sein, damit `Jet<T, N>` transparent durchgereicht werden kann (Abschnitt 7) |

---

## 7. Entwurfsnotizen für Phase 0b und Phase 1

### 7.1 `Jet<T, N>`: algebraische Eigenschaften und Konsequenzen

`Jet<T, N>` bildet einen **kommutativen Ring mit Nilpotenten**, keinen Körper: Reine Infinitesimale (`value == 0`,
Gradient `≠ 0`) sind Nullteiler, denn `(0, d)·(0, d') = (0, 0)`. Daraus folgt:

- Division durch ein `Jet` mit `value ≠ 0` ist erlaubt (Quotientenregel). Für `value == 0` gibt es kein Inverses; die
  Operatoren verhalten sich dann wie bei Fließkommazahlen (Wert `±∞`/NaN), damit generischer Code über `T: Real` unverändert
  läuft.
- Zusätzlich eine ausdrückliche, **geprüfte** Variante `checked_recip(&self) -> Option<Self>`, die bei `value == 0`
  `None` liefert (keine stille Division durch einen Nullteiler).
- `Real` verlangt `PartialOrd`, `abs`, `min`, `max`, `floor`, `signum` und Konstanten: Vergleiche und Rundungen beziehen sich
  auf den **Wert**, Ableitungen der stückweise konstanten Funktionen (`floor`, `round`, …) sind 0, die von `abs` ist
  `signum(value)·d`. Offen: ob `PartialEq` nur den Wert oder auch den Gradienten vergleicht (üblich in Ceres: nur den Wert).
- Als Skalar von Verfahren wie `newton_system` muss `Jet` die komplette `Real`-Oberfläche erfüllen (`sqrt`, `sin`, `exp`, `powf`,
  `powi`, `atan2`, … mit Kettenregel).

### 7.2 und 7.3 Funktionsschnittstelle für Autodiff und Arbeitsspeicher-Muster

Beide Fragen sind inzwischen mit einem **kompilierten Prototyp** untersucht und in
[`docs/design/0001-autodiff-und-arbeitsspeicher.md`](docs/design/0001-autodiff-und-arbeitsspeicher.md) beschrieben.
Kurzfassung der Ergebnisse:

- `Jet<T, N>` erfüllt `Real` ohne Änderung am Trait.
- Ein Trait `VectorFn` (ohne Typparameter, generische Methode `eval<S: Real>`) plus **Adapter** (`jacobian_ad`) versorgt die
  **unveränderten** Löser mit Ableitungen; neue `*_ad`-Löservarianten sind nicht nötig.
- Laufzeitgrößen ohne Heap gehen über Strip-Mining (`⌈n/K⌉` Auswertungen). Der Jet-Puffer hat einen anderen Elementtyp als
  `work: &mut [T]`.
- Exakte Jacobi-Matrizen ändern die Iterationszahl bei gut konditionierten Problemen **nicht**; der echte Gewinn sind zweite
  Ableitungen (Hessematrix: Fehler 0 gegenüber 3·10⁻⁶).
- Ein typisierter Workspace ist ein reiner Aufsatz auf die Slice-Schnittstelle und kann jederzeit ohne Breaking Change
  ergänzt werden. Empfehlung: Slice mit `const fn`-Länge als Grundmuster, Zustandsobjekte für zustandsbehaftete Verfahren.
