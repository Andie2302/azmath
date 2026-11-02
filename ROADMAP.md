# Roadmap und ToDo-Liste

Dieses Dokument hält fest, was **fertig** ist, was **noch fehlt**, was wovon **abhängt** und wie die Bibliothek
langfristig in **Crates** aufgeteilt werden kann. Es ist eine lebende Liste: Ideen kommen jederzeit dazu.

Legende: ✅ fertig · ⚠️ teilweise · ⬜ offen · 🔶 Entscheidung nötig
Aufwand (grob, relativ zu den bisherigen Schritten): **S** = klein (wie die Nullstellensuche), **M** = mittel (wie
GMRES oder Interpolation), **L** = groß (wie ODE oder Multigrid), **XL** = Projekt für sich.

---

## 1. Stand heute (Version 0.8.0)

| Bereich | Fertig |
|---|---|
| Grundlage | `Real` für `f32`/`f64` mit den Backends `std`, `libm`, `micromath`; `no_std`, optional `alloc` |
| Lineare Algebra, direkt | LU (Pivot), Cholesky, QR (Householder), Dreieckslöser, Tridiagonal (Thomas, zyklisch), SVD, symmetrische Eigenwerte (Jacobi) |
| Lineare Algebra, iterativ | CG, BiCGSTAB, GMRES(m), Jacobi-Vorkonditionierer, Potenzmethode, inverse Iteration |
| Nullstellen (1D) | Bisektion, Brent, Newton, Sekante |
| Nichtlinear (mehrdimensional) | Newton-System mit Schrittweitensteuerung (gedämpft), Gauß-Newton, Levenberg-Marquardt |
| Integration | Trapez, Simpson, Gauß-Legendre, adaptives Gauß-Kronrod, Messdaten |
| Interpolation | linear, kubische Splines (4 Randbedingungen), PCHIP, Lagrange |
| ODE | Euler, RK4, Dormand-Prince (adaptiv), implizites Euler, implizite Trapezregel |
| PDE | geometrisches Multigrid für Poisson (1D/2D), als Löser und als CG-Vorkonditionierer |

---

## 2. Gesamtliste (deine Brainstorming-Liste plus Ergänzungen)

### A. Grundbausteine
| | Eintrag | Aufwand | Hängt ab von | Ermöglicht |
|---|---|---|---|---|
| ⬜ | **`Complex<T>`** (komplexe Zahlen) | S–M | `Real` | Polynom-Nullstellen, nichtsymmetrische Eigenwerte, FFT |
| ⬜ | **Automatisches Differenzieren**: `Dual<T>` (1. Ableitung), `Jet<T, N>` (Mehrfach-Richtungen/Taylor), geschachtelt für Hessians | M–L | `Real` | exakte Jacobi-/Gradientenmatrizen, SQP/IPOPT, IAPWS-Ableitungen, Taylor-ODE |
| ⬜ | Matrixnormen (1, ∞, Frobenius, Spektralnorm über SVD), **Normieren/Skalieren/Äquilibrieren** | S | `linalg` | bessere Konditionierung, Fehlerabschätzung |
| ⬜ | Orthonormalisieren (Gram-Schmidt, Polarzerlegung `R = U·Vᵀ` über SVD) | S | SVD ✅ | stabile Rotationsmatrizen |
| ⬜ | `Real` ergänzen (`copysign`, `from_usize`, `fma`-Varianten …) | S | – | bei Bedarf |
| ⬜ | Sparse-Formate (COO/CSR), Bandmatrizen | M | `alloc` | große Systeme, FEM, Innere-Punkte |

### B. Lineare Algebra – Lücken
| | Eintrag | Aufwand | Hängt ab von |
|---|---|---|---|
| ⚠️ | **Gauß-Elimination** (als benannte Funktion; steckt in LU) | S | LU ✅ |
| ⬜ | Banded-LU/-Cholesky | M | Bandformat |
| ⬜ | Links-Vorkonditionierer, ILU/IC, SSOR | M | Sparse |
| ⬜ | Iterative Verfeinerung, Konditionsschätzer, QR mit Spaltenpivot, LDLᵀ | S–M | – |
| ⬜ | Eigenwerte **nichtsymmetrisch** (Hessenberg + QR-Algorithmus, Schur), verallgemeinert | L | `Complex` |
| ⬜ | Lanczos / Arnoldi (Teilspektren großer Matrizen) | M | Sparse/`LinearOperator` ✅ |
| ⬜ | Algebraisches Multigrid (AMG) | XL | Sparse |

### C. Nullstellen und Nichtlineares
| | Eintrag | Aufwand | Hängt ab von |
|---|---|---|---|
| ⬜ | **Regula Falsi** (+ Illinois-Variante) | S | – |
| ⬜ | **Quasi-Newton: BFGS** (L-BFGS), **Broyden** (Jacobi-Update) | M | Linesearch, optional Autodiff |
| ⚠️ | **Gedämpftes Newton** – grundlegend vorhanden (Schrittweiten-Halbierung); fehlen: Armijo/Wolfe-Linesearch, Trust-Region (Dogleg), Jacobi-Wiederverwendung | M | Autodiff (für exakte Jacobi) |
| ⬜ | **Polynom-Nullstellen**: Durand-Kerner, Aberth-Ehrlich, Jenkins-Traub, Begleitmatrix | M–L | `Complex`; Begleitmatrix zusätzlich nichtsymmetrische Eigenwerte |
| ⬜ | Minimierer ohne Ableitung (Golden Section, Nelder-Mead) | S–M | – |

### D. Differentialgleichungen
| | Eintrag | Aufwand | Hängt ab von |
|---|---|---|---|
| ⬜ | **Adams-Bashforth** (und -Moulton als Prädiktor-Korrektor) | M | RK-Anlaufrechnung ✅ |
| ⬜ | **BDF** (variable Ordnung/Schrittweite) | L | Newton ✅ (ode → optimize), Autodiff nützlich |
| ⬜ | **Radau IIA** | L | Newton, lineare Systeme mit Blockstruktur |
| ⬜ | **Rodas** (linear-implizites Rosenbrock) | L | Jacobi (Autodiff), LU |
| ⬜ | Dichte Ausgabe, Ereignissuche (Nulldurchgänge) | M | `rk45` ✅, Nullstellen ✅ |
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
| ⬜ | **LP: Simplex** (primal/dual, Beschränkungen) | M–L | LU ✅ |
| ⬜ | **LP: Innere-Punkte (Primal-Dual)** | L | Cholesky/LDLᵀ, Sparse |
| ⬜ | **QP** (Baustein für SQP) | L | Simplex/Innere-Punkte, KKT-Löser |
| ⬜ | **NLP: SQP** | XL | QP, Linesearch, **Autodiff (Hessians)**, BFGS |
| ⬜ | **NLP: Innere-Punkte (IPOPT-artig)** | XL | Sparse LDLᵀ, Filter-Linesearch, Autodiff |
| ⬜ | **MILP: Branch-and-Bound**, dann **Branch-and-Cut** | L–XL | Simplex (Warmstart), `alloc` |

### G. Geometrie und Rotationen
| | Eintrag | Aufwand | Hängt ab von |
|---|---|---|---|
| ⬜ | `Vec3`, `Mat3`, **Rotationsmatrizen (SO(3))**, **Quaternionen** (Einheitsquaternion, SLERP/NLERP), Achse-Winkel, Rodrigues, Exponential-/Logarithmus-Abbildung | M | `Real`, kleine feste Matrizen ✅ |
| ⬜ | **Euler-Winkel-Umrechnung mit Gimbal-Lock-Behandlung** (die Winkel selbst vermeiden; nur Ein-/Ausgabe) | S | Quaternion |
| ⬜ | Dual-Quaternionen, SE(3), Transformationsketten | M | Quaternion |
| ⬜ | Re-Orthonormalisierung von Rotationsmatrizen | S | Polarzerlegung (SVD ✅) |

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
   brauchen alle Ableitungen. Mit `Dual<T>`/`Jet` bekommen sie diese exakt statt über Differenzen (`_fd`-Varianten,
   Genauigkeit ~`√ε`).
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

## 4. Vorschlag: Aufteilung in Crates (Cargo-Workspace)

Heute ist alles **ein** Crate mit Modulen. Die Modulabhängigkeiten sind aber schon sauber geschichtet, sodass sich das
Crate ohne Umbau der Algorithmen aufteilen lässt:

```
azmath-real       Real, Backends; später Complex, Dual/Jet           [no_std]
azmath-linalg     Vector/Matrix, Zerlegungen, iterativ, eigen, Sparse [no_std, alloc optional]
azmath-numerics   roots, integrate, interp, ode, optimize            [no_std]
azmath-pde        multigrid, FDM, FEM, FVM, Navier-Stokes            [überwiegend alloc]
azmath-opt        LP, QP, MILP, NLP                                  [alloc]
azmath-geometry   Rotationen, Quaternionen, Transformationen         [no_std]
azmath-particles  SPH (netzfrei)                                     [alloc]   (oder Teil von -pde)
iapws             Thermodynamik Wasser/Dampf                         [no_std]
azsat             DPLL, CDCL, SMT                                    [alloc]   (eigenständig, ohne Real)
azmath            Fassade: Features pro Teil-Crate, behält die heutigen Pfade (azmath::linalg::…)
```

Bestehende interne Kopplungen (aus dem Code ermittelt):

| Von | nach | wie |
|---|---|---|
| `optimize` | `linalg` | **vier interne Kernfunktionen** (`qr_decompose`, `qr_solve`, `cholesky_decompose`, `cholesky_solve`) – bei Aufteilung öffentlich (`#[doc(hidden)]`) machen |
| `ode` | `optimize` | `newton_system`, `OptimizeOptions` (öffentliche API) |
| `interp` | `linalg` | `solve_tridiagonal`, `solve_cyclic_tridiagonal` (öffentlich) |
| `multigrid` | `linalg` | `LinearOperator`, `Preconditioner` (öffentlich) |
| `roots`, `integrate` | – | nur `Real` |

**Vorteile:** schnellere Übersetzung, schlankere Abhängigkeiten (wer nur Quaternionen will, zieht keine PDE-Löser),
unabhängige Versionen, `azsat` kann einen eigenen Namen tragen. **Kosten:** Features (`std`/`libm`/`micromath`/`alloc`)
müssen durch alle Crates durchgereicht werden, die CI-Matrix wächst, und die Kerne müssen gezielt veröffentlicht werden.

**Empfehlung:** den Workspace-Umbau **vor** den nächsten großen Gebieten machen (Phase 0), solange nur vier interne
Kopplungen anzupassen sind. Die Fassade `azmath` hält bestehende Importpfade stabil.

---

## 5. Empfohlene Reihenfolge

| Phase | Inhalt | Warum jetzt |
|---|---|---|
| **0** | Workspace-Aufteilung, `CHANGELOG`, Lizenz-/Metadaten-Check, `cargo publish --dry-run` | billig, solange wenig gekoppelt ist |
| **1** | `Complex`, **`Dual`/`Jet` (Autodiff)** samt API-Entscheidung (🔶 Punkt 2), Matrixnormen/-Normierung | Grundbausteine, auf die vieles aufbaut |
| **2** | Schnelle Gewinne: Regula Falsi, Rotationen/Quaternionen, benannte Gauß-Elimination, Orthonormalisierung | wenig Abhängigkeiten, sofort nutzbar |
| **3** | Nichtlinear ausbauen: BFGS/L-BFGS, Broyden, Trust-Region/Linesearch, **Polynom-Nullstellen** (Durand-Kerner, Aberth) | Autodiff und `Complex` sind da |
| **4** | Sparse/Band, Links-Vorkonditionierer/ILU, Lanczos/Arnoldi | öffnet große Systeme |
| **5** | Steife ODE: Adams-Bashforth, **BDF**, **Radau IIA**, **Rodas**, Ereignissuche | baut auf Newton und Autodiff |
| **6** | Nichtsymmetrische Eigenwerte (QR-Algorithmus), Begleitmatrix, Jenkins-Traub | braucht `Complex`; sehr aufwendig |
| **7** | Optimierung: Simplex → Innere-Punkte → QP → SQP → IPOPT-artig → Branch-and-Bound/-Cut | LP ist Voraussetzung für alles danach |
| **8** | IAPWS-IF97, dann IAPWS-95 (mit `Jet`) | Anwendungsgebiet, nutzt die Bausteine |
| **9** | PDE: Multigrid-Erweiterung (Neumann) → FDM-Gerüst → Navier-Stokes; FEM; FVM; SPH | größte Projekte, zuletzt |
| **A** | SAT/SMT (eigenes Crate, jederzeit unabhängig möglich) | keine Abhängigkeit zum Rest |

---

## 6. Offene Fragen (🔶)

1. **„Normieren von Matrizen“** – welche Bedeutung ist gemeint? (a) *Normen berechnen* (1, ∞, Frobenius, Spektralnorm);
   (b) *Skalieren/Äquilibrieren* zur besseren Konditionierung; (c) *Orthonormalisieren* (z. B. eine verrutschte
   Rotationsmatrix wieder orthogonal machen). Alle drei sind klein; ich habe sie unter A eingetragen.
2. **„Jet“** – ich lese es als *Jet im Sinne der Autodiff-Bibliotheken* (z. B. Ceres): eine Zahl, die einen Wert und
   mehrere Ableitungskomponenten mitführt, bzw. abgeschnittene Taylorreihen. Falls etwas anderes gemeint war, bitte
   korrigieren.
3. **IAPWS:** IF97 (schnell, Industrie) oder IAPWS-95 (genauer, Helmholtz-basiert) oder beides? Beides ist in Phase 8
   eingetragen.
4. **Navier-Stokes:** nur inkompressibel in 2D (Lid-driven Cavity) oder auch kompressibel/3D? Kompressibel würde FVM mit
   Riemann-Lösern verlangen.
5. **Aufteilung:** gleich in Phase 0, oder erst wenn ein zweites großes Gebiet beginnt?
6. **`no_std`-Anspruch:** Gelten alle Teil-Crates als `no_std`, oder dürfen FEM/SPH/MILP/SAT bewusst `alloc` (bzw. `std`)
   voraussetzen? Vorschlag: ja, die „großen“ Crates dürfen `alloc` voraussetzen.
