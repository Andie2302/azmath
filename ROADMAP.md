# azmath: Stand und Roadmap

Version laut `Cargo.toml`: **0.0.1**.

Die gemeinsame kurz-, mittel- und langfristige Planung von azmath und azphysics
wird in [aztodo](https://github.com/Andie2302/aztodo) gepflegt:
[Roadmap](https://github.com/Andie2302/aztodo/blob/main/ROADMAP.md),
[Aufgaben und Abnahmekriterien](https://github.com/Andie2302/aztodo/blob/main/BACKLOG.md),
[Architektur](https://github.com/Andie2302/aztodo/blob/main/ARCHITECTURE.md).

Die bisherige vollständige Liste ist als
[Archiv](https://github.com/Andie2302/aztodo/blob/main/docs/archive/azmath-roadmap-2026-11-06.md)
erhalten. Alte Hinweise wie „bewusst ausgelassen“ sind offene Prüfpunkte und
keine dauerhaften Ausschlüsse.

## Vorhandene Grundlagen

| Bereich | Implementiert |
|---|---|
| Zahlengrundlage | Real f32/f64, std/libm/micromath, no_std, optional alloc, Complex/Scalar |
| Direkte lineare Algebra | LU, pivotierte QR, Cholesky, SVD, Bunch–Kaufman-LDLT, Refinement/Konditionsschätzung |
| Iterative lineare Algebra | CG/PCG, MINRES/PMINRES, GMRES, BiCGSTAB, Vorkonditionierer; Arnoldi und Lanczos mit Invariantentests |
| Sparse | COO/CSR, Umordnungen, direkte Löser, Analyse-/Musterwiederverwendung und Refaktorisierung |
| Nichtlineares/Optimierung | Brent/Newton, gedämpfte Systeme, LM, BFGS/L-BFGS, QP/LP, SQP/NLP |
| Analysis | Autodiff, Integration/Quadratur, Interpolation, ODE/DAE/BVP und grundlegende PDE-Diskretisierung |
| Weitere Mathematik | FFT, Spezialfunktionen, Verteilungen/Quantile, Statistik, mathematische Geometrie |

„Implementiert“ bedeutet nicht für beliebige Parameter vollständig abgesichert.
Die Nachweise und Grenzen stehen beim jeweiligen Modul und unter
[docs/quality](docs/quality/README.md).

## Aktueller Abschluss und nächste Aufgaben

**AZM-010** (technische Release-Vorbereitung) und **AZM-020** (MINRES/PMINRES)
sind mit lokalen Nachweisen abgeschlossen. Das Crate wurde nicht veröffentlicht.
Details: [Release- und MINRES-Prüfung](docs/quality/release-minres.md).

1. **AZM-030/031:** Verlet und implizites Gauss–Legendre-Runge–Kutta.
   Gauss–Legendre-Quadratur ist bereits vorhanden und davon verschieden.
2. **AZM-040/041:** Neumann-/periodische Operatoren, Nullräume und FEM-Randbedingungen.

Weitere offene Arbeiten wie Chandrupatla, AMG, NLP-Wiederherstellung,
reelle Schur-Vektoren/Eigenvektoren, komplexe Sparse-Verfahren und größere
Spezialfunktionsbereiche sind im zentralen Backlog erfasst bzw. im Archiv
aufbewahrt. Physikalische Formulierungen wie APIC/FLIP, XPBD, Materialmodelle
und Druckprojektion sind Aufgaben von azphysics; allgemeine Löser bleiben hier.
