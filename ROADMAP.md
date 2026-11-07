# azmath: Stand und Roadmap

Version laut `Cargo.toml`: **0.0.1**.

Die gemeinsame kurz-, mittel- und langfristige Planung von azmath und azphysics
wird in [aztodo](https://github.com/Andie2302/aztodo) gepflegt:
[Roadmap](https://github.com/Andie2302/aztodo/blob/master/ROADMAP.md),
[Aufgaben und Abnahmekriterien](https://github.com/Andie2302/aztodo/blob/master/BACKLOG.md),
[Architektur](https://github.com/Andie2302/aztodo/blob/master/ARCHITECTURE.md).

Die bisherige vollständige Liste ist als
[Archiv](https://github.com/Andie2302/aztodo/blob/master/docs/archive/azmath-roadmap-2026-11-06.md)
erhalten. Alte Hinweise wie „bewusst ausgelassen“ sind offene Prüfpunkte und
keine dauerhaften Ausschlüsse.

AZM-031 ergänzt das implizite zweistufige Gauss–Legendre-RK-Verfahren
(Ordnung 4) für allgemeine ODE und nicht separierbare Hamilton-Systeme.
Die symplektischen Eigenschaften setzen hinreichend genau gelöste Stufen voraus.

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

**AZM-030/031** sind ebenfalls abgeschlossen: Verlet und implizites
Gauss–Legendre-RK4 (Ordnung 4), mit CI-Nachweisen und Oszillatorvergleich.
Details: [Gauss–Legendre-Prüfung](docs/quality/gauss-legendre.md).
Gauss–Legendre-Quadratur ist davon unabhängig.

**Als nächster mathematischer Ausbau: AZM-040/041:** Neumann-/periodische Operatoren, Nullräume und FEM-Randbedingungen.

Weitere offene Arbeiten wie Chandrupatla, AMG, NLP-Wiederherstellung,
reelle Schur-Vektoren/Eigenvektoren, komplexe Sparse-Verfahren und größere
Spezialfunktionsbereiche sind im zentralen Backlog erfasst bzw. im Archiv
aufbewahrt. Physikalische Formulierungen wie APIC/FLIP, XPBD, Materialmodelle
und Druckprojektion sind Aufgaben von azphysics; allgemeine Löser bleiben hier.
