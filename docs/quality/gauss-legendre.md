# Gauss–Legendre-RK4: Prüfung (2026-11-07)

AZM-031: `ode::gauss_legendre4`, `gauss_legendre4_fd` und
`gauss_legendre4_work_len`. Allokationsfreie zweistufige Kollokation, Ordnung 4,
festes Zeitraster, gekoppelte Newton-Stufenzustände. Analytische Jacobi-Matrix
oder Vorwärtsdifferenzen des gekoppelten Systems. Speicher: `5*n*n + 21*n`
Real-Elemente, O(n²); dichter Newton-Löser, nicht für beliebig große Systeme.

## Nachweis

[PR #7](https://github.com/Andie2302/azmath/pull/7), geprüfter Code
`d2ca1ad4cb28b92d976410dcb9ac9874976ed1fa`, Merge
`75c5dc04c51ddf02568f15f4ab3c46f7ba4c2c7c`.
Alle 18 Jobs im [CI-Lauf](https://github.com/Andie2302/azmath/actions/runs/42320120483)
erfolgreich: Backend-Matrix, Embedded, MSRV 1.85, Format/Clippy/Rustdoc,
Coverage, Fuzz-Harness und Paket-Dry-Run. Keine lokalen Tests ausgeführt.

25 neue Tests in `tests/ode_gauss_legendre.rs`: zwölf je f32/f64 plus eine
Workspace-Größenprüfung. Geprüft werden Ordnung 4, Rückwärtsintegration und
Zeitumkehr, quadratische Invarianten, nicht separierbare Hamilton-Funktion,
nichtlineare analytische/FD-Jacobi-Matrizen, nichtautonome Stufenzeiten,
Beobachter, leeres Intervall, ungültige Eingaben, NaN/Inf, Newton-Limit,
nicht darstellbare Schritte und Erhalt des letzten vollständigen Zustands.
Ein zusätzlicher Doctest dokumentiert die Verwendung.

## Grenzen

A-stabil, nicht L-stabil; keine adaptive Fehlerschätzung. Newton akzeptiert
nur erfüllte Residuen; `tol_f` ist absolut in Zustandseinheiten und gilt für
die euklidische Norm der unskalierten Stufengleichungen. Die Toleranz muss für
den gewünschten Diskretisierungsfehler passend gewählt werden.
Symplektizität und Zeitumkehr gelten für exakt gelöste Hamilton-Stufen;
Newton-Toleranz und Rundung begrenzen die praktische Strukturtreue.
Quadratische Invarianten werden in exakter Arithmetik erhalten, allgemeine
nichtlineare Energie nicht exakt. Genaue Referenztests gelten für std/libm;
reines micromath erhält hier keine zusätzliche Genauigkeitsfreigabe.

## Physikalischer Vergleich

[azphysics PR #5](https://github.com/Andie2302/azphysics/pull/5) vergleicht
Verlet mit GL4 am harmonischen Oszillator: Ordnung 2/4 und Langzeitenergie
für f32/f64. Das Beispiel `02_integrator_comparison` wird in dessen CI ausgeführt
und zeigt Fehler sowie RHS-/Jacobi-Auswertungen. Gleiche Schrittzahlen sind
kein gleicher Rechenaufwand; es ist kein Laufzeitbenchmark.
