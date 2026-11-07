# Changelog

Format angelehnt an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).
Das Crate ist noch nicht veröffentlicht (Version 0.0.1).

## [Unveröffentlicht]

### Hinzugefügt
- Allokationsfreies zweistufiges Gauss–Legendre-RK4 (`ode::gauss_legendre4`,
  `gauss_legendre4_fd`): gekoppelte Newton-Stufen, feste Schritte, analytische oder
  numerische Jacobi-Matrix; f32/f64-Tests zu Ordnung, Zeitumkehr, quadratischen
  Invarianten, nicht separierbaren Hamilton-Systemen und Fehlerpfaden (AZM-031).
- Symplektisches Störmer-Verlet-Verfahren `ode::verlet` und `ode::verlet_separable` (Ordnung 2, zeitumkehrbar, allokationsfrei) mit Tests (`tests/ode_verlet.rs`), AZM-030.
- MINRES/PMINRES für reelle symmetrisch-indefinite Systeme, konstante 8*n
  Arbeitselemente, Matrix-/DynMatrix-Hilfsmethoden und f32/f64-Regressionen.
- Paket-/Publish-Dry-Run und Tests aus dem entpackten Crate in CI;
  MSRV-Builds auch mit allen Features und mit libm+alloc.
- Direkte Überlebensfunktionen (`sf`) und inverse obere Tails (`isf`) für die
  Verteilungen sowie `normal_sf`/`normal_isf` und `beta_inc_complement`.
- 606 hochpräzise f32/f64-Referenzen für Verteilungen und Quantile.
- Lanczos-Invarianten und Ritz-Residualtests für f32/f64, einschließlich
  früher Terminierung, extremer Skalierung und ungültiger Eingaben.
- Native CSR-Eingaben für QP/LP (`interior_point_qp_sparse`, `interior_point_lp_sparse`),
  persistenter `QpWorkspace` mit Warmstarts.
- Wiederverwendbare Sparse-Analyse und numerische Refaktorisierung; Bunch-Kaufman-LDLᵀ.
- LU-Verfeinerung, Konditionsschätzung, spaltenpivotierte QR.
- Referenztests (mpmath, ULP) und Fuzz-Target für `special`; Coverage-Lauf in CI.
- Arnoldi-Invariantentests, micromath-Smoke-Test, Embedded-Build (`thumbv7em-none-eabihf`) in CI.
- Eigenschaftstests mit festem Seed für `linalg` und `fft` (`tests/properties_linalg_fft.rs`).
- Eigenschaftstests mit festem Seed für `sparse` (`tests/sparse_properties.rs`): COO→CSR, Umordnungen, LDLᵀ/Cholesky/LU, Refaktorisierung und Erholung nach Fehlern.
- Robustheitstests für `QpWorkspace` (`tests/opt_qp_workspace_robustness.rs`): Zufallsfolgen gegen Kaltstarts für alle KKT-Backends, Zustand nach ungültigen Eingaben, Verwerfen des Warmstarts nach Fehlschlägen, Iterationslimit.

- Fuzz-Targets `fft`, `sparse` und `qp` (`fuzz/`).

### Behoben
- Extrem kleine Normalquantile, Auflösung kleiner Gamma-/Beta-Quantile,
  Student-t-/F-Tails bei großen Argumenten und kleine t-/χ²-p-Werte.
- NaN-Weitergabe, unendliche Argumentgrenzen und breite Gleichverteilungsintervalle.
- Lanczos: skalierte Normen für große/kleine Start- und Restvektoren;
  Ritz-Auswertung skaliert die Tridiagonalmatrix und validiert alle benutzten
  alpha-/beta-Werte einschließlich der Restnorm.
- FFT-Fuzzer: Längen 256 und 257 sind jetzt erreichbar. QP-Fuzzer prüfen
  zusätzlich Stationarität, Dualzulässigkeit und Komplementarität.
- QP-Presolve (`independent_rows`): Widersprüchliche Gleichungen wurden bei sehr unterschiedlich skalierten
  Zeilen (Einträge ~1e300) als „redundant“ verworfen und `Optimal` gemeldet. Nullzeilen mit rechter Seite ≠ 0
  sind jetzt immer widersprüchlich; die Toleranz der Restgleichungen ist skalenfrei und zeilenweise.

### Geändert
- Verteilungsparameter müssen endlich sein; vorher teils akzeptierte unendliche
  Parameter liefern nun NaN. Die gemeinsame Zukunftsplanung liegt in aztodo.
- Härtung von Wurzelsuche, Quadratur, ODE-Grundlöser, Interpolation, Krylov-Löser.
- `special`: genauere Reflexionsformel für `gamma`/`ln_gamma`, Reihenentwicklung von `erf` nahe 0.
