# Changelog

Format angelehnt an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).
Das Crate ist noch nicht veröffentlicht (Version 0.0.1).

## [Unveröffentlicht]

### Hinzugefügt
- Native CSR-Eingaben für QP/LP (`interior_point_qp_sparse`, `interior_point_lp_sparse`),
  persistenter `QpWorkspace` mit Warmstarts.
- Wiederverwendbare Sparse-Analyse und numerische Refaktorisierung; Bunch-Kaufman-LDLᵀ.
- LU-Verfeinerung, Konditionsschätzung, spaltenpivotierte QR.
- Referenztests (mpmath, ULP) und Fuzz-Target für `special`; Coverage-Lauf in CI.
- Arnoldi-Invariantentests, micromath-Smoke-Test, Embedded-Build (`thumbv7em-none-eabihf`) in CI.
- Eigenschaftstests mit festem Seed für `linalg` und `fft` (`tests/properties_linalg_fft.rs`).
- Eigenschaftstests mit festem Seed für `sparse` (`tests/sparse_properties.rs`): COO→CSR, Umordnungen, LDLᵀ/Cholesky/LU, Refaktorisierung und Erholung nach Fehlern.
- Robustheitstests für `QpWorkspace` (`tests/opt_qp_workspace_robustness.rs`): Zufallsfolgen gegen Kaltstarts für alle KKT-Backends, Zustand nach ungültigen Eingaben, Verwerfen des Warmstarts nach Fehlschlägen, Iterationslimit.

### Geändert
- Härtung von Wurzelsuche, Quadratur, ODE-Grundlöser, Interpolation, Krylov-Löser.
- `special`: genauere Reflexionsformel für `gamma`/`ln_gamma`, Reihenentwicklung von `erf` nahe 0.
