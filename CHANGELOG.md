# Änderungsprotokoll

Format: nach Versionen, neueste zuerst. Noch nicht veröffentlicht.

## Unveröffentlicht (seit 0.8.0)

### Neu
- **Struktur:** wieder ein einziges Crate mit Modulen in Unterordnern (statt Workspace).
- **`autodiff`:** `Jet<T, N>` (Vorwärtsmodus), `VectorFn`, `jacobian_ad`, `hessian_ad`.
- **`complex`:** `Complex<T>`.
- **`geometry`:** Quaternionen, `exp_so3`/`log_so3`, Euler-Winkel (Z-Y-X).
- **`linalg`:** Matrixnormen, Äquilibrierung (Ruiz, Sinkhorn), Orthonormalisierung (MGS, `thin_q`), Polarzerlegung,
  Gauß-Elimination (`gauss_solve`, `rank`, `row_echelon`), nichtsymmetrische Eigenwerte (`eigenvalues`, `hessenberg_form`),
  Lanczos/Arnoldi, Links-Vorkonditionierung (`pgmres_left`).
- **`sparse`:** CSR/COO/`CsrView`, Bandmatrizen mit LU/Cholesky, ILU(0)/IC(0)/SSOR, direkte LDLᵀ/Cholesky mit RCM-Umordnung (`SparseLdl`, `SparseCholesky`).
- **`roots`:** Regula Falsi, Illinois, Polynom-Nullstellen (Aberth, Durand-Kerner, Jenkins-Traub, Begleitmatrix).
- **`optimize`:** BFGS, L-BFGS, Wolfe-Schrittweitensuche, Broyden, Dogleg.
- **`ode`:** Adams-Bashforth-Moulton, BDF 1–5, Radau IIA, ROS2, Hermite-Ausgabe, Ereignissuche (RK4).
- **`opt`:** Simplex (mit Variablenschranken), Innere-Punkte-LP/QP, SQP, Innere-Punkte-NLP, Branch-and-Bound.
- **`diff`:** Differenzenquotienten, Ridders-Extrapolation, Gradient, Hessematrix, Jacobi-Matrix per Differenzen.
- **`optimize`:** zusätzlich `golden_section`, `brent_min`, `nelder_mead` (ableitungsfrei).
- **`special`:** `gamma`, `ln_gamma`, `digamma`, `beta`, `erf`, `erfc`, `erf_inv`, `normal_cdf`, `normal_quantile`, `gamma_p`, `gamma_q`, `beta_inc`, `bessel_j`.
- **`fft`:** FFT beliebiger Länge (Radix-2 und Bluestein), reelle FFT, Faltung, `fft_freq`.
- **`scalar`:** `Scalar`-Trait (f32, f64, `Complex`), `ScalarLu` (LU mit Teilpivotisierung, reell und komplex).
- **`Real`:** `to_f64`, `is_finite`, `copysign`, `sin_cos`, `exp2`, `log`, `exp_m1`, `ln_1p`, `asinh`, `acosh`, `atanh`,
  `rem_euclid`, `from_usize`.

### Geändert
- `Real` hat die neue **Pflichtmethode** `to_f64`; eigene Implementierungen des Traits müssen sie ergänzen.
- `RootError` und `OdeError` haben neue Varianten (`InvalidInput`, `EventNotLocated`); beide sind `#[non_exhaustive]`.

### Bekannte Grenzen
- Phase-5-Löser (BDF, Radau) arbeiten mit fester Schrittweite; Branch-and-Bound ohne Warmstart; SQP/Innere-Punkte-NLP ohne
  Wiederherstellungsphase.
