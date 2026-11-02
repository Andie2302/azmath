# azmath

Ein kleines `no_std`-Crate, das **ein einziges Trait `Real`** für `f32` und `f64` bereitstellt.
Mathematische Funktionen (`sqrt`, `sin`, `exp`, …) werden je nach Feature-Flag an ein
Backend weitergeleitet. So lässt sich numerischer Code einmal generisch schreiben und auf
Desktop wie auf Mikrocontrollern verwenden.

## Aufbau

Alles liegt in **einem** Crate, gegliedert in Module (`src/<modul>/`). Das Crate ist `#![no_std]`; die Features
(`std`, `libm`, `micromath`, `alloc`) gelten für alle Module gemeinsam.

| Modul | Inhalt |
|---|---|
| `azmath::Real` | `Real`-Trait, Backends (`std`, `libm`, `micromath`) |
| `azmath::linalg` | Zerlegungen, iterative Löser, Eigenwerte, SVD, Tridiagonal, Matrixnormen, Äquilibrierung, Orthonormalisierung, Polarzerlegung |
| `azmath::roots` | Nullstellen in einer Veränderlichen |
| `azmath::integrate` | numerische Integration |
| `azmath::interp` | Interpolation, Splines |
| `azmath::optimize` | Newton, Gauß-Newton, Levenberg-Marquardt |
| `azmath::ode` | gewöhnliche Differentialgleichungen |
| `azmath::multigrid` | Multigrid für Poisson |
| `azmath::complex` | komplexe Zahlen `Complex<T>` (Arithmetik, Polarform, `sqrt`, `exp`, `ln`, Potenzen, Trigonometrie) |
| `azmath::autodiff` | Automatisches Differenzieren (`Jet`, exakte Jacobi-/Hessematrizen) |

Tests liegen in `tests/`, Beispiele in `examples/`.

## Features / Backends

| Feature     | Backend      | `f32` | `f64` | Hinweis                                           |
|-------------|--------------|:-----:|:-----:|---------------------------------------------------|
| `std`       | Standardbib. |  ✅   |  ✅   | **Default**, genaueste Ergebnisse                 |
| `libm`      | [`libm`]     |  ✅   |  ✅   | `no_std`, genau                                   |
| `micromath` | [`micromath`]|  ✅   |  ❌   | `no_std`, schnell, aber nur eine Näherung         |

Zusätzlich gibt es das Feature **`alloc`** (dynamisch große Matrizen und Vektoren, siehe
[Lineare Algebra](#lineare-algebra)). Es ist unabhängig von den Backends und braucht nur einen Allokator.

Sind mehrere Features aktiv, gilt die Priorität **`std` → `libm` → `micromath`**.
Ohne jedes Feature stehen keine `Real`-Implementierungen zur Verfügung.

Für `no_std` muss das Default-Feature abgeschaltet werden:

```toml
[dependencies]
azmath = { version = "0.8", default-features = false, features = ["libm"] }
```

[`libm`]: https://crates.io/crates/libm
[`micromath`]: https://crates.io/crates/micromath

## Verwendung

```rust
use azmath::Real;

fn hypotenuse<T: Real>(a: T, b: T) -> T {
    (a * a + b * b).sqrt()
}

fn celsius_zu_fahrenheit<T: Real>(c: T) -> T {
    c * T::from_f64(1.8) + T::from_f64(32.0)
}

fn main() {
    println!("{}", hypotenuse(3.0f32, 4.0f32)); // 5
    println!("{}", hypotenuse(3.0f64, 4.0f64)); // 5
    println!("{}", celsius_zu_fahrenheit(100.0f64)); // 212
}
```

## Das Trait `Real`

- **Konstanten:** `ZERO`, `ONE`, `PI`, `E`, `EPSILON`, `NAN`, `INFINITY`, `NEG_INFINITY`, `MIN_POSITIVE`
- **Operatoren:** `+ - * /`, Negation und die zugehörigen Zuweisungsoperatoren, Vergleiche
- **Basis:** `abs`, `signum`, `min`, `max`, `clamp`, `is_nan`, `is_infinite`, `recip`
- **Höhere Funktionen:** `sqrt`, `cbrt`, `hypot`, `mul_add`, `sin`, `cos`, `tan`, `asin`, `acos`,
  `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `exp`, `ln`, `log10`, `log2`, `powf`, `powi`
- **Rundung:** `floor`, `ceil`, `round`, `trunc`, `fract`
- **Winkel:** `to_radians`, `to_degrees`
- **Konvertierung:** `from_f64` für Literale in generischen Formeln

`sinh`, `cosh`, `tanh`, `cbrt`, `fract`, `recip`, `clamp` und die Winkelumrechnung haben eine
Standardimplementierung im Trait; mit `std` und `libm` werden `sinh`/`cosh`/`tanh`/`cbrt`
direkt an das Backend weitergereicht, mit `micromath` (das sie nicht anbietet) greift die
Standardimplementierung über `exp`/`powf`.

## Lineare Algebra

Das Modul `azmath::linalg` enthält Vektoren, Matrizen und einen LU-Löser (mit Teilpivotisierung).
Es nutzt nur das `Real`-Trait und funktioniert daher mit jedem Backend.

- `Vector<T, N>` und `Matrix<T, R, C>`: feste Größe über Const Generics, liegen auf dem Stack, kein
  Allokator nötig. Operatoren `+`, `-`, Negation, Skalar-, Matrix-Vektor- und Matrix-Matrix-Produkt.
- **LU** (`Matrix::solve`, `determinant`, `inverse`, `lu()`): für beliebige quadratische Matrizen, mit
  Teilpivotisierung. Eine `LuDecomposition` löst beliebig viele rechte Seiten günstig.
- **Cholesky** (`Matrix::cholesky`, `solve_spd`): für symmetrisch positiv definite Matrizen,
  schneller als LU. Symmetrie und positive Definitheit werden geprüft.
- **Dreieckslöser** (`solve_lower`, `solve_upper`, jeweils auch als `_unit`- und
  `_transposed`-Variante): Vorwärts- und Rückwärtssubstitution als öffentlicher Baustein. Das jeweils
  andere Dreieck wird ignoriert. Die `_unit`-Varianten nehmen eine Einheitsdiagonale an (wie `L` bei
  LU), die `_transposed`-Varianten lösen `Lᵀ·x = b` bzw. `Uᵀ·x = b`, ohne die Matrix zu
  transponieren (so löst man etwa Cholesky in zwei Schritten). Ein Diagonalelement 0 oder NaN
  ergibt `LinalgError::Singular`.
- **QR** (`Matrix::qr`, `least_squares`): Householder-Verfahren für `M × N`-Matrizen mit `M ≥ N`.
  Löst überbestimmte Systeme im Sinne der kleinsten Quadrate (Ausgleichsgeraden, Polynomfits)
  und liefert `Q`, `R` und die Residuumsnorm.
- **Iterative Löser** (`azmath::linalg::iterative`): CG (`cg`, `pcg`) für große symmetrisch positiv
  definite Systeme, **matrixfrei** über das Trait `LinearOperator` (dichte `Matrix`/`DynMatrix`,
  eigene Stencils oder beliebige Closures über `FnOperator`). Der Arbeitsspeicher wird als Slice
  übergeben, daher läuft das ohne Allokator. Optional mit Vorkonditionierer (`Jacobi`,
  `DynJacobi`, eigene über das Trait `Preconditioner`); bequem auch als `Matrix::solve_cg` und
  `solve_pcg`. Abbruch bei relativer Residuumsnorm `‖b − A·x‖ ≤ tol · ‖b‖`
  (`IterOptions`), sonst `LinalgError::NoConvergence`.
- **BiCGSTAB** (`bicgstab`, `pbicgstab`, `Matrix::solve_bicgstab`): für beliebige, auch
  **nichtsymmetrische** reguläre Systeme, ebenfalls matrixfrei (benötigt nur `A·x`, kein `Aᵀ·x`)
  und mit Vorkonditionierer. Arbeitsspeicher: `8·n`. In seltenen Fällen bricht das Verfahren
  zusammen (`LinalgError::Breakdown`). Bei sehr schlecht konditionierten Matrizen sagt das
  Residuum mehr aus als der Fehler der Lösung selbst.
- **Eigenwerte und SVD** (`azmath::linalg::eigen`): `Matrix::symmetric_eigen` liefert **alle** Eigenwerte
  (aufsteigend) und orthonormale Eigenvektoren einer symmetrischen Matrix (zyklisches Jacobi-Verfahren,
  sehr genau auch für kleine Eigenwerte). `Matrix::svd` berechnet die dünne SVD `A = U·Σ·Vᵀ` (`M ≥ N`,
  einseitiges Jacobi-Verfahren) samt `rank()`, `condition_number()`, `pseudo_inverse()` und `solve()`, der
  Lösung **kleinster Norm** auch bei Rangmangel (wo QR `Singular` meldet). Matrixfrei gibt es
  `power_iteration` (betragsgrößter Eigenwert, nur `A·x` nötig) und `inverse_iteration` (Eigenwert nahe
  einem `shift`, bequem als `Matrix::eigenpair_near`). Mit `alloc` stehen `DynMatrix::symmetric_eigen`,
  `svd`, `dominant_eigenpair` und `eigenpair_near` bereit. Nicht enthalten: Eigenwerte nichtsymmetrischer
  Matrizen (komplexe Eigenwerte) und Teilspektren großer Matrizen (Lanczos, Arnoldi).
- **Tridiagonal** (`solve_tridiagonal`, `solve_cyclic_tridiagonal`): Thomas-Algorithmus in `O(n)`, auch
  mit zyklischen Ecken (Sherman-Morrison); stabil für diagonaldominante bzw. symmetrisch positiv
  definite Matrizen, sonst besser LU. Grundlage der Splines.
- **GMRES(m)** (`gmres`, `pgmres`, `gmres_work_len`, mit `alloc` auch `DynMatrix::solve_gmres`): für
  beliebige reguläre Systeme, **monoton und ohne Zusammenbruch**, auch bei symmetrisch indefiniten
  und schiefsymmetrischen Matrizen, bei denen CG (`NotPositiveDefinite`) und BiCGSTAB (`Breakdown`)
  versagen. Matrixfrei, eine Anwendung von `A` pro Iteration, mit **Rechts-Vorkonditionierer**
  (wirkt auf die Spalten, Jacobi hilft also gegen ungleichmäßige Spaltenskalierung). Es speichert
  `restart + 1` Basisvektoren; ein kleines `restart` spart Speicher, kostet aber Iterationen (bei der
  1D-Konvektions-Diffusion mit `n = 100`: volles GMRES 100 Anwendungen von `A`, GMRES(20) etwa 500,
  BiCGSTAB 214). Der Arbeitsspeicher kommt als Slice (`gmres_work_len(n, restart)` als `const fn`),
  für feste `Matrix`-Typen gibt es daher keine Komfortmethode.
- Mit `--features alloc`: `DynMatrix<T>`, `DynVector<T>`, `DynLu<T>`, `DynCholesky<T>` und
  `DynQr<T>` mit zur Laufzeit festgelegter Größe (gleiche Algorithmen, Dimensionsfehler als
  `LinalgError::DimensionMismatch`).
- Fehler (`LinalgError`, `#[non_exhaustive]`): `Singular` (bei QR: Rangdefizit), `NotFinite`
  (NaN/Unendlich im Eingang), `DimensionMismatch`, `NotSymmetric`, `NotPositiveDefinite`, `NoConvergence`, `Breakdown`.

```rust
use azmath::linalg::{Matrix, Vector};

// 2x + y = 5,  x + 3y = 10
let a = Matrix::new([[2.0f64, 1.0], [1.0, 3.0]]);
let b = Vector::new([5.0, 10.0]);
let x = a.solve(&b).unwrap(); // x = 1, y = 3
```

Der LU-Löser verwendet nur `+ - * /`, `abs` und `max`. Diese sind in allen Backends exakt, die
Näherungen von `micromath` wirken sich hier also nicht aus. **Cholesky und QR benötigen dagegen
`sqrt`**: mit `micromath` (bewusst schnell, aber ungenau) können die Ergebnisse deutlich abweichen
(`sqrt(0)` ist dort z. B. nur ≈ 8e-20). Für genaue Werte `std` oder `libm` verwenden.

## Nullstellen

Das Modul `azmath::roots` löst `f(x) = 0` für Funktionen einer Veränderlichen. Die Funktionen
werden als Closures übergeben, alle Verfahren sind generisch über `Real` und brauchen keinen
Allokator.

| Funktion | Braucht | Eigenschaft |
|---|---|---|
| `brent` | Intervall mit Vorzeichenwechsel | schnell **und** sicher, die erste Wahl |
| `bisection` | Intervall mit Vorzeichenwechsel | langsam, aber absolut sicher |
| `newton` | Funktion, Ableitung, Startwert | quadratisch konvergent, kann divergieren |
| `secant` | zwei Startwerte | ohne Ableitung, überlinear konvergent |

Abbruch bei Intervallbreite bzw. Schritt `≤ tol · max(1, |x|)` (`RootOptions`, Standard: `8·ε`
und 500 Iterationen). Das Ergebnis (`RootResult`) enthält Nullstelle, `f(x)` und Iterationszahl;
Fehler (`RootError`, `#[non_exhaustive]`): `NotBracketed`, `NoConvergence`, `NotFinite`,
`ZeroDerivative`.

```rust
use azmath::roots::{brent, RootOptions};
use azmath::Real;

// Lösung von cos(x) = x
let r = brent(|x: f64| Real::cos(x) - x, 0.0, 1.0, RootOptions::default()).unwrap();
assert!((r.root - 0.7390851332151607).abs() < 1e-14);
```

Hinweis: Funktionen wie `cos` in der Closure am besten als `Real::cos(x)` aufrufen. Bei einem
`f64` würde `x.cos()` sonst die Methode der Standardbibliothek statt des gewählten Backends nehmen.

## Numerische Integration

Das Modul `azmath::integrate` berechnet `∫ₐᵇ f(x) dx`. Die Funktion wird als Closure übergeben,
alle Verfahren sind generisch über `Real` und brauchen keinen Allokator.

| Funktion | Art | Eigenschaft |
|---|---|---|
| `adaptive` | adaptiv | Gauß-Kronrod (G7/K15) mit Fehlerschätzung, die erste Wahl |
| `gauss_legendre` | feste Regel | 1 bis 5 Punkte pro Teilintervall, exakt bis Grad `2k−1` |
| `simpson` | feste Regel | exakt bis Grad 3, Fehler ∝ h⁴ |
| `trapezoid` | feste Regel | exakt für Geraden, Fehler ∝ h² |
| `trapezoid_samples`, `simpson_samples`, `trapezoid_xy` | Messdaten | Integral aus Stützwerten (gleichmäßig bzw. beliebig verteilt) |

`adaptive` teilt jeweils das Intervall mit dem größten Fehler, bis die Gesamtabweichung unter
`max(abs_tol, rel_tol · |I|)` liegt (`IntegrateOptions`, Standard: `ε` und `ε^0,75`). Es wertet die
Funktion nicht an den Intervallenden aus, daher sind Randsingularitäten wie `√x`, `ln x` oder `1/√x`
bei 0 erlaubt. Das Ergebnis (`IntegralResult`) enthält Wert, Fehlerschätzung und Anzahl der
Auswertungen; Fehler (`IntegrateError`, `#[non_exhaustive]`): `InvalidInput`, `NotFinite`,
`NoConvergence`.

```rust
use azmath::integrate::{adaptive, IntegrateOptions};
use azmath::Real;

// ∫₀^π sin(x) dx = 2
let r = adaptive(|x: f64| Real::sin(x), 0.0, core::f64::consts::PI, IntegrateOptions::default())
    .unwrap();
assert!((r.value - 2.0).abs() < 1e-13);
```

Grenzen: Unendliche Integrationsgrenzen werden noch nicht unterstützt. Die Fehlerschätzung ist
bei glatten Funktionen vorsichtig, bei Randsingularitäten nur ein Richtwert. Singularitäten im
Inneren des Intervalls begrenzen die erreichbare Genauigkeit; dann das Intervall an der
Singularität teilen.

## Nichtlineare Gleichungen und Ausgleichsrechnung

Das Modul `azmath::optimize` löst nichtlineare Gleichungssysteme und nichtlineare
Ausgleichsprobleme `min ‖r(x)‖₂`.

| Funktion | Problem | Verfahren |
|---|---|---|
| `newton_system` | `F(x) = 0`, `n` Gleichungen und Unbekannte | Newton mit Schrittweitensteuerung (QR) |
| `gauss_newton` | `min ‖r(x)‖` mit `m ≥ n` Residuen | Gauß-Newton (QR) mit Schrittweitensteuerung |
| `levenberg_marquardt` | `min ‖r(x)‖` | gedämpft, robust bei schlechtem Startwert und Rangmangel |

Zu jedem gibt es eine Variante `*_fd` mit Jacobi-Matrix aus Vorwärtsdifferenzen
(`numerical_jacobian`). Die Funktionen arbeiten **auf Slices** und brauchen keinen Allokator: Der
Arbeitsspeicher wird übergeben (`*_work_len`, als `const fn` auch für Arrays nutzbar). Dadurch
passen sie zu festen Größen wie zu zur Laufzeit bekannten Datenmengen (Kurvenanpassung).
`x` ist Startwert und Ergebnis; das Ergebnis (`OptimizeResult`) nennt Iterationen, Residuumsnorm und
den Abbruchgrund (`Termination`: `Residual`, `Step`, `Gradient`, `Cost`). Fehler
(`OptimizeError`, `#[non_exhaustive]`): `InvalidInput`, `NotFinite`, `SingularJacobian`,
`NoConvergence`.

```rust
use azmath::optimize::{levenberg_marquardt_fd, levenberg_marquardt_work_len, OptimizeOptions};
use azmath::Real;

// Anpassung y = a·exp(b·t) an exakte Daten (a = 2, b = −1)
let t = [0.0, 0.5, 1.0, 1.5, 2.0];
let y = t.map(|t| 2.0 * Real::exp(-t));
let residual = |p: &[f64], r: &mut [f64]| {
    for i in 0..5 {
        r[i] = p[0] * Real::exp(p[1] * t[i]) - y[i];
    }
};
let mut p = [1.0, 0.0];
let mut work = [0.0; levenberg_marquardt_work_len(5, 2)];
levenberg_marquardt_fd(residual, &mut p, 5, OptimizeOptions::default(), &mut work).unwrap();
assert!((p[0] - 2.0).abs() < 1e-7 && (p[1] + 1.0).abs() < 1e-7);
```

Hinweise: Alle Verfahren sind **lokal**; sie finden eine Lösung nahe am Startwert (bei Plateaus oder
mehreren Lösungen nicht zwingend die beste). Levenberg-Marquardt löst die gedämpften
Normalengleichungen per Cholesky (die Konditionszahl wird quadriert), Gauß-Newton und Newton
verwenden QR.

## Differentialgleichungen

Das Modul `azmath::ode` löst Anfangswertprobleme `y' = f(t, y)`, `y(t₀) = y₀`, auch Systeme und
rückwärts in der Zeit. Wie bei `optimize` arbeiten die Funktionen auf Slices ohne Allokator: Der
Zustand wird in-place fortgeschrieben, der Arbeitsspeicher übergeben (`*_work_len` als `const fn`).

| Funktion | Art | Ordnung | Geeignet für |
|---|---|---|---|
| `rk45` | explizit, **adaptiv** (Dormand-Prince) | 5(4) | die erste Wahl für nicht steife Probleme |
| `rk4`, `rk4_step` | explizit, feste Schrittweite | 4 | glatte Probleme mit bekannter Schrittweite |
| `euler` | explizit, feste Schrittweite | 1 | Demonstrationen |
| `implicit_euler` | implizit (Newton), feste Schrittweite | 1 | **steife** Probleme, L-stabil |
| `implicit_trapezoid` | implizit (Newton), feste Schrittweite | 2 | steife Probleme, genauer, aber nur A-stabil |

Ein **Observer** `|t, y| …` wird mit dem Anfangszustand und nach jedem (akzeptierten) Schritt
aufgerufen, um den Verlauf aufzuzeichnen, ohne dass das Verfahren Speicher belegt. `rk45` steuert die
Schrittweite über `Rk45Options` (`rtol`, `atol`, `h_init`, `h_max`, `max_steps`); die impliziten Verfahren
lösen je Schritt mit `optimize::newton_system` (analytische Jacobi-Matrix oder `*_fd`). Fehler
(`OdeError`, `#[non_exhaustive]`): `InvalidInput`, `NotFinite`, `StepSizeTooSmall`,
`MaxStepsExceeded`, `NewtonFailed`.

```rust
use azmath::ode::{rk45, rk45_work_len, Rk45Options};

// Harmonischer Oszillator y'' = −y als System (y, v)' = (v, −y), y(0) = 0, v(0) = 1
let f = |_t: f64, y: &[f64], dy: &mut [f64]| {
    dy[0] = y[1];
    dy[1] = -y[0];
};
let mut y = [0.0, 1.0];
let mut work = [0.0; rk45_work_len(2)];
rk45(f, 0.0, core::f64::consts::PI / 2.0, &mut y, Rk45Options::default(), &mut work, |_, _| {})
    .unwrap();
assert!((y[0] - 1.0).abs() < 1e-5 && y[1].abs() < 1e-5);
```

Steife Probleme (sehr schnell abklingende Anteile) zwingen explizite Verfahren zu winzigen Schritten;
dort sind die impliziten Verfahren im Vorteil. Die Trapezregel dämpft steife Einschwingvorgänge nur
schwach (sie schwingt nach), das implizite Euler-Verfahren dagegen sofort. Nicht enthalten sind
bisher Verfahren höherer Ordnung für steife Probleme (BDF, Radau) und Ereignissuche.

## Interpolation

Das Modul `azmath::interp` verbindet Stützwerte `(xᵢ, yᵢ)` (mit **streng aufsteigendem** `x`) durch
Funktionen. Alle Verfahren arbeiten auf Slices ohne Allokator: `CubicSpline` und `Pchip` **leihen** sich
die Daten und legen ihre Koeffizienten in einem übergebenen Slice ab; mit `alloc` gibt es die
besitzenden Varianten `CubicSplineBuf` und `PchipBuf`.

| Typ / Funktion | Eigenschaft |
|---|---|
| `linear`, `locate` | stückweise linear, die sicherste Wahl; `locate` findet das Teilintervall (binäre Suche) |
| `CubicSpline` | zweimal stetig differenzierbar; Randbedingungen `Natural`, `Clamped`, `NotAKnot`, `Periodic` |
| `Pchip` | stückweise kubisch und **monoton** (kein Überschwingen), nur einmal stetig differenzierbar |
| `lagrange` | ein Polynom durch alle Punkte (Neville); nur für wenige, gut verteilte Punkte |

Splines liefern `value`, `derivative`, `second_derivative` und das exakte `integral`. Außerhalb der
Stützstellen wird extrapoliert (das letzte Polynom fortgesetzt; `contains` prüft den Bereich). Fehler
(`InterpError`, `#[non_exhaustive]`): `InvalidInput`, `NotIncreasing`, `NotFinite`, `NotPeriodic`,
`Singular`.

```rust
use azmath::interp::{cubic_spline_work_len, BoundaryCondition, CubicSpline};

let x = [0.0f64, 1.0, 2.0, 3.0, 4.0];
let y = [0.0f64, 1.0, 4.0, 9.0, 16.0]; // y = x²
let (mut m, mut work) = ([0.0f64; 5], [0.0f64; cubic_spline_work_len(5)]);
let s = CubicSpline::new(&x, &y, BoundaryCondition::NotAKnot, &mut m, &mut work).unwrap();
assert!((s.value(2.5) - 6.25).abs() < 1e-12);
assert!((s.integral(0.0, 4.0) - 64.0 / 3.0).abs() < 1e-12);
```

Hinweise: Mit der Randbedingung hängt die Genauigkeit am Rand zusammen. Not-a-knot reproduziert
Polynome bis Grad 3 und ist ein guter Standard, wenn man den Rand nicht kennt. Eingespannt ist am
genauesten, wenn die Steigungen bekannt sind; natürlich passt, wenn die Krümmung am Rand 0 ist. Alle
drei konvergieren asymptotisch mit `h⁴` (bei groben Gittern kann not-a-knot wegen des Randfehlers
besser oder schlechter als die anderen sein). Ein Polynom durch viele gleichmäßige Punkte zeigt das
Runge-Phänomen (Beispiel: 1/(1+25x²) mit 11 Punkten, Fehler 1,9 bei Spline 0,001).

## Multigrid

Das Modul `azmath::multigrid` löst die Poisson-Gleichung `−Δu = f` mit Dirichlet-Rand auf dem Einheitsintervall
(1D) bzw. Einheitsquadrat (2D, 5-Punkte-Stern) mit **geometrischem Multigrid**: Rechenaufwand `O(N)` und eine
von der Gittergröße **unabhängige** Konvergenzrate. Gemessen bei `−Δu = 1` mit Gauß-Seidel V(2,2) auf Gittern
von `7×7` bis `127×127`: durchgehend 12 Zyklen bis `1e-12`, während CG von 9 auf 264 Iterationen wächst.

| Funktion / Typ | Aufgabe |
|---|---|
| `solve_1d`, `solve_2d` | Löser bis zur relativen Residuumstoleranz (`SolveOptions`) |
| `cycle_1d`, `cycle_2d` | ein einzelner V- oder W-Zyklus |
| `MultigridPreconditioner` | symmetrischer Zyklus als Vorkonditionierer für `pcg`; konstant etwa 7 Iterationen |
| `Poisson1d`, `Poisson2d` | der Operator als `LinearOperator` (für CG, Eigenwerte, …) |
| `solve_1d_vec`, `solve_2d_vec` | wie die Löser, mit eigener Speicherverwaltung (`alloc`) |

Bausteine: Glätter `GaussSeidel` (vorwärts vor, rückwärts nach der Grobgitterkorrektur, damit der Zyklus
symmetrisch bleibt) oder `WeightedJacobi(ω)` (1D: `2/3`, 2D: `4/5`), Restriktion per Full Weighting, lineare bzw.
bilineare Prolongation, exakte Lösung auf dem gröbsten Gitter, `CycleType::V` oder `W`
(`MultigridOptions`). Die Gittergröße muss `n = 2ᵏ − 1` sein (`grid_levels`); der Arbeitsspeicher kommt als Slice
(`work_len_1d`, `work_len_2d`, `const fn`). Fehler (`MultigridError`, `#[non_exhaustive]`): `InvalidGridSize`,
`InvalidInput`, `NotFinite`, `NoConvergence`.

```rust
use azmath::multigrid::{solve_1d, work_len_1d, MultigridOptions, SolveOptions};

// −u'' = π²·sin(πx) auf (0, 1) hat die Lösung sin(πx)
const N: usize = 63;
let h = 1.0 / (N as f64 + 1.0);
let pi = core::f64::consts::PI;
let f: [f64; N] = core::array::from_fn(|i| pi * pi * (pi * (i + 1) as f64 * h).sin());
let mut u = [0.0; N];
let mut work = [0.0; work_len_1d(N)];
let r = solve_1d(&f, &mut u, N, MultigridOptions::default(), SolveOptions::default(), &mut work).unwrap();
assert!(r.cycles <= 15 && (u[N / 2] - 1.0).abs() < 1e-3);
```

Hinweise: Das Verfahren ist auf den Poisson-Operator mit konstanten Koeffizienten zugeschnitten (kein
algebraisches Multigrid, keine variablen Koeffizienten, keine anisotropen Gitter, keine anderen Randbedingungen).
Die Diskretisierung selbst ist von zweiter Ordnung: der Fehler zur kontinuierlichen Lösung fällt mit `h²`
(Faktor 4 beim Halbieren der Maschenweite), unabhängig davon, wie genau das Gleichungssystem gelöst wird.
Als Vorkonditionierer muss `pre_smooth == post_smooth` gelten, sonst ist der Zyklus nicht symmetrisch.

## Beispiele

Die Beispiele liegen in [`examples/`](examples) und laufen standardmäßig mit `std`:

```bash
cargo run --example 01_generic_functions
cargo run --example 03_f64_ops
cargo run --example 04_backends --no-default-features --features libm
cargo run --example 06_linear_system --features alloc
cargo run --example 07_least_squares
cargo run --example 08_conjugate_gradient
cargo run --example 09_roots
cargo run --example 10_bicgstab
cargo run --example 11_integrate
cargo run --example 12_curve_fit
cargo run --example 13_ode
cargo run --example 14_gmres
cargo run --example 15_interpolation
cargo run --example 16_eigen_svd
cargo run --release --example 17_multigrid
```

| Beispiel                    | Inhalt                                              |
|-----------------------------|-----------------------------------------------------|
| `01_generic_functions`      | Generische Funktionen für `f32` und `f64`           |
| `02_f32_ops`                | Arithmetik, Rundung, Trigonometrie mit `f32`        |
| `03_f64_ops`                | Quadratische Gleichung, `exp`, `ln`, `powf` in `f64`|
| `04_backends`               | Anzeige und Vergleich der Backends                  |
| `05_convert`                | `from_f64` und Einheitenumrechnung                  |
| `06_linear_system`          | LU-Löser, Determinante, Inverse (mit `alloc`: dynamisch) |
| `07_least_squares`          | Ausgleichsgerade und Polynomfit mit QR, Cholesky       |
| `08_conjugate_gradient`     | CG matrixfrei (Poisson) und Jacobi-Vorkonditionierer   |
| `09_roots`                  | Nullstellen: Bisektion, Brent, Newton, Sekante, Kepler |
| `10_bicgstab`               | BiCGSTAB für nichtsymmetrische Systeme                 |
| `11_integrate`              | Integration: feste Regeln, adaptiv, Messdaten          |
| `12_curve_fit`              | Kurvenanpassung (LM, Gauß-Newton) und Newton-System    |
| `13_ode`                    | ODE: RK45, RK4, Euler, steifes Problem mit implizitem Euler |
| `14_gmres`                  | GMRES(m): schiefsymmetrisch, Neustart-Kosten, Vorkonditionierer |
| `15_interpolation`          | Interpolation: Runge-Phänomen, Randbedingungen, PCHIP, Integral |
| `16_eigen_svd`              | Eigenwerte (Jacobi, Potenzmethode), SVD, PCA, Rangmangel |
| `17_multigrid`              | Multigrid: gitterunabhängige Konvergenz, Vorkonditionierer, O(h²) |
| `18_autodiff`               | Autodiff: exakte Jacobi-Matrix für Levenberg-Marquardt, Hessematrix |

## Tests

Die Integrationstests liegen in `tests/` (Präfix = Modul):

```bash
cargo test                                              # std
cargo test --no-default-features --features libm       # libm
cargo test --features alloc                            # inkl. dynamische Matrizen
cargo test --test linalg_qr                            # nur eine Testdatei
```

Die Doctests laufen mit `cargo test --doc`.

**Die Tests laufen nur mit den genauen Backends `std` und `libm`.** `micromath` rechnet bewusst
mit Bit-Tricks und ist ungenau; Fehler akkumulieren sich in längeren Rechnungen, sodass Ergebnisse
dort nur zufällig stimmen und keine sinnvollen Zahlenwerte geprüft werden können. Mit
`micromath` allein werden Bibliothek, Beispiele und Tests daher nur **gebaut**, nicht ausgeführt.

Die Continuous Integration ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) baut alle
Feature-Kombinationen, testet sie (außer `micromath` allein) und prüft `cargo fmt`,
`cargo clippy` und `cargo doc`.

## Hinweise

- **`micromath` ist eine bewusste Entscheidung für Geschwindigkeit vor Genauigkeit.** Es arbeitet
  mit Bit-Tricks und liefert Näherungen mit Fehlern im Prozentbereich. Das Crate verbessert diese
  Ergebnisse absichtlich nicht; wer genaue Werte braucht, nimmt `std` oder `libm`.

- Beispiel: `sqrt(0.7)` ergibt mit `micromath` ≈ 0.85 statt 0.8367.
- `micromath` 2.1.0 bricht bei `ln` negativer Werte in Debug-Builds mit einem
  Integer-Überlauf ab (Upstream-Verhalten).
- Edition 2024, benötigt eine entsprechend aktuelle Rust-Version.

## Roadmap

Was bereits umgesetzt ist, was noch fehlt (Optimierung, steife ODE, FEM/FVM/SPH, Navier-Stokes, Autodiff, Rotationen,
Polynom-Nullstellen, SAT/SMT u. a.), wovon es abhängt und wie das Projekt gegliedert werden soll,
steht in [`ROADMAP.md`](ROADMAP.md).

## Lizenz

Dieses Projekt steht wahlweise unter einer der folgenden Lizenzen:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT-Lizenz](LICENSE-MIT)

nach Wahl der Nutzenden.
