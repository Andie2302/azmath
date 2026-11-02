# azmath

Ein kleines `no_std`-Crate, das **ein einziges Trait `Real`** für `f32` und `f64` bereitstellt.
Mathematische Funktionen (`sqrt`, `sin`, `exp`, …) werden je nach Feature-Flag an ein
Backend weitergeleitet. So lässt sich numerischer Code einmal generisch schreiben und auf
Desktop wie auf Mikrocontrollern verwenden.

## Features / Backends

| Feature     | Backend      | `f32` | `f64` | Hinweis                                           |
|-------------|--------------|:-----:|:-----:|---------------------------------------------------|
| `std`       | Standardbib. |  ✅   |  ✅   | **Default**, genaueste Ergebnisse                 |
| `libm`      | [`libm`]     |  ✅   |  ✅   | `no_std`, genau                                   |
| `micromath` | [`micromath`]|  ✅   |  ❌   | `no_std`, schnell, aber nur eine Näherung         |

Sind mehrere Features aktiv, gilt die Priorität **`std` → `libm` → `micromath`**.
Ohne jedes Feature stehen keine `Real`-Implementierungen zur Verfügung.

Für `no_std` muss das Default-Feature abgeschaltet werden:

```toml
[dependencies]
azmath = { version = "0.1", default-features = false, features = ["libm"] }
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

## Beispiele

Die Beispiele liegen in [`examples/`](examples) und laufen standardmäßig mit `std`:

```bash
cargo run --example 01_generic_functions
cargo run --example 03_f64_ops
cargo run --example 04_backends --no-default-features --features libm
```

| Beispiel                    | Inhalt                                              |
|-----------------------------|-----------------------------------------------------|
| `01_generic_functions`      | Generische Funktionen für `f32` und `f64`           |
| `02_f32_ops`                | Arithmetik, Rundung, Trigonometrie mit `f32`        |
| `03_f64_ops`                | Quadratische Gleichung, `exp`, `ln`, `powf` in `f64`|
| `04_backends`               | Anzeige und Vergleich der Backends                  |
| `05_convert`                | `from_f64` und Einheitenumrechnung                  |

## Tests

Die Integrationstests liegen in [`tests/`](tests):

```bash
cargo test                                              # std
cargo test --no-default-features --features libm        # libm
cargo test --no-default-features --features micromath   # micromath (nur f32)
```

Zusätzlich laufen die Doctests mit `cargo test --doc`. Die Continuous Integration
([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) prüft alle Feature-Kombinationen,
`cargo fmt` und `cargo clippy`.

## Hinweise

- **`micromath` ist eine bewusste Entscheidung für Geschwindigkeit vor Genauigkeit.** Es arbeitet
  mit Bit-Tricks und liefert Näherungen mit Fehlern im Prozentbereich. Das Crate verbessert diese
  Ergebnisse absichtlich nicht; wer genaue Werte braucht, nimmt `std` oder `libm`.

- Beispiel: `sqrt(0.7)` ergibt mit `micromath` ≈ 0.85 statt 0.8367; die Tests verwenden dafür eine
  größere Toleranz.
- `micromath` 2.1.0 bricht bei `ln` negativer Werte in Debug-Builds mit einem
  Integer-Überlauf ab (Upstream-Verhalten).
- Edition 2024, benötigt eine entsprechend aktuelle Rust-Version.

## Lizenz

Dieses Projekt steht wahlweise unter einer der folgenden Lizenzen:

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT-Lizenz](LICENSE-MIT)

nach Wahl der Nutzenden.
