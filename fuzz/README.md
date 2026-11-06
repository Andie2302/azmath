# Spezialfunktionen mit libFuzzer prüfen

Benötigt Nightly und `cargo install cargo-fuzz --locked`:

```sh
cargo +nightly fuzz run special -- -max_total_time=60 -timeout=5 -max_len=64
```

Der Target prüft f32/f64-Invarianten in einem beschränkten numerischen Bereich
und rohe f64-Bitmuster für kleine Argumente und nichtendliche Eingaben.
Ordnungen und Bessel-Argumente sind begrenzt, damit ein einzelner Fall nicht
unverhältnismäßig lange rechnet. Dies ist kein Nachweis für sämtliche
Spezialfunktionen oder unbeschränkte Argumente. Fehler landen in `artifacts/`;
einen gefundenen Fehler zusätzlich als dauerhaften Integrationstest aufnehmen.

In Containern mit gesperrtem `ptrace` kann LeakSanitizer beim Prozessende
scheitern. Für diesen Fall denselben Befehl mit `ASAN_OPTIONS=detect_leaks=0`
starten. AddressSanitizer bleibt aktiv; Leak-Prüfung ist dann nicht enthalten.

## Weitere Targets

```sh
cargo +nightly fuzz run fft    -- -max_total_time=60 -timeout=5 -max_len=512
cargo +nightly fuzz run sparse -- -max_total_time=60 -timeout=5 -max_len=512
cargo +nightly fuzz run qp     -- -max_total_time=60 -timeout=5 -max_len=512
```

- `fft`: beliebige Längen 0–257; Roundtrip, Parseval, reelle FFT, Faltung gegen direkte Summe,
  Fehler bei zu kurzem Zwischenspeicher.
- `sparse`: COO→CSR mit ungültigen Indizes und nichtendlichen Werten, Umordnungen, LDLᵀ/Cholesky/LU
  dürfen Fehler melden, aber nie paniken; diagonaldominante Matrizen müssen genau gelöst werden.
- `qp`: `QpWorkspace` und `interior_point_qp_sparse` mit extremen Werten; bei `Optimal` müssen
  `A·x = b` und `G·x ≤ h` (relativ zur Summe der Termbeträge) erfüllt sein.

Gemeinsame Hilfen stehen in `fuzz_targets/common.rs`. Gefundene Fehler als Integrationstest festhalten.
