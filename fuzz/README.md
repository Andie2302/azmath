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
