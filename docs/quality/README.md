# Numerische Qualität prüfen

Die normalen Integrationstests enthalten eingecheckte Referenzwerte und brauchen
keine zusätzlichen Python- oder Rust-Abhängigkeiten. Für Spezialfunktionen:

```sh
cargo test --test special_reference --test special_properties -- --nocapture
cargo test --no-default-features --features libm --test special_reference --test special_properties -- --nocapture
```

`special_reference` gibt den größten beobachteten ULP-Abstand je Funktion und
Datentyp aus. `special_properties` prüft mit festem Seed jeweils 512 erzeugte
f32/f64-Eingaben gegen Gamma-Rekursion, Fehlerfunktions-Symmetrie,
Komplementarität der unvollständigen Gammafunktionen und Bessel-Rekursionen.
Das prüft einen begrenzten Bereich, keine globale Genauigkeitsgarantie.

Referenzdaten neu erzeugen (mpmath 1.3.0):

```sh
python scripts/generate_special_refs.py
cargo fmt --all --check
```

Der Generator rundet Eingaben zuerst auf den jeweiligen IEEE-Typ. Anschließend
berechnet er den Sollwert mit 100 und 200 Dezimalstellen und verlangt identische,
direkt auf den Zieltyp gerundete Ergebnisse. Die Tabellen speichern Bitmuster,
damit Rusts Dezimalparser nicht Teil des Vergleichs wird. ULP bedeutet hier den
Abstand in darstellbaren IEEE-Werten; für `ln_gamma` nahe seinen Nullstellen wird
stattdessen ein absoluter Fehler von höchstens `8 * EPSILON` geprüft. Unendliche
Sollwerte müssen exakt übereinstimmen. Die funktionsabhängigen ULP-Budgets stehen
im Test; die Algorithmen versprechen keine korrekte Rundung auf ein ULP.

Coverage reproduzieren:

```sh
rustup component add llvm-tools-preview
cargo install cargo-llvm-cov --version 0.9.1 --locked
cargo llvm-cov --all-features --json --output-path coverage.json
```

CI lädt `coverage.json` als Artefakt hoch. Die Messung umfasst die Bibliotheksquellen
mit allen Features (dabei hat `std` Vorrang); sie ersetzt weder den separaten
libm-Testlauf noch micromath-Smoke-Tests. Stable liefert hier Zeilen-, Funktions-
und Regionenabdeckung, keine Branch- oder MC/DC-Abdeckung. Noch gibt es keinen
Coverage-Schwellenwert: Erst Messwerte und relevante Lücken stabilisieren.

Für laufzeitbegrenztes libFuzzer-Fuzzing siehe [fuzz/README.md](../../fuzz/README.md).

Weitere Messung und Regressionen: [Lanczos und Fuzzer-Prüfungen](lanczos-fuzz.md).
