# Lanczos und Fuzzer-Prüfungen (2026-11-06)

Basis ist master `70be6eb`, einschließlich der sieben zuvor übernommenen
Commits mit FFT-/Sparse-/QP-Tests und weiteren Fuzz-Targets.

## Lanczos

14 Tests prüfen unabhängig für f32 und f64:

- Orthonormalität und Projektion `Qᵀ A Q = T` sowie die Rekonstruktion
  `A Q = Q T + r eₘᵀ` bei abgeschnittener Zerlegung.
- Ritz-Paare: Rekonstruktion des Ritz-Vektors im ursprünglichen Raum und
  Vergleich von `‖Ay − θy‖` mit der gemeldeten Residualabschätzung.
- Bekanntes vollständiges Diagonalspektrum, invarianter Unterraum und Nulloperator.
- Große/kleine Startvektoren, Restvektoren und Tridiagonalmatrizen
  (f32: 1e±30, f64: 1e±200).
- Leere/singleton Ritz-Probleme, zu kurze Puffer, Nullstart, NaN/Inf und
  nicht darstellbare Ritz-Werte.

Dabei behoben: Quadratsummen für Start- und Restnorm konnten über- oder
unterlaufen. Lanczos verwendet jetzt die skalierte Norm. Die Ritz-Auswertung
validiert auch die letzte Restnorm und skaliert die Tridiagonalmatrix vor dem
Jacobi-Verfahren. Ein überlaufender Ergebniswert liefert `NotFinite`.
Die Abbruchskala wird ohne Addition zweier potenziell großer Beträge gebildet.

Zeilenabdeckung von `src/linalg/eigen/lanczos.rs`: vorher 64/176 (36,36 %),
jetzt 180/188 (95.74 %).

Validiert: 1.093 Unit-/Integrationstests mit allen Features, 92 zusätzliche
std-Doctests, 1.185 Tests inklusive Doctests mit libm+alloc, beide Fuzzer-Harness-
Tests mit Stable und Nightly, Embedded-Build mit libm+alloc, Clippy (Crate und
Fuzz-Paket), Rustdoc mit Warnungen als Fehlern und Formatierung.

Messung: Rust 1.99.1, cargo-llvm-cov 0.9.1, `--all-features` (std-Backend).
Die Datei enthält auch Arnoldi; dessen bestehende Tests gehören zur Messung.
Das ist Zeilenabdeckung, keine Garantie numerischer Genauigkeit oder Branch-Coverage.
Die alte Baseline steht in [coverage-special.md](coverage-special.md).

## Fuzz-Targets

`Bytes::range(0, 257)` liest jetzt zwei Bytes statt einem; Längen 256 und 257
sind erreichbar. Kleine Bereiche behalten ihre bisherige Corpus-Interpretation.
Der QP-Prüfer ergänzt endliche Multiplikatoren, Dualzulässigkeit, Stationarität
und Komplementarität. Skalierte Produkte vermeiden Zwischenüberläufe; Toleranzen
berücksichtigen das normweise Abbruchkriterium und den internen Schlupf.
Extreme Größenordnungen bleiben eine begrenzte Prüfung auf notwendige Bedingungen.

Zwei separate Tests prüfen den Generator an den FFT-Randlängen und den
QP-Prüfer an bekannten Lösungen, einschließlich absichtlich falscher,
aber zulässiger oder stationärer Ergebnisse. Diese Tests laufen auch in CI.
Anleitung: [fuzz/README.md](../../fuzz/README.md).

| Target | Durchläufe | Dauer | Ergebnis |
|---|---:|---:|---|
| qp | 8,192 | 61 s | Kein Fund |
| fft | 7,548 | 61 s | Kein Fund |
| qp_extreme | 7,079 | 61 s | Kein Fund |

Die Läufe verwendeten Nightly mit AddressSanitizer. LeakSanitizer war wegen
der ptrace-Beschränkung dieser Umgebung deaktiviert. Diese kurzen Läufe
ersetzen nicht den vom Anwender gemeldeten einstündigen Lauf des vorherigen
Targets (3.411.026 Durchläufe ohne Fund); der galt für dessen damalige Prüfungen.

Nächster sinnvoller Schwerpunkt: Referenztests für Verteilungsfunktionen und
Quantile in `stats::dist`, besonders kleine Schwanzwahrscheinlichkeiten,
f32-Grenzen und Rundungsfehler nahe 0 und 1.
