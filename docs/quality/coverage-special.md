# Coverage- und Genauigkeitsbaseline (2026-11-06)

Ausgangsstand: `c990a6c` auf master; anschließend Referenz- und Invariantentests
sowie die unten beschriebenen Korrekturen. Rust 1.99.1, cargo-llvm-cov 0.9.1,
`--all-features`, LLVM-Instrumentierung, Bibliotheksquellen ohne Testcode.
Aufruf und Reproduktion stehen in [README.md](README.md). Die Coverage-Messung
führt Unit- und Integrationstests aus; Doctests sind nicht instrumentiert.
Zusätzlich separat erfolgreich geprüft: 92 std-Doctests, 1.148 Tests inklusive
Doctests mit libm+alloc, drei micromath-Smoke-Tests, Embedded-Build mit
libm+alloc, Clippy, Formatierung und Rustdoc mit Warnungen als Fehlern.

| Bereich | Vorher, Zeilen | Nachher, Zeilen |
|---|---:|---:|
| Bibliothek | 18645/20095 (92.78 %) | 18706/20163 (92.77 %) |
| special | 410/428 (95.79 %) | 471/496 (94.96 %) |

Schon der Ausgangsstand hatte alle 26 Funktionen von `special` mindestens einmal
ausgeführt. Trotzdem scheiterten neue Genauigkeits- und Randfallprüfungen.
Coverage beweist daher weder die Genauigkeit noch die Robustheit eines Verfahrens.
Die Prozentwerte vergleichen unterschiedliche Quelltextumfänge, weil die
Korrekturen zusätzliche Fallbehandlungen enthalten.

## Referenzgenauigkeit

356 eingecheckte Fälle für Gamma, Log-Gamma, erf/erfc und J/I/Y/K, unabhängig für
f32 und f64 gerundet. Sollwerte aus mpmath 1.3.0, bei 100 und 200 Dezimalstellen
identisch auf den Zieltyp gerundet. Bereiche: Gamma bis 170,5 und negative
Halbzahlargumente bis −171,5; Nähe negativer Pole; erf/erfc bis Betrag 20 und
kleine Werte bis 1e−200; Bessel-Ordnungen 0, 1, 2, 5, 10 bei 0,001 bis 20 sowie
kleine J-Argumente. Zusätzlich der durch Fuzzing gefundene Y₃-f32-Fall bei
0,23702975. Nicht jede Kombination gehört zur Tabelle.

| Funktion | f32, max. ULP | f64, max. ULP |
|---|---:|---:|
| bessel_i | 2 | 3 |
| bessel_j | 12 | 21 |
| bessel_k | 36 | 37 |
| bessel_y | 10 | 6 |
| erf | 5 | 5 |
| erfc | 21 | 116 |
| gamma | 11 | 725 |
| ln_gamma | 46 | 17 |

Das sind beobachtete Maxima über std und libm an diesen Punkten, keine globalen Fehlergrenzen.
Die ULP-Budgets im Test erlauben Unterschiede zwischen std und libm. Nahe den
Nullstellen von `ln_gamma` gilt die dokumentierte absolute Fehlerschranke.
Die verbleibende Gamma-Abweichung zeigt, dass Lanczos und die logarithmische
Reflexion keine Garantie auf korrekte Rundung geben.

## Gefundene und behobene Fehler

- Gamma/Log-Gamma für +∞ sowie f32-Gamma bei Überlauf lieferten NaN.
- Die Gamma-Reflexion verlor nahe negativen ganzzahligen Polen Genauigkeit
  durch `sin(pi*x)`. Jetzt wird das Argument vor der Multiplikation reduziert.
  Ein überlaufendes positives Gamma konnte außerdem eine noch darstellbare
  negative-Argument-Lösung zu null machen; dafür gibt es einen logarithmischen Pfad.
- Sehr kleine erf-Argumente gingen beim Quadrieren verloren. Eine lokale
  Potenzreihe erhält diese Werte.
- Miller-Rekursion für J versagte bei sehr kleinen Argumenten; bei kleinen
  J₀-Argumenten war die Startordnung zu niedrig. Für Betrag kleiner eins wird
  jetzt die konvergente Potenzreihe verwendet.
- Die Quadratur für Y war bei größeren Argumenten zu grob. Die f32-Toleranz
  war unerreichbar; ein erster Korrekturversuch blieb unter der Rundungsgrenze
  von `adaptive` (50*EPSILON) und scheiterte am Fuzz-Fund. Die Toleranz
  berücksichtigt jetzt diese Grenze. +∞ wird vor der Panelberechnung behandelt.
- Die K- und Y-Integranden liefen bei f32 in Zwischenüberläufe, obwohl das
  Ergebnis darstellbar war. Exponenten werden kombiniert und Gewichte bzw.
  Normierung vor der Exponentialfunktion angewendet.

## Fuzzing und weitere Prioritäten

20.000 libFuzzer-Durchläufe des finalen Targets mit Nightly
1.101.0-nightly (2026-11-05), cargo-fuzz 0.13.2, AddressSanitizer und begrenzten
Argumenten/Ordnungen. Ohne gefundenen Fehler abgeschlossen. LeakSanitizer musste
wegen der ptrace-Beschränkung dieser Umgebung deaktiviert werden; Speicherlecks
wurden damit nicht geprüft. Das Target prüft rohe f64-Bitmuster für kleine und
nichtendliche Argumente, f32/f64-Invarianten und endliche Bessel-Ergebnisse.

Die größte einzelne Zeilenlücke der Ausgangsmessung liegt in
`src/linalg/eigen/lanczos.rs`: 112 nicht ausgeführte Zeilen, 36,36 % Abdeckung.
Danach folgen `stats/dist.rs` (70) und `roots/newton.rs` (56). Sinnvolle nächste
Schritte sind Lanczos-Invarianten einschließlich Fehlerfällen für f32/f64 und
Referenztests für Verteilungsschwänze. Bei `special` fehlen insbesondere breitere
Referenzraster, große Bessel-Ordnungen/-Argumente, Nullstellennähe und unabhängige
Referenzen für unvollständige Gamma-/Beta- und elliptische Funktionen. Fuzzing
auf Sparse, FFT und ODE ausweiten. Stable-Branch-Coverage wurde nicht gemessen.
