# Verteilungen und Quantile (2026-11-06)

Ausgangsstand: azmath `7481f81`. Die neuen Regressionstests reproduzierten
nichtendliche Normalquantile für kleine positive p und verschluckte NaN-Eingaben.
Zusätzlich begrenzten 200 Halbierungsschritte die Auflösung kleiner Gamma-/Beta-
Quantile auf einen absoluten Bereich um 1e-60.

## Verhalten und API

- Verteilungen bieten direkte Überlebensfunktionen `*_sf(x)=P(X>x)` und deren
  Inverse `*_isf(q)`. Für Normalverteilungen mit Mittelwert/Standardabweichung
  heißen die Varianten `normal_sf_with` und `normal_isf_with`.
- `special` ergänzt `normal_sf`, `normal_isf` und `beta_inc_complement`.
- Kleine Normalquantile werden ohne `2*p-1` invertiert. Positive Verteilungen
  invertieren den kleineren Tail mit skalierter/expandierender Klammer und
  Abbruch an benachbarten Gleitkommazahlen (höchstens 1200 Halbierungen).
- Student-t-Tails vermeiden den Verlust durch `x*x` bei großen Argumenten;
  F-Tails arbeiten mit der passenden Beta-Seite und stabilen Verhältnissen.
- t- und χ²-Hypothesentests benutzen direkte obere Tails für kleine p-Werte.
- Ungültige oder nichtendliche Verteilungsparameter liefern NaN; endliche
  gültige Parameter mit Argument ±Inf haben die dokumentierten Stützbereichs-
  grenzen. NaN wird vor Clamping geprüft. Breite Gleichverteilungsintervalle
  werden ohne überlaufende Intervallbreite behandelt.

Bei einem auf eins gerundeten p ist die verlorene obere Wahrscheinlichkeit
nicht rekonstruierbar. Deshalb `isf(q)` direkt aufrufen. Auch ein korrektes
Beta-Quantil darf auf einen Randwert runden, wenn sein Abstand von 0 oder 1
im Zieltyp nicht darstellbar ist.

## Referenzverfahren

606 eingecheckte CDF-/SF-/Quantil-/ISF-Fälle für Normal-, Exponential-, Gleich-,
Gamma-, χ²-, Beta-, Student-t- und F-Verteilungen. Eingaben und Parameter werden
zuerst exakt auf f32/f64 gerundet. mpmath 1.3.0 berechnet die Referenzen bei
100 und 200 Dezimalstellen; die direkt auf IEEE gerundeten Sollwerte müssen
übereinstimmen. Numerische Referenzinversion arbeitet in log(x), damit kleine
Quantile relativ und nicht nur absolut aufgelöst werden.

```sh
python scripts/generate_dist_refs.py
cargo test --test stats_dist_reference --test stats_dist_robustness -- --nocapture
cargo test --no-default-features --features libm,alloc --test stats_dist_reference -- --nocapture
```

Normale Cargo-Tests benötigen weder Python noch mpmath. Die Generatoren teilen
`reference_ieee.py`; die bisherigen 356 special-Referenzen wurden unverändert
reproduziert. Unendliche Sollwerte müssen exakt übereinstimmen. Es gibt kein
pauschales absolutes Fehlerbudget, das kleine Tails verdeckt.

| Familie | f32 max. ULP | f64 max. ULP |
|---|---:|---:|
| beta | 59 | 152 |
| chi2 | 31 | 37 |
| exponential | 1 | 1 |
| f | 50 | 301 |
| gamma | 39 | 733 |
| normal | 27 | 51 |
| student_t | 31 | 324 |
| uniform | 1 | 1 |

Beobachtete Maxima über std und libm, keine globalen Fehlergrenzen. Die Tabelle
fasst die Referenzpunkte je Familie zusammen. Prüfbudgets: Gleichverteilung
4 ULP, Exponentialverteilung 8, Normalquantile/ISF 8, übrige f32-Fälle 128 und
übrige f64-Fälle 1024. Die eigentlichen Werte und Einzelmaxima stehen in den
Tests bzw. deren Ausgabe. Wahrscheinlichkeiten bis 1e-300 (f64) und 1e-30
(f32) sowie nächste untersuchte Wahrscheinlichkeiten unter eins sind enthalten;
auf null/eins gerundete Eingaben werden als Randfälle behandelt.

## Validierung und Grenzen

- Je 1.195 Tests einschließlich 92 Doctests mit allen Features bzw. libm+alloc.
- Neun zusätzliche Robustheits-/Invariantentests plus ein Referenztest.
- Clippy, Formatierung, Rustdoc mit Warnungen als Fehlern, Embedded-Build mit
  libm+alloc und drei micromath-Smoke-Tests erfolgreich.
- Die Genauigkeitsprüfungen gelten für std/libm. Micromath wird dadurch nicht
  als hochpräziser Statistik-Backend freigegeben.

Die Raster betreffen ausgewählte moderate Formparameter, nicht beliebig große
oder verschwindend kleine Freiheitsgrade/Formparameter. Weiter offen sind
breitere Referenzraster, Konvergenzdiagnostik der unvollständigen Funktionen,
logarithmische Tails und größere Parameterbereiche (AZM-080 in aztodo).
Dieses Arbeitspaket führt keine generelle Genauigkeitsgarantie für alle
Verteilungsparameter ein.
