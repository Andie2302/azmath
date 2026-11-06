# Native CSR-QP-Eingaben: Zeit, Speicher und Residuen

```sh
cargo run --release --example 29_qp_csr_bench --features alloc
```

Lokale verwaltete Linux-Umgebung, Rust 1.99.1. Median aus fünf Ausführungen,
ohne parallel laufende Builds. Dichte und CSR-Eingaben lösen identische QPs
über denselben Auto-KKT-Backend und denselben Solver-Kern.
Alle Fälle wählen Sparse, ohne dichten Rückfall.

`chain`: tridiagonales Q, Schranken 0 ≤ x ≤ 0,4, keine Gleichungen.
`diagonal_equalities`: diagonales Q, dieselben Schranken und n/4 Gleichungen
x[4r]=0,2. CSR-Daten entstehen direkt aus Triplets, nicht aus dichten Matrizen.
Die dichten Daten werden ebenfalls direkt erzeugt.

Die Aufbauzeit enthält Koeffizienten- und Vektoraufbau. Die Lösungslaufzeit
enthält Validierung, Presolve, symbolische Analyse, alle Iterationen, Faktoren,
Zielfunktion und Residuenprüfungen. Beide Zeiten sind Millisekunden.
Der Eingabespeicher umfasst die gleichzeitig gehaltenen Matrix-/Vektorpuffer
vor dem Lösen. Gesamt-Peak ist der zusätzlich lebende Heap vom Beginn des
Eingabeaufbaus bis einschließlich Ergebnis, also diesmal **mit** Eingabedaten.
Ein zählender Systemallokator erfasst angeforderte Layoutgrößen; nicht RSS,
Allocator-Metadaten oder Stack. Peak- und Eingabewerte sind Bytes.

| Fall | n | Gleichungen | Eingabe | Aufbau ms | Lösen ms | Eingabe-Bytes | Gesamt-Peak-Bytes | Iterationen |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| chain | 128 | 0 | Dense | 0.1114 | 1.8276 | 396288 | 514096 | 6 |
| chain | 128 | 0 | CSR | 0.0337 | 0.2979 | 16376 | 134184 | 6 |
| chain | 512 | 0 | Dense | 1.7934 | 28.4633 | 6303744 | 6775216 | 6 |
| chain | 512 | 0 | CSR | 0.1229 | 1.3759 | 65528 | 537000 | 6 |
| diagonal_equalities | 256 | 64 | Dense | 0.3838 | 6.5093 | 1710592 | 1931488 | 6 |
| diagonal_equalities | 256 | 64 | CSR | 0.0314 | 0.6074 | 26648 | 247544 | 6 |

Alle Fälle melden Optimal. Unabhängig nachgerechnete maximale Residuen:
Stationarität 1.810e-11, primale Verletzung einschließlich Gleichungen
1.630e-13, Komplementarität 1.684e-09. Die Nachrechnung liegt außerhalb
der Messintervalle. Tests ergänzen f32, Gleichungsredundanz/-widersprüche,
Form-/Symmetriefehler, nichtendliche Werte und dichten Rückfall.

Der Kettenvergleich bei n=512 reduziert den gemessenen Gesamt-Peak von rund
6,78 MB auf 0,54 MB. Das Ergebnis gilt für diese dünnen Muster. CSR-Matrizen,
Sparse-Produkte und KKT-Assembly vermeiden pauschale dichte Kopien, aber
Presolve/Faktoren können Fill-in entwickeln und Dense/Auto bzw. numerische
Rückfälle dürfen eine dichte KKT-Matrix erzeugen. Der Presolve und die genaue
Minimum-Degree-Umordnung sind nicht allgemein linear in der Eintragszahl.
Zeiten unterliegen Cloud-Auslastung und Cache-Effekten; die kleinen synthetischen
Probleme liefern keine universelle Beschleunigungs- oder Speichergrenze.
