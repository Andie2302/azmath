# QP-KKT: End-to-end-Vergleich

```sh
cargo run --release --example 28_qp_kkt_bench --features alloc
```

Lokale Messung in der verwalteten Linux-Umgebung, Rust 1.99.1. Median aus fünf
separaten Ausführungen, ohne parallel laufende Builds. Jeder Aufruf löst das
vollständige QP inklusive Musteraufbau, symbolischer Analyse, aller Iterationen,
Faktorisierungen und linearer Residuenprüfungen. Die Modi werden in der Reihenfolge
Dense, Sparse, Auto gemessen. `chain` hat eine tridiagonale Hessematrix,
`dense_hessian` eine dichte. Nebenbedingungen: 0 ≤ x ≤ 0,4 (keine Gleichungen).

Peak-Heap ist der zusätzliche gleichzeitige Heap-Bedarf ab Eintritt in den QP-Aufruf.
Die zuvor aufgebauten dichten Q/G-Eingaben sind nicht enthalten, auch keine
Allocator-Metadaten, Stacks oder RSS. Allokationszahlen zählen alloc/realloc;
Peak-Bytes verwenden deren angeforderte Layoutgrößen. Zeiten sind Millisekunden.

| Fall | n | Backend | Zeit ms | zusätzlicher Peak-Heap Bytes | Allokationen | Iterationen | Analysen | Refaktorisierungen | Rückfälle |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| chain | 64 | Dense | 1.1712 | 88120 | 420 | 6 | 0 | 0 | 0 |
| chain | 64 | Sparse | 0.3661 | 58856 | 855 | 6 | 1 | 5 | 0 |
| chain | 64 | Auto | 0.3563 | 58856 | 855 | 6 | 1 | 5 | 0 |
| chain | 128 | Dense | 14.8833 | 307320 | 615 | 6 | 0 | 0 | 0 |
| chain | 128 | Sparse | 1.2016 | 117800 | 1499 | 6 | 1 | 5 | 0 |
| chain | 128 | Auto | 1.2387 | 117800 | 1499 | 6 | 1 | 5 | 0 |
| dense_hessian | 48 | Dense | 0.8558 | 93424 | 566 | 7 | 0 | 0 | 0 |
| dense_hessian | 48 | Sparse | 1.9473 | 362944 | 1488 | 7 | 1 | 6 | 0 |
| dense_hessian | 48 | Auto | 0.6815 | 93424 | 566 | 7 | 0 | 0 | 0 |

Alle Fälle melden Optimal. Maximale Stationaritätsabweichung: 1.807e-11,
maximale Komplementaritätsabweichung: 3.112e-08, keine positive primale
Schrankenverletzung. Größter akzeptierter linearer Rückwärtsfehler: 1.170e-15.
Diese unabhängigen QP-Residuen werden außerhalb des Messintervalls nachgerechnet.
Die Tests ergänzen Gleichungen, redundante Gleichungen, f32 und einen tatsächlich
ausgelösten dichten Rückfall bei einem Nullpivot.

Auto wählt in den Kettenbeispielen Sparse und bei der dichten Hessematrix Dense.
Die 25-%-Strukturheuristik ist kein universell optimaler Performance-Schwellwert.
Sie vermeidet hier den Speicher-/Zeitnachteil erzwungener Sparse-Faktorisierung
für das dichte Beispiel. Zeiten können durch Cloud-Auslastung und Cache-Effekte
schwanken; kleine synthetische Fälle sind keine allgemeine Skalierungsgarantie.
Beide Modi verwenden dieselbe CSR-Assemblierung und Residuenprüfung.
Problemkoeffizienten und übrige Matrix-Vektor-Produkte bleiben im QP-Frontend dicht;
Gesamtspeicher großer Probleme wird durch dieses Backend allein nicht vollständig sparse.
