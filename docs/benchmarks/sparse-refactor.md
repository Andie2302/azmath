# Wiederverwendung von Sparse-Faktorisierungen

Aufruf:

```sh
cargo run --release --example 27_sparse_refactor_bench --features alloc -- 30
```

Lokale Messung in der verwalteten Linux-Umgebung, Rust 1.99.1
(58cf8f9c3, 2026-10-13). Median aus fünf Läufen, jeweils 30 Matrizen mit
unterschiedlichen Diagonalwerten und identischem CSR-Muster. Alle drei Fälle
verwenden `Ordering::MinimumDegree`. Zeiten in Millisekunden für alle 30
Faktorisierungen zusammen, ohne Matrixaufbau und ohne Lösungsschritte.

| Fall | Dimension | Neuaufbau | Refaktorisierung | Verhältnis | Allokationen Neuaufbau | Allokationen Refaktorisierung |
|---|---:|---:|---:|---:|---:|---:|
| 2D-Poisson, LDLᵀ | 144 | 12,497 | 0,242 | 51,69 | 34.500 | 0 |
| Quasi-definites KKT-System, LDLᵀ | 150 | 4,358 | 0,044 | 98,16 | 28.350 | 0 |
| I−hJ für 1D-Advektion/Diffusion, LU | 160 | 5,913 | 3,130 | 1,89 | 29.430 | 0 |

Neuaufbau umfasst Analyse, Speicheraufbau und numerische Faktorisierung.
Die wiederverwendete Variante besitzt vor Messbeginn Analyse und Arbeitspuffer;
bei LU wurden alle Varianten zum Aufwärmen durchlaufen, um benötigte
Spaltenkapazitäten bereits bereitzustellen. Der Benchmark meldet zusätzlich die
Zeit einer einmaligen Analyse. Er misst Allokations- und Reallokationsaufrufe
über einen zählenden Systemallokator; Freigaben werden nicht mitgezählt.
Residuen aller Lösungen werden außerhalb der Zeitmessung geprüft.

Diese kleinen synthetischen Fälle sind keine Messung der gesamten QP-/ODE-Löser
und keine allgemeine Beschleunigungsgarantie. Größere Systeme, andere Muster,
Pivots, Fill-in und Hardware können andere Ergebnisse liefern.
LDLᵀ/Cholesky benötigen bei `refactor` grundsätzlich keine Allokationen.
LU verwendet erneut numerische Zeilenpivotierung; seine Spaltenpuffer können bei
verändertem Fill-in wachsen. Die Messung von null Allokationen gilt für die
hier getesteten, aufgewärmten Varianten.
