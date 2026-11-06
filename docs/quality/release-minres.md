# Basisprüfung und MINRES (2026-11-06)

Basis: azmath `8e01f6a`. Die dortigen 17 GitHub-CI-Checks waren erfolgreich.
Dieses Arbeitspaket schließt AZM-010 (technische Release-Vorbereitung) und
AZM-020 (MINRES) ab. Version bleibt 0.0.1. Es wurde nichts auf crates.io
veröffentlicht; eine Veröffentlichung ist eine eigene Entscheidung (AZM-011).

## Technische Basis- und Paketprüfung

| Prüfung | Lokales Ergebnis |
|---|---|
| `cargo test --all-features` | 1.211 Tests einschließlich 93 Doctests bestanden |
| `cargo test --no-default-features --features libm,alloc` | 1.211 Tests einschließlich 93 Doctests bestanden |
| Rust 1.85.0, alle Features und libm+alloc | Beide Builds erfolgreich; 15 MINRES-Tests mit allen Features bestanden |
| `thumbv7em-none-eabihf`, libm / libm+alloc | Beide no_std-Builds erfolgreich |
| Micromath mit/ohne alloc | Je drei Laufzeit-Smoke-Tests bestanden |
| Ohne Features | `cargo test --no-default-features --no-run` erfolgreich |
| Formatierung / Clippy / Rustdoc | Erfolgreich, Warnungen als Fehler bei Clippy/Rustdoc |
| `cargo publish --dry-run --locked --allow-dirty` | Paket erstellt und verifiziert; Upload vom Dry-Run abgebrochen |
| Tests aus entpacktem Crate, alle Features, offline | 1.211 Tests einschließlich Doctests bestanden |

Die lokale Arbeitskopie erfordert `--allow-dirty`; das ist keine Freigabe eines
ungeprüften Release-Artefakts. CI führt den Dry-Run ohne diesen Zusatz auf dem
sauberen Checkout aus und testet danach das entpackte Paket. Das Paket enthält
Bibliothek, Beispiele, Tests, Referenzdaten, Lizenzen und Changelog; docs,
Designentwürfe, Fuzz-Workspace und Build-Ausgaben werden ausgeschlossen.
Die Referenzgeneratoren sind offline verwendbare Hilfsmittel; normale Tests
benötigen weder Python noch mpmath.

Reproduzieren im sauberen Checkout:

```sh
cargo publish --dry-run --locked
cargo test --manifest-path target/package/azmath-0.0.1/Cargo.toml --all-features --offline
cargo +1.85.0 build --all-features --locked
cargo +1.85.0 build --no-default-features --features libm,alloc --locked
```

Die CI ermittelt die Paketversion aus Cargo-Metadaten statt einen Dateipfad mit
fest eingebauter Versionsnummer zu pflegen. Diese Prüfung bestätigt technische
Paketfähigkeit und die dokumentierten Testbereiche, nicht die Genauigkeit
jedes Algorithmus für beliebige Parameter. Versionswahl, Tag und tatsächliche
Veröffentlichung folgen erst mit gesondertem Auftrag.

## MINRES und PMINRES

Neue APIs in `linalg::iterative`: `minres`, `pminres`, `minres_work_len`.
Die Komfortmethoden `solve_minres` und `solve_pminres` gibt es für Matrix und
(hinter alloc) DynMatrix. CSR sowie andere LinearOperator-Implementierungen
funktionieren über die matrixfreie API.

- Reelle symmetrische A, auch indefinite; für PMINRES muss M linear, symmetrisch
  und positiv definit sein. Das sind Voraussetzungen des Aufrufers. Die
  auftretenden Formen `rᵀ M⁻¹ r` werden geprüft; eine vollständige Symmetrie-/SPD-
  Prüfung eines beliebigen Operators ist damit nicht möglich.
- Kurze Lanczos-/Givens-Rekursion, Arbeitsbuffer `8*n`, keine Allokation im Kern.
- Erfolg ausschließlich am neu berechneten ursprünglichen Residuum
  `‖b-Ax‖ <= tol*‖b‖`. Dafür je Iteration zwei Anwendungen von A.
- Bei einer kleinen rekursiven Schätzung mit zu großem wahrem Residuum wird
  aus dem wahren Residuum neu gestartet; das gesamte Iterationsbudget bleibt
  erhalten. Diese Korrektur war für den f32-Regressionstest nötig.
- Nichtpositive Vorkonditioniererformen, NaN/Inf, falsche Dimensionen,
  unzureichendes Budget und invariante Unterräume ohne ausreichendes Residuum
  werden diagnostiziert. Die letzte Näherung bleibt bei Budgetende erhalten.
- Kein MINRES-QLP und keine allgemeine Minimalnorm-/Least-Squares-Garantie.

Grundlage: Paige und Saunders, *Solution of Sparse Indefinite Systems of Linear
Equations*, SIAM J. Numer. Anal. 12 (1975), DOI
[10.1137/0712047](https://doi.org/10.1137/0712047).

## Numerische Nachweise

15 Tests: pro Float-Typ sieben Tests, dazu ein CSR-/DynMatrix-Test. Enthalten
sind 32 erzeugte symmetrisch-indefinite 4×4-Systeme pro Typ im Vergleich mit LU,
Zero-Diagonal-Systeme, echte Residuen, Startnäherungen, leere und singuläre
Probleme, ungültige Vorkonditionierer und große/kleine Skalierungen
(f32 1e±30, f64 1e±200).

| Beispiel: Diagonale ±0,01, ±1, ±100 | Ohne Vorkonditionierer | Mit positiver Betragsdiagonale |
|---|---:|---:|
| f32 | 14 Iterationen | 2 Iterationen |
| f64 | 10 Iterationen | 2 Iterationen |

Das ist ein gezielter Regressionstest und kein allgemeines Leistungsversprechen.
Die signierte Diagonale eines indefiniten A darf nicht ohne SPD-Prüfung als
MINRES-Vorkonditionierer verwendet werden. Bei starker Konditionierung oder
Rundungsfehlern sind weitere Iterationen/Neustarts oder `NoConvergence` möglich;
die Erfolgskontrolle darf dadurch keinen falschen Erfolg melden.

Nächster geplanter Schritt: AZM-030, separables symplektisches Verlet mit
Ordnungs-, Zeitumkehr- und Energieverhaltenstests.
