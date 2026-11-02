# design/

Wegwerf-Prototypen, die API-Entwürfe prüfen. **Nicht Teil des Workspaces**, nicht in der CI, nicht für die Veröffentlichung.
Sie können veralten. Die Ergebnisse stehen in [`docs/design/`](../docs/design).

| Prototyp | Zweck | Entwurf |
|---|---|---|
| `spike-autodiff/` | `Jet<T, N>` als `Real`, Funktionsschnittstelle `VectorFn`, Autodiff-Adapter, typisierter Workspace | [0001](../docs/design/0001-autodiff-und-arbeitsspeicher.md) |

Prüfen: `cd design/spike-autodiff && cargo test`.
