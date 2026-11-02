# Entwurf 0001: Autodiff-Schnittstelle und Arbeitsspeicher-Muster

**Status:** Vorschlag, Entscheidung steht aus · **Phase:** 0b · **Prototyp:** [`design/spike-autodiff`](../../design/spike-autodiff)

## 1. Fragen

1. Wie bekommen die Löser exakte Ableitungen (Autodiff), obwohl sie heute Closures für *einen* Zahlentyp nehmen?
2. Wie reichen Verfahren ihren Arbeitsspeicher an den Aufrufer weiter (heute: `work: &mut [T]` plus `*_work_len`)?
   Aus der Entscheidung zum Workspace-Umbau: Ein einheitliches Scratchpad-/Workspace-Muster (genannt wurden
   *„strukturierte Context/Scratch-Objekte oder const-berechnete Längen“*) soll spätere Breaking Changes quer durch alle Crates
   verhindern. Zu klären ist, ob dafür das heutige Muster genügt.

Rahmen aus den Entscheidungen: Basis-Crates bleiben `#![no_std]` **ohne** Allokator nutzbar; Autodiff arbeitet mit
`Jet<T, N>` (Wert plus Gradient `[T; N]` fester Größe auf dem Stack).

## 2. Methode

Statt Varianten nur zu beschreiben, wurde ein Wegwerf-Prototyp gebaut und gegen die **echten, unveränderten** Crates
getestet (`design/spike-autodiff`, 10 Tests, bestehen alle). Er liegt bewusst außerhalb des Workspaces.

## 3. Ergebnisse (alle im Prototyp geprüft)

### 3.1 `Jet<T, N>` passt in `Real`, ohne dass `Real` geändert wird

`Jet<T, N>` implementiert die komplette Oberfläche (Operatoren, 9 Konstanten, `sqrt`, `sin` … `atan2`, `powf`, `powi`, `hypot`,
`mul_add`, `min`/`max`, `abs`, Rundungen). `sinh`, `cosh`, `tanh`, `cbrt`, `recip` und `to_radians` kommen **gratis** aus den
Standardimplementierungen des Traits und liefern korrekte Ableitungen. Jede Funktion wurde gegen zentrale Differenzen geprüft.

Festlegungen, die sich aus dem Trait ergeben:

| Thema | Wahl | Grund |
|---|---|---|
| `PartialEq`, `PartialOrd` | nur über den **Wert** (`same()` vergleicht alles) | `Real` verlangt Ordnung; wie Ceres |
| `is_nan`, `is_infinite` | wahr bei Wert **oder** Gradient | Löser erkennen so nicht endliche Ableitungen |
| `floor`, `ceil`, `round`, `trunc`, `signum` | Gradient 0 | stückweise konstant |
| `abs` | Ableitung `signum(wert)·d` | bei 0 gilt `signum(+0) = +1` |
| Division | Quotientenregel; durch Wert 0 wie bei Gleitkommazahlen | generischer Code läuft unverändert |
| `checked_recip(&self) -> Option<Self>` | `None` bei Wert 0 | Nullteiler `(0,d)·(0,d') = 0` (getestet), kein Körper |

### 3.2 Funktionsschnittstelle: ein Trait **ohne** Typparameter

```rust
pub trait VectorFn {
    fn eval<S: Real>(&self, x: &[S], out: &mut [S]); // für S = f64 und S = Jet<f64, K>
}
```

Ein erster Entwurf mit `VectorFn<T>` war überflüssig und machte Aufrufe mehrdeutig; die generische Methode genügt. Eine
Closure kann das nicht sein (Closures können keine generische Methode haben), deshalb ein Typ oder ein Makro:

```rust
vector_fn!(Rosenbrock, |x, out| {
    let c = |v: f64| S::from_f64(v);          // Literale müssen über S laufen
    out[0] = c(10.0) * (x[1] - x[0] * x[0]);
    out[1] = c(1.0) - x[0];
});
```

**Preis:** Konstanten in der Funktion brauchen `S::from_f64(…)` (hier als kleine Hilfsclosure `c`). Das lässt sich ohne
Änderung an `Real` nicht vermeiden, weil `S * 10.0` für generisches `S` nicht erlaubt ist.

### 3.3 Der Adapter genügt: **keine neuen `*_ad`-Löser nötig**

Aus einer `VectorFn` entstehen die beiden Closures, die `newton_system`, `gauss_newton` und `levenberg_marquardt` schon heute
verlangen. Die Löser bleiben **unverändert**:

```rust
let mut scratch = [Jet::<f64, 2>::constant(0.0); ad_scratch_len(m, n)];
levenberg_marquardt(
    |p, r| f.eval::<f64>(p, r),                                          // Funktionswert
    |p, j| jacobian_ad::<f64, 2, _>(&f, p, m, j, &mut scratch),          // exakte Jacobi-Matrix
    &mut x, m, opts, &mut work,
)
```

Das entkoppelt Autodiff von den Lösern: Jede künftige Löserfamilie (BFGS, SQP, BDF …) profitiert, ohne ein zweites
Schnittstellenpaar zu bekommen. Geprüft mit `levenberg_marquardt` und `newton_system`.

### 3.4 Laufzeitgrößen ohne Heap: Strip-Mining

`Jet<T, K>` hat feste Größe `K`, die Zahl der Unbekannten `n` steht oft erst zur Laufzeit fest. `jacobian_ad` verarbeitet die
`n` Richtungen in **Blöcken von `K`** und braucht `⌈n/K⌉` Auswertungen. Gemessen an `n = 10`:

| `K` | 1 | 3 | 4 | 10 |
|---|---|---|---|---|
| Auswertungen | 10 | 4 | 3 | 1 |

Alle Varianten liefern dieselbe Jacobi-Matrix (auf `1e-14`). `K` wählt der Aufrufer (Abwägung: Stapelgröße gegen Anzahl der
Aufrufe).

### 3.5 Der AD-Puffer hat einen anderen Elementtyp

`jacobian_ad` braucht Arbeitsspeicher aus **`Jet<T, K>`**, nicht aus `T` (Länge `n + m`, `ad_scratch_len`). Ein einziger
`work: &mut [T]` reicht dafür also nicht. Das spricht für getrennte, typisierte Puffer-Parameter statt eines einzigen
Einheitspuffers.

### 3.6 Zweite Ableitungen durch Schachtelung

`Jet<Jet<T, K>, K>` liefert die Hessematrix (`hessian_ad`), ohne dass `Jet` etwas davon weiß: dieselbe `VectorFn`, ein
anderes `S`. Das ist die Grundlage für SQP und Innere-Punkte.

### 3.7 Was Autodiff bringt – und was nicht

| Größe (Rosenbrock bzw. Exponentialanpassung) | Autodiff | Differenzen |
|---|---|---|
| Fehler der Jacobi-Matrix | **0** | 9,9·10⁻⁸ |
| Fehler der Hessematrix | **0** | 3,1·10⁻⁶ (zentral, beste Schrittweite) |
| Iterationen von Levenberg-Marquardt | 5 | **5** (gleich) |
| Iterationen von Newton (Kreis ∩ Gerade) | 5 | ähnlich |

**Ehrliche Einordnung:** Bei gut konditionierten Problemen ändert eine exakte Jacobi-Matrix die Iterationszahl **nicht**; LM und
Newton vertragen eine um `1e-7` ungenaue Jacobi-Matrix. Der Gewinn liegt in (a) **zweiten Ableitungen** (Differenzen verlieren
dort etwa die Hälfte der Stellen), (b) dem Wegfall der Schrittweitenwahl und der Auslöschung, (c) Verfahren, die Ableitungen als
Ergebnis brauchen (Empfindlichkeiten, IAPWS-Eigenschaften aus der Helmholtz-Energie), und (d) Bequemlichkeit.
Rechenaufwand wurde **nicht gemessen**; grob kostet ein Jet-Aufruf das `(1+K)`-fache einer Gleitkommaauswertung, ähnlich wie
`K` Differenzen-Aufrufe.

### 3.8 Ein typisierter Workspace ist ein reiner Aufsatz

`LmWorkspace::new(m, n, &mut buf)?` prüft die Länge einmal und ruft intern **denselben** `levenberg_marquardt` mit dem Slice
auf. Er musste am Löser nichts ändern und lässt sich mehrfach wiederverwenden. Eine solche Hülle kann also jederzeit
**nachträglich und ohne Breaking Change** ergänzt werden.

## 4. Empfehlung

| # | Entscheidung | Begründung |
|---|---|---|
| **D1** | Löser behalten ihre Closure-/Slice-Schnittstelle. Autodiff kommt über **Adapter** (`jacobian_ad`, später `gradient_ad`, `hessian_ad`), **nicht** über neue Löservarianten. | 3.3: funktioniert unverändert, kein doppelter API-Satz |
| **D2** | `VectorFn` ohne Typparameter mit `eval<S: Real>`, dazu ein `vector_fn!`-Makro. | 3.2 |
| **D3** | `K` (Blockgröße) ist ein const-generischer Parameter des Adapters; Puffer als eigener Slice `&mut [Jet<T, K>]` mit `const fn ad_scratch_len`. | 3.4, 3.5 |
| **D4** | `Jet`-Semantik wie in 3.1 (Wert-Gleichheit, `is_nan` auch für Gradient, `checked_recip`). `Real` bleibt unverändert. | 3.1 |
| **D5** | **Arbeitsspeicher:** Der Slice mit `const fn`-Länge bleibt das stabile Grundmuster für **zustandslose** Löser. **Zustandsbehaftete** Verfahren (BDF-Historie, L-BFGS-Speicher, SQP, Krylov mit Neustart über mehrere Aufrufe) bekommen ein **Zustandsobjekt** (`Stepper::new(…, &mut buf)?`). Typisierte Hüllen für zustandslose Löser sind optional und jederzeit ergänzbar. | 3.5, 3.8: nichts muss jetzt umgebaut werden |
| **D6** | Regel für Aufrufer: Längen **immer** über die `*_work_len`-Funktionen bestimmen, nie fest eintragen. | Dann ändert sich bei wachsendem Bedarf nur ein Wert, keine Signatur |

**Zur Entscheidung „einheitliches Muster, um Breaking Changes zu vermeiden“:** Genannt waren als Beispiele sowohl
Context-/Scratch-Objekte als auch *const-berechnete Längen*. Die const-berechneten Längen (`*_work_len` als `const fn`) sind
genau das heutige Muster, es steht also schon. Es erfüllt das Ziel, solange Aufrufer die Längen über diese Funktionen
bestimmen (D6): Wächst der Bedarf eines Lösers, ändert sich nur ein Wert, keine Signatur. Ein *einziges* Objekt für alle
Löser wäre dagegen ein Breaking Change quer durch alle Crates und brächte bei zustandslosen Lösern wenig, denn die
Längenprüfung passiert so oder so zur Laufzeit. Wo ein Verfahren **Zustand über mehrere Aufrufe** hält (BDF-Historie,
L-BFGS-Speicher, SQP), ist ein Zustandsobjekt dagegen sinnvoll. Wer ein einheitliches Objekt für alles bevorzugt, kann das
nach 3.8 nachträglich bekommen; es wäre aber ein größerer Umbau.

---

## 5. Offene Punkte und Risiken

- **Leistung** nicht gemessen (nur Zählung der Auswertungen). Vor Phase 3 sollte eine kleine Messung folgen.
- **Stapelgröße:** `Jet<f64, 8>` hat 72 Byte; geschachtelte Jets wachsen quadratisch. `K` bewusst klein halten.
- **Hessematrix** ist im Prototyp nur für `K = n` (alle Richtungen auf einmal) umgesetzt, nicht blockweise.
- **Nicht glatte Funktionen** (`abs`, `min`, `max`, `floor` …): Ableitung folgt dem gewählten Ast, an Knickstellen ist sie
  willkürlich; das gehört in die Doku.
- **`powf`** mit Basis ≤ 0 liefert wie bei Gleitkommazahlen NaN; der `ln`-Term wird nur gerechnet, wenn der Exponent selbst
  von den Variablen abhängt (sonst `0·NaN`).
- **Ergonomie der Literale** (3.2) bleibt ein Nachteil; ein Makro für Konstanten wäre denkbar.
- **Entscheidung D5** ist eine Empfehlung; ein einheitliches Objekt für alle Löser wäre die Alternative.

## 6. Folgen für die Umsetzung (Phase 1)

1. Neues Crate `azmath-autodiff` mit `Jet<T, N>` (aus dem Prototyp), `VectorFn`, `vector_fn!`, `jacobian_ad`, `gradient_ad`,
   `hessian_ad`, `ad_scratch_len`.
2. Kein Eingriff in `azmath-optimize`/`azmath-ode`; Beispiele und Tests zeigen die Adapter mit den vorhandenen Lösern.
3. `PartialEq`-Semantik und `is_nan`-Regel in der Doku festhalten (3.1).
4. Danach `Complex` (`azmath-core`), Matrixnormen, Äquilibrierung, Orthonormalisierung.
