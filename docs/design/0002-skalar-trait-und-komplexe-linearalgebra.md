# Entwurf 0002: Skalar-Trait und komplexe lineare Algebra

**Status:** teilweise umgesetzt (Schritte 1–5 in minimaler Form, `azmath::scalar`) · **Abhängig von:** `Real`, `Complex<T>`, `linalg`, `sparse`

## 1. Frage

Alle Löser in `linalg` und `sparse` verlangen `T: Real`. `Complex<T>` ist bewusst kein `Real` (keine Ordnung).
Komplexe Gleichungssysteme, komplexe LU/QR/Cholesky und komplexe Eigenwerte (Frequenzbereich, Quantenmechanik,
Spektralverfahren) sind deshalb nicht möglich. Wie öffnen wir die Algorithmen für `Complex<T>`, ohne `Real` aufzuweichen
und ohne die bestehende API zu brechen?

## 2. Was die Algorithmen wirklich brauchen

| Operation | Real | Komplex | Folgerung |
|---|---|---|---|
| `+ − × ÷`, `0`, `1` | ja | ja | Körper-Grundlage |
| Pivotwahl / Größenvergleich | `abs`, `<` | `modulus` (reell) | Betrag liefert immer `Real` |
| Skalarprodukt, Normen | `x·y` | `conj(x)·y` | `conj` nötig |
| Hermitesche statt symmetrische Struktur | `Aᵀ` | `Aᴴ` | `adjoint` statt `transpose` |
| Toleranzen, `EPSILON`, `sqrt` von Normen | `T` | `T::Real` | assoziierter Typ |

Reell sind also nur Beträge, Toleranzen und Wurzeln von Normen; alles andere ist Körperarithmetik plus Konjugation.

## 3. Vorschlag

```rust
pub trait Scalar: Copy + PartialEq + Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self>
    + Div<Output=Self> + Neg<Output=Self> + AddAssign + SubAssign + MulAssign + DivAssign
{
    /// Zugehöriger reeller Typ (für Beträge, Normen, Toleranzen).
    type Real: Real;
    const ZERO: Self;
    const ONE: Self;
    fn from_real(r: Self::Real) -> Self;
    fn re(self) -> Self::Real;
    fn im(self) -> Self::Real;       // bei Real: 0
    fn conj(self) -> Self;           // bei Real: Identität
    fn modulus(self) -> Self::Real;  // |z|, bei Real: abs
    fn modulus_sq(self) -> Self::Real;
    fn is_finite(self) -> bool;
}
```

- **`impl<T: Real> Scalar for T`** ist wegen der Kohärenzregeln nicht gleichzeitig mit `impl Scalar for Complex<T>` möglich.
  Deshalb: Implementierung **einzeln** für `f32`, `f64` und `Complex<T: Real>` (sowie `Jet`, falls gewünscht). `Real` bleibt
  unverändert und wird **kein** Subtrait von `Scalar`.
- Kein Bruch bestehender API: vorhandene Funktionen behalten `T: Real`.

## 4. Einführung in Schritten (jeweils eigener Schritt)

1. **`Scalar`** in `src/scalar/`, Implementierungen für `f32`, `f64`, `Complex<f32|f64>`, Tests der Axiome.
2. **Komplexe dichte Löser** als neue Funktionen/Typen neben den reellen, nicht als Umbau:
   `lu_solve_c`-artige Namen vermeiden; stattdessen generische interne Kerne `*_scalar<T: Scalar>`, auf die die reellen
   Funktionen später verweisen können. Zuerst: LU mit Teilpivotisierung (Pivot nach `modulus`), Rückwärts-/Vorwärtssubstitution,
   Determinante, Inverse.
3. **QR (Householder)** und **Cholesky (hermitesch)** komplex.
4. **Eigenwerte komplexer Matrizen:** Hessenberg + komplexes QR-Verfahren (Einzelschritt mit Wilkinson-Shift), danach
   hermitesche Eigenwerte (reell) über die reelle symmetrische Einbettung oder direkt.
5. **Sparse:** `CsrMatrix<T: Scalar>` und iterative Löser (GMRES/BiCGSTAB) generisch; `LinearOperator` über `Scalar`.
6. Erst danach prüfen, ob die reellen Löser selbst auf `Scalar` umgestellt werden (nur wenn ohne Leistungs- und
   Signaturbruch möglich; sonst bleibt es bei der Doppelstruktur).

## 5. Entscheidungen / offene Punkte

| Punkt | Entscheidung |
|---|---|
| `Real: Scalar`? | nein (Kohärenz, kein Bruch) |
| Duplizierter Code reell/komplex | akzeptiert; gemeinsame Kerne generisch über `Scalar`, reelle Wrapper bleiben |
| Skalarprodukt-Konvention | linear im zweiten Argument, konjugiert im ersten (`xᴴy`) |
| `Matrix<T,R,C>` / `DynMatrix<T>` | Schranke auf `Scalar` lockern, wo nur Körperarithmetik nötig ist (Speicher, Produkt) |
| Offen | Soll `Jet<Complex<T>, N>` unterstützt werden? Vorerst nein. |

## 6. Nicht-Ziele

Keine Quaternionen-Algebra als Skalar, keine exakte/rationale Arithmetik (gehört in eigene Crates).
