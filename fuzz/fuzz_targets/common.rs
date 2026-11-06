// Gemeinsame Hilfen der Fuzz-Targets (per `include!`/`#[path]` eingebunden).

/// Bytestrom mit sicherem Lesen: fehlende Bytes liefern 0.
pub struct Bytes<'a>(pub &'a [u8]);

impl Bytes<'_> {
    pub fn u8(&mut self) -> u8 {
        match self.0.split_first() {
            Some((&b, rest)) => {
                self.0 = rest;
                b
            }
            None => 0,
        }
    }
    /// Wert in `lo..=hi`.
    pub fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.u8() as usize % (hi - lo + 1)
    }
    /// Endlicher, moderater Wert in etwa [-8, 8].
    pub fn moderate(&mut self) -> f64 {
        (self.u8() as f64 - 128.0) / 16.0
    }
    /// Wert aus einer Tabelle mit Sonderfällen; `special` erlaubt NaN/Inf/Extreme.
    #[allow(dead_code)]
    pub fn value(&mut self, special: bool) -> f64 {
        self.value_with(special, true)
    }
    /// Wie [`value`](Self::value); mit `extreme = false` bleiben nur NaN, ±Inf und ±0 als Sonderwerte,
    /// alle übrigen Werte liegen in etwa in [-8, 8].
    pub fn value_with(&mut self, special: bool, extreme: bool) -> f64 {
        let b = self.u8();
        if special {
            match b % 24 {
                0 => return f64::NAN,
                1 => return f64::INFINITY,
                2 => return f64::NEG_INFINITY,
                3 if extreme => return 1e300,
                4 if extreme => return -1e300,
                5 if extreme => return 1e-300,
                6 if extreme => return 5e-324,
                7 => return 0.0,
                8 => return -0.0,
                _ => {}
            }
        }
        (self.u8() as f64 - 128.0) / 16.0
    }
}
