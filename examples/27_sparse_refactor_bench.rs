//! Reproduzierbarer Vergleich: Neuaufbau gegenüber Analyse-/Pufferwiederverwendung.
//! cargo run --release --example 27_sparse_refactor_bench --features alloc -- 30
//! Zeiten sind lokale Messwerte, keine zugesicherte Beschleunigung.
#[cfg(all(feature = "std", feature = "alloc"))]
mod benchmark {
    use azmath::sparse::{
        CooMatrix, CsrMatrix, Ordering, SparseLdl, SparseLdlSymbolic, SparseLu, SparseLuSymbolic,
    };
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        hint::black_box,
        sync::atomic::{AtomicUsize, Ordering as AtomicOrdering},
        time::Instant,
    };
    struct CountAllocator;
    static ALLOCS: AtomicUsize = AtomicUsize::new(0);
    #[global_allocator]
    static ALLOCATOR: CountAllocator = CountAllocator;
    // Zählt tatsächliche Allokations-/Reallokationsaufrufe, nicht Freigaben.
    unsafe impl GlobalAlloc for CountAllocator {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            ALLOCS.fetch_add(1, AtomicOrdering::Relaxed);
            unsafe { System.alloc(l) }
        }
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            ALLOCS.fetch_add(1, AtomicOrdering::Relaxed);
            unsafe { System.alloc_zeroed(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            unsafe { System.dealloc(p, l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
            ALLOCS.fetch_add(1, AtomicOrdering::Relaxed);
            unsafe { System.realloc(p, l, n) }
        }
    }
    fn measure(mut operation: impl FnMut()) -> (f64, usize) {
        let allocations = ALLOCS.load(AtomicOrdering::Relaxed);
        let start = Instant::now();
        operation();
        (
            start.elapsed().as_secs_f64() * 1000.0,
            ALLOCS.load(AtomicOrdering::Relaxed) - allocations,
        )
    }
    fn grid(side: usize) -> CsrMatrix<f64> {
        let n = side * side;
        let mut c = CooMatrix::new(n, n);
        for i in 0..n {
            c.push(i, i, 4.0).unwrap();
            for j in [
                if i % side > 0 { Some(i - 1) } else { None },
                if i >= side { Some(i - side) } else { None },
            ]
            .into_iter()
            .flatten()
            {
                c.push(i, j, -1.0).unwrap();
                c.push(j, i, -1.0).unwrap();
            }
        }
        CsrMatrix::from_coo(&c)
    }
    fn kkt(n: usize) -> CsrMatrix<f64> {
        let mut c = CooMatrix::new(n + n / 2, n + n / 2);
        for i in 0..n {
            c.push(i, i, 3.0).unwrap();
            if i > 0 {
                c.push(i, i - 1, -0.5).unwrap();
                c.push(i - 1, i, -0.5).unwrap();
            }
        }
        for i in 0..n / 2 {
            c.push(n + i, n + i, -0.1).unwrap();
            c.push(n + i, 2 * i, 1.0).unwrap();
            c.push(2 * i, n + i, 1.0).unwrap();
        }
        CsrMatrix::from_coo(&c)
    }
    fn ode(n: usize) -> CsrMatrix<f64> {
        let mut c = CooMatrix::new(n, n);
        // I − hJ für eine nicht symmetrische Advektions-Diffusions-Diskretisierung.
        for i in 0..n {
            c.push(i, i, 4.0).unwrap();
            if i > 0 {
                c.push(i, i - 1, -1.5).unwrap();
                c.push(i - 1, i, -0.5).unwrap();
            }
        }
        CsrMatrix::from_coo(&c)
    }
    fn variants(a: &CsrMatrix<f64>, count: usize) -> Vec<CsrMatrix<f64>> {
        (0..count)
            .map(|k| {
                let mut m = a.clone();
                for i in 0..m.rows() {
                    for p in m.row_ptr()[i]..m.row_ptr()[i + 1] {
                        if m.col_idx()[p] == i {
                            let v = m.values()[p];
                            m.values_mut()[p] = v * (1.0 + k as f64 * 0.001);
                        }
                    }
                }
                m
            })
            .collect()
    }
    fn check(a: &CsrMatrix<f64>, x: &[f64]) {
        for i in 0..a.rows() {
            let value = a.row(i).map(|(j, v)| v * x[j]).sum::<f64>();
            assert!((value - 1.0).abs() < 1e-8);
        }
    }
    fn ldl_case(label: &str, a: CsrMatrix<f64>, count: usize) {
        let matrices = variants(&a, count);
        let start = Instant::now();
        let plan = SparseLdlSymbolic::analyze(&a, Ordering::MinimumDegree).unwrap();
        let analysis_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut factor = plan.factorize(&a).unwrap();
        for m in &matrices {
            factor.refactor(m).unwrap();
            check(m, &factor.solve(&vec![1.0; m.rows()]).unwrap());
        }
        let fresh = measure(|| {
            for m in &matrices {
                black_box(SparseLdl::new(black_box(m), Ordering::MinimumDegree).unwrap());
            }
        });
        let reused = measure(|| {
            for m in &matrices {
                factor.refactor(black_box(m)).unwrap();
                black_box(factor.d());
            }
        });
        println!(
            "{label},{},{count},{analysis_ms:.6},{:.6},{},{:.6},{}",
            a.rows(),
            fresh.0,
            fresh.1,
            reused.0,
            reused.1
        );
        assert_eq!(reused.1, 0, "LDL refactor must not allocate");
    }
    pub fn run() {
        let count = std::env::args()
            .nth(1)
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(30)
            .max(1);
        println!("case,n,repeats,analysis_ms,fresh_ms,fresh_allocs,refactor_ms,refactor_allocs");
        ldl_case("poisson_ldl", grid(12), count);
        ldl_case("kkt_ldl", kkt(100), count);
        let a = ode(160);
        let matrices = variants(&a, count);
        let start = Instant::now();
        let plan = SparseLuSymbolic::analyze(&a, Ordering::MinimumDegree).unwrap();
        let analysis_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut factor = plan.factorize(&a).unwrap();
        for m in &matrices {
            factor.refactor(m).unwrap();
            check(m, &factor.solve(&vec![1.0; m.rows()]).unwrap());
        }
        let fresh = measure(|| {
            for m in &matrices {
                black_box(SparseLu::new(black_box(m), Ordering::MinimumDegree).unwrap());
            }
        });
        let reused = measure(|| {
            for m in &matrices {
                factor.refactor(black_box(m)).unwrap();
                black_box(factor.nnz_l());
            }
        });
        println!(
            "ode_lu,{},{count},{analysis_ms:.6},{:.6},{},{:.6},{}",
            a.rows(),
            fresh.0,
            fresh.1,
            reused.0,
            reused.1
        );
        // LU darf bei veränderten Pivots/Fill-in neue Kapazität benötigen.
    }
}
#[cfg(all(feature = "std", feature = "alloc"))]
fn main() {
    benchmark::run();
}
#[cfg(not(all(feature = "std", feature = "alloc")))]
fn main() {
    println!("Bitte mit std und alloc bauen.");
}
