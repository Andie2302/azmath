//! Dense Eingaben gegenüber nativen CSR-Eingaben, gleicher Auto-KKT-Backend.
//! cargo run --release --example 29_qp_csr_bench --features alloc
#[cfg(all(feature = "std", feature = "alloc"))]
mod bench {
    use azmath::{
        opt::{QpOptions, QpStatus, interior_point_qp, interior_point_qp_sparse},
        sparse::{CooMatrix, CsrMatrix},
    };
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        sync::atomic::{AtomicUsize, Ordering},
        time::Instant,
    };
    static LIVE: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    struct Allocator;
    #[global_allocator]
    static ALLOCATOR: Allocator = Allocator;
    fn add(bytes: usize) {
        let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
        PEAK.fetch_max(live, Ordering::Relaxed);
    }
    unsafe impl GlobalAlloc for Allocator {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            CALLS.fetch_add(1, Ordering::Relaxed);
            add(l.size());
            unsafe { System.alloc(l) }
        }
        unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
            CALLS.fetch_add(1, Ordering::Relaxed);
            add(l.size());
            unsafe { System.alloc_zeroed(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            LIVE.fetch_sub(l.size(), Ordering::Relaxed);
            unsafe { System.dealloc(p, l) }
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
            CALLS.fetch_add(1, Ordering::Relaxed);
            if n >= l.size() {
                add(n - l.size());
            } else {
                LIVE.fetch_sub(l.size() - n, Ordering::Relaxed);
            }
            unsafe { System.realloc(p, l, n) }
        }
    }

    enum Problem {
        Dense {
            q: Vec<f64>,
            a: Vec<f64>,
            g: Vec<f64>,
        },
        Csr {
            q: CsrMatrix<f64>,
            a: CsrMatrix<f64>,
            g: CsrMatrix<f64>,
        },
    }
    fn matrix(
        rows: usize,
        cols: usize,
        entries: impl IntoIterator<Item = (usize, usize, f64)>,
    ) -> CsrMatrix<f64> {
        let mut c = CooMatrix::new(rows, cols);
        for (i, j, v) in entries {
            c.push(i, j, v).unwrap();
        }
        CsrMatrix::from_coo(&c)
    }
    fn case(n: usize, equalities: bool, sparse: bool) {
        let base = LIVE.load(Ordering::Relaxed);
        PEAK.store(base, Ordering::Relaxed);
        let calls = CALLS.load(Ordering::Relaxed);
        let build = Instant::now();
        let p = if equalities { n / 4 } else { 0 };
        let c: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { -1.5 } else { -1.0 })
            .collect();
        let b = vec![0.2; p];
        let mut h = vec![0.0; 2 * n];
        h[n..].fill(0.4);
        let problem = if sparse {
            let q = matrix(
                n,
                n,
                (0..n).flat_map(|i| {
                    let mut row = vec![(i, i, 2.0)];
                    if !equalities && i > 0 {
                        row.push((i, i - 1, -0.2));
                        row.push((i - 1, i, -0.2));
                    }
                    row
                }),
            );
            let a = matrix(p, n, (0..p).map(|r| (r, 4 * r, 1.0)));
            let g = matrix(
                2 * n,
                n,
                (0..n).flat_map(|i| [(i, i, -1.0), (n + i, i, 1.0)]),
            );
            Problem::Csr { q, a, g }
        } else {
            let mut q = vec![0.0; n * n];
            for i in 0..n {
                q[i * n + i] = 2.0;
                if !equalities && i > 0 {
                    q[i * n + i - 1] = -0.2;
                    q[(i - 1) * n + i] = -0.2;
                }
            }
            let mut a = vec![0.0; p * n];
            for r in 0..p {
                a[r * n + 4 * r] = 1.0;
            }
            let mut g = vec![0.0; 2 * n * n];
            for i in 0..n {
                g[i * n + i] = -1.0;
                g[(n + i) * n + i] = 1.0;
            }
            Problem::Dense { q, a, g }
        };
        let build_ms = build.elapsed().as_secs_f64() * 1000.0;
        let input_bytes = LIVE.load(Ordering::Relaxed) - base;
        let start = Instant::now();
        let sol = match &problem {
            Problem::Dense { q, a, g } => {
                interior_point_qp(q, &c, a, &b, g, &h, QpOptions::default())
            }
            Problem::Csr { q, a, g } => {
                interior_point_qp_sparse(q, &c, a, &b, g, &h, QpOptions::default())
            }
        }
        .unwrap();
        let solve_ms = start.elapsed().as_secs_f64() * 1000.0;
        let peak = PEAK.load(Ordering::Relaxed) - base;
        let calls = CALLS.load(Ordering::Relaxed) - calls;
        assert_eq!(sol.status, QpStatus::Optimal);
        let (mut dual, mut primal, mut comp) = (0.0f64, 0.0f64, 0.0f64);
        for (i, &ci) in c.iter().enumerate() {
            let mut qx = 2.0 * sol.x[i];
            if !equalities {
                if i > 0 {
                    qx -= 0.2 * sol.x[i - 1];
                }
                if i + 1 < n {
                    qx -= 0.2 * sol.x[i + 1];
                }
            }
            let y = if equalities && i % 4 == 0 {
                sol.y[i / 4]
            } else {
                0.0
            };
            dual = dual.max((qx + ci + y - sol.z[i] + sol.z[n + i]).abs());
            primal = primal.max((-sol.x[i]).max(sol.x[i] - 0.4));
            comp = comp
                .max((sol.z[i] * sol.x[i]).abs())
                .max((sol.z[n + i] * (0.4 - sol.x[i])).abs());
        }
        for r in 0..p {
            primal = primal.max((sol.x[4 * r] - 0.2).abs());
        }
        assert!(dual < 1e-6 && primal < 1e-6 && comp < 1e-6);
        println!(
            "{},{n},{p},{},{build_ms:.6},{solve_ms:.6},{input_bytes},{peak},{calls},{},{},{},{dual:.3e},{primal:.3e},{comp:.3e}",
            if equalities {
                "diagonal_equalities"
            } else {
                "chain"
            },
            if sparse { "CSR" } else { "Dense" },
            sol.iterations,
            sol.kkt.symbolic_analyses,
            sol.kkt.dense_fallbacks
        );
    }
    pub fn run() {
        println!(
            "case,n,equalities,input,build_ms,solve_ms,input_bytes,total_peak_bytes,alloc_calls,iterations,analyses,fallbacks,dual_residual,primal_violation,complementarity"
        );
        for (n, eq) in [(128, false), (512, false), (256, true)] {
            case(n, eq, false);
            case(n, eq, true);
        }
    }
}
#[cfg(all(feature = "std", feature = "alloc"))]
fn main() {
    bench::run();
}
#[cfg(not(all(feature = "std", feature = "alloc")))]
fn main() {
    println!("Bitte mit std und alloc bauen.");
}
