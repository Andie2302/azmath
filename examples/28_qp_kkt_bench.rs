//! End-to-end-QP-Vergleich inklusive Musteranalyse, Faktorisierung und Residuenprüfung.
//! cargo run --release --example 28_qp_kkt_bench --features alloc
#[cfg(all(feature = "std", feature = "alloc"))]
mod bench {
    use azmath::opt::{QpKktBackend, QpOptions, QpStatus, interior_point_qp};
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        hint::black_box,
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
    fn case(n: usize, dense_q: bool) {
        let mut q = vec![0.0; n * n];
        let c: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { -1.5 } else { -1.0 })
            .collect();
        for i in 0..n {
            q[i * n + i] = 2.0;
            if dense_q {
                for j in 0..i {
                    q[i * n + j] = 0.01;
                    q[j * n + i] = 0.01;
                }
            } else if i > 0 {
                q[i * n + i - 1] = -0.2;
                q[(i - 1) * n + i] = -0.2;
            }
        }
        let mut g = vec![0.0; 2 * n * n];
        let mut h = vec![0.0; 2 * n];
        for i in 0..n {
            g[i * n + i] = -1.0;
            g[(n + i) * n + i] = 1.0;
            h[n + i] = 0.4;
        }
        for mode in [
            QpKktBackend::Dense,
            QpKktBackend::Sparse,
            QpKktBackend::Auto,
        ] {
            let base = LIVE.load(Ordering::Relaxed);
            PEAK.store(base, Ordering::Relaxed);
            let calls = CALLS.load(Ordering::Relaxed);
            let start = Instant::now();
            let sol = interior_point_qp(
                black_box(&q),
                &c,
                &[],
                &[],
                &g,
                &h,
                QpOptions {
                    kkt_backend: mode,
                    ..QpOptions::default()
                },
            )
            .unwrap();
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            let peak = PEAK.load(Ordering::Relaxed) - base;
            let calls = CALLS.load(Ordering::Relaxed) - calls;
            black_box(&sol);
            assert_eq!(sol.status, QpStatus::Optimal);
            let mut dual = 0.0f64;
            let mut primal = 0.0f64;
            let mut comp = 0.0f64;
            for i in 0..n {
                let r = (0..n).map(|j| q[i * n + j] * sol.x[j]).sum::<f64>() + c[i] - sol.z[i]
                    + sol.z[n + i];
                dual = dual.max(r.abs());
                primal = primal.max((-sol.x[i]).max(sol.x[i] - 0.4));
                comp = comp
                    .max((sol.z[i] * sol.x[i]).abs())
                    .max((sol.z[n + i] * (0.4 - sol.x[i])).abs());
            }
            assert!(dual < 1e-6 && primal < 1e-6 && comp < 1e-6);
            println!(
                "{},{n},{mode:?},{ms:.6},{peak},{calls},{},{},{},{},{:.3e},{dual:.3e},{primal:.3e},{comp:.3e}",
                if dense_q { "dense_hessian" } else { "chain" },
                sol.iterations,
                sol.kkt.symbolic_analyses,
                sol.kkt.sparse_refactorizations,
                sol.kkt.dense_fallbacks,
                sol.kkt.max_linear_backward_error
            );
        }
    }
    pub fn run() {
        println!(
            "case,n,backend,ms,peak_extra_bytes,alloc_calls,iterations,analyses,refactors,fallbacks,linear_backward_error,dual_residual,primal_violation,complementarity"
        );
        case(64, false);
        case(128, false);
        case(48, true);
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
