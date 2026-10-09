#![allow(non_snake_case)]
//! Stiff QP benchmark: bending of a clamped elastic rod (Euler-Bernoulli beam)
//! resting on a rigid obstacle.
//!
//!   minimize   1/2 sum_i EI_i * h * kappa_i^2  -  sum_i h f_i w_i
//!   subject to w_0 = w_1 = 0, w_N-1 = w_N = 0          (clamped ends)
//!              w_i >= g(x_i)                           (obstacle contact)
//!
//! with kappa_i = (w_{i-1} - 2 w_i + w_{i+1}) / h^2.  The Hessian scales like
//! EI/h^3 and has a condition number ~ N^4 * max(EI)/min(EI), i.e. the problem
//! is very stiff.  The bending stiffness jumps by 4 orders of magnitude along
//! the rod to make it worse.
use clarabel::algebra::*;
use clarabel::solver::*;
use std::time::Instant;

struct Rod {
    P: CscMatrix<f64>,
    q: Vec<f64>,
    A: CscMatrix<f64>,
    b: Vec<f64>,
    cones: Vec<SupportedConeT<f64>>,
}

fn build_rod(nel: usize, eir: f64) -> Rod {
    let n = nel + 1; // nodes
    let h = 1.0 / nel as f64;
    let x = |i: usize| i as f64 * h;
    // bending stiffness: stiff core segment with soft ends
    let ei = |xi: f64| if (0.3..0.7).contains(&xi) { eir } else { 1.0 };
    // load: uniform gravity plus a concentrated load near x = 0.2
    let f = |xi: f64| -1.0e2 - if (xi - 0.2).abs() < 0.02 { 1.0e4 } else { 0.0 };
    // obstacle: a smooth bump below the rod
    let g = |xi: f64| -4.0e-3 + 1.2e-2 * (xi - 0.45) * (xi - 0.45);

    // P = sum_i EI_i/h^3 * d_i d_i', d_i = e_{i-1} - 2 e_i + e_{i+1}
    // assemble upper triangle in CSC (pentadiagonal)
    let mut diag = vec![[0.0f64; 3]; n]; // diag[j][k] = P[j-k, j]
    let stencil = [1.0, -2.0, 1.0];
    for i in 1..n - 1 {
        let c = ei(x(i)) / (h * h * h);
        let idx = [i - 1, i, i + 1];
        for a in 0..3 {
            for bb in a..3 {
                let (r, col) = (idx[a], idx[bb]);
                diag[col][col - r] += c * stencil[a] * stencil[bb];
            }
        }
    }
    let mut colptr = vec![0usize];
    let mut rowval = vec![];
    let mut nzval = vec![];
    for j in 0..n {
        for k in (0..3).rev() {
            if j >= k && (k == 0 || diag[j][k] != 0.0) {
                rowval.push(j - k);
                nzval.push(diag[j][k]);
            }
        }
        colptr.push(rowval.len());
    }
    let P = CscMatrix::new(n, n, colptr, rowval, nzval);
    let q: Vec<f64> = (0..n).map(|i| -h * f(x(i))).collect();

    // A = [ E_bc ; -I ],  b = [0 ; -g]
    let bc = [0, 1, n - 2, n - 1];
    let m = bc.len() + n;
    let mut trip: Vec<(usize, usize, f64)> = vec![];
    for (r, &j) in bc.iter().enumerate() {
        trip.push((r, j, 1.0));
    }
    for j in 0..n {
        trip.push((bc.len() + j, j, -1.0));
    }
    trip.sort_by_key(|t| (t.1, t.0));
    let mut colptr = vec![0usize; n + 1];
    for t in &trip {
        colptr[t.1 + 1] += 1;
    }
    for j in 0..n {
        colptr[j + 1] += colptr[j];
    }
    let rowval = trip.iter().map(|t| t.0).collect();
    let nzval = trip.iter().map(|t| t.2).collect();
    let A = CscMatrix::new(m, n, colptr, rowval, nzval);
    let mut b = vec![0.0; bc.len()];
    b.extend((0..n).map(|i| -g(x(i))));
    let cones = vec![ZeroConeT(bc.len()), NonnegativeConeT(n)];
    Rod { P, q, A, b, cones }
}

fn solve(rod: &Rod, verbose: bool) -> DefaultSolver<f64> {
    let settings = DefaultSettingsBuilder::default()
        .verbose(verbose)
        .build()
        .unwrap();
    let mut solver =
        DefaultSolver::new(&rod.P, &rod.q, &rod.A, &rod.b, &rod.cones, settings).unwrap();
    solver.solve();
    solver
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let nel: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(1000);
    let reps: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(20);
    let verbose = args.get(3).map(|s| s == "v").unwrap_or(false);

    let eir: f64 = std::env::var("ROD_EI_RATIO").ok().map(|s| s.parse().unwrap()).unwrap_or(1.0e1);
    let rod = build_rod(nel, eir);
    let s = solve(&rod, verbose);
    let x = &s.solution.x;
    let n = x.len();
    let (imin, wmin) = x
        .iter()
        .enumerate()
        .fold((0, f64::MAX), |a, (i, &v)| if v < a.1 { (i, v) } else { a });
    let contacts = s.solution.z[4..].iter().filter(|&&z| z > 1e-6).count();
    println!(
        "nel={} status={:?} iters={} obj={:.12e} min w={:.9e} at x={:.4} contacts={} w(mid)={:.9e}",
        nel,
        s.solution.status,
        s.solution.iterations,
        s.solution.obj_val,
        wmin,
        imin as f64 / nel as f64,
        contacts,
        x[n / 2]
    );
    let digest: f64 = x.iter().enumerate().map(|(i, v)| v * (1.0 + (i % 7) as f64)).sum();
    println!("digest={:.12e}", digest);

    if reps == 0 {
        return;
    }
    let mut times = vec![];
    let mut tsetup = vec![];
    for _ in 0..reps {
        let t0 = Instant::now();
        let settings = DefaultSettingsBuilder::default().verbose(false).build().unwrap();
        let mut s =
            DefaultSolver::new(&rod.P, &rod.q, &rod.A, &rod.b, &rod.cones, settings).unwrap();
        tsetup.push(t0.elapsed().as_secs_f64() * 1e3);
        if std::env::var("SETUP_ONLY").is_ok() {
            times.push(0.0);
            continue;
        }
        s.solve();
        std::hint::black_box(&s.solution.x);
        times.push(t0.elapsed().as_secs_f64() * 1e3);
        if std::env::var("TIMERS").is_ok() && times.len() == reps {
            s.timers.as_ref().unwrap().print();
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    tsetup.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "setup+solve time: median {:.3} ms  min {:.3} ms  (setup median {:.3} ms, {} reps)",
        times[reps / 2],
        times[0],
        tsetup[reps / 2],
        reps
    );
}
