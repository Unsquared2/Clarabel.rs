//! Tests of the bound eliminating KKT solver, through the public solver
//! interface.  Every problem is solved with the "auto" method (bound
//! elimination, banded or QDLDL backend) and checked against the "qdldl"
//! method, which uses the full quasidefinite KKT system.  The test hooks
//! force the less common backends: QDLDL instead of banded, and the single
//! chain instead of the twisted banded factorization.
#![allow(non_snake_case)]
#![allow(clippy::needless_range_loop, clippy::field_reassign_with_default, clippy::type_complexity)]

use super::banded::test_hooks::{NO_BANDED, NO_SCHUR, NO_TWIST};
use crate::algebra::*;
use crate::solver::*;

struct Prob {
    P: CscMatrix<f64>,
    q: Vec<f64>,
    A: CscMatrix<f64>,
    b: Vec<f64>,
    cones: Vec<SupportedConeT<f64>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Path {
    Upstream,
    Auto,
    AutoQdldlBackend,
    AutoSingleChain,
}

const PATHS: [Path; 4] = [Path::Upstream, Path::Auto, Path::AutoQdldlBackend, Path::AutoSingleChain];

struct Out {
    status: SolverStatus,
    s: Vec<f64>,
    x: Vec<f64>,
    z: Vec<f64>,
    obj: f64,
    solver_name: String,
}

fn settings(path: Path) -> DefaultSettings<f64> {
    let mut s = DefaultSettings::default();
    s.verbose = false;
    s.presolve_enable = false;
    s.direct_solve_method = if path == Path::Upstream { "qdldl" } else { "auto" }.into();
    s
}

fn solve_with(p: &Prob, path: Path, s: DefaultSettings<f64>) -> Out {
    NO_BANDED.with(|c| c.set(path == Path::AutoQdldlBackend));
    NO_SCHUR.with(|c| c.set(path == Path::AutoQdldlBackend));
    NO_TWIST.with(|c| c.set(path == Path::AutoSingleChain));
    let mut solver = DefaultSolver::new(&p.P, &p.q, &p.A, &p.b, &p.cones, s).unwrap();
    solver.solve();
    NO_BANDED.with(|c| c.set(false));
    NO_SCHUR.with(|c| c.set(false));
    NO_TWIST.with(|c| c.set(false));
    Out {
        status: solver.solution.status,
        s: solver.solution.s.clone(),
        x: solver.solution.x.clone(),
        z: solver.solution.z.clone(),
        obj: solver.solution.obj_val,
        solver_name: solver.info.linsolver.name.clone(),
    }
}

fn solve(p: &Prob, path: Path) -> Out {
    solve_with(p, path, settings(path))
}

fn csc(m: usize, n: usize, mut t: Vec<(usize, usize, f64)>) -> CscMatrix<f64> {
    t.sort_by_key(|e| (e.1, e.0));
    let mut colptr = vec![0usize; n + 1];
    for e in &t {
        colptr[e.1 + 1] += 1;
    }
    for j in 0..n {
        colptr[j + 1] += colptr[j];
    }
    CscMatrix::new(m, n, colptr, t.iter().map(|e| e.0).collect(), t.iter().map(|e| e.2).collect())
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        self.f() * 2.0 - 1.0
    }
}

/// clamped rod on an obstacle, `nel` elements (as examples/rust/example_rod.rs)
fn rod(nel: usize, ratio: f64) -> Prob {
    let n = nel + 1;
    let h = 1.0 / nel as f64;
    let mut t = vec![];
    for i in 1..n - 1 {
        let x = i as f64 * h;
        let c = if (0.3..0.7).contains(&x) { ratio } else { 1.0 } / (h * h * h);
        let idx = [i - 1, i, i + 1];
        let st = [1.0, -2.0, 1.0];
        for a in 0..3 {
            for bb in a..3 {
                t.push((idx[a], idx[bb], c * st[a] * st[bb]));
            }
        }
    }
    let P = csc(n, n, t).to_triu();
    let q = (0..n).map(|i| h * (100.0 + if (i as f64 * h - 0.2).abs() < 0.02 { 1e4 } else { 0.0 })).collect();
    let bc = [0, 1, n - 2, n - 1];
    let mut ta = vec![];
    for (r, &j) in bc.iter().enumerate() {
        ta.push((r, j, 1.0));
    }
    for j in 0..n {
        ta.push((4 + j, j, -1.0));
    }
    let A = csc(4 + n, n, ta);
    let mut b = vec![0.0; 4];
    b.extend((0..n).map(|i| {
        let x = i as f64 * h;
        -(-4.0e-3 + 1.2e-2 * (x - 0.45) * (x - 0.45))
    }));
    Prob { P, q, A, b, cones: vec![ZeroConeT(4), NonnegativeConeT(n)] }
}

/// random sparse feasible QP (or LP) with general rows and box bounds of
/// assorted coefficients, built around a known point
fn random_qp(seed: u64, n: usize, lp: bool) -> Prob {
    let mut g = Lcg(seed);
    let x0: Vec<f64> = (0..n).map(|_| g.n()).collect();
    let mut tp = vec![];
    if !lp {
        for _ in 0..2 * n {
            let (i, j) = ((g.f() * n as f64) as usize, (g.f() * n as f64) as usize);
            let (a, b) = (g.n(), g.n());
            // rank one term (a e_i + b e_j)(a e_i + b e_j)'
            tp.push((i, i, a * a));
            tp.push((j, j, b * b));
            if i != j {
                tp.push((i.min(j), i.max(j), a * b));
            }
        }
    }
    let mut P = csc(n, n, tp);
    P.canonicalize().unwrap();
    let q = (0..n).map(|_| g.n()).collect();
    let mut rows: Vec<(Vec<(usize, f64)>, f64, bool)> = vec![];
    let dot = |r: &Vec<(usize, f64)>, x: &[f64]| r.iter().map(|&(j, a)| a * x[j]).sum::<f64>();
    for k in 0..n / 2 {
        let r: Vec<(usize, f64)> = (0..3).map(|_| ((g.f() * n as f64) as usize, g.n())).collect();
        let mut r2: Vec<(usize, f64)> = vec![];
        for (j, a) in r {
            match r2.iter_mut().find(|e| e.0 == j) {
                Some(e) => e.1 += a,
                None => r2.push((j, a)),
            }
        }
        let rhs = dot(&r2, &x0);
        let eq = k % 3 == 0;
        rows.push((r2, if eq { rhs } else { rhs + g.f() }, eq));
    }
    for j in 0..n {
        let a = if g.f() < 0.7 { 1.0 } else { 0.1 + 3.0 * g.f() };
        rows.push((vec![(j, a)], a * x0[j] + g.f(), false));
        let a = if g.f() < 0.7 { 1.0 } else { 0.1 + 3.0 * g.f() };
        rows.push((vec![(j, -a)], -a * x0[j] + g.f(), false));
    }
    rows.sort_by_key(|r| !r.2);
    let neq = rows.iter().filter(|r| r.2).count();
    let mut t = vec![];
    for (i, (r, _, _)) in rows.iter().enumerate() {
        for &(j, a) in r {
            t.push((i, j, a));
        }
    }
    let A = csc(rows.len(), n, t);
    let b = rows.iter().map(|r| r.1).collect();
    let mut cones = vec![];
    if neq > 0 {
        cones.push(ZeroConeT(neq));
    }
    cones.push(NonnegativeConeT(rows.len() - neq));
    Prob { P, q, A, b, cones }
}

fn assert_close(a: &Out, b: &Out, tol: f64, what: &str) {
    assert_eq!(a.status, b.status, "{what}: status {:?} vs {:?}", a.status, b.status);
    if a.status == SolverStatus::Solved {
        let rel = (a.obj - b.obj).abs() / (1.0 + b.obj.abs());
        assert!(rel < tol, "{what}: objective {} vs {} ({:.1e})", a.obj, b.obj, rel);
        let dx = a.x.iter().zip(&b.x).fold(0.0f64, |m, (u, v)| m.max((u - v).abs()));
        let sx = b.x.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        assert!(dx / sx < tol.sqrt(), "{what}: x differs by {dx:.1e}");
    }
}

fn is_ok(st: SolverStatus) -> bool {
    matches!(st, SolverStatus::Solved | SolverStatus::AlmostSolved)
}

/// independent optimality check of a returned solution: primal
/// feasibility, cone membership, stationarity and complementarity
fn assert_kkt(p: &Prob, o: &Out, tol: f64, what: &str) {
    let inf = |v: &[f64]| v.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let mut rp = p.b.clone();
    p.A.gemv(&mut rp, &o.x, -1.0, 1.0); // b - Ax = s
    let ep = rp.iter().zip(&o.s).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    let mut rd = p.q.clone();
    p.P.sym_up().symv(&mut rd, &o.x, 1.0, 1.0);
    p.A.t().gemv(&mut rd, &o.z, 1.0, 1.0);
    let nz = match p.cones[0] {
        ZeroConeT(k) => k,
        _ => 0,
    };
    let (mut cone, mut comp) = (0.0f64, 0.0f64);
    for i in 0..p.b.len() {
        if i < nz {
            cone = cone.max(o.s[i].abs());
        } else {
            cone = cone.max((-o.s[i]).max(0.0)).max((-o.z[i]).max(0.0));
            comp = comp.max((o.s[i] * o.z[i]).abs());
        }
    }
    let sc = 1.0 + inf(&p.b).max(inf(&p.q)).max(inf(&o.x)).max(inf(&o.z));
    for (name, v) in [("primal", ep), ("dual", inf(&rd)), ("cone", cone), ("compl", comp)] {
        assert!(v / sc < tol, "{what}: {name} residual {:.1e}", v / sc);
    }
}

/// Solve on every path.  Where upstream succeeds the new paths must agree
/// with it; where it fails the default path must still find a solution
/// that passes an independent KKT check.
fn check_all_paths(p: &Prob, tol: f64, what: &str) -> Vec<Out> {
    let outs: Vec<Out> = PATHS.iter().map(|&path| solve(p, path)).collect();
    let up_ok = is_ok(outs[0].status)
        || matches!(outs[0].status, SolverStatus::PrimalInfeasible | SolverStatus::DualInfeasible);
    for (k, o) in outs.iter().enumerate().skip(1) {
        let w = format!("{what} {:?}", PATHS[k]);
        if up_ok {
            assert_close(o, &outs[0], tol, &w);
        } else if is_ok(o.status) {
            assert_kkt(p, o, 1e-6, &w);
        } else {
            assert!(PATHS[k] != Path::Auto, "{w}: upstream {:?}, new {:?}", outs[0].status, o.status);
        }
    }
    outs
}

#[test]
fn test_rods_all_paths() {
    // sizes either side of the twisted threshold, and stiffer variants
    for &(nel, ratio) in &[(6, 1.0), (12, 10.0), (35, 10.0), (36, 1.0), (100, 10.0), (400, 100.0)] {
        let outs = check_all_paths(&rod(nel, ratio), 1e-6, &format!("rod {nel} {ratio}"));
        assert!(outs[1].solver_name.starts_with("banded"), "{}", outs[1].solver_name);
        assert!(outs[2].solver_name.starts_with("qdldl+boundelim"), "{}", outs[2].solver_name);
    }
}

#[test]
fn test_random_qp_lp_all_paths() {
    for seed in 0..40 {
        let n = 3 + (seed as usize * 7) % 30;
        let p = random_qp(seed, n, seed % 3 == 0);
        check_all_paths(&p, 1e-6, &format!("random seed {seed} n {n}"));
    }
}

#[test]
fn test_all_rows_bounds() {
    // no general constraints at all: the reduced system is just P + D
    let n = 20;
    let mut p = random_qp(7, n, false);
    let nb = 2 * n;
    let m = p.A.m;
    let keep: Vec<usize> = (m - nb..m).collect();
    let mut t = vec![];
    for j in 0..n {
        for k in p.A.colptr[j]..p.A.colptr[j + 1] {
            let r = p.A.rowval[k];
            if r >= m - nb {
                t.push((r - (m - nb), j, p.A.nzval[k]));
            }
        }
    }
    p.A = csc(nb, n, t);
    p.b = keep.iter().map(|&r| p.b[r]).collect();
    p.cones = vec![NonnegativeConeT(nb)];
    check_all_paths(&p, 1e-6, "all bounds");
}

#[test]
fn test_infeasible_and_unbounded() {
    // contradictory bounds  x0 <= -1, x0 >= 1
    let P = CscMatrix::<f64>::identity(3);
    let A = csc(3, 3, vec![(0, 0, 1.0), (1, 0, -1.0), (2, 1, 1.0)]);
    let p = Prob { P, q: vec![1.0; 3], A, b: vec![-1.0, -1.0, 1.0], cones: vec![NonnegativeConeT(3)] };
    let outs = check_all_paths(&p, 1e-6, "infeasible");
    assert_eq!(outs[1].status, SolverStatus::PrimalInfeasible);

    // unbounded LP: min -x0 with only x0 >= 0
    let p = Prob {
        P: CscMatrix::zeros((2, 2)),
        q: vec![-1.0, 0.0],
        A: csc(2, 2, vec![(0, 0, -1.0), (1, 1, -1.0)]),
        b: vec![0.0, 0.0],
        cones: vec![NonnegativeConeT(2)],
    };
    let outs = check_all_paths(&p, 1e-6, "unbounded");
    assert_eq!(outs[1].status, SolverStatus::DualInfeasible);
}

#[test]
fn test_noncanonical_input() {
    // duplicate entries in P (diagonal and off diagonal) and in A, an
    // explicit zero bound coefficient, and an empty row of A
    let P = CscMatrix::new(
        3,
        3,
        vec![0, 2, 5, 6],
        vec![0, 0, 0, 1, 1, 2],
        vec![1.0, 1.0, 0.25, 1.0, 0.5, 2.0],
    );
    let A = CscMatrix::new(
        6,
        3,
        vec![0, 3, 5, 7],
        vec![0, 0, 4, 1, 2, 3, 5],
        vec![-0.5, -0.5, 0.0, -1.0, 1.0, -1.0, 0.0],
    );
    let p = Prob {
        P,
        q: vec![1.0, -2.0, 0.5],
        A,
        b: vec![1.0, 1.0, 2.0, 1.0, 1.0, 0.0],
        cones: vec![NonnegativeConeT(6)],
    };
    check_all_paths(&p, 1e-6, "noncanonical");
}

#[test]
fn test_repeat_solve_and_settings() {
    let p = rod(80, 10.0);
    for variant in 0..5 {
        let tweak = |s: &mut DefaultSettings<f64>| match variant {
            0 => s.static_regularization_enable = false,
            1 => s.iterative_refinement_enable = false,
            2 => s.equilibrate_enable = false,
            3 => s.dynamic_regularization_enable = false,
            _ => {}
        };
        let mut su = settings(Path::Upstream);
        tweak(&mut su);
        let up = solve_with(&p, Path::Upstream, su);
        for path in [Path::Auto, Path::AutoQdldlBackend] {
            let mut s = settings(path);
            tweak(&mut s);
            let o = solve_with(&p, path, s);
            let w = format!("{path:?} variant {variant}");
            if is_ok(up.status) {
                assert!(is_ok(o.status), "{w}: {:?}", o.status);
                let rel = (o.obj - up.obj).abs() / (1.0 + up.obj.abs());
                assert!(rel < 1e-4, "{w}: {rel:.1e}");
            } else if path == Path::Auto {
                // e.g. without static regularization upstream fails at
                // once, but the banded path needs none on this problem
                assert_eq!(o.status, SolverStatus::Solved, "{w}");
                assert_kkt(&p, &o, 1e-6, &w);
            }
        }
    }
    for path in [Path::Auto, Path::AutoQdldlBackend] {
        // solving twice gives the same answer
        let mut solver = DefaultSolver::new(&p.P, &p.q, &p.A, &p.b, &p.cones, settings(path)).unwrap();
        solver.solve();
        let x1 = solver.solution.x.clone();
        solver.solve();
        assert_eq!(x1, solver.solution.x);
    }
}

#[test]
fn test_data_updates() {
    let p = random_qp(11, 15, false);
    let mut g = Lcg(99);
    let P2: Vec<f64> = p.P.nzval.iter().map(|v| v * (1.0 + 0.3 * g.f())).collect();
    let A2: Vec<f64> = p.A.nzval.iter().map(|v| v * (1.0 + 0.1 * g.f())).collect();
    let q2: Vec<f64> = p.q.iter().map(|v| v + 0.2 * g.n()).collect();
    let b2: Vec<f64> = p.b.iter().map(|v| v + 0.5 + 0.1 * g.f()).collect();
    let mut p2 = Prob { P: p.P.clone(), q: q2.clone(), A: p.A.clone(), b: b2.clone(), cones: p.cones.clone() };
    p2.P.nzval = P2.clone();
    p2.A.nzval = A2.clone();
    for path in [Path::Auto, Path::AutoQdldlBackend, Path::Upstream] {
        let fresh = solve(&p2, path);
        let mut solver = DefaultSolver::new(&p.P, &p.q, &p.A, &p.b, &p.cones, settings(path)).unwrap();
        solver.solve();
        solver.update_P(&P2).unwrap();
        solver.update_A(&A2).unwrap();
        solver.update_q(&q2).unwrap();
        solver.update_b(&b2).unwrap();
        solver.solve();
        let updated = Out {
            status: solver.solution.status,
            s: solver.solution.s.clone(),
            x: solver.solution.x.clone(),
            z: solver.solution.z.clone(),
            obj: solver.solution.obj_val,
            solver_name: String::new(),
        };
        assert_close(&updated, &fresh, 1e-6, &format!("updates {path:?}"));
    }
}

#[test]
fn test_f32() {
    // single precision, on both backends, against upstream in f32
    let cv = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let cm = |M: &CscMatrix<f64>| CscMatrix::new(M.m, M.n, M.colptr.clone(), M.rowval.clone(), cv(&M.nzval));
    let mut chain = rod(40, 1.0);
    // a mild "rod": scale the stiffness down so that f32 can resolve it
    chain.P.nzval.iter_mut().for_each(|v| *v *= 1e-4);
    for (k, p) in [random_qp(3, 20, false), chain].iter().enumerate() {
        let solve32 = |method: &str| {
            // tolerances suited to single precision
            let mut s = DefaultSettings::<f32>::default();
            s.verbose = false;
            s.direct_solve_method = method.into();
            s.tol_feas = 1e-4;
            s.tol_gap_abs = 1e-4;
            s.tol_gap_rel = 1e-4;
            s.reduced_tol_feas = 1e-3;
            s.reduced_tol_gap_abs = 1e-3;
            s.reduced_tol_gap_rel = 1e-3;
            let mut solver = DefaultSolver::new(&cm(&p.P), &cv(&p.q), &cm(&p.A), &cv(&p.b), &p
                .cones
                .iter()
                .map(|c| match c {
                    ZeroConeT(n) => ZeroConeT(*n),
                    NonnegativeConeT(n) => NonnegativeConeT(*n),
                    _ => unreachable!(),
                })
                .collect::<Vec<_>>(), s)
            .unwrap();
            solver.solve();
            (solver.solution.status, solver.solution.obj_val)
        };
        // f32 is fragile upstream as well (regularization constants are
        // tuned for f64): require no regression relative to it
        let (su, ou) = solve32("qdldl");
        let (sn, on) = solve32("auto");
        println!("problem {k}: f32 upstream {su:?} {ou}, auto {sn:?} {on}");
        if is_ok(su) {
            assert!(is_ok(sn), "problem {k}: auto f32 {sn:?}");
            let rel = (on - ou).abs() / (1.0 + ou.abs());
            assert!(rel < 1e-3, "problem {k}: f32 objective {on} vs {ou}");
        }
    }
}

#[test]
fn test_duals_satisfy_kkt() {
    // the multipliers of eliminated rows are recovered from the reduced
    // solution: check stationarity P x + q + A'z = 0 directly
    for (k, p) in [rod(150, 10.0), random_qp(5, 25, false), random_qp(6, 25, true)].iter().enumerate() {
        let o = solve(p, Path::Auto);
        assert_eq!(o.status, SolverStatus::Solved, "problem {k}");
        let mut r = p.q.clone();
        p.P.sym_up().symv(&mut r, &o.x, 1.0, 1.0);
        p.A.t().gemv(&mut r, &o.z, 1.0, 1.0);
        let scale = 1.0 + p.q.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let err = r.iter().fold(0.0f64, |m, v| m.max(v.abs())) / scale;
        assert!(err < 1e-6, "problem {k}: stationarity residual {err:.1e}");
    }
}

#[test]
fn test_small_rods() {
    // small enough to run under Miri: single chain and twisted, both
    // backends
    for &nel in &[8, 40] {
        check_all_paths(&rod(nel, 10.0), 1e-6, &format!("small rod {nel}"));
    }
}

#[cfg(feature = "faer-sparse")]
#[test]
fn test_faer_backend() {
    // a dense QP with bounds: heavy enough that the "auto" rule selects
    // faer for the reduced system
    let n = 150;
    let mut g = Lcg(42);
    let B: Vec<Vec<f64>> = (0..n).map(|_| (0..n).map(|_| g.n()).collect()).collect();
    let mut t = vec![];
    for i in 0..n {
        for j in i..n {
            let v: f64 = (0..n).map(|k| B[k][i] * B[k][j]).sum::<f64>() / n as f64;
            t.push((i, j, v));
        }
    }
    let P = csc(n, n, t);
    let q = (0..n).map(|_| g.n()).collect();
    let mut ta = vec![];
    for j in 0..n {
        ta.push((2 * j, j, 1.0));
        ta.push((2 * j + 1, j, -1.0));
    }
    let A = csc(2 * n, n, ta);
    let b = vec![0.5; 2 * n];
    let p = Prob { P, q, A, b, cones: vec![NonnegativeConeT(2 * n)] };
    let up = solve(&p, Path::Upstream);
    let new = solve(&p, Path::Auto);
    assert!(new.solver_name.starts_with("faer"), "{}", new.solver_name);
    assert_close(&new, &up, 1e-6, "faer backend");
    assert_kkt(&p, &new, 1e-6, "faer backend");
}

#[test]
fn test_data_updates_banded() {
    // updates through the banded backend (band data and its reversed copy
    // must follow changes of P and A), single chain and twisted
    for &nel in &[10, 60] {
        let p = rod(nel, 10.0);
        let mut g = Lcg(5);
        // (a uniform scaling keeps P positive semidefinite)
        let P2: Vec<f64> = p.P.nzval.iter().map(|v| v * 1.7).collect();
        let A2: Vec<f64> = p.A.nzval.iter().map(|v| v * (1.0 + 0.2 * g.f())).collect();
        let q2: Vec<f64> = p.q.iter().map(|v| v * (1.0 + 0.3 * g.f())).collect();
        let mut p2 = Prob { P: p.P.clone(), q: q2.clone(), A: p.A.clone(), b: p.b.clone(), cones: p.cones.clone() };
        p2.P.nzval = P2.clone();
        p2.A.nzval = A2.clone();
        for path in [Path::Auto, Path::AutoSingleChain] {
            NO_TWIST.with(|c| c.set(path == Path::AutoSingleChain));
            let mut solver = DefaultSolver::new(&p.P, &p.q, &p.A, &p.b, &p.cones, settings(path)).unwrap();
            NO_TWIST.with(|c| c.set(false));
            assert!(solver.info.linsolver.name.starts_with("banded"));
            solver.solve();
            solver.update_P(&P2).unwrap();
            solver.update_A(&A2).unwrap();
            solver.update_q(&q2).unwrap();
            solver.solve();
            let updated = Out {
                status: solver.solution.status,
                s: solver.solution.s.clone(),
                x: solver.solution.x.clone(),
                z: solver.solution.z.clone(),
                obj: solver.solution.obj_val,
                solver_name: String::new(),
            };
            let fresh = solve(&p2, Path::Upstream);
            let w = format!("banded updates {nel} {path:?}");
            if is_ok(fresh.status) {
                assert_close(&updated, &fresh, 1e-6, &w);
            } else {
                // upstream fails on this one: check the answer directly
                assert_eq!(updated.status, SolverStatus::Solved, "{w}");
                assert_kkt(&p2, &updated, 1e-6, &w);
            }
        }
    }
}

/// the rod as a modelling tool states it: auxiliary variables t = W D w
/// for the bending energy ½‖t‖², with P = I on t only
fn rod_lsq(nel: usize, ratio: f64) -> Prob {
    let n = nel + 1;
    let nt = n - 2;
    let h = 1.0 / nel as f64;
    let P = csc(n + nt, n + nt, (0..nt).map(|i| (n + i, n + i, 1.0)).collect());
    let mut q: Vec<f64> = (0..n).map(|i| h * (100.0 + if (i as f64 * h - 0.2).abs() < 0.02 { 1e4 } else { 0.0 })).collect();
    q.resize(n + nt, 0.0);
    let mut ta = vec![];
    for (r, &j) in [0, 1, n - 2, n - 1].iter().enumerate() {
        ta.push((r, j, 1.0));
    }
    for i in 1..n - 1 {
        let x = i as f64 * h;
        let c = (if (0.3..0.7).contains(&x) { ratio } else { 1.0 } / (h * h * h)).sqrt();
        let r = 4 + i - 1;
        ta.push((r, n + i - 1, 1.0));
        for (k, s) in [(i - 1, 1.0), (i, -2.0), (i + 1, 1.0)] {
            ta.push((r, k, -c * s));
        }
    }
    let m0 = 4 + nt;
    for j in 0..n {
        ta.push((m0 + j, j, -1.0));
    }
    let A = csc(m0 + n, n + nt, ta);
    let mut b = vec![0.0; m0];
    b.extend((0..n).map(|i| {
        let x = i as f64 * h;
        -(-4.0e-3 + 1.2e-2 * (x - 0.45) * (x - 0.45))
    }));
    Prob { P, q, A, b, cones: vec![ZeroConeT(m0), NonnegativeConeT(n)] }
}

/// random feasible least-squares problem: boxed variables x and k
/// auxiliary variables t with diagonal weights, each in an equality row
/// t = Fx, an inequality row, a single-entry equality, or sharing that
/// equality with another auxiliary variable; plus general rows
fn random_lsq(seed: u64, n: usize, k: usize, lp_x: bool) -> Prob {
    let mut g = Lcg(seed);
    let x0: Vec<f64> = (0..n).map(|_| g.n()).collect();
    let mut tp = vec![];
    if !lp_x {
        for _ in 0..n {
            let (i, j) = ((g.f() * n as f64) as usize, (g.f() * n as f64) as usize);
            let (a, b) = (g.n(), g.n());
            tp.push((i, i, a * a));
            tp.push((j, j, b * b));
            if i != j {
                tp.push((i.min(j), i.max(j), a * b));
            }
        }
    }
    for i in 0..k {
        tp.push((n + i, n + i, 0.1 + 2.0 * g.f()));
    }
    let mut P = csc(n + k, n + k, tp);
    P.canonicalize().unwrap();
    let mut q: Vec<f64> = (0..n).map(|_| g.n()).collect();
    q.resize(n + k, 0.0);

    // rows: (entries, rhs, equality)
    let mut rows: Vec<(Vec<(usize, f64)>, f64, bool)> = vec![];
    let dot = |r: &Vec<(usize, f64)>, x: &[f64]| r.iter().map(|&(j, a)| a * x[j]).sum::<f64>();
    let rand_row = |g: &mut Lcg| {
        let mut r: Vec<(usize, f64)> = vec![];
        for _ in 0..3 {
            let (j, a) = ((g.f() * n as f64) as usize, g.n());
            match r.iter_mut().find(|e| e.0 == j) {
                Some(e) => e.1 += a,
                None => r.push((j, a)),
            }
        }
        r
    };
    let mut last_single = None;
    for i in 0..k {
        let a = if g.f() < 0.5 { 1.0 } else { 0.2 + 2.0 * g.n() };
        match i % 4 {
            0 | 1 => {
                let mut r = rand_row(&mut g);
                let v = dot(&r, &x0);
                r.push((n + i, a));
                if i % 4 == 0 {
                    rows.push((r, v + a * g.n(), true));
                } else {
                    let t0 = g.n();
                    rows.push((r, v + a * t0 + g.f(), false));
                }
                last_single = None;
            }
            2 => {
                last_single = Some(rows.len());
                rows.push((vec![(n + i, a)], g.n(), true));
            }
            _ => match last_single {
                Some(ri) => rows[ri].0.push((n + i, a)),
                None => rows.push((vec![(n + i, a)], g.n(), true)),
            },
        }
    }
    for _ in 0..n / 3 {
        let r = rand_row(&mut g);
        let v = dot(&r, &x0);
        rows.push((r, v + g.f(), false));
    }
    for j in 0..n {
        rows.push((vec![(j, 1.0)], x0[j] + g.f(), false));
        rows.push((vec![(j, -1.0)], -x0[j] + g.f(), false));
    }
    rows.sort_by_key(|r| !r.2);
    let neq = rows.iter().filter(|r| r.2).count();
    let mut t = vec![];
    for (i, (r, _, _)) in rows.iter().enumerate() {
        for &(j, a) in r {
            t.push((i, j, a));
        }
    }
    let A = csc(rows.len(), n + k, t);
    let b = rows.iter().map(|r| r.1).collect();
    let mut cones = vec![];
    if neq > 0 {
        cones.push(ZeroConeT(neq));
    }
    cones.push(NonnegativeConeT(rows.len() - neq));
    Prob { P, q, A, b, cones }
}

#[test]
fn test_aux_variables_rod() {
    // the auxiliary variables leave the system, so the band is narrow
    for &(nel, ratio) in &[(6, 1.0), (40, 10.0), (300, 10.0)] {
        let outs = check_all_paths(&rod_lsq(nel, ratio), 1e-6, &format!("rod_lsq {nel} {ratio}"));
        let name = &outs[1].solver_name;
        let kb: usize = name.trim_start_matches("banded(kb=").split(')').next().unwrap().parse().unwrap();
        assert!(kb <= 3, "{name}");
        assert_kkt(&rod_lsq(nel, ratio), &outs[1], 1e-6, "rod_lsq kkt");
    }
}

#[test]
fn test_aux_variables_random() {
    for seed in 0..40 {
        let (n, k) = (3 + (seed as usize * 5) % 20, 1 + (seed as usize * 3) % 12);
        let p = random_lsq(seed, n, k, seed % 3 == 0);
        let outs = check_all_paths(&p, 1e-6, &format!("random_lsq seed {seed} n {n} k {k}"));
        if is_ok(outs[1].status) {
            assert_kkt(&p, &outs[1], 1e-6, &format!("random_lsq seed {seed}"));
        }
    }
}

#[test]
fn test_aux_variables_updates() {
    // new weights and coefficients, one weight set to zero (that variable
    // then needs a regularized pivot), on every backend
    for (p, what) in [(random_lsq(3, 12, 9, false), "random"), (rod_lsq(30, 10.0), "rod")] {
        let mut g = Lcg(17);
        let n = p.q.len();
        let mut P2: Vec<f64> = p.P.nzval.iter().map(|v| v * (1.0 + 0.5 * g.f())).collect();
        // the last diagonal entry is an auxiliary weight
        *P2.last_mut().unwrap() = 0.0;
        let A2: Vec<f64> = p.A.nzval.iter().map(|v| v * (1.0 + 0.1 * g.f())).collect();
        let q2: Vec<f64> = p.q.iter().map(|v| v * (1.0 + 0.2 * g.f())).collect();
        assert!(p.P.rowval[p.P.nnz() - 1] == n - 1 && p.P.colptr[n] - p.P.colptr[n - 1] == 1);
        let mut p2 = Prob { P: p.P.clone(), q: q2.clone(), A: p.A.clone(), b: p.b.clone(), cones: p.cones.clone() };
        p2.P.nzval = P2.clone();
        p2.A.nzval = A2.clone();
        let fresh = solve(&p2, Path::Upstream);
        for path in [Path::Auto, Path::AutoQdldlBackend, Path::AutoSingleChain] {
            NO_BANDED.with(|c| c.set(path == Path::AutoQdldlBackend));
            NO_SCHUR.with(|c| c.set(path == Path::AutoQdldlBackend));
            NO_TWIST.with(|c| c.set(path == Path::AutoSingleChain));
            let mut solver = DefaultSolver::new(&p.P, &p.q, &p.A, &p.b, &p.cones, settings(path)).unwrap();
            NO_BANDED.with(|c| c.set(false));
            NO_SCHUR.with(|c| c.set(false));
            NO_TWIST.with(|c| c.set(false));
            solver.solve();
            solver.update_P(&P2).unwrap();
            solver.update_A(&A2).unwrap();
            solver.update_q(&q2).unwrap();
            solver.solve();
            let updated = Out {
                status: solver.solution.status,
                s: solver.solution.s.clone(),
                x: solver.solution.x.clone(),
                z: solver.solution.z.clone(),
                obj: solver.solution.obj_val,
                solver_name: String::new(),
            };
            let w = format!("aux updates {what} {path:?}");
            if is_ok(fresh.status) {
                assert_close(&updated, &fresh, 1e-6, &w);
            } else {
                assert!(is_ok(updated.status), "{w}: {:?}", updated.status);
                assert_kkt(&p2, &updated, 1e-6, &w);
            }
        }
    }
}

#[test]
fn test_aux_variables_small() {
    // small enough for Miri: every backend, with an auxiliary variable in
    // each kind of row, and the direct backend's two-lane residual
    check_all_paths(&rod_lsq(6, 1.0), 1e-6, "rod_lsq 6");
    check_all_paths(&random_lsq(1, 5, 4, false), 1e-6, "random_lsq small");
}

/// long-only factor-model portfolio as a modelling tool states it: risk
/// ‖Fᵀw‖² through k auxiliary variables plus diagonal idiosyncratic risk,
/// a budget and box bounds; with `sectors`, some sparse sector caps too,
/// and with `dup_budget` the budget row twice (dependent equalities)
fn portfolio(seed: u64, n: usize, k: usize, sectors: usize, dup_budget: bool) -> Prob {
    let mut g = Lcg(seed);
    let nv = n + k;
    let mut tp: Vec<(usize, usize, f64)> = (0..n).map(|i| (i, i, 0.1 * g.f())).collect();
    tp.extend((0..k).map(|i| (n + i, n + i, 1.0)));
    let P = csc(nv, nv, tp);
    let mut q: Vec<f64> = (0..n).map(|_| -0.05 * g.n()).collect();
    q.resize(nv, 0.0);
    let mut t = vec![];
    let mut b = vec![];
    // t = Fᵀw
    for i in 0..k {
        t.push((i, n + i, 1.0));
        for j in 0..n {
            t.push((i, j, -g.n() / (k as f64).sqrt()));
        }
        b.push(0.0);
    }
    let nbud = if dup_budget { 2 } else { 1 };
    for r in 0..nbud {
        for j in 0..n {
            t.push((k + r, j, 1.0));
        }
        b.push(1.0);
    }
    let neq = k + nbud;
    let mut m = neq;
    for s in 0..sectors {
        for j in (s..n).step_by(sectors.max(1) * 3) {
            t.push((m, j, 1.0));
        }
        b.push(0.4);
        m += 1;
    }
    for j in 0..n {
        t.push((m, j, -1.0));
        b.push(0.0);
        t.push((m + 1, j, 1.0));
        b.push(0.05_f64.max(2.0 / n as f64));
        m += 2;
    }
    let A = csc(m, nv, t);
    Prob { P, q, A, b, cones: vec![ZeroConeT(neq), NonnegativeConeT(m - neq)] }
}

#[test]
fn test_schur_backend() {
    // dense factor rows (dense mode), with sparse sector rows, and with a
    // repeated budget row, whose S is singular (exact factorization fails,
    // the regularized one is used)
    for (p, what) in [
        (portfolio(1, 60, 5, 0, false), "portfolio"),
        (portfolio(2, 80, 8, 3, false), "portfolio + sectors"),
        (portfolio(3, 40, 4, 0, true), "portfolio, repeated budget"),
        (portfolio(4, 200, 3, 12, false), "portfolio, many sectors"),
    ] {
        let outs = check_all_paths(&p, 1e-6, what);
        assert!(outs[1].solver_name.starts_with("schur"), "{what}: {}", outs[1].solver_name);
        assert!(outs[2].solver_name.starts_with("qdldl+boundelim"), "{what}: {}", outs[2].solver_name);
        assert_kkt(&p, &outs[1], 1e-6, what);
    }
    // LPs with every variable bounded also qualify (P = 0), unless the
    // banded backend takes them
    let mut nschur = 0;
    for seed in 0..10 {
        let p = random_qp(100 + seed, 30 + 3 * seed as usize, true);
        let outs = check_all_paths(&p, 1e-6, &format!("lp {seed}"));
        nschur += outs[1].solver_name.starts_with("schur") as usize;
    }
    assert!(nschur >= 5, "{nschur} of 10 LPs on the Schur backend");
}

#[test]
fn test_schur_refinement_fallback() {
    // with refinement tolerances of zero no solve counts as refined, so
    // every exact Schur solve falls back to the regularized factors (and
    // solves again); the answers must not change
    let p = portfolio(6, 60, 5, 2, false);
    let tight = |path: Path| {
        let mut s = settings(path);
        s.iterative_refinement_reltol = 0.0;
        s.iterative_refinement_abstol = 0.0;
        s
    };
    let up = solve_with(&p, Path::Upstream, tight(Path::Upstream));
    let new = solve_with(&p, Path::Auto, tight(Path::Auto));
    assert!(new.solver_name.starts_with("schur"), "{}", new.solver_name);
    assert_close(&new, &up, 1e-6, "schur fallback");
    assert_kkt(&p, &new, 1e-6, "schur fallback");
}

#[test]
fn test_schur_updates() {
    // new risk weights, factor loadings and returns, on the Schur backend
    // and on QDLDL, against a fresh solve
    let p = portfolio(5, 50, 4, 2, false);
    let mut g = Lcg(23);
    let P2: Vec<f64> = p.P.nzval.iter().map(|v| v * (1.0 + 0.5 * g.f())).collect();
    let A2: Vec<f64> = p.A.nzval.iter().map(|v| v * (1.0 + 0.1 * g.f())).collect();
    let q2: Vec<f64> = p.q.iter().map(|v| v + 0.02 * g.n()).collect();
    let mut p2 = Prob { P: p.P.clone(), q: q2.clone(), A: p.A.clone(), b: p.b.clone(), cones: p.cones.clone() };
    p2.P.nzval = P2.clone();
    p2.A.nzval = A2.clone();
    let fresh = solve(&p2, Path::Upstream);
    for path in [Path::Auto, Path::AutoQdldlBackend] {
        NO_SCHUR.with(|c| c.set(path == Path::AutoQdldlBackend));
        let mut solver = DefaultSolver::new(&p.P, &p.q, &p.A, &p.b, &p.cones, settings(path)).unwrap();
        NO_SCHUR.with(|c| c.set(false));
        let name = solver.info.linsolver.name.clone();
        assert_eq!(name.starts_with("schur"), path == Path::Auto, "{name}");
        solver.solve();
        solver.update_P(&P2).unwrap();
        solver.update_A(&A2).unwrap();
        solver.update_q(&q2).unwrap();
        solver.solve();
        let updated = Out {
            status: solver.solution.status,
            s: solver.solution.s.clone(),
            x: solver.solution.x.clone(),
            z: solver.solution.z.clone(),
            obj: solver.solution.obj_val,
            solver_name: String::new(),
        };
        assert_close(&updated, &fresh, 1e-6, &format!("schur updates {path:?}"));
    }
}
