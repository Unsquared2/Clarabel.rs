# Benchmarks

`cvxpy_cached.py` times Clarabel through CVXPY with the compilation cached
(DPP): every problem has a `Parameter`, so after the first solve each
`problem.solve()` only refills the problem data, updates the solver CVXPY
keeps between solves, and solves.  It needs `cvxpy`, `numpy` and `scipy`.

    python benchmarks/cvxpy_cached.py [reps] [problem ...]

Problems: `rod` (stiff clamped rod on an obstacle), `mpc` (linear MPC),
`portfolio` (long-only factor-model portfolio), `lp_box` (sparse LP with box
bounds).  Pin it to one core (`taskset -c 0`) for stable timings.

For the Rust-level rod benchmark see `examples/rust/example_rod.rs`
(`cargo run --release --example rod`).
