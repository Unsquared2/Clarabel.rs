"""cvxpy benchmarks with the compilation cached (DPP).

Each problem has a Parameter, so the first solve compiles it and later
solves reuse the cached canonicalization: they refill the problem data,
update the Clarabel solver that cvxpy keeps between solves, and solve.
Reports, over re-solves with fresh parameter values, the median of
  solve()     problem.solve() end to end
  clarabel    the solver's own solve time
plus the first (compiling) solve.

  rod        stiff clamped rod on an obstacle (parameter: the load)
  mpc        linear MPC with input/state bounds (parameter: initial state)
  portfolio  long-only factor-model portfolio (parameter: expected returns)
  lp_box     sparse LP with box bounds (parameter: right-hand side)

Usage: python cvxpy_cached.py [reps] [problem ...]
"""
import sys
import time

import numpy as np
import scipy.sparse as sp

import cvxpy as cp


def rod(nel=1000, ratio=10.0, seed=0):
    rng = np.random.default_rng(seed)
    n = nel + 1
    h = 1.0 / nel
    x = np.arange(n) * h
    ei = np.where((x > 0.3) & (x < 0.7), ratio, 1.0)
    D = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    W = sp.diags(np.sqrt(ei[1:-1] / h**3))
    g = -4e-3 + 1.2e-2 * (x - 0.45) ** 2
    f = cp.Parameter(n)
    w = cp.Variable(n)
    obj = 0.5 * cp.sum_squares(W @ D @ w) - h * f @ w
    cons = [w[[0, 1, n - 2, n - 1]] == 0, w >= g]

    def fresh():
        f.value = -100.0 * (1 + 0.1 * rng.standard_normal(n)) \
            - np.where(np.abs(x - 0.2) < 0.02, 1e4, 0.0)
    return cp.Problem(cp.Minimize(obj), cons), fresh


def mpc(T=300, nx=4, nu=2, seed=0):
    rng = np.random.default_rng(seed)
    A = np.eye(nx) + 0.1 * rng.standard_normal((nx, nx)) / np.sqrt(nx)
    B = rng.standard_normal((nx, nu))
    x0 = cp.Parameter(nx)
    x = cp.Variable((nx, T + 1))
    u = cp.Variable((nu, T))
    cons = [x[:, 0] == x0]
    cons += [x[:, 1:] == A @ x[:, :-1] + B @ u]
    cons += [cp.abs(u) <= 1, cp.abs(x) <= 5]
    obj = cp.sum_squares(x) + 0.1 * cp.sum_squares(u)

    def fresh():
        x0.value = rng.standard_normal(nx)
    return cp.Problem(cp.Minimize(obj), cons), fresh


def portfolio(n=500, k=20, seed=0):
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((n, k)) / np.sqrt(k)
    d = rng.uniform(0.0, 0.1, n)
    mu = cp.Parameter(n)
    wv = cp.Variable(n)
    risk = cp.sum_squares(F.T @ wv) + cp.sum(cp.multiply(d, cp.square(wv)))
    cons = [cp.sum(wv) == 1, wv >= 0, wv <= 0.05]

    def fresh():
        mu.value = rng.standard_normal(n) * 0.05
    return cp.Problem(cp.Minimize(risk - mu @ wv), cons), fresh


def lp_box(m=300, n=600, seed=0):
    rng = np.random.default_rng(seed)
    A = sp.random(m, n, density=0.02, random_state=seed, format="csr")
    x0 = rng.uniform(0, 1, n)
    c = rng.standard_normal(n)
    b = cp.Parameter(m)
    xv = cp.Variable(n)

    def fresh():
        b.value = A @ x0 + rng.uniform(0, 1, m)
    return cp.Problem(cp.Minimize(c @ xv), [A @ xv <= b, xv >= 0, xv <= 1]), fresh


PROBLEMS = {"rod": rod, "mpc": mpc, "portfolio": portfolio, "lp_box": lp_box}


def bench(make, reps):
    prob, fresh = make()
    assert prob.is_dpp()
    fresh()
    t0 = time.perf_counter()
    prob.solve(solver=cp.CLARABEL)
    first = time.perf_counter() - t0
    tot, sol = [], []
    for _ in range(reps):
        fresh()
        t0 = time.perf_counter()
        prob.solve(solver=cp.CLARABEL)
        tot.append(time.perf_counter() - t0)
        sol.append(prob.solver_stats.solve_time)
        assert prob.status == cp.OPTIMAL, prob.status
    return prob, first, float(np.median(tot)), float(np.median(sol))


if __name__ == "__main__":
    import clarabel

    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    names = sys.argv[2:] or list(PROBLEMS)
    print(f"clarabel {clarabel.__version__}")
    for name in names:
        prob, first, t, ts = bench(PROBLEMS[name], reps)
        print(f"{name:10s} iters {prob.solver_stats.num_iters:3d}  first {first*1e3:7.1f} ms  "
              f"cached solve() {t*1e3:7.2f} ms  clarabel {ts*1e3:7.2f} ms")
