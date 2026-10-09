#![allow(non_snake_case)]
// index loops are the clearest form for these numeric kernels
#![allow(clippy::needless_range_loop)]
//! Banded LDLᵀ factorisation for quasidefinite matrices whose sparsity
//! pattern admits a small bandwidth after (reverse) Cuthill-McKee ordering.
//!
//! The band is stored densely, row-major, with KB leading and trailing
//! padding rows so that all kernels are branch free.  The bandwidth is a
//! const generic so that the inner loops are fully unrolled and the sliding
//! window of the recurrence stays in registers.  Solves operate on NR
//! right-hand sides at once, interleaved as `[T; NR]`, which turns the
//! latency-bound substitution recurrences into SIMD operations across the
//! right-hand sides.
//!
//! Memory safety: the hot loops index without bounds checks.  Every kernel
//! asserts the lengths of its arrays on entry (band arrays of
//! (n + 2 kb) kb entries, reversed band arrays of (nb + 3 kb) kb entries,
//! vectors of n + 2 kb padded rows), and the loop bounds keep all row
//! indices inside those lengths; the derivations are given at each loop.

use crate::algebra::*;

pub const BANDED_MAX_KB: usize = 8;

// Test-only switches that force the less common code paths.  They compile
// to constants outside of `cfg(test)`.
pub(crate) mod test_hooks {
    #[cfg(test)]
    thread_local! {
        pub(crate) static NO_TWIST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        pub(crate) static NO_BANDED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    #[inline]
    pub(crate) fn no_twist() -> bool {
        #[cfg(test)]
        return NO_TWIST.with(|c| c.get());
        #[cfg(not(test))]
        false
    }
    #[inline]
    pub(crate) fn no_banded() -> bool {
        #[cfg(test)]
        return NO_BANDED.with(|c| c.get());
        #[cfg(not(test))]
        false
    }
}

pub struct BandedLDL<T> {
    pub n: usize,
    pub kb: usize,
    // perm[new] = old, iperm[old] = new
    #[allow(dead_code)]
    pub perm: Vec<usize>,
    pub iperm: Vec<usize>,
    // true matrix values: band (row-major, padded) and diagonal
    pub a_band: Vec<T>,
    pub a_diag: Vec<T>,
    // static offsets applied to the diagonal for factorisation only
    pub d_shift: Vec<T>,
    // factors
    l_band: Vec<T>,
    dinv: Vec<T>,
    // expected signs (permuted)
    signs: Vec<T>,
    reg_eps: T,
    reg_delta: T,
    pub regularize_count: usize,
    // smallest |pivot| in the last factorization
    pub min_pivot: T,

    // Twisted ("burn at both ends") mode: rows [0,h) are eliminated top
    // down and rows [h+kb,n) bottom up as two independent chains, so that
    // the latency-bound recurrences of the factorization and of both
    // substitutions run two at a time.  The kb separator rows [h,h+kb) are
    // eliminated last.  The bottom chain is stored as a reversed band:
    // reversed row t is original row n-1-t, and slot q of it holds the
    // entry in original column (n-1-t)+kb-q.
    twisted: Option<usize>,
    a_rev: Vec<T>,
    lb_rev: Vec<T>,
    // a_rev must be rebuilt from a_band before the next factorization
    rev_dirty: bool,
}

/// Reverse Cuthill-McKee ordering of a symmetric quasidefinite pattern
/// given in triangular CSC form, with every negative-sign node (a
/// constraint row, zero diagonal before regularization) moved to just
/// after its last neighbour so that its pivot is a proper Schur complement
/// rather than the bare regularization term.   Returns (perm, bandwidth).
pub fn rcm_ordering<T: FloatT>(K: &CscMatrix<T>, signs: &[i8]) -> (Vec<usize>, usize) {
    let n = K.n;
    // build full adjacency lists
    let mut deg = vec![0usize; n];
    for j in 0..n {
        for k in K.colptr[j]..K.colptr[j + 1] {
            let i = K.rowval[k];
            if i != j {
                deg[i] += 1;
                deg[j] += 1;
            }
        }
    }
    let mut ptr = vec![0usize; n + 1];
    for i in 0..n {
        ptr[i + 1] = ptr[i] + deg[i];
    }
    let mut fill = ptr.clone();
    let mut adj = vec![0usize; ptr[n]];
    for j in 0..n {
        for k in K.colptr[j]..K.colptr[j + 1] {
            let i = K.rowval[k];
            if i != j {
                adj[fill[i]] = j;
                fill[i] += 1;
                adj[fill[j]] = i;
                fill[j] += 1;
            }
        }
    }
    for i in 0..n {
        let a = &mut adj[ptr[i]..ptr[i + 1]];
        if a.len() > 1 {
            a.sort_unstable_by_key(|&v| deg[v]);
        }
    }

    // BFS from root over unvisited nodes, appending to `order`.  Nodes
    // are marked with `stamp`, so trial sweeps need no clearing.
    let mut mark = vec![0u32; n];
    let mut level = vec![0usize; n];
    let bfs = |root: usize, stamp: u32, mark: &mut [u32], order: &mut Vec<usize>, level: &mut [usize]| {
        let start = order.len();
        mark[root] = stamp;
        level[root] = 0;
        order.push(root);
        let mut head = start;
        while head < order.len() {
            let v = order[head];
            head += 1;
            for &w in &adj[ptr[v]..ptr[v + 1]] {
                // nodes with mark 1 were placed by an earlier component
                if mark[w] != stamp && mark[w] != 1 {
                    mark[w] = stamp;
                    level[w] = level[v] + 1;
                    order.push(w);
                }
            }
        }
    };

    let mut order = Vec::with_capacity(n);
    let mut trial = Vec::with_capacity(n);
    let mut stamp = 1u32;
    for comp_root in 0..n {
        if mark[comp_root] == 1 {
            continue;
        }
        // pseudo-peripheral node search: up to two extra BFS sweeps
        let mut root = comp_root;
        let mut best_depth = 0;
        for sweep in 0..3 {
            stamp += 1;
            trial.clear();
            bfs(root, stamp, &mut mark, &mut trial, &mut level);
            let last_level = level[*trial.last().unwrap()];
            if sweep > 0 && last_level <= best_depth {
                break;
            }
            best_depth = last_level;
            let first = trial.len() - trial.iter().rev().take_while(|&&v| level[v] == last_level).count();
            root = trial[first..]
                .iter()
                .min_by_key(|&&v| deg[v])
                .copied()
                .unwrap();
        }
        let start = order.len();
        bfs(root, u32::MAX, &mut mark, &mut order, &mut level);
        for &v in &order[start..] {
            mark[v] = 1;
        }
    }
    order.reverse();

    let bandwidth = |order: &[usize]| -> usize {
        let mut ip = vec![0usize; n];
        for (p, &v) in order.iter().enumerate() {
            ip[v] = p;
        }
        (0..n)
            .flat_map(|v| adj[ptr[v]..ptr[v + 1]].iter().map(move |&w| (v, w)))
            .map(|(v, w)| ip[v].abs_diff(ip[w]))
            .max()
            .unwrap_or(0)
    };

    // Placing the negative nodes can only widen the band slightly, so
    // when plain RCM is far from banded the matrix goes to a general
    // sparse solver and the placement is skipped.
    let bw0 = bandwidth(&order);
    if bw0 > 2 * BANDED_MAX_KB {
        return (order, bw0);
    }

    // Twisted mode (see BandedLDL) eliminates the bottom half bottom up,
    // so negative nodes there go before their neighbours.  If the result
    // is too short for twisted mode, place every node after them instead.
    let mut placed = order.clone();
    place_negative_nodes(&mut placed, signs, &adj, &ptr, true);
    let mut bw = bandwidth(&placed);
    if n < 4 * bw.max(1) + 4 {
        placed = order;
        place_negative_nodes(&mut placed, signs, &adj, &ptr, false);
        bw = bandwidth(&placed);
    }
    (placed, bw)
}

/// Move negative nodes (constraint rows, zero diagonal before
/// regularization) so that they are eliminated after their positive
/// neighbours, making their pivots proper Schur complements.  With
/// `twisted`, nodes in the bottom half of the ordering are placed before
/// their first neighbour instead, since that half is eliminated bottom up.
///
/// Inserting a node between two positive nodes stretches the band around
/// it, so with few negative nodes each one tries a few offsets from its
/// anchor and keeps the best, judged on the edges near the insertion point
/// (the only ones whose length changes): bandwidth first, then the number
/// of edges at that bandwidth, then total squared edge length.  With many
/// negative nodes they are placed directly at their anchors by a sort.
fn place_negative_nodes(order: &mut Vec<usize>, signs: &[i8], adj: &[usize], ptr: &[usize], twisted: bool) {
    let n = order.len();
    let negs: Vec<usize> = order.iter().copied().filter(|&v| signs[v] < 0).collect();
    if negs.is_empty() {
        return;
    }
    let nbrs = |v: usize| adj[ptr[v]..ptr[v + 1]].iter().copied().filter(|&w| signs[w] > 0);

    if negs.len() > 64 {
        // keys 2p+1 for positions p; anchors at even keys around them
        let mut key = vec![0usize; n];
        for (p, &v) in order.iter().enumerate() {
            key[v] = 2 * p + 1;
        }
        let mut dkey = key.clone();
        for &v in &negs {
            let (lo, hi) = (nbrs(v).map(|w| key[w]).min(), nbrs(v).map(|w| key[w]).max());
            if let (Some(lo), Some(hi)) = (lo, hi) {
                dkey[v] = if twisted && key[v] >= n { lo - 1 } else { hi + 1 };
            }
        }
        order.sort_by_key(|&v| dkey[v]);
        return;
    }

    let mut ip = vec![0usize; n];
    let mut moved = false;
    for (cnt, &v) in negs.iter().cycle().take(negs.len() * 3).enumerate() {
        // stop once a full sweep has left every node in place
        if cnt > 0 && cnt % negs.len() == 0 {
            if !moved {
                break;
            }
            moved = false;
        }
        let cur = order.iter().position(|&w| w == v).unwrap();
        order.remove(cur);
        for (p, &w) in order.iter().enumerate() {
            ip[w] = p;
        }
        let (lo, hi) = match (nbrs(v).map(|w| ip[w]).min(), nbrs(v).map(|w| ip[w]).max()) {
            (Some(lo), Some(hi)) => (lo, hi),
            _ => {
                order.insert(cur, v);
                continue;
            }
        };
        // candidate insertion indices (v is inserted before order[k])
        let before = twisted && cur >= order.len() / 2;
        let cands: Vec<usize> = if before {
            (0..4).map(|d| lo.saturating_sub(d)).collect()
        } else {
            (0..4).map(|d| (hi + 1 + d).min(order.len())).collect()
        };
        const L: usize = BANDED_MAX_KB + 4;
        let (cmin, cmax) = (*cands.iter().min().unwrap(), *cands.iter().max().unwrap());
        let window = &order[cmin.saturating_sub(L)..(cmax + L).min(order.len())];
        let mut best = cands[0];
        let mut bestscore = (usize::MAX, 0, 0);
        for &k in &cands {
            let pos = |w: usize| if w == v { k } else { ip[w] + (ip[w] >= k) as usize };
            let (mut bw, mut cnt, mut sq) = (0, 0, 0);
            for &u in window.iter().chain(std::iter::once(&v)) {
                for &w in &adj[ptr[u]..ptr[u + 1]] {
                    let d = pos(u).abs_diff(pos(w));
                    if d > bw {
                        bw = d;
                        cnt = 0;
                    }
                    cnt += (d == bw) as usize;
                    sq += d * d;
                }
            }
            if (bw, cnt, sq) < bestscore {
                bestscore = (bw, cnt, sq);
                best = k;
            }
        }
        moved |= best != cur;
        order.insert(best, v);
    }
}

/// position of a matrix entry in the band storage
#[derive(Clone, Copy)]
pub enum BandPos {
    Diag(usize),
    Off(usize),
}

impl<T: FloatT> BandedLDL<T> {
    pub fn new(perm: Vec<usize>, kb: usize, signs: &[i8], reg_eps: T, reg_delta: T) -> Self {
        let mut out = Self::new_single(perm, kb, signs, reg_eps, reg_delta);
        let (n, kb) = (out.n, out.kb);
        if n >= 4 * kb + 4 && !test_hooks::no_twist() {
            let h = (n - kb) / 2;
            let nb = n - h - kb;
            out.twisted = Some(h);
            out.a_rev = vec![T::zero(); (nb + 3 * kb) * kb];
            out.lb_rev = vec![T::zero(); (nb + 3 * kb) * kb];
        }
        out
    }

    fn new_single(perm: Vec<usize>, kb: usize, signs: &[i8], reg_eps: T, reg_delta: T) -> Self {
        let n = perm.len();
        let kb = kb.max(1);
        let mut iperm = vec![0usize; n];
        for (new, &old) in perm.iter().enumerate() {
            iperm[old] = new;
        }
        let rows = n + 2 * kb;
        let mut psigns = vec![T::one(); n];
        for (new, &old) in perm.iter().enumerate() {
            psigns[new] = if signs[old] < 0 { -T::one() } else { T::one() };
        }
        Self {
            n,
            kb,
            perm,
            iperm,
            a_band: vec![T::zero(); rows * kb],
            a_diag: vec![T::zero(); n],
            d_shift: vec![T::zero(); n],
            l_band: vec![T::zero(); rows * kb],
            dinv: vec![T::zero(); rows],
            signs: psigns,
            reg_eps,
            reg_delta,
            regularize_count: 0,
            min_pivot: T::infinity(),
            twisted: None,
            a_rev: vec![],
            lb_rev: vec![],
            rev_dirty: true,
        }
    }

    /// band storage position of entry (i,j) in original (unpermuted) indices
    pub fn position(&self, i: usize, j: usize) -> BandPos {
        let (a, b) = (self.iperm[i], self.iperm[j]);
        let (r, c) = if a > b { (a, b) } else { (b, a) };
        if r == c {
            BandPos::Diag(r)
        } else {
            debug_assert!(r - c <= self.kb);
            BandPos::Off((r + self.kb) * self.kb + (c + self.kb - r))
        }
    }

    /// To be called after the off-diagonal band data `a_band` changes.
    /// (Changes to the diagonal need no notification.)
    pub fn band_changed(&mut self) {
        self.rev_dirty = true;
    }

    pub fn refactor(&mut self) -> bool {
        if self.twisted.is_some() {
            return match self.kb {
                1 => self.factor_twisted::<1>(),
                2 => self.factor_twisted::<2>(),
                3 => self.factor_twisted::<3>(),
                4 => self.factor_twisted::<4>(),
                5 => self.factor_twisted::<5>(),
                6 => self.factor_twisted::<6>(),
                7 => self.factor_twisted::<7>(),
                8 => self.factor_twisted::<8>(),
                _ => unreachable!(),
            };
        }
        match self.kb {
            1 => self.factor_kb::<1>(),
            2 => self.factor_kb::<2>(),
            3 => self.factor_kb::<3>(),
            4 => self.factor_kb::<4>(),
            5 => self.factor_kb::<5>(),
            6 => self.factor_kb::<6>(),
            7 => self.factor_kb::<7>(),
            8 => self.factor_kb::<8>(),
            _ => unreachable!(),
        }
    }


    // pivot regularization and bookkeeping shared by all factor kernels
    #[inline(always)]
    fn finish_pivot(&self, di: T, i: usize, count: &mut usize, min_piv: &mut T) -> Option<T> {
        let sign = self.signs[i];
        // dynamic regularization of pivots that are small or of the wrong
        // sign, branch free so the pivot recurrence stays short.  A NaN
        // pivot is left alone (as in QDLDL) so that the failure surfaces.
        let bad = di * sign < self.reg_eps;
        let di = if bad { self.reg_delta * sign } else { di };
        *count += bad as usize;
        if di == T::zero() {
            return None;
        }
        let a = di.abs();
        *min_piv = if a < *min_piv || a.is_nan() { a } else { *min_piv };
        Some(T::recip(di))
    }

    #[inline(never)]
    fn factor_twisted<const KB: usize>(&mut self) -> bool {
        let n = self.n;
        let h = self.twisted.unwrap();
        let nb = n - h - KB;
        let idx = |r: usize, q: usize| (r + KB) * KB + q;

        // reversed band of the bottom part (and of the separator rows),
        // rebuilt only when the band data has changed
        if self.rev_dirty {
            for t in 0..nb + KB {
                let i = n - 1 - t;
                for q in 0..KB {
                    self.a_rev[idx(t, q)] = self.a_band[idx(i + KB - q, q)];
                }
            }
            self.rev_dirty = false;
        }

        let mut count = 0;
        let mut min_neg = T::infinity();

        // one row of the left-looking banded LDL recurrence
        #[inline(always)]
        fn row<T: FloatT, const KB: usize>(
            a: &[T],
            mut di: T,
            win_l: &[[T; KB]; KB],
            win_dinv: &[T; KB],
        ) -> ([T; KB], T) {
            let mut u = [T::zero(); KB];
            let mut l = [T::zero(); KB];
            for s in 0..KB {
                let mut acc = a[s];
                for t in 0..s {
                    acc = fmadd(-u[t], win_l[s][t + KB - s], acc);
                }
                u[s] = acc;
                let lij = acc * win_dinv[s];
                l[s] = lij;
                di = fmadd(-lij, acc, di);
            }
            (l, di)
        }
        #[inline(always)]
        fn shift<T: Copy, const KB: usize>(
            win_l: &mut [[T; KB]; KB],
            win_dinv: &mut [T; KB],
            l: [T; KB],
            d: T,
        ) {
            for q in 0..KB - 1 {
                win_l[q] = win_l[q + 1];
                win_dinv[q] = win_dinv[q + 1];
            }
            win_l[KB - 1] = l;
            win_dinv[KB - 1] = d;
        }

        // the two chains, interleaved
        let mut wa_l = [[T::zero(); KB]; KB];
        let mut wa_d = [T::zero(); KB];
        let mut wb_l = [[T::zero(); KB]; KB];
        let mut wb_d = [T::zero(); KB];
        for t in 0..h.max(nb) {
            if t < h {
                let di = self.a_diag[t] + self.d_shift[t];
                let (l, di) = row::<T, KB>(&self.a_band[idx(t, 0)..idx(t, KB)], di, &wa_l, &wa_d);
                let Some(dinv) = self.finish_pivot(di, t, &mut count, &mut min_neg) else {
                    return false;
                };
                self.l_band[idx(t, 0)..idx(t, KB)].copy_from_slice(&l);
                self.dinv[t + KB] = dinv;
                shift(&mut wa_l, &mut wa_d, l, dinv);
            }
            if t < nb {
                let i = n - 1 - t;
                let di = self.a_diag[i] + self.d_shift[i];
                let (l, di) = row::<T, KB>(&self.a_rev[idx(t, 0)..idx(t, KB)], di, &wb_l, &wb_d);
                let Some(dinv) = self.finish_pivot(di, i, &mut count, &mut min_neg) else {
                    return false;
                };
                self.lb_rev[idx(t, 0)..idx(t, KB)].copy_from_slice(&l);
                self.dinv[i + KB] = dinv;
                shift(&mut wb_l, &mut wb_d, l, dinv);
            }
        }

        // separator rows s = h + j.  Their couplings to the top chain sit
        // in band slots q < KB - j (columns < h) of l_band, and to the
        // bottom chain in reversed slots q <= j of lb_rev.
        let mut ua = [[T::zero(); KB]; KB]; // ua[j][q] = L(s, s-KB+q) d
        let mut ub = [[T::zero(); KB]; KB]; // ub[j][q] = LB(s, s+KB-q) d
        for j in 0..KB {
            let s = h + j;
            let ts = n - 1 - s;
            for q in 0..KB - j {
                let c = s - KB + q;
                let mut acc = self.a_band[idx(s, q)];
                for q2 in 0..q {
                    acc -= ua[j][q2] * self.l_band[idx(c, q2 + KB - q)];
                }
                ua[j][q] = acc;
                self.l_band[idx(s, q)] = acc * self.dinv[c + KB];
            }
            for q in 0..=j {
                let c = ts - KB + q; // reversed column
                let mut acc = self.a_rev[idx(ts, q)];
                for q2 in 0..q {
                    acc -= ub[j][q2] * self.lb_rev[idx(c, q2 + KB - q)];
                }
                ub[j][q] = acc;
                self.lb_rev[idx(ts, q)] = acc * self.dinv[n - 1 - c + KB];
            }
        }
        // dense LDL of the separator Schur complement
        let mut us = [[T::zero(); KB]; KB];
        let mut ls = [[T::zero(); KB]; KB];
        let mut ds = [T::zero(); KB];
        for j in 0..KB {
            let s = h + j;
            for j2 in 0..=j {
                let s2 = h + j2;
                let mut m = if j2 == j {
                    self.a_diag[s] + self.d_shift[s]
                } else {
                    self.a_band[idx(s, s2 + KB - s)]
                };
                // top chain columns common to both rows: c in [s-KB, h)
                for q in 0..KB - j {
                    let c = s - KB + q;
                    m -= ua[j][q] * self.l_band[idx(s2, c + KB - s2)];
                }
                // bottom chain columns common to both: b in [h+KB, s2+KB]
                for q in 0..=j2 {
                    // b = s2 + KB - q in row s2; in row s it is slot q + j - j2
                    m -= ub[j][q + j - j2] * self.lb_rev[idx(n - 1 - s2, q)];
                }
                if j2 < j {
                    let mut acc = m;
                    for k in 0..j2 {
                        acc -= us[j][k] * ls[j2][k];
                    }
                    us[j][j2] = acc;
                    ls[j][j2] = acc / ds[j2];
                } else {
                    let mut di = m;
                    for k in 0..j {
                        di -= us[j][k] * ls[j][k];
                    }
                    let Some(dinv) = self.finish_pivot(di, s, &mut count, &mut min_neg) else {
                        return false;
                    };
                    ds[j] = T::recip(dinv);
                    self.dinv[s + KB] = dinv;
                }
            }
            for j2 in 0..j {
                self.l_band[idx(s, j2 + KB - j)] = ls[j][j2];
            }
        }

        self.regularize_count = count;
        self.min_pivot = min_neg;
        self.dinv[KB..KB + n].iter().all(|d| d.is_finite())
    }

    #[inline(never)]
    fn solve_twisted<const KB: usize, const NR: usize>(&self, x: &mut [[T; NR]]) {
        let n = self.n;
        let h = self.twisted.unwrap();
        let nb = n - h - KB;
        let idx = |r: usize, q: usize| (r + KB) * KB + q;
        let (lb, lr, dinv) = (&self.l_band[..], &self.lb_rev[..], &self.dinv[..]);
        // All indexing in the loops below is of rows r < n + 2 KB of the
        // band and of x, and of reversed rows t < nb + 3 KB of lb_rev
        // (see the derivations alongside), so these bounds make the
        // unchecked accesses in the chain loops sound.
        assert!(x.len() >= n + 2 * KB && dinv.len() >= n + 2 * KB);
        assert!(lb.len() >= (n + 2 * KB) * KB && lr.len() >= (nb + 3 * KB) * KB);
        let lb_at = |k: usize| unsafe { *lb.get_unchecked(k) };
        let lr_at = |k: usize| unsafe { *lr.get_unchecked(k) };

        #[inline(always)]
        fn fwd<T: FloatT, const KB: usize, const NR: usize>(
            l: [T; KB],
            mut y: [T; NR],
            win: &mut [[T; NR]; KB],
        ) -> [T; NR] {
            for s in 0..KB {
                for r in 0..NR {
                    y[r] = fmadd(-l[s], win[s][r], y[r]);
                }
            }
            for q in 0..KB - 1 {
                win[q] = win[q + 1];
            }
            win[KB - 1] = y;
            y
        }

        // forward substitution on both chains
        let mut wa = [[T::zero(); NR]; KB];
        let mut wb = [[T::zero(); NR]; KB];
        // SAFETY: t < h < n and t < nb, rows of x below n + KB
        for t in 0..h.max(nb) {
            if t < h {
                let l = std::array::from_fn(|q| lb_at(idx(t, q)));
                unsafe {
                    let xt = x.get_unchecked_mut(t + KB);
                    *xt = fwd::<T, KB, NR>(l, *xt, &mut wa);
                }
            }
            if t < nb {
                let i = n - 1 - t;
                let l = std::array::from_fn(|q| lr_at(idx(t, q)));
                unsafe {
                    let xi = x.get_unchecked_mut(i + KB);
                    *xi = fwd::<T, KB, NR>(l, *xi, &mut wb);
                }
            }
        }
        // separator rows
        for j in 0..KB {
            let s = h + j;
            let ts = n - 1 - s;
            let mut y = x[s + KB];
            for q in 0..KB {
                let (la, lbq) = (lb[idx(s, q)], lr[idx(ts, q)]);
                let (xa, xb) = (x[s + q], x[s + 2 * KB - q]);
                for r in 0..NR {
                    y[r] -= la * xa[r] + lbq * xb[r];
                }
            }
            x[s + KB] = y;
        }
        // diagonal
        for (xi, &d) in x[KB..KB + n].iter_mut().zip(&dinv[KB..KB + n]) {
            for r in 0..NR {
                xi[r] *= d;
            }
        }
        // backward: separator rows, using separator couplings only
        for j in (0..KB).rev() {
            let s = h + j;
            let mut v = x[s + KB];
            for q in 0..KB - 1 - j {
                let k = s + 1 + q;
                let lks = lb[idx(k, KB - 1 - q)];
                for r in 0..NR {
                    v[r] -= lks * x[k + KB][r];
                }
            }
            x[s + KB] = v;
        }
        // backward on both chains.  wa[q] holds x of row i+1+q of the top
        // chain, wb[q] of reversed row t+1+q of the bottom chain.
        let mut wa = [[T::zero(); NR]; KB];
        let mut wb = [[T::zero(); NR]; KB];
        for q in 0..KB {
            wa[q] = x[h + q + KB];
            wb[q] = x[n - 1 - (nb + q) + KB];
        }
        // SAFETY: band rows i + 1 + q <= h - 1 + KB < n + 2 KB, reversed
        // rows t + 1 + q <= nb + KB - 1, and rows of x below n + KB
        for step in 0..h.max(nb) {
            if step < h {
                let i = h - 1 - step;
                let mut v = unsafe { *x.get_unchecked(i + KB) };
                for q in 0..KB {
                    let lki = lb_at(idx(i + 1 + q, KB - 1 - q));
                    for r in 0..NR {
                        v[r] = fmadd(-lki, wa[q][r], v[r]);
                    }
                }
                unsafe { *x.get_unchecked_mut(i + KB) = v };
                for q in (1..KB).rev() {
                    wa[q] = wa[q - 1];
                }
                wa[0] = v;
            }
            if step < nb {
                let t = nb - 1 - step;
                let i = n - 1 - t;
                let mut v = unsafe { *x.get_unchecked(i + KB) };
                for q in 0..KB {
                    let lki = lr_at(idx(t + 1 + q, KB - 1 - q));
                    for r in 0..NR {
                        v[r] = fmadd(-lki, wb[q][r], v[r]);
                    }
                }
                unsafe { *x.get_unchecked_mut(i + KB) = v };
                for q in (1..KB).rev() {
                    wb[q] = wb[q - 1];
                }
                wb[0] = v;
            }
        }
    }

    #[inline(never)]
    fn factor_kb<const KB: usize>(&mut self) -> bool {
        let n = self.n;
        let (eps, delta) = (self.reg_eps, self.reg_delta);
        let a_band = &self.a_band[..];
        let l_band = &mut self.l_band[..];
        let dinv = &mut self.dinv[..];
        let mut count = 0;
        let mut min_neg = T::infinity();

        // sliding window of the KB previous rows of U = L*D and L,
        // and of the previous dinv values.  win_l[q] holds row (i-KB+q).
        let mut win_l = [[T::zero(); KB]; KB];
        let mut win_dinv = [T::zero(); KB];

        for i in 0..n {
            let ri = i + KB; // storage row
            let row: [T; KB] = a_band[ri * KB..ri * KB + KB].try_into().unwrap();
            let mut u = [T::zero(); KB]; // u[s] = L_{i,c} d_c, c = i-KB+s
            let mut l = [T::zero(); KB];
            let mut di = self.a_diag[i] + self.d_shift[i];
            for s in 0..KB {
                let mut acc = row[s];
                // L_{j,c} for j = i-KB+s and c = i-KB+t, t < s,
                // stored in row j at slot t - s + KB
                for t in 0..s {
                    acc = fmadd(-u[t], win_l[s][t + KB - s], acc);
                }
                u[s] = acc;
                let lij = acc * win_dinv[s];
                l[s] = lij;
                di = fmadd(-lij, acc, di);
            }
            let sign = self.signs[i];
            if di * sign < eps {
                di = delta * sign;
                count += 1;
            }
            if di == T::zero() {
                return false;
            }
            min_neg = min_neg.min(di.abs());
            let dinv_i = T::recip(di);
            l_band[ri * KB..ri * KB + KB].copy_from_slice(&l);
            dinv[ri] = dinv_i;

            // shift windows
            for q in 0..KB - 1 {
                win_l[q] = win_l[q + 1];
                win_dinv[q] = win_dinv[q + 1];
            }
            win_l[KB - 1] = l;
            win_dinv[KB - 1] = dinv_i;
        }
        self.regularize_count = count;
        self.min_pivot = min_neg;
        dinv[KB..KB + n].iter().all(|d| d.is_finite())
    }

    /// Solve in place on NR interleaved right hand sides, in permuted
    /// coordinates.  `x` must have length (n + 2 kb) with zero padding.
    pub fn solve_permuted<const NR: usize>(&self, x: &mut [[T; NR]]) {
        if self.twisted.is_some() {
            return match self.kb {
                1 => self.solve_twisted::<1, NR>(x),
                2 => self.solve_twisted::<2, NR>(x),
                3 => self.solve_twisted::<3, NR>(x),
                4 => self.solve_twisted::<4, NR>(x),
                5 => self.solve_twisted::<5, NR>(x),
                6 => self.solve_twisted::<6, NR>(x),
                7 => self.solve_twisted::<7, NR>(x),
                8 => self.solve_twisted::<8, NR>(x),
                _ => unreachable!(),
            };
        }
        match self.kb {
            1 => self.solve_kb::<1, NR>(x),
            2 => self.solve_kb::<2, NR>(x),
            3 => self.solve_kb::<3, NR>(x),
            4 => self.solve_kb::<4, NR>(x),
            5 => self.solve_kb::<5, NR>(x),
            6 => self.solve_kb::<6, NR>(x),
            7 => self.solve_kb::<7, NR>(x),
            8 => self.solve_kb::<8, NR>(x),
            _ => unreachable!(),
        }
    }

    #[inline(never)]
    fn solve_kb<const KB: usize, const NR: usize>(&self, x: &mut [[T; NR]]) {
        let n = self.n;
        let l_band = &self.l_band[..];
        let dinv = &self.dinv[..];

        // forward: y_i = b_i - sum_s L_{i,i-KB+s} y_{i-KB+s}
        let mut win = [[T::zero(); NR]; KB];
        for i in 0..n {
            let ri = i + KB;
            let l: [T; KB] = l_band[ri * KB..ri * KB + KB].try_into().unwrap();
            let mut y = x[ri];
            for s in 0..KB {
                for r in 0..NR {
                    y[r] = fmadd(-l[s], win[s][r], y[r]);
                }
            }
            x[ri] = y;
            for q in 0..KB - 1 {
                win[q] = win[q + 1];
            }
            win[KB - 1] = y;
        }
        // diagonal
        for i in 0..n {
            let ri = i + KB;
            let d = dinv[ri];
            for r in 0..NR {
                x[ri][r] *= d;
            }
        }
        // backward: x_i = y_i - sum_{k=i+1}^{i+KB} L_{k,i} x_k
        // win[q] holds x_{i+1+q}
        let mut win = [[T::zero(); NR]; KB];
        for i in (0..n).rev() {
            let ri = i + KB;
            let mut v = x[ri];
            for q in 0..KB {
                let rk = ri + 1 + q;
                // L_{k,i} is at row k slot i - k + KB = KB - 1 - q
                let lki = l_band[rk * KB + KB - 1 - q];
                for r in 0..NR {
                    v[r] = fmadd(-lki, win[q][r], v[r]);
                }
            }
            x[ri] = v;
            for q in (1..KB).rev() {
                win[q] = win[q - 1];
            }
            win[0] = v;
        }
    }

    /// e -= A x for NR right hand sides, permuted coordinates, padded.
    /// Returns the inf norm of the result per rhs (NaN if it has any).
    pub fn residual_permuted<const NR: usize>(&self, e: &mut [[T; NR]], x: &[[T; NR]]) -> [T; NR] {
        match self.kb {
            1 => self.residual_kb::<1, NR>(e, x),
            2 => self.residual_kb::<2, NR>(e, x),
            3 => self.residual_kb::<3, NR>(e, x),
            4 => self.residual_kb::<4, NR>(e, x),
            5 => self.residual_kb::<5, NR>(e, x),
            6 => self.residual_kb::<6, NR>(e, x),
            7 => self.residual_kb::<7, NR>(e, x),
            8 => self.residual_kb::<8, NR>(e, x),
            _ => unreachable!(),
        }
    }

    #[inline(never)]
    fn residual_kb<const KB: usize, const NR: usize>(&self, e: &mut [[T; NR]], x: &[[T; NR]]) -> [T; NR] {
        let n = self.n;
        let a = &self.a_band[..];
        let rows = n + 2 * KB;
        assert!(a.len() >= rows * KB && self.a_diag.len() >= n);
        assert!(e.len() >= rows && x.len() >= rows);
        let mut norm = [T::zero(); NR];
        // SAFETY: for i < n the rows touched are ri - KB .. ri + KB <
        // n + 2 KB, and band slots (ri + 1 + q) * KB + KB - 1 - q are below
        // rows * KB; both bounded by the asserts above
        unsafe {
            for i in 0..n {
                let ri = i + KB;
                let d = *self.a_diag.get_unchecked(i);
                let mut acc = *e.get_unchecked(ri);
                let xi = *x.get_unchecked(ri);
                for r in 0..NR {
                    acc[r] -= d * xi[r];
                }
                // lower part of row i
                for s in 0..KB {
                    let v = *a.get_unchecked(ri * KB + s);
                    let xs = *x.get_unchecked(ri - KB + s);
                    for r in 0..NR {
                        acc[r] -= v * xs[r];
                    }
                }
                // upper part of row i = column i below diagonal
                for q in 0..KB {
                    let rk = ri + 1 + q;
                    let aki = *a.get_unchecked(rk * KB + KB - 1 - q);
                    let xk = *x.get_unchecked(rk);
                    for r in 0..NR {
                        acc[r] -= aki * xk[r];
                    }
                }
                *e.get_unchecked_mut(ri) = acc;
                for r in 0..NR {
                    // NaN propagates (a NaN compares false, so test it)
                    let v = acc[r].abs();
                    norm[r] = if v > norm[r] || v.is_nan() { v } else { norm[r] };
                }
            }
        }
        norm
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // random banded quasidefinite matrix, solved with the band LDL in its
    // natural ordering, must give a residual at machine precision
    #[test]
    fn test_banded_exact() {
        for (kb, n) in (1..=BANDED_MAX_KB).flat_map(|kb| [(kb, 4 * kb + 4), (kb, 4 * kb + 5), (kb, 50), (kb, 51)]) {
            let mut seed = 12345u64 + kb as u64;
            let mut rnd = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
            };
            let signs: Vec<i8> = (0..n).map(|i| if i % 5 == 4 { -1 } else { 1 }).collect();
            let perm: Vec<usize> = (0..n).collect();
            let _ = &perm;
            let mut band = BandedLDL::<f64>::new(perm, kb, &signs, 1e-13, 1e-7);
            // fill
            let mut dense = vec![vec![0.0; n]; n];
            for i in 0..n {
                let d = (signs[i] as f64) * (5.0 + rnd());
                band.a_diag[i] = d;
                dense[i][i] = d;
                for j in i.saturating_sub(kb)..i {
                    let v = rnd();
                    match band.position(i, j) {
                        BandPos::Off(q) => band.a_band[q] = v,
                        _ => panic!(),
                    }
                    dense[i][j] = v;
                    dense[j][i] = v;
                }
            }
            assert_eq!(band.twisted.is_some(), n >= 4 * kb + 4);
            assert!(band.refactor());
            assert_eq!(band.regularize_count, 0);
            let mut x = vec![[0.0f64; 2]; n + 2 * kb];
            let mut b = x.clone();
            for i in 0..n {
                b[i + kb] = [rnd(), rnd()];
            }
            x.copy_from_slice(&b);
            band.solve_permuted::<2>(&mut x);
            // dense residual
            for r in 0..2 {
                let mut maxres: f64 = 0.0;
                for i in 0..n {
                    let mut acc = b[i + kb][r];
                    for j in 0..n {
                        acc -= dense[i][j] * x[j + kb][r];
                    }
                    maxres = maxres.max(acc.abs());
                }
                assert!(maxres < 1e-12, "kb={} dense residual {}", kb, maxres);
            }
            let mut e = vec![[0.0f64; 2]; n + 2 * kb];
            e.copy_from_slice(&b);
            let n2 = band.residual_permuted::<2>(&mut e, &x);
            let norm = e.iter().fold(0.0f64, |m, v| m.max(v[0].abs()).max(v[1].abs()));
            assert!(n2[0].max(n2[1]) == norm);
            assert!(norm < 1e-12, "kb={} band residual {:?}", kb, norm);
        }
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    // cargo test --release --lib banded::bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_factor_solve() {
        let (n, kb) = (1005, 3);
        for twisted in [false, true] {
            let signs: Vec<i8> = (0..n).map(|i| if i % 250 == 3 { -1 } else { 1 }).collect();
            let perm: Vec<usize> = (0..n).collect();
            let mut band = BandedLDL::<f64>::new(perm, kb, &signs, 1e-13, 1e-7);
            if !twisted {
                band.twisted = None;
            }
            for i in 0..n {
                band.a_diag[i] = signs[i] as f64 * 10.0;
                for j in i.saturating_sub(kb)..i {
                    if let BandPos::Off(q) = band.position(i, j) {
                        band.a_band[q] = 0.3;
                    }
                }
            }
            let reps = 500;
            let (mut tf, mut ts) = (f64::MAX, f64::MAX);
            let mut x = vec![[1.0f64; 2]; n + 2 * kb];
            for _ in 0..10 {
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    assert!(band.refactor());
                }
                tf = tf.min(t0.elapsed().as_secs_f64() / reps as f64 * 1e6);
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    band.solve_permuted::<2>(&mut x);
                    std::hint::black_box(&mut x);
                }
                ts = ts.min(t0.elapsed().as_secs_f64() / reps as f64 * 1e6);
            }
            println!("twisted={} factor {:.2} us  solve {:.2} us", twisted, tf, ts);
        }
    }
}
