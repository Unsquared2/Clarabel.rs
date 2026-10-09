#![allow(non_snake_case)]
// index loops are the clearest form for these numeric kernels
#![allow(clippy::needless_range_loop)]
//! KKT solver for problems with only zero and nonnegative cones that
//! eliminates simple bound rows before factoring.
//!
//! A row of A belonging to a nonnegative cone with a single nonzero entry
//! `a` in column `j` is a variable bound.  Its block of the KKT system
//!
//!   [ P    A_B'  A_R' ] [x  ]   [rx  ]
//!   [ A_B -H_B        ] [z_B] = [rz_B]
//!   [ A_R       -H_R  ] [z_R]   [rz_R]
//!
//! can be eliminated exactly via z_B = H_B⁻¹(A_B x - rz_B), leaving
//!
//!   [ P + D   A_R' ] [x  ]   [rx + A_B'H_B⁻¹ rz_B]
//!   [ A_R    -H_R  ] [z_R] = [rz_R               ]
//!
//! with D = A_B'H_B⁻¹A_B diagonal.  Since H_B = s/z > 0 for nonnegative
//! cones the eliminated block needs no regularization, and a variable with
//! an eliminated bound has a strictly positive diagonal entry in the PSD
//! x-block, so on banded problems the reduced system can often be factored
//! with no static regularization at all.
//!
//! Recovering z_B for a strongly active bound (H_B → 0) amplifies the
//! rounding error of A_B x - rz_B by H_B⁻¹, so the reduced factorization
//! is used as the solver inside iterative refinement on the *full* KKT
//! system: corrections are small and are recovered accurately.  Residuals
//! are formed with K₀ = [P A_R'; A_R -H_R], i.e. without D, which avoids
//! cancellation against the large entries of D.
//!
//! The reduced system is factored with a dense-band LDLᵀ when a reverse
//! Cuthill-McKee ordering gives a small bandwidth (1D structures such as
//! rods, beams, chains and time-stepped control problems), and otherwise
//! with the LDL solver selected by the settings, as for the full system
//! (QDLDL, or faer for heavy factorizations when that feature is enabled).  Vectors are kept in the backend's (permuted) ordering.

use super::banded::*;
use crate::algebra::*;
use super::ldlsolvers::amd_order;
use super::ldlsolvers::qdldl::QDLDLDirectLDLSolver;
use super::BoxedDirectLDLSolver;
use crate::solver::core::cones::{CompositeCone, Cone, SupportedConeAsTag, SupportedConeTag};
use crate::solver::core::kktsolvers::{HasLinearSolverInfo, KKTSolver, LinearSolverInfo};
use crate::solver::core::CoreSettings;

// number of interleaved right hand sides
const NR: usize = 2;

// p_map marker for a duplicate diagonal entry of P (summed into the first)
const DUP_DIAG: usize = usize::MAX - 1;

enum Backend<T> {
    Direct(BoxedDirectLDLSolver<T>),
    Banded(Box<BandedLDL<T>>),
}

pub struct BoundElimKKTSolver<T> {
    // eliminated bound rows: constraint row, vector index of the
    // variable, A nz index, coefficient and its square
    b_row: Vec<usize>,
    b_pos: Vec<usize>,
    b_nz: Vec<usize>,
    b_coef: Vec<T>,
    b_coef2: Vec<T>,
    // H_B, H_B⁻¹ and a H_B⁻¹ for eliminated rows
    b_h: Vec<T>,
    b_hinv: Vec<T>,
    b_ch: Vec<T>,
    // eliminated bounds of variable j are b_*[bptr[j]..bptr[j+1]]
    bptr: Vec<usize>,

    // remaining rows: original constraint row index, vector index
    r_row: Vec<usize>,
    r_pos: Vec<usize>,
    // vector index of each variable
    x_pos: Vec<usize>,
    // vector index = backend position + offset (banded padding)
    offset: usize,

    // A nz index -> KKT nz index for entries of A_R
    a_map: Vec<(usize, usize)>,
    // P nz index -> KKT nz index (usize::MAX if lower triangular,
    // DUP_DIAG for repeated diagonal entries)
    p_map: Vec<usize>,
    // backend position -> KKT nz index of the diagonal
    diag_idx: Vec<usize>,
    // P diagonal values in backend order (zero for constraint rows)
    p_diag: Vec<T>,
    // static regularization direction per backend position (+1 for
    // variable rows, -1 for constraint rows), as in the full KKT solver
    reg_sign: Vec<T>,
    // variables without an eliminated bound (their rows may be singular,
    // so are always regularized)
    unbounded: Vec<bool>,

    // K₀ (triu, natural order).  The diagonal holds P_jj for variable rows
    // and -H for constraint rows.  The factors are of K₀ + D + shifts.
    KKT: CscMatrix<T>,
    backend: Backend<T>,
    // band position of every KKT nonzero (banded backend only)
    band_map: Vec<BandPos>,

    // the cone Hs diagonal for all m rows
    hs: Vec<T>,
    // diagonal of K₀, and the diagonal D added to it for factorization,
    // in backend order
    k0_diag: Vec<T>,
    shift: Vec<T>,

    // NR interleaved lanes, in backend order (padded for the banded
    // backend): reduced rhs, K₀ rhs [rx; rz_R], solution, residual, work
    b: Vec<[T; NR]>,
    b0: Vec<[T; NR]>,
    x: Vec<[T; NR]>,
    e: Vec<[T; NR]>,
    dx: Vec<[T; NR]>,
    // per lane, for the eliminated rows: rhs, solution, residual, work
    rzb: [Vec<T>; NR],
    zb: [Vec<T>; NR],
    eb: [Vec<T>; NR],
    dzb: [Vec<T>; NR],
    // scalar work for the general sparse backend
    w0: Vec<T>,
    w1: Vec<T>,
    // inf norm of the full rhs per lane, set by reduce_rhs
    normb: [T; NR],
}

impl<T: FloatT> BoundElimKKTSolver<T> {
    /// Returns true if the cones are all zero / nonnegative and at
    /// least one bound row can be eliminated.
    pub fn is_applicable(A: &CscMatrix<T>, cones: &CompositeCone<T>) -> bool {
        let mut nonneg = vec![false; A.m];
        for (cone, rng) in cones.iter().zip(cones.rng_cones.iter()) {
            match cone.as_tag() {
                SupportedConeTag::ZeroCone => {}
                SupportedConeTag::NonnegativeCone => nonneg[rng.clone()].fill(true),
                _ => return false,
            }
        }
        let mut rowcnt = vec![0usize; A.m];
        A.rowval.iter().for_each(|&r| rowcnt[r] += 1);
        (0..A.m).any(|r| nonneg[r] && rowcnt[r] == 1)
    }

    pub fn new(
        P: &CscMatrix<T>,
        A: &CscMatrix<T>,
        cones: &CompositeCone<T>,
        m: usize,
        n: usize,
        settings: &CoreSettings<T>,
    ) -> Self {
        assert!(P.n == n && P.m == n && A.n == n && A.m == m);
        assert!(P.rowval.iter().all(|&i| i < n) && A.rowval.iter().all(|&i| i < m));

        // classify rows
        let mut nonneg = vec![false; m];
        for (cone, rng) in cones.iter().zip(cones.rng_cones.iter()) {
            if cone.as_tag() == SupportedConeTag::NonnegativeCone {
                nonneg[rng.clone()].fill(true);
            }
        }
        let mut rowcnt = vec![0usize; m];
        A.rowval.iter().for_each(|&r| rowcnt[r] += 1);

        // bound rows, and remaining rows
        let mut is_bound = vec![false; m];
        let mut ridx = vec![usize::MAX; m];
        let mut r_row = vec![];
        for r in 0..m {
            if nonneg[r] && rowcnt[r] == 1 {
                is_bound[r] = true;
            } else {
                ridx[r] = r_row.len();
                r_row.push(r);
            }
        }
        let mr = r_row.len();

        // bounds are collected column by column, so are sorted by variable
        let (mut b_row, mut b_col, mut b_nz, mut b_coef) = (vec![], vec![], vec![], vec![]);
        // A_R in coordinate form sorted by row: (row, col, A nz index)
        let mut ar: Vec<(usize, usize, usize)> = Vec::with_capacity(A.nnz());
        for j in 0..n {
            for k in A.colptr[j]..A.colptr[j + 1] {
                let r = A.rowval[k];
                if is_bound[r] {
                    b_row.push(r);
                    b_col.push(j);
                    b_nz.push(k);
                    b_coef.push(A.nzval[k]);
                } else {
                    ar.push((ridx[r], j, k));
                }
            }
        }
        ar.sort_unstable();

        // assemble K₀ (triu) of dimension n + mr
        let dim = n + mr;
        let nnz_est = P.nnz() + n + ar.len() + mr;
        let mut colptr = Vec::with_capacity(dim + 1);
        let mut rowval = Vec::with_capacity(nnz_est);
        let mut nzval = Vec::with_capacity(nnz_est);
        let mut p_map = vec![usize::MAX; P.nnz()];
        let mut diag_nat = vec![0usize; dim];
        let mut p_diag_nat = vec![T::zero(); dim];
        colptr.push(0);
        for j in 0..n {
            let mut has_diag = false;
            for k in P.colptr[j]..P.colptr[j + 1] {
                let i = P.rowval[k];
                if i > j {
                    continue; // ignore any lower triangle data
                }
                if i == j {
                    p_diag_nat[j] += P.nzval[k];
                    if has_diag {
                        p_map[k] = DUP_DIAG;
                        continue;
                    }
                    has_diag = true;
                    diag_nat[j] = rowval.len();
                }
                p_map[k] = rowval.len();
                rowval.push(i);
                nzval.push(P.nzval[k]);
            }
            if !has_diag {
                diag_nat[j] = rowval.len();
                rowval.push(j);
                nzval.push(T::zero());
            }
            colptr.push(rowval.len());
        }
        let mut a_map = Vec::with_capacity(ar.len());
        let mut q = 0;
        for r in 0..mr {
            while q < ar.len() && ar[q].0 == r {
                let (_, j, k) = ar[q];
                a_map.push((k, rowval.len()));
                rowval.push(j);
                nzval.push(A.nzval[k]);
                q += 1;
            }
            diag_nat[n + r] = rowval.len();
            rowval.push(n + r);
            nzval.push(T::zero());
            colptr.push(rowval.len());
        }
        let mut KKT = CscMatrix::new(dim, dim, colptr, rowval, nzval);
        for j in 0..n {
            KKT.nzval[diag_nat[j]] = p_diag_nat[j];
        }

        let mut signs = vec![1_i8; dim];
        signs[n..].fill(-1);
        let mut has_bound = vec![false; dim];
        b_col.iter().for_each(|&j| has_bound[j] = true);

        // Choose a backend.  Narrow bands (up to 8) always use the banded
        // factorization.  Wider ones (up to BANDED_MAX_KB) use it if its
        // dense band costs at most twice the flops of a sparse
        // factorization, estimated from the AMD analysis that the sparse
        // backend needs anyway.
        let (perm, bw) = rcm_ordering(&KKT, &signs);
        // AMD on the pattern of the full KKT system; see below
        let full_amd = || amd_order(&full_kkt_pattern(P, A));
        let mut amd = None;
        let use_banded = !test_hooks::no_banded()
            && (bw <= 8
                || (bw <= BANDED_MAX_KB && {
                    let (fperm, iperm, info) = full_amd();
                    let sparse_flops = (info.n_div + info.n_mult_subs_ldl) as f64;
                    let band_flops = (dim * bw * (bw + 1) / 2) as f64;
                    amd = Some((fperm, iperm, info));
                    band_flops <= 2.0 * sparse_flops
                }));

        let (backend, band_map, iperm, lanes_len, offset) = if use_banded {
            let band = BandedLDL::new(
                perm,
                bw,
                &signs,
                settings.dynamic_regularization_eps,
                settings.dynamic_regularization_delta,
            );
            let mut band_map = Vec::with_capacity(KKT.nnz());
            for j in 0..dim {
                for k in KKT.colptr[j]..KKT.colptr[j + 1] {
                    band_map.push(band.position(KKT.rowval[k], j));
                }
            }
            let iperm = band.iperm.clone();
            let kb = band.kb;
            (Backend::Banded(Box::new(band)), band_map, iperm, dim + 2 * kb, kb)
        } else {
            // Order with AMD on the pattern of the full KKT system, as
            // the full solver does, and drop the bound rows from that
            // order.  A bound row has a single neighbour, so eliminating
            // it creates no fill among the other nodes wherever it is
            // placed: the factor of the reduced system has no more fill
            // than that of the full one.  (Ordering the reduced matrix
            // directly can be noticeably worse.)
            let (fperm, _, info) = amd.unwrap_or_else(full_amd);
            let perm: Vec<usize> = fperm
                .iter()
                .filter_map(|&f| match f.checked_sub(n) {
                    None => Some(f),
                    Some(r) if is_bound[r] => None,
                    Some(r) => Some(n + ridx[r]),
                })
                .collect();
            let ldl = direct_ldl(&KKT, &signs, settings, perm, &info);
            (Backend::Direct(ldl), vec![], (0..dim).collect(), dim, 0)
        };

        // vector index of every reduced index
        let pos: Vec<usize> = iperm.iter().map(|&p| p + offset).collect();
        let x_pos = pos[0..n].to_vec();
        let r_pos = pos[n..].to_vec();
        let b_pos: Vec<usize> = b_col.iter().map(|&j| pos[j]).collect();

        // diagonal data in backend order
        let mut diag_idx = vec![0usize; dim];
        let mut p_diag = vec![T::zero(); dim];
        let mut reg_sign = vec![T::zero(); dim];
        let mut unbounded = vec![false; dim];
        for i in 0..dim {
            let r = iperm[i];
            diag_idx[r] = diag_nat[i];
            p_diag[r] = p_diag_nat[i];
            reg_sign[r] = if signs[i] < 0 { -T::one() } else { T::one() };
            unbounded[r] = signs[i] > 0 && !has_bound[i];
        }

        let nb = b_row.len();
        let mut bptr = vec![0usize; n + 1];
        b_col.iter().for_each(|&j| bptr[j + 1] += 1);
        for j in 0..n {
            bptr[j + 1] += bptr[j];
        }
        let b_coef2 = b_coef.iter().map(|&a| a * a).collect();
        let zl = [T::zero(); NR];
        let lane = |len: usize| -> [Vec<T>; NR] { std::array::from_fn(|_| vec![T::zero(); len]) };
        let wlen = if use_banded { 0 } else { dim };
        let mut s = Self {
            b_row,
            b_pos,
            b_nz,
            b_coef,
            b_coef2,
            b_h: vec![T::zero(); nb],
            b_hinv: vec![T::zero(); nb],
            b_ch: vec![T::zero(); nb],
            bptr,
            r_row,
            r_pos,
            x_pos,
            offset,
            a_map,
            p_map,
            diag_idx,
            p_diag,
            reg_sign,
            unbounded,
            KKT,
            backend,
            band_map,
            hs: vec![T::zero(); m],
            k0_diag: vec![T::zero(); dim],
            shift: vec![T::zero(); dim],
            b: vec![zl; lanes_len],
            b0: vec![zl; lanes_len],
            x: vec![zl; lanes_len],
            e: vec![zl; lanes_len],
            dx: vec![zl; lanes_len],
            rzb: lane(nb),
            zb: lane(nb),
            eb: lane(nb),
            dzb: lane(nb),
            w0: vec![T::zero(); wlen],
            w1: vec![T::zero(); wlen],
            normb: [T::zero(); NR],
        };
        s.check_indices(m);
        s.scatter_to_band();
        s
    }

    // Validate the index invariants that the unchecked indexing in the
    // per-iteration kernels relies on.  Called once, from new().
    fn check_indices(&self, m: usize) {
        let len = self.b.len();
        assert!([&self.b0, &self.x, &self.e, &self.dx].iter().all(|v| v.len() == len));
        assert!(self.x_pos.iter().chain(&self.r_pos).chain(&self.b_pos).all(|&p| p < len));
        assert!(self.b_row.iter().chain(&self.r_row).all(|&r| r < m) && self.hs.len() == m);
        let nb = self.b_row.len();
        for v in [&self.b_pos, &self.b_nz] {
            assert!(v.len() == nb);
        }
        for v in [&self.b_coef, &self.b_coef2, &self.b_h, &self.b_hinv, &self.b_ch] {
            assert!(v.len() == nb);
        }
        for l in 0..NR {
            for v in [&self.rzb[l], &self.zb[l], &self.eb[l], &self.dzb[l]] {
                assert!(v.len() == nb);
            }
        }
        let n = self.x_pos.len();
        assert!(self.bptr.len() == n + 1 && self.bptr[n] == nb);
        assert!(self.bptr.windows(2).all(|w| w[0] <= w[1]));
    }

    // copy all K₀ values into the band storage
    fn scatter_to_band(&mut self) {
        if let Backend::Banded(band) = &mut self.backend {
            // summed, so that duplicate entries mean the same as in CSC
            band.a_diag.fill(T::zero());
            band.a_band.fill(T::zero());
            for (k, p) in self.band_map.iter().enumerate() {
                match *p {
                    BandPos::Diag(r) => band.a_diag[r] += self.KKT.nzval[k],
                    BandPos::Off(q) => band.a_band[q] += self.KKT.nzval[k],
                }
            }
            band.band_changed();
        }
    }

    // Reduced rhs (into b), K₀ rhs (into b0) and eliminated-row rhs (into
    // rzb) for L lanes.  Unused lanes are zeroed.
    //
    // SAFETY (unchecked indexing here, in expand_lhs and in the kernels
    // below): the index arrays and buffer lengths are validated once in
    // new() by check_indices(), and the caller's slices are checked here.
    fn reduce_rhs<const L: usize>(&mut self, rhs: [(&[T], &[T]); L]) {
        let m = self.hs.len();
        for (rx, rz) in &rhs {
            assert!(rx.len() == self.x_pos.len() && rz.len() == m);
        }
        let (b, b0) = (&mut self.b, &mut self.b0);
        let (b_row, b_ch, bptr) = (&self.b_row, &self.b_ch, &self.bptr);
        let [rzb0, rzb1] = &mut self.rzb;
        unsafe {
            for (j, &p) in self.x_pos.iter().enumerate() {
                let mut v = [T::zero(); NR];
                for l in 0..L {
                    v[l] = *rhs[l].0.get_unchecked(j);
                }
                *b0.get_unchecked_mut(p) = v;
                for i in *bptr.get_unchecked(j)..*bptr.get_unchecked(j + 1) {
                    let r = *b_row.get_unchecked(i);
                    let c = *b_ch.get_unchecked(i);
                    let t0 = *rhs[0].1.get_unchecked(r);
                    *rzb0.get_unchecked_mut(i) = t0;
                    v[0] = fmadd(c, t0, v[0]);
                    if L > 1 {
                        let t1 = *rhs[L - 1].1.get_unchecked(r);
                        *rzb1.get_unchecked_mut(i) = t1;
                        v[1] = fmadd(c, t1, v[1]);
                    }
                }
                *b.get_unchecked_mut(p) = v;
            }
            for (&p, &row) in self.r_pos.iter().zip(&self.r_row) {
                let mut v = [T::zero(); NR];
                for l in 0..L {
                    v[l] = *rhs[l].1.get_unchecked(row);
                }
                *b.get_unchecked_mut(p) = v;
                *b0.get_unchecked_mut(p) = v;
            }
        }
        if L < 2 {
            rzb1.fill(T::zero());
        }
        let mut normb = [T::zero(); NR];
        for (l, nb) in normb.iter_mut().enumerate().take(L) {
            *nb = rhs[l].0.norm_inf().max(rhs[l].1.norm_inf());
        }
        self.normb = normb;
    }

    // recover full (x,z) for L lanes
    fn expand_lhs<const L: usize>(&self, lhs: [(&mut [T], &mut [T]); L]) {
        let m = self.hs.len();
        for (lx, lz) in &lhs {
            assert!(lx.len() == self.x_pos.len() && lz.len() == m);
        }
        let x = &self.x;
        unsafe {
            for (j, &p) in self.x_pos.iter().enumerate() {
                let xj = *x.get_unchecked(p);
                for l in 0..L {
                    *lhs[l].0.get_unchecked_mut(j) = xj[l];
                }
            }
            for (&p, &row) in self.r_pos.iter().zip(&self.r_row) {
                let xp = *x.get_unchecked(p);
                for l in 0..L {
                    *lhs[l].1.get_unchecked_mut(row) = xp[l];
                }
            }
            for (i, &r) in self.b_row.iter().enumerate() {
                for l in 0..L {
                    *lhs[l].1.get_unchecked_mut(r) = *self.zb[l].get_unchecked(i);
                }
            }
        }
    }

    // Solve the reduced system in place on all lanes of `v`
    fn reduced_solve(
        backend: &mut Backend<T>,
        KKT: &CscMatrix<T>,
        v: &mut [[T; NR]],
        nlanes: usize,
        w0: &mut [T],
        w1: &mut [T],
    ) {
        match backend {
            Backend::Banded(band) => band.solve_permuted::<NR>(v),
            Backend::Direct(ldl) => {
                for l in 0..nlanes {
                    for (wi, vi) in w0.iter_mut().zip(v.iter()) {
                        *wi = vi[l];
                    }
                    ldl.solve(KKT, w1, w0);
                    for (wi, vi) in w1.iter().zip(v.iter_mut()) {
                        vi[l] = *wi;
                    }
                }
            }
        }
    }

    // z_B = H⁻¹(a x_j - r) for each eliminated row and lane
    fn recover_zb(&self, x: &[[T; NR]], r: &[Vec<T>; NR], zb: &mut [Vec<T>; NR]) {
        let [z0, z1] = zb;
        let [r0, r1] = r;
        assert!(x.len() == self.b.len());
        // SAFETY: lengths and b_pos < b.len() == x.len() checked in new()
        unsafe {
            for i in 0..self.b_row.len() {
                let h = *self.b_hinv.get_unchecked(i);
                let a = *self.b_coef.get_unchecked(i);
                let xp = *x.get_unchecked(*self.b_pos.get_unchecked(i));
                *z0.get_unchecked_mut(i) = h * (a * xp[0] - *r0.get_unchecked(i));
                *z1.get_unchecked_mut(i) = h * (a * xp[1] - *r1.get_unchecked(i));
            }
        }
    }

    // Residual of the full KKT system for (x, zb), into e (variable and
    // remaining rows, backend order) and eb (eliminated rows).  Returns
    // its inf norm per lane (NaN if it has any).
    fn full_residual(&mut self, use_dx: bool, nlanes: usize) -> [T; NR] {
        let (x, zb) = if use_dx { (&self.dx, &self.dzb) } else { (&self.x, &self.zb) };
        let e = &mut self.e;
        e.copy_from_slice(&self.b0);
        // eliminated rows: their columns of A_B' in the variable rows, and
        // the rows themselves
        let ([z0, z1], [eb0, eb1], [r0, r1]) = (zb, &mut self.eb, &self.rzb);
        let mut nb = [T::zero(); NR];
        // SAFETY: lengths and b_pos < e.len() == x.len() checked in new()
        unsafe {
            for i in 0..self.b_row.len() {
                let a = *self.b_coef.get_unchecked(i);
                let h = *self.b_h.get_unchecked(i);
                let p = *self.b_pos.get_unchecked(i);
                let (zi0, zi1) = (*z0.get_unchecked(i), *z1.get_unchecked(i));
                let xp = *x.get_unchecked(p);
                let ep = e.get_unchecked_mut(p);
                ep[0] -= a * zi0;
                ep[1] -= a * zi1;
                let v0 = *r0.get_unchecked(i) - (a * xp[0] - h * zi0);
                let v1 = *r1.get_unchecked(i) - (a * xp[1] - h * zi1);
                *eb0.get_unchecked_mut(i) = v0;
                *eb1.get_unchecked_mut(i) = v1;
                nb[0] = if v0.abs() > nb[0] || v0.is_nan() { v0.abs() } else { nb[0] };
                nb[1] = if v1.abs() > nb[1] || v1.is_nan() { v1.abs() } else { nb[1] };
            }
        }
        // then the K₀ part, e -= K₀ x
        let ne = match &self.backend {
            Backend::Banded(band) => band.residual_permuted::<NR>(e, x),
            Backend::Direct(_) => {
                let Ksym = self.KKT.sym_up();
                let (w0, w1) = (&mut self.w0, &mut self.w1);
                for l in 0..nlanes {
                    for i in 0..w0.len() {
                        w0[i] = x[i][l];
                        w1[i] = e[i][l];
                    }
                    Ksym.symv(w1, w0, -T::one(), T::one());
                    for i in 0..w1.len() {
                        e[i][l] = w1[i];
                    }
                }
                lane_norm_inf(e, [T::zero(); NR])
            }
        };
        let mut norm = [T::zero(); NR];
        for l in 0..NR {
            norm[l] = if ne[l].is_nan() || nb[l].is_nan() { T::nan() } else { ne[l].max(nb[l]) };
        }
        norm
    }

    // Solve all lanes of the system whose rhs was set by reduce_rhs, with
    // iterative refinement on the full system.
    fn solve_lanes(&mut self, nlanes: usize, settings: &CoreSettings<T>) -> bool {
        let finite = |s: &Self| {
            s.x.iter().all(|v| v.iter().take(nlanes).all(|t| t.is_finite()))
                && s.zb.iter().take(nlanes).all(|v| v.iter().all(|t| t.is_finite()))
        };

        self.x.copy_from_slice(&self.b);
        Self::reduced_solve(&mut self.backend, &self.KKT, &mut self.x, nlanes, &mut self.w0, &mut self.w1);
        let mut zb = std::mem::take(&mut self.zb);
        self.recover_zb(&self.x, &self.rzb, &mut zb);
        self.zb = zb;

        if !settings.iterative_refinement_enable {
            return finite(self);
        }
        let reltol = settings.iterative_refinement_reltol;
        let abstol = settings.iterative_refinement_abstol;
        let maxiter = settings.iterative_refinement_max_iter;
        let stopratio = settings.iterative_refinement_stop_ratio;

        let normb = self.normb;

        let mut norme = self.full_residual(false, nlanes);
        let mut active = [false; NR];
        for l in 0..nlanes {
            if !norme[l].is_finite() {
                return false;
            }
            active[l] = norme[l] > abstol + reltol * normb[l];
        }

        for _ in 0..maxiter {
            if !active.iter().any(|&a| a) {
                break;
            }
            // correction: solve K dξ = residual, with the residual of the
            // eliminated rows folded in exactly as for a rhs
            self.dx.copy_from_slice(&self.e);
            for v in self.dx.iter_mut() {
                for l in 0..NR {
                    if !active[l] {
                        v[l] = T::zero();
                    }
                }
            }
            for l in 0..nlanes {
                if active[l] {
                    for i in 0..self.b_pos.len() {
                        let p = self.b_pos[i];
                        self.dx[p][l] = fmadd(self.b_ch[i], self.eb[l][i], self.dx[p][l]);
                    }
                }
            }
            Self::reduced_solve(&mut self.backend, &self.KKT, &mut self.dx, nlanes, &mut self.w0, &mut self.w1);
            let mut dzb = std::mem::take(&mut self.dzb);
            self.recover_zb(&self.dx, &self.eb, &mut dzb);
            // candidate = current + correction
            for l in 0..nlanes {
                for (d, z) in dzb[l].iter_mut().zip(&self.zb[l]) {
                    *d += *z;
                }
            }
            self.dzb = dzb;
            for (d, x) in self.dx.iter_mut().zip(self.x.iter()) {
                for l in 0..nlanes {
                    d[l] += x[l];
                }
            }

            let newe = self.full_residual(true, nlanes);
            for l in 0..nlanes {
                if !active[l] {
                    continue;
                }
                if !newe[l].is_finite() {
                    return false;
                }
                let ratio = norme[l] / newe[l];
                if ratio > T::one() {
                    for (x, d) in self.x.iter_mut().zip(self.dx.iter()) {
                        x[l] = d[l];
                    }
                    self.zb[l].copy_from_slice(&self.dzb[l]);
                }
                if ratio < stopratio {
                    active[l] = false;
                } else {
                    norme[l] = newe[l];
                    active[l] = newe[l] > abstol + reltol * normb[l];
                }
            }
        }
        // every accepted iterate had a finite residual, so is finite
        true
    }
}

// lane-wise inf norm of v, combined with `init`
fn lane_norm_inf<T: FloatT>(v: &[[T; NR]], init: [T; NR]) -> [T; NR] {
    let mut m = init;
    for x in v {
        for l in 0..NR {
            let a = x[l].abs();
            // NaN propagates: `a > m` is false for NaN, so test explicitly
            m[l] = if a > m[l] || a.is_nan() { a } else { m[l] };
        }
    }
    m
}

impl<T: FloatT> HasLinearSolverInfo for BoundElimKKTSolver<T> {
    fn linear_solver_info(&self) -> LinearSolverInfo {
        match &self.backend {
            Backend::Direct(ldl) => {
                let mut info = ldl.linear_solver_info();
                info.name = format!("{}+boundelim", info.name);
                info
            }
            Backend::Banded(b) => LinearSolverInfo {
                name: format!("banded(kb={})+boundelim", b.kb),
                threads: 1,
                direct: true,
                nnzA: self.KKT.nnz(),
                nnzL: b.n * b.kb,
            },
        }
    }
}

impl<T: FloatT> KKTSolver<T> for BoundElimKKTSolver<T> {
    fn update(&mut self, cones: &CompositeCone<T>, settings: &CoreSettings<T>) -> bool {
        cones.get_Hs(&mut self.hs);
        let off = self.offset;

        // K₀ diagonal, H for the bound rows (floored so that a²/H cannot
        // overflow), and the exact D = A_B'H_B⁻¹A_B
        let tiny = T::min_positive_value().sqrt();
        self.k0_diag.copy_from(&self.p_diag);
        for (&p, &row) in self.r_pos.iter().zip(&self.r_row) {
            self.k0_diag[p - off] = -self.hs[row];
        }
        for (h, &r) in self.b_h.iter_mut().zip(&self.b_row) {
            let v = self.hs[r];
            *h = if v > tiny { v } else { tiny };
        }
        let eps_of = |maxdiag: T| {
            if settings.static_regularization_enable {
                settings.static_regularization_constant
                    + settings.static_regularization_proportional * maxdiag
            } else {
                T::zero()
            }
        };

        match &mut self.backend {
            Backend::Banded(band) => {
                self.shift.fill(T::zero());
                for i in 0..self.b_h.len() {
                    let hinv = T::recip(self.b_h[i]);
                    let ch = self.b_coef[i] * hinv;
                    self.b_hinv[i] = hinv;
                    self.b_ch[i] = ch;
                    self.shift[self.b_pos[i] - off] += self.b_coef[i] * ch;
                }
                // the largest reduced diagonal, while staging it in the band
                let mut maxdiag = T::zero();
                for (ad, (&k, &d)) in band.a_diag.iter_mut().zip(self.k0_diag.iter().zip(&self.shift)) {
                    *ad = k;
                    let v = (k + d).abs();
                    maxdiag = if v > maxdiag || v.is_nan() { v } else { maxdiag };
                }
                let eps = eps_of(maxdiag);
                // The banded ordering places every constraint row after its
                // variables, so its pivot is a Schur complement, and bounded
                // variables have a positive diagonal term.  First factor
                // with only the rows of unbounded variables regularized and
                // accept that if no pivot is small; otherwise regularize
                // every row.  (The exact a²/H is used in both cases, which
                // tests better on banded, stiff problems than the form of
                // the general backend below.)  Refinement on the full system
                // removes the perturbation.
                for ((s, &d), (&u, &rs)) in band
                    .d_shift
                    .iter_mut()
                    .zip(&self.shift)
                    .zip(self.unbounded.iter().zip(&self.reg_sign))
                {
                    *s = if u { d + rs * eps } else { d };
                }
                let hundred: T = (100.).as_T();
                let threshold = hundred * settings.static_regularization_constant.max(T::epsilon().sqrt());
                if band.refactor() && band.regularize_count == 0 && band.min_pivot > threshold {
                    return true;
                }
                for ((s, &d), &rs) in band.d_shift.iter_mut().zip(&self.shift).zip(&self.reg_sign) {
                    *s = d + rs * eps;
                }
                band.refactor()
            }
            Backend::Direct(ldl) => {
                // Factor the reduced form of exactly the regularized full
                // KKT matrix: ε sized as in the full solver (from its
                // diagonal: P_jj, and -H for every constraint row), and the
                // bound rows' diagonal -(H + ε) eliminated, which adds
                // a²/(H + ε).  The solve (rhs folding and recovery of z_B)
                // uses the same H + ε, so that it is an exact solve of that
                // system; refinement measures residuals of the true one.
                let maxdiag = self.p_diag.iter().chain(self.hs.iter()).fold(T::zero(), |m, &v| m.max(v.abs()));
                let eps = eps_of(maxdiag);
                self.shift.fill(T::zero());
                for i in 0..self.b_h.len() {
                    let hinv = T::recip(self.b_h[i] + eps);
                    self.b_hinv[i] = hinv;
                    self.b_ch[i] = self.b_coef[i] * hinv;
                    self.shift[self.b_pos[i] - off] += self.b_coef2[i] * hinv;
                }
                for (&idx, &v) in self.diag_idx.iter().zip(self.k0_diag.iter()) {
                    self.KKT.nzval[idx] = v;
                }
                let vals: Vec<T> = (0..self.k0_diag.len())
                    .map(|i| self.k0_diag[i] + self.shift[i] + self.reg_sign[i] * eps)
                    .collect();
                ldl.update_values(&self.diag_idx, &vals);
                ldl.refactor(&self.KKT)
            }
        }
    }

    fn setrhs(&mut self, rhsx: &[T], rhsz: &[T]) {
        self.reduce_rhs::<1>([(rhsx, rhsz)]);
    }

    fn solve(
        &mut self,
        lhsx: Option<&mut [T]>,
        lhsz: Option<&mut [T]>,
        settings: &CoreSettings<T>,
    ) -> bool {
        if !self.solve_lanes(1, settings) {
            return false;
        }
        let (n, m) = (self.x_pos.len(), self.hs.len());
        match (lhsx, lhsz) {
            (Some(x), Some(z)) => self.expand_lhs::<1>([(x, z)]),
            (x, z) => {
                let (mut wx, mut wz) = (vec![T::zero(); n], vec![T::zero(); m]);
                self.expand_lhs::<1>([(&mut wx, &mut wz)]);
                if let Some(x) = x {
                    x.copy_from(&wx);
                }
                if let Some(z) = z {
                    z.copy_from(&wz);
                }
            }
        }
        true
    }

    fn solve_pair(
        &mut self,
        rhs: [(&[T], &[T]); 2],
        lhs: [(&mut [T], &mut [T]); 2],
        settings: &CoreSettings<T>,
    ) -> bool {
        self.reduce_rhs::<2>(rhs);
        if !self.solve_lanes(2, settings) {
            return false;
        }
        self.expand_lhs::<2>(lhs);
        true
    }

    fn update_P(&mut self, P: &CscMatrix<T>) {
        let off = self.offset;
        let mut idx = vec![];
        let mut vals = vec![];
        for &p in &self.x_pos {
            self.p_diag[p - off] = T::zero();
        }
        for j in 0..P.n {
            for k in P.colptr[j]..P.colptr[j + 1] {
                let kk = self.p_map[k];
                if kk == usize::MAX {
                    continue;
                }
                if P.rowval[k] == j {
                    // diagonals are written to K₀ and the factors on the
                    // next update()
                    self.p_diag[self.x_pos[j] - off] += P.nzval[k];
                } else {
                    self.KKT.nzval[kk] = P.nzval[k];
                    idx.push(kk);
                    vals.push(P.nzval[k]);
                }
            }
        }
        for &p in &self.x_pos {
            self.KKT.nzval[self.diag_idx[p - off]] = self.p_diag[p - off];
        }
        if let Backend::Direct(ldl) = &mut self.backend {
            ldl.update_values(&idx, &vals);
        }
        self.scatter_to_band();
    }

    fn update_A(&mut self, A: &CscMatrix<T>) {
        for i in 0..self.b_nz.len() {
            let a = A.nzval[self.b_nz[i]];
            self.b_coef[i] = a;
            self.b_coef2[i] = a * a;
        }
        let mut idx = Vec::with_capacity(self.a_map.len());
        let mut vals = Vec::with_capacity(self.a_map.len());
        for &(k, kk) in &self.a_map {
            self.KKT.nzval[kk] = A.nzval[k];
            idx.push(kk);
            vals.push(A.nzval[k]);
        }
        if let Backend::Direct(ldl) = &mut self.backend {
            ldl.update_values(&idx, &vals);
        }
        self.scatter_to_band();
    }
}

// Pattern (upper triangle) of the full KKT matrix [P A'; A 0], for ordering
fn full_kkt_pattern<T: FloatT>(P: &CscMatrix<T>, A: &CscMatrix<T>) -> CscMatrix<T> {
    let (n, m) = (A.n, A.m);
    let mut colptr = vec![0usize; n + m + 1];
    for j in 0..n {
        let above = (P.colptr[j]..P.colptr[j + 1]).filter(|&k| P.rowval[k] < j).count();
        colptr[j + 1] = above + 1;
    }
    for &r in &A.rowval {
        colptr[n + r + 1] += 1;
    }
    for r in 0..m {
        colptr[n + r + 1] += 1; // diagonal
    }
    for c in 0..n + m {
        colptr[c + 1] += colptr[c];
    }
    let mut next = colptr.clone();
    let mut rowval = vec![0usize; colptr[n + m]];
    for j in 0..n {
        for k in P.colptr[j]..P.colptr[j + 1] {
            if P.rowval[k] < j {
                rowval[next[j]] = P.rowval[k];
                next[j] += 1;
            }
        }
        rowval[next[j]] = j;
        next[j] += 1;
    }
    // columns of A' in increasing variable order are sorted
    for j in 0..n {
        for k in A.colptr[j]..A.colptr[j + 1] {
            let c = n + A.rowval[k];
            rowval[next[c]] = j;
            next[c] += 1;
        }
    }
    for r in 0..m {
        rowval[next[n + r]] = n + r;
        next[n + r] += 1;
    }
    let nzval = vec![T::one(); rowval.len()];
    CscMatrix::new(n + m, n + m, colptr, rowval, nzval)
}

// The general sparse LDL solver for the reduced system, with the given
// ordering.  The choice between QDLDL and faer follows the "auto" rule of
// the full KKT solver, using the AMD analysis of the full system.
fn direct_ldl<T: FloatT>(
    KKT: &CscMatrix<T>,
    signs: &[i8],
    settings: &CoreSettings<T>,
    perm: Vec<usize>,
    info: &amd::Info,
) -> BoxedDirectLDLSolver<T> {
    #[cfg(feature = "faer-sparse")]
    {
        use super::ldlsolvers::faer_ldl::FaerDirectLDLSolver;
        let flops = (info.n_div + info.n_mult_subs_ldl) as f64;
        if flops / (info.lnz as f64) >= 40.0 {
            return Box::new(FaerDirectLDLSolver::<T>::new(KKT, signs, settings, Some(perm)));
        }
    }
    let _ = info;
    Box::new(QDLDLDirectLDLSolver::<T>::new(KKT, signs, settings, Some(perm)))
}
