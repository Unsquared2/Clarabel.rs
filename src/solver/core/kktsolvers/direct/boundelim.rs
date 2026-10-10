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
//! Auxiliary variables are eliminated as well: a variable j with no bound
//! and no off-diagonal entries in P, P_jj > 0, and a single entry a in A,
//! in a remaining row r.  Modelling tools introduce these for every
//! least-squares term (t = Fx, minimizing ‖t‖²).  Its variable row
//! P_jj x_j + a z_r = rx_j gives x_j = (rx_j - a z_r)/P_jj, which turns
//! row r into  A_r x - (H_r + a²/P_jj) z_r = rz_r - (a/P_jj) rx_j : the
//! variable leaves the system, and row r gains a nonzero pivot even when
//! it is an equality.  As for bound rows, refinement measures residuals of
//! the full system.
//!
//! The reduced system is factored with a dense-band LDLᵀ when a reverse
//! Cuthill-McKee ordering gives a small bandwidth (1D structures such as
//! rods, beams, chains and time-stepped control problems), and otherwise
//! with the LDL solver selected by the settings, as for the full system
//! (QDLDL, or faer for heavy factorizations when that feature is enabled).  Vectors are kept in the backend's (permuted) ordering.

use super::banded::*;
use super::schur::SchurLDL;
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
    Schur(Box<SchurLDL<T>>),
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

    // eliminated auxiliary variables: variable, its constraint row and
    // that row's vector index, A nz index, coefficient, P_jj, and 1/π and
    // a/π for the pivot π used (P_jj, regularized if small)
    v_var: Vec<usize>,
    v_row: Vec<usize>,
    v_rpos: Vec<usize>,
    v_nz: Vec<usize>,
    v_coef: Vec<T>,
    v_p: Vec<T>,
    v_pinv: Vec<T>,
    v_ca: Vec<T>,
    // index into v_* of every variable (usize::MAX if not eliminated)
    elim_of: Vec<usize>,

    // remaining rows: original constraint row index, vector index
    r_row: Vec<usize>,
    r_pos: Vec<usize>,
    // the variables that are not eliminated, and their vector indices
    x_var: Vec<usize>,
    x_pos: Vec<usize>,
    // vector index of every variable (usize::MAX if eliminated)
    var_pos: Vec<usize>,
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
    // the same for the eliminated auxiliary variables
    rxv: [Vec<T>; NR],
    xv: [Vec<T>; NR],
    exv: [Vec<T>; NR],
    dxv: [Vec<T>; NR],
    // scalar work for the general sparse backend, lanes for the Schur one
    w0: Vec<T>,
    w1: Vec<T>,
    sw: Vec<[T; NR]>,
    // diagonal values for the factorization (sparse and Schur backends)
    kvals: Vec<T>,
    // Schur backend: whether the current factors are of the unregularized
    // system, and the ε of the regularized one
    schur_exact: bool,
    schur_eps: T,
    // whether the last solve_lanes() met the refinement tolerance
    ir_converged: bool,
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

        // auxiliary variables to eliminate: no off-diagonal entries in P
        // (upper triangle; any lower triangle data is ignored throughout),
        // P_jj > 0, and a single entry in A, in a remaining row
        let mut p_offdiag = vec![false; n];
        let mut pjj = vec![T::zero(); n];
        for j in 0..n {
            for k in P.colptr[j]..P.colptr[j + 1] {
                let i = P.rowval[k];
                if i < j {
                    p_offdiag[i] = true;
                    p_offdiag[j] = true;
                } else if i == j {
                    pjj[j] += P.nzval[k];
                }
            }
        }
        let mut elim_of = vec![usize::MAX; n];
        let mut v_var = vec![];
        for j in 0..n {
            let (k0, k1) = (A.colptr[j], A.colptr[j + 1]);
            if !p_offdiag[j] && pjj[j] > T::zero() && k1 == k0 + 1 && !is_bound[A.rowval[k0]] {
                elim_of[j] = v_var.len();
                v_var.push(j);
            }
        }
        // reduced index of every variable that is not eliminated
        let mut vidx = vec![usize::MAX; n];
        let x_var: Vec<usize> = (0..n).filter(|&j| elim_of[j] == usize::MAX).collect();
        x_var.iter().enumerate().for_each(|(i, &j)| vidx[j] = i);
        let nr = x_var.len();

        // bounds are collected column by column, so are sorted by variable
        let (mut b_row, mut b_col, mut b_nz, mut b_coef) = (vec![], vec![], vec![], vec![]);
        let (mut v_row, mut v_nz, mut v_coef) = (vec![], vec![], vec![]);
        // A_R in coordinate form sorted by row: (row, reduced col, A nz index)
        let mut ar: Vec<(usize, usize, usize)> = Vec::with_capacity(A.nnz());
        for j in 0..n {
            for k in A.colptr[j]..A.colptr[j + 1] {
                let r = A.rowval[k];
                if is_bound[r] {
                    b_row.push(r);
                    b_col.push(j);
                    b_nz.push(k);
                    b_coef.push(A.nzval[k]);
                } else if elim_of[j] != usize::MAX {
                    v_row.push(r);
                    v_nz.push(k);
                    v_coef.push(A.nzval[k]);
                } else {
                    ar.push((ridx[r], vidx[j], k));
                }
            }
        }
        // sort by row: a stable counting sort of these column-ordered
        // entries, so columns stay ascending within each row
        let mut rptr = vec![0usize; mr + 1];
        ar.iter().for_each(|e| rptr[e.0 + 1] += 1);
        for r in 0..mr {
            rptr[r + 1] += rptr[r];
        }
        let mut sorted = vec![(0, 0, 0); ar.len()];
        for &e in &ar {
            sorted[rptr[e.0]] = e;
            rptr[e.0] += 1;
        }
        let ar = sorted;
        let v_p: Vec<T> = v_var.iter().map(|&j| pjj[j]).collect();

        // assemble K₀ (triu) of dimension nr + mr
        let dim = nr + mr;
        let nnz_est = P.nnz() + nr + ar.len() + mr;
        let mut colptr = Vec::with_capacity(dim + 1);
        let mut rowval = Vec::with_capacity(nnz_est);
        let mut nzval = Vec::with_capacity(nnz_est);
        let mut p_map = vec![usize::MAX; P.nnz()];
        let mut diag_nat = vec![0usize; dim];
        let mut p_diag_nat = vec![T::zero(); dim];
        colptr.push(0);
        for (jr, &j) in x_var.iter().enumerate() {
            let mut has_diag = false;
            for k in P.colptr[j]..P.colptr[j + 1] {
                let i = P.rowval[k];
                if i > j {
                    continue; // ignore any lower triangle data
                }
                // an eliminated variable has no off-diagonal entries
                let ir = vidx[i];
                if i == j {
                    p_diag_nat[jr] += P.nzval[k];
                    if has_diag {
                        p_map[k] = DUP_DIAG;
                        continue;
                    }
                    has_diag = true;
                    diag_nat[jr] = rowval.len();
                }
                p_map[k] = rowval.len();
                rowval.push(ir);
                nzval.push(P.nzval[k]);
            }
            if !has_diag {
                diag_nat[jr] = rowval.len();
                rowval.push(jr);
                nzval.push(T::zero());
            }
            colptr.push(rowval.len());
        }
        let mut a_map = Vec::with_capacity(ar.len());
        let mut q = 0;
        for r in 0..mr {
            while q < ar.len() && ar[q].0 == r {
                let (_, jr, k) = ar[q];
                a_map.push((k, rowval.len()));
                rowval.push(jr);
                nzval.push(A.nzval[k]);
                q += 1;
            }
            diag_nat[nr + r] = rowval.len();
            rowval.push(nr + r);
            nzval.push(T::zero());
            colptr.push(rowval.len());
        }
        let mut KKT = CscMatrix::new(dim, dim, colptr, rowval, nzval);
        for jr in 0..nr {
            KKT.nzval[diag_nat[jr]] = p_diag_nat[jr];
        }

        let mut signs = vec![1_i8; dim];
        signs[nr..].fill(-1);
        let mut has_bound = vec![false; dim];
        b_col.iter().for_each(|&j| has_bound[vidx[j]] = true);

        // Choose a backend.  Narrow bands (up to 8) always use the banded
        // factorization.  Wider ones (up to BANDED_MAX_KB) use it if its
        // dense band costs at most twice the flops of a sparse
        // factorization, estimated from the AMD analysis that the sparse
        // backend needs anyway.
        // (rows with an eliminated variable have the pivot -(H + a²/P_jj),
        // which is not small, so the ordering need not place them after
        // their variables: sign 0)
        let mut ord_signs = signs.clone();
        v_row.iter().for_each(|&r| ord_signs[nr + ridx[r]] = 0);
        let (perm, bw) = rcm_ordering(&KKT, &ord_signs);
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
        } else if let Some(schur) = schur_backend(&KKT, nr, |j| p_diag_nat[j] > T::zero() || has_bound[j], settings) {
            (Backend::Schur(Box::new(schur)), vec![], (0..dim).collect(), dim, 0)
        } else {
            // Order with AMD on the pattern of the full KKT system, as
            // the full solver does, and drop the bound rows from that
            // order, and the eliminated variables.  Each of those has a
            // single neighbour, so eliminating it creates no fill among
            // the other nodes wherever it is placed: the factor of the
            // reduced system has no more fill than that of the full one.
            // (Ordering the reduced matrix directly can be noticeably
            // worse.)
            let (fperm, _, info) = amd.unwrap_or_else(full_amd);
            let perm: Vec<usize> = fperm
                .iter()
                .filter_map(|&f| match f.checked_sub(n) {
                    None if elim_of[f] != usize::MAX => None,
                    None => Some(vidx[f]),
                    Some(r) if is_bound[r] => None,
                    Some(r) => Some(nr + ridx[r]),
                })
                .collect();
            let ldl = direct_ldl(&KKT, &signs, settings, perm, &info);
            (Backend::Direct(ldl), vec![], (0..dim).collect(), dim, 0)
        };

        // vector index of every reduced index
        let pos: Vec<usize> = iperm.iter().map(|&p| p + offset).collect();
        let x_pos = pos[0..nr].to_vec();
        let r_pos = pos[nr..].to_vec();
        let var_pos: Vec<usize> = vidx.iter().map(|&i| if i == usize::MAX { i } else { pos[i] }).collect();
        let b_pos: Vec<usize> = b_col.iter().map(|&j| var_pos[j]).collect();
        let v_rpos: Vec<usize> = v_row.iter().map(|&r| pos[nr + ridx[r]]).collect();

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
        let wlen = if matches!(backend, Backend::Direct(_)) { dim } else { 0 };
        let mr_schur = if let Backend::Schur(s) = &backend { s.mr } else { 0 };
        let nv = v_var.len();
        let mut s = Self {
            v_var,
            v_row,
            v_rpos,
            v_nz,
            v_coef,
            v_p,
            v_pinv: vec![T::zero(); nv],
            v_ca: vec![T::zero(); nv],
            elim_of,
            x_var,
            var_pos,
            rxv: lane(nv),
            xv: lane(nv),
            exv: lane(nv),
            dxv: lane(nv),
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
            sw: vec![zl; mr_schur],
            kvals: vec![T::zero(); dim],
            schur_exact: false,
            schur_eps: T::zero(),
            ir_converged: true,
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
        // K₀ structure, for symv_lanes_sub
        let K = &self.KKT;
        assert!(K.colptr.len() == K.n + 1 && K.colptr.windows(2).all(|w| w[0] <= w[1]));
        assert!(K.rowval.iter().all(|&i| i < K.n) && K.colptr[K.n] <= K.rowval.len());
        let n = self.var_pos.len();
        assert!(self.bptr.len() == n + 1 && self.bptr[n] == nb);
        assert!(self.bptr.windows(2).all(|w| w[0] <= w[1]));
        assert!(self.x_var.len() == self.x_pos.len() && self.x_var.iter().all(|&j| j < n));
        let nv = self.v_var.len();
        assert!(self.v_var.iter().all(|&j| j < n) && self.v_rpos.iter().all(|&p| p < len));
        assert!(self.elim_of.len() == n && self.x_var.len() + nv == n);
        for v in [&self.v_row, &self.v_rpos, &self.v_nz] {
            assert!(v.len() == nv);
        }
        for v in [&self.v_coef, &self.v_p, &self.v_pinv, &self.v_ca] {
            assert!(v.len() == nv);
        }
        for l in 0..NR {
            for v in [&self.rxv[l], &self.xv[l], &self.exv[l], &self.dxv[l]] {
                assert!(v.len() == nv);
            }
        }
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
            assert!(rx.len() == self.var_pos.len() && rz.len() == m);
        }
        let (b, b0) = (&mut self.b, &mut self.b0);
        let (b_row, b_ch, bptr) = (&self.b_row, &self.b_ch, &self.bptr);
        let [rzb0, rzb1] = &mut self.rzb;
        unsafe {
            for (&j, &p) in self.x_var.iter().zip(&self.x_pos) {
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
            // eliminated variables: rz_r - (a/π) rx_j in their rows
            for e in 0..self.v_var.len() {
                let j = *self.v_var.get_unchecked(e);
                let c = *self.v_ca.get_unchecked(e);
                let bp = b.get_unchecked_mut(*self.v_rpos.get_unchecked(e));
                for l in 0..L {
                    let t = *rhs[l].0.get_unchecked(j);
                    *self.rxv[l].get_unchecked_mut(e) = t;
                    bp[l] = fmadd(-c, t, bp[l]);
                }
            }
        }
        if L < 2 {
            rzb1.fill(T::zero());
            self.rxv[1].fill(T::zero());
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
            assert!(lx.len() == self.var_pos.len() && lz.len() == m);
        }
        let x = &self.x;
        unsafe {
            for (&j, &p) in self.x_var.iter().zip(&self.x_pos) {
                let xj = *x.get_unchecked(p);
                for l in 0..L {
                    *lhs[l].0.get_unchecked_mut(j) = xj[l];
                }
            }
            for (e, &j) in self.v_var.iter().enumerate() {
                for l in 0..L {
                    *lhs[l].0.get_unchecked_mut(j) = *self.xv[l].get_unchecked(e);
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
        sw: &mut [[T; NR]],
    ) {
        match backend {
            Backend::Banded(band) => band.solve_permuted::<NR>(v),
            Backend::Schur(s) => s.solve::<NR>(v, sw, nlanes),
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

    // x_j = (r - a z_r)/π for each eliminated variable and lane
    fn recover_xv(&self, x: &[[T; NR]], r: &[Vec<T>; NR], xv: &mut [Vec<T>; NR]) {
        assert!(x.len() == self.b.len());
        // SAFETY: lengths and v_rpos < b.len() == x.len() checked in new()
        unsafe {
            for e in 0..self.v_var.len() {
                let pinv = *self.v_pinv.get_unchecked(e);
                let a = *self.v_coef.get_unchecked(e);
                let zr = *x.get_unchecked(*self.v_rpos.get_unchecked(e));
                for l in 0..NR {
                    *xv[l].get_unchecked_mut(e) = pinv * (*r[l].get_unchecked(e) - a * zr[l]);
                }
            }
        }
    }

    // Residual of the full KKT system for (x, zb, xv), into e (variable
    // and remaining rows, backend order), eb (eliminated rows) and exv
    // (eliminated variables).  Returns its inf norm per lane (NaN if it
    // has any).
    fn full_residual(&mut self, use_dx: bool, nlanes: usize) -> [T; NR] {
        let (x, zb) = if use_dx { (&self.dx, &self.dzb) } else { (&self.x, &self.zb) };
        let xv = if use_dx { &self.dxv } else { &self.xv };
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
            // eliminated variables: their column of A in their row, and
            // their own rows rx_j - P_jj x_j - a z_r
            for i in 0..self.v_var.len() {
                let a = *self.v_coef.get_unchecked(i);
                let pjj = *self.v_p.get_unchecked(i);
                let p = *self.v_rpos.get_unchecked(i);
                let zr = *x.get_unchecked(p);
                let ep = e.get_unchecked_mut(p);
                for l in 0..NR {
                    let xi = *xv[l].get_unchecked(i);
                    ep[l] -= a * xi;
                    let v = *self.rxv[l].get_unchecked(i) - (pjj * xi + a * zr[l]);
                    *self.exv[l].get_unchecked_mut(i) = v;
                    nb[l] = if v.abs() > nb[l] || v.is_nan() { v.abs() } else { nb[l] };
                }
            }
        }
        // then the K₀ part, e -= K₀ x
        let ne = match &mut self.backend {
            Backend::Banded(band) => band.residual_permuted::<NR>(e, x),
            Backend::Schur(s) if s.is_dense() => {
                // K₀ = [diag(P) A_R'; A_R -H_R], held by the factorization
                s.residual_sub::<NR>(e, x, &self.k0_diag, nlanes);
                lane_norm_inf(e, [T::zero(); NR])
            }
            Backend::Direct(_) | Backend::Schur(_) => {
                symv_lanes_sub(&self.KKT, x, e);
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
        let ok = self.solve_lanes_once(nlanes, settings);
        // An exact Schur factorization whose solve does not refine to the
        // tolerance (badly scaled problems): switch to the regularized one
        // for the rest of this iteration and solve again.
        if self.schur_exact && !(ok && self.ir_converged) {
            self.schur_exact = false;
            self.fold(self.schur_eps);
            if !self.schur_factor().0 {
                return false;
            }
            self.refold_rhs();
            return self.solve_lanes_once(nlanes, settings);
        }
        ok
    }

    fn solve_lanes_once(&mut self, nlanes: usize, settings: &CoreSettings<T>) -> bool {
        self.ir_converged = true;
        let finite = |s: &Self| {
            s.x.iter().all(|v| v.iter().take(nlanes).all(|t| t.is_finite()))
                && s.zb.iter().take(nlanes).all(|v| v.iter().all(|t| t.is_finite()))
                && s.xv.iter().take(nlanes).all(|v| v.iter().all(|t| t.is_finite()))
        };

        self.x.copy_from_slice(&self.b);
        Self::reduced_solve(&mut self.backend, &self.KKT, &mut self.x, nlanes, &mut self.w0, &mut self.w1, &mut self.sw);
        let mut zb = std::mem::take(&mut self.zb);
        self.recover_zb(&self.x, &self.rzb, &mut zb);
        self.zb = zb;
        let mut xv = std::mem::take(&mut self.xv);
        self.recover_xv(&self.x, &self.rxv, &mut xv);
        self.xv = xv;

        if !settings.iterative_refinement_enable {
            return finite(self);
        }
        let reltol = settings.iterative_refinement_reltol;
        let abstol = settings.iterative_refinement_abstol;
        let maxiter = settings.iterative_refinement_max_iter;
        let stopratio = settings.iterative_refinement_stop_ratio;

        let normb = self.normb;

        let mut norme = self.full_residual(false, nlanes);
        // residual of the accepted iterate
        let mut best = norme;
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
                    for i in 0..self.v_rpos.len() {
                        let p = self.v_rpos[i];
                        self.dx[p][l] = fmadd(-self.v_ca[i], self.exv[l][i], self.dx[p][l]);
                    }
                }
            }
            Self::reduced_solve(&mut self.backend, &self.KKT, &mut self.dx, nlanes, &mut self.w0, &mut self.w1, &mut self.sw);
            let mut dzb = std::mem::take(&mut self.dzb);
            self.recover_zb(&self.dx, &self.eb, &mut dzb);
            let mut dxv = std::mem::take(&mut self.dxv);
            self.recover_xv(&self.dx, &self.exv, &mut dxv);
            // candidate = current + correction
            for l in 0..nlanes {
                for (d, z) in dzb[l].iter_mut().zip(&self.zb[l]) {
                    *d += *z;
                }
                for (d, v) in dxv[l].iter_mut().zip(&self.xv[l]) {
                    *d += *v;
                }
            }
            self.dzb = dzb;
            self.dxv = dxv;
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
                    best[l] = newe[l];
                    for (x, d) in self.x.iter_mut().zip(self.dx.iter()) {
                        x[l] = d[l];
                    }
                    self.zb[l].copy_from_slice(&self.dzb[l]);
                    self.xv[l].copy_from_slice(&self.dxv[l]);
                }
                if ratio < stopratio {
                    active[l] = false;
                } else {
                    norme[l] = newe[l];
                    active[l] = newe[l] > abstol + reltol * normb[l];
                }
            }
        }
        self.ir_converged = (0..nlanes).all(|l| best[l] <= abstol + reltol * normb[l]);
        // every accepted iterate had a finite residual, so is finite
        true
    }
}

// e -= K x on all lanes, for K symmetric with its upper triangle given,
// in one pass over K
fn symv_lanes_sub<T: FloatT>(K: &CscMatrix<T>, x: &[[T; NR]], e: &mut [[T; NR]]) {
    let n = K.n;
    assert!(x.len() == n && e.len() == n && K.colptr.len() == n + 1);
    assert!(K.colptr[n] <= K.rowval.len() && K.colptr[n] <= K.nzval.len());
    // SAFETY: K is built in new() with every row index below n and
    // nondecreasing colptr (checked there); lengths are checked above
    unsafe {
        for j in 0..n {
            let xj = *x.get_unchecked(j);
            let mut acc = [T::zero(); NR];
            for k in *K.colptr.get_unchecked(j)..*K.colptr.get_unchecked(j + 1) {
                let i = *K.rowval.get_unchecked(k);
                let v = *K.nzval.get_unchecked(k);
                let ei = e.get_unchecked_mut(i);
                for l in 0..NR {
                    ei[l] = fmadd(-v, xj[l], ei[l]);
                }
                if i != j {
                    let xi = *x.get_unchecked(i);
                    for l in 0..NR {
                        acc[l] = fmadd(v, xi[l], acc[l]);
                    }
                }
            }
            let ej = e.get_unchecked_mut(j);
            for l in 0..NR {
                ej[l] -= acc[l];
            }
        }
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
            Backend::Schur(s) => LinearSolverInfo {
                name: format!("schur(m={})+boundelim", s.mr),
                threads: 1,
                direct: true,
                nnzA: self.KKT.nnz(),
                nnzL: s.mr * (s.mr + 1) / 2,
            },
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
                let mut maxdiag = self.v_p.iter().fold(T::zero(), |m, &v| m.max(v.abs()));
                for (ad, (&k, &d)) in band.a_diag.iter_mut().zip(self.k0_diag.iter().zip(&self.shift)) {
                    *ad = k;
                    let v = (k + d).abs();
                    maxdiag = if v > maxdiag || v.is_nan() { v } else { maxdiag };
                }
                let eps = eps_of(maxdiag);
                let hundred: T = (100.).as_T();
                let threshold = hundred * settings.static_regularization_constant.max(T::epsilon().sqrt());
                // eliminated variables: the exact P_jj as pivot, unless it
                // is small (after a data update), then P_jj + ε
                for i in 0..self.v_p.len() {
                    let p = self.v_p[i];
                    let p = if p > threshold { p } else { p + eps };
                    let pinv = T::recip(if p > tiny { p } else { tiny });
                    let ca = self.v_coef[i] * pinv;
                    self.v_pinv[i] = pinv;
                    self.v_ca[i] = ca;
                    self.shift[self.v_rpos[i] - off] -= self.v_coef[i] * ca;
                }
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
                if band.refactor() && band.regularize_count == 0 && band.min_pivot > threshold {
                    return true;
                }
                for ((s, &d), &rs) in band.d_shift.iter_mut().zip(&self.shift).zip(&self.reg_sign) {
                    *s = d + rs * eps;
                }
                band.refactor()
            }
            Backend::Direct(_) => {
                // Factor the reduced form of exactly the regularized full
                // KKT matrix: ε sized as in the full solver (from its
                // diagonal: P_jj, and -H for every constraint row), and the
                // bound rows' diagonal -(H + ε) eliminated, which adds
                // a²/(H + ε).  The solve (rhs folding and recovery of z_B)
                // uses the same H + ε, so that it is an exact solve of that
                // system; refinement measures residuals of the true one.
                let eps = eps_of(self.regularization_scale());
                self.fold(eps);
                let Backend::Direct(ldl) = &mut self.backend else { unreachable!() };
                ldl.update_values(&self.diag_idx, &self.kvals);
                ldl.refactor(&self.KKT)
            }
            Backend::Schur(_) => {
                // As for the sparse backend, but first with ε = 0, since
                // all the variables have a positive diagonal (P_jj > 0 or a
                // bound).  Solving the exact system saves the refinement
                // step that the O(ε) perturbation needs on every solve.
                // Small or replaced pivots, and any solve that does not
                // refine to the tolerance (see solve_lanes), fall back to
                // the regularized system.
                let eps = eps_of(self.regularization_scale());
                self.schur_eps = eps;
                if settings.iterative_refinement_enable {
                    self.fold(T::zero());
                    let (ok, safe) = self.schur_factor();
                    if ok && safe {
                        self.schur_exact = true;
                        return true;
                    }
                }
                self.schur_exact = false;
                self.fold(eps);
                self.schur_factor().0
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
        let (n, m) = (self.var_pos.len(), self.hs.len());
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
        self.v_p.fill(T::zero());
        for j in 0..P.n {
            for k in P.colptr[j]..P.colptr[j + 1] {
                let e = self.elim_of[j];
                if e != usize::MAX {
                    if P.rowval[k] == j {
                        self.v_p[e] += P.nzval[k];
                    }
                    continue;
                }
                let kk = self.p_map[k];
                if kk == usize::MAX {
                    continue;
                }
                if P.rowval[k] == j {
                    // diagonals are written to K₀ and the factors on the
                    // next update()
                    self.p_diag[self.var_pos[j] - off] += P.nzval[k];
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
        for i in 0..self.v_nz.len() {
            self.v_coef[i] = A.nzval[self.v_nz[i]];
        }
        if let Backend::Schur(s) = &mut self.backend {
            s.invalidate_values();
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

// The dense Schur complement backend, if the reduced system suits it: no
// off-diagonal entries of P among the remaining variables (each variable
// column of K₀ holds only its diagonal), every one of them with a positive
// diagonal term (`pos_diag`: P_jj > 0 or a bound), and a dense factor of the rows'
// Schur complement that costs no more than about twice forming it (which
// any elimination of the variables first must do), or few rows anyway.
fn schur_backend<T: FloatT>(
    KKT: &CscMatrix<T>,
    nr: usize,
    pos_diag: impl Fn(usize) -> bool,
    settings: &CoreSettings<T>,
) -> Option<SchurLDL<T>> {
    const SCHUR_MAX_M: usize = 1000;
    const SCHUR_SMALL_M: usize = 48;
    let dim = KKT.n;
    let mr = dim - nr;
    if test_hooks::no_schur() || mr == 0 || mr > SCHUR_MAX_M {
        return None;
    }
    if (0..nr).any(|j| KKT.colptr[j + 1] - KKT.colptr[j] != 1 || !pos_diag(j)) {
        return None;
    }
    // A by variable, from the row columns of K₀ (their off-diagonals)
    let mut cols: Vec<Vec<(usize, usize)>> = vec![vec![]; nr];
    for r in 0..mr {
        for k in KKT.colptr[nr + r]..KKT.colptr[nr + r + 1] {
            let i = KKT.rowval[k];
            if i < nr {
                cols[i].push((r, k));
            }
        }
    }
    let form: f64 = cols.iter().map(|c| (c.len() * (c.len() + 1) / 2) as f64).sum();
    let cube = (mr as f64).powi(3) / 3.0;
    if mr > SCHUR_SMALL_M && cube > 2.0 * form {
        return None;
    }
    Some(SchurLDL::new(
        nr,
        mr,
        &cols,
        settings.dynamic_regularization_eps,
        settings.dynamic_regularization_delta,
    ))
}

impl<T: FloatT> BoundElimKKTSolver<T> {
    // the largest diagonal of the full KKT system, for sizing ε
    fn regularization_scale(&self) -> T {
        (self.p_diag.iter().chain(&self.v_p).chain(&self.hs)).fold(T::zero(), |m, &v| m.max(v.abs()))
    }

    // Diagonal values to factor (into kvals), and the coefficients that
    // fold the eliminated rows and variables into the reduced rhs, for the
    // full system with every diagonal regularized by ε = e (sparse and
    // Schur backends, whose positions are natural).  Also sets K₀'s
    // diagonal.
    fn fold(&mut self, e: T) {
        let tiny = T::min_positive_value().sqrt();
        for (&idx, &v) in self.diag_idx.iter().zip(self.k0_diag.iter()) {
            self.KKT.nzval[idx] = v;
        }
        self.shift.fill(T::zero());
        for i in 0..self.b_h.len() {
            let hinv = T::recip(self.b_h[i] + e);
            self.b_hinv[i] = hinv;
            self.b_ch[i] = self.b_coef[i] * hinv;
            self.shift[self.b_pos[i]] += self.b_coef2[i] * hinv;
        }
        // likewise the eliminated variables' diagonal P_jj + ε
        for i in 0..self.v_p.len() {
            let p = self.v_p[i] + e;
            let pinv = T::recip(if p > tiny { p } else { tiny });
            let ca = self.v_coef[i] * pinv;
            self.v_pinv[i] = pinv;
            self.v_ca[i] = ca;
            self.shift[self.v_rpos[i]] -= self.v_coef[i] * ca;
        }
        for i in 0..self.kvals.len() {
            self.kvals[i] = self.k0_diag[i] + self.shift[i] + self.reg_sign[i] * e;
        }
    }

    // Factor kvals with the Schur backend: (finite, no small or replaced
    // pivot).  "Small" is relative to S's largest diagonal and stricter
    // than numerical safety alone would need: when S is that close to
    // singular, the regularized system's damping of its near null space
    // matters to the interior point iterations (badly scaled problems
    // reached wrong verdicts with exact solves that refined fine).
    fn schur_factor(&mut self) -> (bool, bool) {
        let Backend::Schur(s) = &mut self.backend else { unreachable!() };
        // positions are natural: variables, then rows, whose diagonal is -G
        let nr = s.nr;
        let vals = &mut self.kvals;
        vals[nr..].iter_mut().for_each(|v| *v = -*v);
        let ok = s.factor(&vals[..nr], &vals[nr..], &self.KKT.nzval);
        let safe = s.regularize_count == 0 && s.min_pivot_ratio() > (1e-6).as_T();
        (ok, safe)
    }

    // Rebuild the reduced rhs from the K₀ rhs and the eliminated parts of
    // the full rhs, after the folding coefficients changed
    fn refold_rhs(&mut self) {
        self.b.copy_from_slice(&self.b0);
        for (&j, &p) in self.x_var.iter().zip(&self.x_pos) {
            for i in self.bptr[j]..self.bptr[j + 1] {
                for l in 0..NR {
                    self.b[p][l] = fmadd(self.b_ch[i], self.rzb[l][i], self.b[p][l]);
                }
            }
        }
        for e in 0..self.v_var.len() {
            let p = self.v_rpos[e];
            for l in 0..NR {
                self.b[p][l] = fmadd(-self.v_ca[e], self.rxv[l][e], self.b[p][l]);
            }
        }
    }
}
