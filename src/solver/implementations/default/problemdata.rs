#![allow(non_snake_case)]
use itertools::izip;

use super::*;
use crate::algebra::*;
use crate::solver::core::{
    cones::{CompositeCone, Cone},
    traits::ProblemData,
};
use crate::solver::SupportedConeT;

#[cfg(feature = "sdp")]
use crate::solver::chordal::ChordalInfo;

// ---------------
// Data type for default problem format
// ---------------

/// Standard-form solver type implementing the [`ProblemData`](crate::solver::core::traits::ProblemData) trait
pub struct DefaultProblemData<T> {
    /// The matrix P in the quadratic objective term
    pub P: CscMatrix<T>,
    /// The vector q in the quadratic objective term
    pub q: Vec<T>,
    /// The matrix A in the constraints
    pub A: CscMatrix<T>,
    /// The vector b in the constraints
    pub b: Vec<T>,
    /// Vector of cones in the problem
    pub cones: Vec<SupportedConeT<T>>,
    /// Number of variables
    pub n: usize,
    /// Number of constraints
    pub m: usize,
    /// Equilibration data for the problem
    pub equilibration: DefaultEquilibrationData<T>,

    // unscaled inf norms of linear terms.  Set to "None"
    // during data updating to allow for multiple updates, and
    // then recalculated during solve if needed
    normq: Option<T>,
    normb: Option<T>,

    pub(crate) presolver: Option<Presolver<T>>,
    dropped_zeros: usize, // number of eliminated structural zeros

    #[cfg(feature = "sdp")]
    pub(crate) chordal_info: Option<ChordalInfo<T>>,

    // full symmetric copy of P for gather-only products (see P_mul)
    P_full: FullSym<T>,
}

// Both triangles of a symmetric matrix stored by column, with the index of
// the source triangular entry of every value so that it can be refreshed
// when the triangular data changes.  y = Px is then a gather-only dot
// product per row, with no scattered read-modify-write traffic.
struct FullSym<T> {
    colptr: Vec<usize>,
    rowval: Vec<usize>,
    nzval: Vec<T>,
    src: Vec<usize>,
}

impl<T: FloatT> FullSym<T> {
    fn new(P: &CscMatrix<T>) -> Self {
        let n = P.n;
        let mut cnt = vec![0usize; n + 1];
        for j in 0..n {
            for k in P.colptr[j]..P.colptr[j + 1] {
                let i = P.rowval[k];
                cnt[j + 1] += 1;
                if i != j {
                    cnt[i + 1] += 1;
                }
            }
        }
        for j in 0..n {
            cnt[j + 1] += cnt[j];
        }
        let nnz = cnt[n];
        let mut next = cnt.clone();
        let mut rowval = vec![0usize; nnz];
        let mut src = vec![0usize; nnz];
        for j in 0..n {
            for k in P.colptr[j]..P.colptr[j + 1] {
                let i = P.rowval[k];
                rowval[next[j]] = i;
                src[next[j]] = k;
                next[j] += 1;
                if i != j {
                    rowval[next[i]] = j;
                    src[next[i]] = k;
                    next[i] += 1;
                }
            }
        }
        assert!(rowval.iter().all(|&r| r < n));
        let mut out = Self { colptr: cnt, rowval, nzval: vec![T::zero(); nnz], src };
        out.refresh(P);
        out
    }

    fn refresh(&mut self, P: &CscMatrix<T>) {
        for (v, &k) in self.nzval.iter_mut().zip(&self.src) {
            *v = P.nzval[k];
        }
    }

    // y = P x
    fn mul(&self, y: &mut [T], x: &[T]) {
        let n = self.colptr.len() - 1;
        assert!(y.len() == n && x.len() == n);
        for (j, yj) in y.iter_mut().enumerate() {
            let mut acc = T::zero();
            for k in self.colptr[j]..self.colptr[j + 1] {
                // SAFETY: k < nnz by colptr construction, and every row
                // index was checked to be < n in new()
                unsafe {
                    let xk = *x.get_unchecked(*self.rowval.get_unchecked(k));
                    acc = fmadd(*self.nzval.get_unchecked(k), xk, acc);
                }
            }
            *yj = acc;
        }
    }
}

impl<T> DefaultProblemData<T>
where
    T: FloatT,
{
    /// Create a new `DefaultProblemData` object
    pub fn new(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        settings: &DefaultSettings<T>,
    ) -> Self {
        // clean up the cones by consolidating repeated NNs,
        // eliminate empty cones, transform singletons etc
        // this makes a locally owned copy of the cones
        let cones = SupportedConeT::new_collapsed(cones);

        // some caution is required to ensure we take a minimal,
        // but nonzero, number of data copies during presolve steps

        let mut P_new: Option<CscMatrix<T>> = None;
        #[allow(unused_mut)] // mut q_new only needed with chordal
        let mut q_new: Option<Vec<T>> = None;
        let mut A_new: Option<CscMatrix<T>> = None;
        let mut b_new: Option<Vec<T>> = None;
        let mut cones_new: Option<Vec<SupportedConeT<T>>> = None;

        if !P.is_triu() {
            P_new = Some(P.to_triu());
        }

        // presolve : return nothing if disabled or no reduction
        // --------------------------------------
        let presolver = try_presolver(A, b, &cones, settings);

        if let Some(ref presolver) = presolver {
            let (_A_new, _b_new, _cones_new) = presolver.presolve(A, b, &cones);
            (A_new, b_new, cones_new) = (Some(_A_new), Some(_b_new), Some(_cones_new));
        }

        // chordal decomposition : return nothing if disabled or no decomp
        // --------------------------------------
        #[cfg(feature = "sdp")]
        let mut chordal_info = try_chordal_info(A, b, &cones, settings);
        #[cfg(feature = "sdp")]
        if let Some(ref mut chordal_info) = chordal_info {
            let (_P_new, _q_new, _A_new, _b_new, _cones_new) = chordal_info.decomp_augment(
                P_new.as_ref().unwrap_or(P),
                unwrap_and_slice_or_else(&q_new, || q),
                A_new.as_ref().unwrap_or(A),
                unwrap_and_slice_or_else(&b_new, || b),
                settings,
            );
            (P_new, q_new, A_new, b_new, cones_new) = (
                Some(_P_new),
                Some(_q_new),
                Some(_A_new),
                Some(_b_new),
                Some(_cones_new),
            );
        }

        // now make sure we have a clean copy of everything if we
        // haven't made one already.   Necessary since we will scale
        // the internal copy and don't want to step on the user

        let mut P_new = P_new.unwrap_or_else(|| P.clone());
        let q_new = q_new.unwrap_or_else(|| q.to_vec());
        let mut A_new = A_new.unwrap_or_else(|| A.clone());
        let mut b_new = b_new.unwrap_or_else(|| b.to_vec());

        // cones was already copied, so can just pass through without cloning
        let cones_new = cones_new.unwrap_or(cones);

        //cap entries in b at INFINITY.  This is important
        //for inf values that were not in a reduced cone
        //this is not considered part of the "presolve", so
        //can always happen regardless of user settings
        let infbound = crate::get_infinity().as_T();
        b_new.scalarop(|x| T::min(x, infbound));

        // this ensures m is the *reduced* size m
        let (m, n) = A_new.size();

        // explicitly dropzeros on the copied data, since dropzeros
        // operates in place.  PJG: revisit this order of operations
        // once a proper presolver is implemented, since it might
        // be preferable to dropzeros then presolve
        let mut dropped_zeros = 0;
        if settings.input_sparse_dropzeros {
            dropped_zeros += P_new.dropzeros() + A_new.dropzeros();
        }

        let equilibration = DefaultEquilibrationData::<T>::new(n, m);

        let normq = Some(q_new.norm_inf());
        let normb = Some(b_new.norm_inf());

        let P_full = FullSym::new(&P_new);
        Self {
            P_full,
            P: P_new,
            q: q_new,
            A: A_new,
            b: b_new,
            cones: cones_new,
            n,
            m,
            equilibration,
            normq,
            normb,
            dropped_zeros,
            presolver,
            #[cfg(feature = "sdp")]
            chordal_info,
        }
    }

    /// y = P x, with P the (scaled) symmetric quadratic term
    pub(crate) fn P_mul(&self, y: &mut [T], x: &[T]) {
        self.P_full.mul(y, x);
    }

    /// to be called after the values of P are changed
    pub(crate) fn refresh_P_full(&mut self) {
        self.P_full.refresh(&self.P);
    }

    pub(crate) fn get_normq(&mut self) -> T {
        if let Some(norm) = self.normq {
            norm
        } else {
            let dinv = &self.equilibration.dinv;
            let cinv = T::recip(self.equilibration.c);
            let norm = self.q.norm_inf_scaled(dinv) * cinv;
            self.normq = Some(norm);
            norm
        }
    }

    pub(crate) fn get_normb(&mut self) -> T {
        if let Some(norm) = self.normb {
            norm
        } else {
            let einv = &self.equilibration.einv;
            let norm = self.b.norm_inf_scaled(einv);
            self.normb = Some(norm);
            norm
        }
    }

    pub(crate) fn clear_normq(&mut self) {
        self.normq = None;
    }

    pub(crate) fn clear_normb(&mut self) {
        self.normb = None;
    }

    // data updating not supported following presolve
    //reduction or chordal decomposition
    pub(crate) fn is_presolved(&self) -> bool {
        self.presolver.is_some()
    }

    // data updating not supported if structural zeros
    // have been eliminated
    pub(crate) fn is_dropped_zeros(&self) -> bool {
        self.dropped_zeros != 0
    }

    #[allow(dead_code)]
    pub(crate) fn is_chordal_decomposed(&self) -> bool {
        #[cfg(feature = "sdp")]
        if self.chordal_info.is_some() {
            return true;
        }
        false
    }
}

impl<T> ProblemData<T> for DefaultProblemData<T>
where
    T: FloatT,
{
    type V = DefaultVariables<T>;
    type C = CompositeCone<T>;
    type SE = DefaultSettings<T>;

    fn equilibrate(&mut self, cones: &CompositeCone<T>, settings: &DefaultSettings<T>) {
        let data = self;
        let equil = &mut data.equilibration;

        // if equilibration is disabled, just return.  Note that
        // the default equilibration structure initializes with
        // identity scaling already.
        if !settings.equilibrate_enable {
            return;
        }

        // references to scaling matrices from workspace
        let (d, e) = (&mut equil.d, &mut equil.e);

        // use the inverse scalings as work vectors
        let dwork = &mut equil.dinv;
        let ework = &mut equil.einv;

        // references to problem data
        // note that P may be triu, but it shouldn't matter
        let (P, A, q, b) = (&mut data.P, &mut data.A, &mut data.q, &mut data.b);

        let scale_min = settings.equilibrate_min_scaling;
        let scale_max = settings.equilibrate_max_scaling;

        // column norms of the scaled P, gathered while scaling
        let mut pcol = vec![T::zero(); P.n];

        // perform scaling operations for a fixed number of steps
        for _ in 0..settings.equilibrate_max_iter {
            kkt_col_norms(P, A, dwork, ework);

            // zero rows or columns should not get scaled, then take
            // 1/sqrt of the norms and bound the cumulative scaling.
            // (one fused, vectorizable pass per vector)
            let fix = |w: &mut [T], cum: &[T]| {
                for (w, &c) in izip!(w.iter_mut(), cum.iter()) {
                    // clip(1/r, min/c, max/c) with r = sqrt(x), using a
                    // single division: c/r < min <=> c < min r, etc.
                    // (branch free, so that the loop vectorizes)
                    let x = if *w == T::zero() { T::one() } else { *w };
                    let r = T::sqrt(x);
                    let (lo, hi) = (c < scale_min * r, c > scale_max * r);
                    let num = if lo { scale_min } else if hi { scale_max } else { T::one() };
                    let den = if lo || hi { c } else { r };
                    *w = num / den;
                }
            };
            fix(dwork, d);
            fix(ework, e);

            // Scale the problem data and update the equilibration
            // matrices, collecting the column norms of the newly scaled P
            // (to compute their mean) in the same pass over P
            lrscale_col_norms(P, dwork, &mut pcol);
            A.lrscale(ework, dwork);
            q.hadamard(dwork);
            b.hadamard(ework);
            d.hadamard(dwork);
            e.hadamard(ework);
            let mean_col_norm_P = pcol.mean();
            let inf_norm_q = q.norm_inf();

            if mean_col_norm_P != T::zero() && inf_norm_q != T::zero() {
                let scale_cost = T::max(inf_norm_q, mean_col_norm_P);
                let ctmp = T::recip(scale_cost);
                let ctmp = T::clip(&ctmp, scale_min / equil.c, scale_max / equil.c);

                // scale the penalty terms and overall scaling
                P.scale(ctmp);
                q.scale(ctmp);
                equil.c *= ctmp;
            }
        } //end Ruiz scaling loop

        // fix scalings in cones for which elementwise
        // scaling can't be applied. Rectification should
        //either do nothing or take a convex combination of
        //scalings over a cone, so shouldn't need to check
        //bounds on the scalings here
        if cones.rectify_equilibration(ework, e) {
            // only rescale again if some cones were rectified
            scale_data(P, A, q, b, None, ework);
            e.hadamard(ework);
        }

        // update the inverse scaling data
        equil.dinv.scalarop_from(T::recip, d);
        equil.einv.scalarop_from(T::recip, e);

        data.P_full.refresh(&data.P);
    }
}

// ---------------
// utilities
// ---------------

fn kkt_col_norms<T: FloatT>(
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    norm_LHS: &mut [T],
    norm_RHS: &mut [T],
) {
    P.col_norms_sym(norm_LHS); // P can be triu
    // column norms of A (incrementally from the P norms) and row norms of
    // A (the column norms of A'), in one pass over A
    norm_RHS.fill(T::zero());
    assert!(norm_LHS.len() == A.n && norm_RHS.len() == A.m);
    for (j, nl) in norm_LHS.iter_mut().enumerate() {
        let mut m = *nl;
        for k in A.colptr[j]..A.colptr[j + 1] {
            let v = A.nzval[k].abs();
            m = if v > m { v } else { m };
            let nr = &mut norm_RHS[A.rowval[k]];
            *nr = if v > *nr { v } else { *nr };
        }
        *nl = m;
    }
}

// P = D P D for diagonal D, and the column norms of the result (of the
// stored triangle, as P.col_norms) into `norms`
fn lrscale_col_norms<T: FloatT>(P: &mut CscMatrix<T>, d: &[T], norms: &mut [T]) {
    assert!(d.len() == P.n && norms.len() == P.n);
    for (j, nj) in norms.iter_mut().enumerate() {
        let (first, last) = (P.colptr[j], P.colptr[j + 1]);
        let dj = d[j];
        let mut m = T::zero();
        for (v, &i) in P.nzval[first..last].iter_mut().zip(&P.rowval[first..last]) {
            *v *= d[i] * dj;
            let a = v.abs();
            m = if a > m { a } else { m };
        }
        *nj = m;
    }
}

fn scale_data<T: FloatT>(
    P: &mut CscMatrix<T>,
    A: &mut CscMatrix<T>,
    q: &mut [T],
    b: &mut [T],
    d: Option<&[T]>,
    e: &[T],
) {
    match d {
        Some(d) => {
            P.lrscale(d, d); // P[:,:] = Ds*P*Ds
            A.lrscale(e, d);
            q.hadamard(d);
        }
        None => {
            A.lscale(e); // A[:,:] = Es*A
        }
    }
    b.hadamard(e);
}

#[cfg(feature = "sdp")]
fn try_chordal_info<T>(
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    settings: &DefaultSettings<T>,
) -> Option<ChordalInfo<T>>
where
    T: FloatT,
{
    if !settings.chordal_decomposition_enable {
        return None;
    }

    // nothing to do if there are no PSD cones or they are all small
    if !cones
        .iter()
        .any(|c| matches!(c, SupportedConeT::PSDTriangleConeT(dim) if *dim > 3))
    {
        return None;
    }

    let chordal_info = ChordalInfo::new(A, b, cones, settings);

    // no decomposition possible
    if !chordal_info.is_decomposed() {
        return None;
    }

    Some(chordal_info)
}

fn try_presolver<T>(
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    settings: &DefaultSettings<T>,
) -> Option<Presolver<T>>
where
    T: FloatT,
{
    if !settings.presolve_enable {
        return None;
    }

    let presolver = Presolver::new(A, b, cones, settings);

    if !presolver.is_reduced() {
        return None;
    }

    Some(presolver)
}

// -- utility function that tries to unwrap and slice a vector, or return
// an alternative.   Necessary since the Options for q and b are &Vec, but
// the user supplied data is a slice &[T]
#[cfg(feature = "sdp")]
pub(crate) fn unwrap_and_slice_or_else<'a, T, F>(opt: &'a Option<Vec<T>>, f: F) -> &'a [T]
where
    F: FnOnce() -> &'a [T],
    T: FloatT,
{
    if opt.is_some() {
        opt.as_ref().unwrap().as_slice()
    } else {
        f()
    }
}
