#![allow(non_snake_case)]
// index loops are the clearest form for these numeric kernels
#![allow(clippy::needless_range_loop)]
//! Dense Schur complement factorization of a quasidefinite system
//!
//!   [ D   A' ] [x]   [bx]
//!   [ A  -G  ] [z] = [bz]
//!
//! with D (nr × nr) and G (mr × mr) diagonal, D > 0 and G ≥ 0, for few
//! rows mr.  Eliminating x gives  S z = A D⁻¹ bx - bz  with
//! S = G + A D⁻¹ A'  symmetric positive definite, then x = D⁻¹(bx - A'z).
//! S is formed densely and factored by Cholesky.  This is the elimination
//! order a sparse LDLᵀ uses for such systems (variables first, the dense
//! block of the rows last), without the symbolic analysis and with
//! contiguous kernels.
//!
//! The reduced KKT systems of bound elimination have this form when P is
//! diagonal, as for models whose quadratic terms are all on separate
//! variables (factor-model portfolios, least squares through auxiliary
//! variables).  Small pivots are replaced as in the other LDL solvers
//! (dynamic regularization); refinement on the full system follows.

use crate::algebra::*;

pub struct SchurLDL<T> {
    pub nr: usize,
    pub mr: usize,
    // the distinct entries of A by variable (CSC, rows ascending within a
    // column); entry e is the sum of values[vlist[vptr[e]..vptr[e+1]]] of
    // the caller's value array (duplicates of the input are summed)
    colptr: Vec<usize>,
    rowidx: Vec<usize>,
    vptr: Vec<usize>,
    vlist: Vec<usize>,
    // no duplicates: entry e is values[vlist[e]]
    simple: bool,
    // one more than the largest value index
    vend: usize,
    aval: Vec<T>,
    // whether aval (and Ad) hold the current values of A
    a_valid: bool,
    // dense mode (A at least a third full): A row major (mr × nr), and
    // A D⁻¹ while factoring
    dense: bool,
    // position of entry e in Ad
    doff: Vec<usize>,
    Ad: Vec<T>,
    Bd: Vec<T>,
    // D⁻¹
    dinv: Vec<T>,
    // Cholesky factor of S, lower triangle, row major (mr × mr)
    L: Vec<T>,
    reg_eps: T,
    reg_delta: T,
    pub regularize_count: usize,
    // smallest Cholesky pivot over the largest diagonal of S
    pivot_ratio: T,
    // scratch for the dense mode: two of length nr, one of length mr
    u: Vec<T>,
    w: Vec<T>,
    tz: Vec<T>,
}

impl<T: FloatT> SchurLDL<T> {
    /// `cols[j]` lists the (row, value index) pairs of column j of A.
    pub fn new(nr: usize, mr: usize, cols: &[Vec<(usize, usize)>], reg_eps: T, reg_delta: T) -> Self {
        assert!(cols.len() == nr);
        let mut colptr = Vec::with_capacity(nr + 1);
        let (mut rowidx, mut vptr, mut vlist) = (vec![], vec![0], vec![]);
        colptr.push(0);
        for c in cols {
            let mut c = c.clone();
            c.sort_unstable();
            for (q, &(r, k)) in c.iter().enumerate() {
                assert!(r < mr);
                if q == 0 || c[q - 1].0 != r {
                    if q > 0 {
                        vptr.push(vlist.len());
                    }
                    rowidx.push(r);
                }
                vlist.push(k);
            }
            if !c.is_empty() {
                vptr.push(vlist.len());
            }
            colptr.push(rowidx.len());
        }
        let nnz = rowidx.len();
        assert!(vptr.len() == nnz + 1);
        let dense = 3 * nnz >= nr * mr;
        let dlen = if dense { nr * mr } else { 0 };
        let mut doff = vec![];
        if dense {
            for j in 0..nr {
                doff.extend(rowidx[colptr[j]..colptr[j + 1]].iter().map(|&r| r * nr + j));
            }
        }
        let simple = vlist.len() == nnz;
        let vend = vlist.iter().max().map_or(0, |&k| k + 1);
        let wlen = if dense { nr } else { 0 };
        Self {
            nr,
            mr,
            colptr,
            rowidx,
            vptr,
            vlist,
            simple,
            vend,
            aval: vec![T::zero(); nnz],
            a_valid: false,
            dense,
            doff,
            Ad: vec![T::zero(); dlen],
            Bd: vec![T::zero(); dlen],
            dinv: vec![T::zero(); nr],
            L: vec![T::zero(); mr * mr],
            reg_eps,
            reg_delta,
            regularize_count: 0,
            pivot_ratio: T::zero(),
            u: vec![T::zero(); wlen],
            w: vec![T::zero(); wlen],
            tz: vec![T::zero(); mr],
        }
    }

    /// The smallest Cholesky pivot of the last factorization relative to
    /// the largest diagonal entry of S (zero if S is zero).
    pub fn min_pivot_ratio(&self) -> T {
        self.pivot_ratio
    }

    /// The values of A changed: gather them again at the next factor().
    pub fn invalidate_values(&mut self) {
        self.a_valid = false;
    }

    /// Whether A is held densely (and residual_sub can be used).
    pub fn is_dense(&self) -> bool {
        self.dense
    }

    /// Factor with diagonals `d` (nr) and `g` (mr), and A's values taken
    /// from `values` through the value indices given to new() (on the
    /// first call, and after invalidate_values()).  Returns false if the
    /// factors are not finite.
    pub fn factor(&mut self, d: &[T], g: &[T], values: &[T]) -> bool {
        let (nr, mr) = (self.nr, self.mr);
        assert!(d.len() == nr && g.len() == mr);
        if !self.a_valid {
            self.gather(values);
            self.a_valid = true;
        }
        self.factor_values(d, g)
    }

    fn gather(&mut self, values: &[T]) {
        assert!(self.vend <= values.len());
        // SAFETY: every value index is below vend (new()), and vptr has
        // aval.len() + 1 nondecreasing entries ending at vlist.len()
        unsafe {
            if self.simple {
                for (a, &k) in self.aval.iter_mut().zip(&self.vlist) {
                    *a = *values.get_unchecked(k);
                }
            } else {
                for (e, a) in self.aval.iter_mut().enumerate() {
                    let (p0, p1) = (*self.vptr.get_unchecked(e), *self.vptr.get_unchecked(e + 1));
                    *a = (p0..p1).fold(T::zero(), |s, p| s + *values.get_unchecked(*self.vlist.get_unchecked(p)));
                }
            }
        }
        if self.dense {
            // (positions without an entry stay zero from new())
            for (&o, &a) in self.doff.iter().zip(&self.aval) {
                self.Ad[o] = a;
            }
        }
    }

    fn factor_values(&mut self, d: &[T], g: &[T]) -> bool {
        let (nr, mr) = (self.nr, self.mr);
        self.regularize_count = 0;
        for (di, &dj) in self.dinv.iter_mut().zip(d) {
            let dj = if dj > self.reg_eps || dj.is_nan() {
                dj
            } else {
                self.regularize_count += 1;
                self.reg_delta
            };
            *di = T::recip(dj);
        }

        // S = G + A D⁻¹ A' (lower triangle)
        let L = &mut self.L;
        L.fill(T::zero());
        if self.dense {
            let (Ad, Bd, dinv) = (&self.Ad, &mut self.Bd, &self.dinv);
            for (a, b) in Ad.chunks_exact(nr).zip(Bd.chunks_exact_mut(nr)) {
                for (bj, (&aj, &dj)) in b.iter_mut().zip(a.iter().zip(dinv)) {
                    *bj = aj * dj;
                }
            }
            for r1 in 0..mr {
                let b1 = &Bd[r1 * nr..(r1 + 1) * nr];
                for r2 in 0..=r1 {
                    L[r1 * mr + r2] = kdot(b1, &Ad[r2 * nr..(r2 + 1) * nr]);
                }
            }
        } else {
            // one column of A at a time
            // SAFETY: rowidx < mr (checked in new()), colptr is
            // nondecreasing and ends at rowidx.len() == aval.len(), and
            // dinv.len() == nr
            unsafe {
                for j in 0..nr {
                    let (k0, k1) = (*self.colptr.get_unchecked(j), *self.colptr.get_unchecked(j + 1));
                    let dj = *self.dinv.get_unchecked(j);
                    for a in k0..k1 {
                        let ra = *self.rowidx.get_unchecked(a);
                        let va = *self.aval.get_unchecked(a) * dj;
                        let row = ra * mr;
                        for b in k0..=a {
                            let rb = *self.rowidx.get_unchecked(b);
                            let s = L.get_unchecked_mut(row + rb);
                            *s = fmadd(va, *self.aval.get_unchecked(b), *s);
                        }
                    }
                }
            }
        }
        for r in 0..mr {
            L[r * mr + r] += g[r];
        }

        // Cholesky, S = L L' (dot product form on contiguous rows)
        let smax = (0..mr).fold(T::zero(), |m, r| m.max(L[r * mr + r]));
        let mut pmin = T::infinity();
        for j in 0..mr {
            let (head, rest) = L.split_at_mut(j * mr);
            let rowj = &mut rest[..mr];
            for k in 0..j {
                let rowk = &head[k * mr..k * mr + k];
                let s = rowj[k] - kdot(&rowj[..k], rowk);
                rowj[k] = s / head[k * mr + k];
            }
            let mut p = rowj[j] - kdot(&rowj[..j], &rowj[..j]);
            pmin = pmin.min(p);
            if p <= self.reg_eps {
                self.regularize_count += 1;
                p = self.reg_delta;
            }
            rowj[j] = p.sqrt();
        }
        self.pivot_ratio = if smax > T::zero() && mr > 0 { pmin / smax } else { T::zero() };
        L.iter().all(|v| v.is_finite()) && self.dinv.iter().all(|v| v.is_finite())
    }

    // solve S z = t in place
    fn chol_solve(L: &[T], mr: usize, t: &mut [T]) {
        for i in 0..mr {
            let s = t[i] - kdot(&L[i * mr..i * mr + i], &t[..i]);
            t[i] = s / L[i * mr + i];
        }
        for i in (0..mr).rev() {
            let s = t[i] / L[i * mr + i];
            t[i] = s;
            kaxpy(-s, &L[i * mr..i * mr + i], &mut t[..i]);
        }
    }

    /// Solve in place on the first `nlanes` lanes of `v`, which holds
    /// [bx; bz] (length nr + mr); `t` is work of length mr.
    pub fn solve<const NR: usize>(&mut self, v: &mut [[T; NR]], t: &mut [[T; NR]], nlanes: usize) {
        let (nr, mr) = (self.nr, self.mr);
        assert!(v.len() == nr + mr && t.len() == mr && nlanes <= NR);
        let (vx, vz) = v.split_at_mut(nr);
        if self.dense {
            let (Ad, u, w, tz) = (&self.Ad, &mut self.u, &mut self.w, &mut self.tz);
            for l in 0..nlanes {
                // t = A D⁻¹ bx - bz
                for j in 0..nr {
                    u[j] = vx[j][l] * self.dinv[j];
                    w[j] = vx[j][l];
                }
                for r in 0..mr {
                    tz[r] = kdot(&Ad[r * nr..(r + 1) * nr], u) - vz[r][l];
                }
                Self::chol_solve(&self.L, mr, tz);
                // x = D⁻¹(bx - A'z)
                for r in 0..mr {
                    kaxpy(-tz[r], &Ad[r * nr..(r + 1) * nr], w);
                }
                for j in 0..nr {
                    vx[j][l] = w[j] * self.dinv[j];
                }
                for r in 0..mr {
                    vz[r][l] = tz[r];
                }
            }
            return;
        }

        // t = A D⁻¹ bx - bz
        for (ti, zi) in t.iter_mut().zip(vz.iter()) {
            for l in 0..NR {
                ti[l] = -zi[l];
            }
        }
        // SAFETY: as in factor()
        unsafe {
            for j in 0..nr {
                let dj = *self.dinv.get_unchecked(j);
                let mut u = *vx.get_unchecked(j);
                for l in 0..NR {
                    u[l] *= dj;
                }
                for k in *self.colptr.get_unchecked(j)..*self.colptr.get_unchecked(j + 1) {
                    let a = *self.aval.get_unchecked(k);
                    let tr = t.get_unchecked_mut(*self.rowidx.get_unchecked(k));
                    for l in 0..NR {
                        tr[l] = fmadd(a, u[l], tr[l]);
                    }
                }
            }
        }
        // S z = t: forward with L, back with L'
        let L = &self.L;
        for i in 0..mr {
            let row = &L[i * mr..i * mr + i];
            let mut s = t[i];
            for (k, &lik) in row.iter().enumerate() {
                for l in 0..NR {
                    s[l] = fmadd(-lik, t[k][l], s[l]);
                }
            }
            let d = L[i * mr + i];
            for l in 0..NR {
                t[i][l] = s[l] / d;
            }
        }
        for i in (0..mr).rev() {
            let d = L[i * mr + i];
            let mut s = t[i];
            for l in 0..NR {
                s[l] /= d;
            }
            t[i] = s;
            for k in 0..i {
                let lik = L[i * mr + k];
                for l in 0..NR {
                    t[k][l] = fmadd(-lik, s[l], t[k][l]);
                }
            }
        }
        // x = D⁻¹(bx - A'z)
        // SAFETY: as in factor()
        unsafe {
            for j in 0..nr {
                let mut s = *vx.get_unchecked(j);
                for k in *self.colptr.get_unchecked(j)..*self.colptr.get_unchecked(j + 1) {
                    let a = *self.aval.get_unchecked(k);
                    let zr = *t.get_unchecked(*self.rowidx.get_unchecked(k));
                    for l in 0..NR {
                        s[l] = fmadd(-a, zr[l], s[l]);
                    }
                }
                let dj = *self.dinv.get_unchecked(j);
                let x = vx.get_unchecked_mut(j);
                for l in 0..NR {
                    x[l] = dj * s[l];
                }
            }
        }
        vz.copy_from_slice(t);
    }

    /// Dense mode only: e -= K x on the first `nlanes` lanes, for
    /// K = [diag(kd_x) A'; A diag(kd_z)] with kd = [kd_x; kd_z] and A as
    /// of the last factorization.
    pub fn residual_sub<const NR: usize>(&mut self, e: &mut [[T; NR]], x: &[[T; NR]], kd: &[T], nlanes: usize) {
        let (nr, mr) = (self.nr, self.mr);
        assert!(self.dense && e.len() == nr + mr && x.len() == nr + mr && kd.len() == nr + mr);
        let (Ad, u, w, tz) = (&self.Ad, &mut self.u, &mut self.w, &mut self.tz);
        for l in 0..nlanes {
            for j in 0..nr {
                u[j] = x[j][l];
                w[j] = kd[j] * u[j];
            }
            for r in 0..mr {
                tz[r] = x[nr + r][l];
            }
            for r in 0..mr {
                let row = &Ad[r * nr..(r + 1) * nr];
                e[nr + r][l] -= kdot(row, u) + kd[nr + r] * tz[r];
                kaxpy(tz[r], row, w);
            }
            for j in 0..nr {
                e[j][l] -= w[j];
            }
        }
    }
}

// Dense kernels.  The library is built for the baseline target (SSE2 on
// x86-64), so for f64 these dispatch at run time to AVX2 + FMA versions
// where the CPU has them.

// x · y
#[inline]
fn kdot<T: FloatT>(x: &[T], y: &[T]) -> T {
    #[cfg(target_arch = "x86_64")]
    if let (Some(xf), Some(yf)) = (simd::as_f64(x), simd::as_f64(y)) {
        if simd::avx2_fma() {
            // SAFETY: the CPU supports the features the function is built
            // with; T is f64, so the value converts exactly
            let s = unsafe { simd::dot_f64(xf, yf) };
            return T::from_f64(s).unwrap();
        }
    }
    let n = x.len().min(y.len());
    let (x, y) = (&x[..n], &y[..n]);
    let mut acc = [T::zero(); 8];
    let (xc, yc) = (x.chunks_exact(8), y.chunks_exact(8));
    let (xr, yr) = (xc.remainder(), yc.remainder());
    for (a, b) in xc.zip(yc) {
        for k in 0..8 {
            acc[k] = a[k] * b[k] + acc[k];
        }
    }
    let mut s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (&a, &b) in xr.iter().zip(yr) {
        s = a * b + s;
    }
    s
}

// y += a x
#[inline]
fn kaxpy<T: FloatT>(a: T, x: &[T], y: &mut [T]) {
    #[cfg(target_arch = "x86_64")]
    if let (Some(xf), Some(af)) = (simd::as_f64(x), a.to_f64()) {
        if simd::avx2_fma() && simd::as_f64(y).is_some() {
            // SAFETY: as above; y has the same type as x (f64)
            let yf = unsafe { std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut f64, y.len()) };
            unsafe { simd::axpy_f64(af, xf, yf) };
            return;
        }
    }
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi = a * xi + *yi;
    }
}

#[cfg(target_arch = "x86_64")]
mod simd {
    use super::*;
    use std::any::TypeId;
    use std::arch::x86_64::*;

    #[inline]
    pub(super) fn avx2_fma() -> bool {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }

    // the slice as f64, if T is f64
    #[inline]
    pub(super) fn as_f64<T: FloatT>(x: &[T]) -> Option<&[f64]> {
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: T is f64
            Some(unsafe { std::slice::from_raw_parts(x.as_ptr() as *const f64, x.len()) })
        } else {
            None
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_f64(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len().min(y.len());
        let (px, py) = (x.as_ptr(), y.as_ptr());
        let mut acc = [_mm256_setzero_pd(); 4];
        let mut i = 0;
        // SAFETY (pointer reads): every index read is below n
        while i + 16 <= n {
            for (k, a) in acc.iter_mut().enumerate() {
                let j = i + 4 * k;
                *a = _mm256_fmadd_pd(_mm256_loadu_pd(px.add(j)), _mm256_loadu_pd(py.add(j)), *a);
            }
            i += 16;
        }
        while i + 4 <= n {
            acc[0] = _mm256_fmadd_pd(_mm256_loadu_pd(px.add(i)), _mm256_loadu_pd(py.add(i)), acc[0]);
            i += 4;
        }
        let v = _mm256_add_pd(_mm256_add_pd(acc[0], acc[1]), _mm256_add_pd(acc[2], acc[3]));
        let h = _mm_add_pd(_mm256_castpd256_pd128(v), _mm256_extractf128_pd(v, 1));
        let mut s = _mm_cvtsd_f64(_mm_add_sd(h, _mm_unpackhi_pd(h, h)));
        while i < n {
            s = (*px.add(i)).mul_add(*py.add(i), s);
            i += 1;
        }
        s
    }

    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn axpy_f64(a: f64, x: &[f64], y: &mut [f64]) {
        let n = x.len().min(y.len());
        let (px, py) = (x.as_ptr(), y.as_mut_ptr());
        let av = _mm256_set1_pd(a);
        let mut i = 0;
        // SAFETY (pointer accesses): every index is below n
        while i + 4 <= n {
            _mm256_storeu_pd(py.add(i), _mm256_fmadd_pd(av, _mm256_loadu_pd(px.add(i)), _mm256_loadu_pd(py.add(i))));
            i += 4;
        }
        while i < n {
            *py.add(i) = a.mul_add(*px.add(i), *py.add(i));
            i += 1;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_schur_exact() {
        // random D, G, A: compare with the residual of the full system
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut modes = [false; 2];
        for &(nr, mr, dens) in &[(1, 1, 0.6), (5, 3, 0.6), (30, 7, 0.6), (40, 1, 0.6), (3, 9, 0.6), (30, 7, 0.15), (60, 12, 0.1)] {
            let d: Vec<f64> = (0..nr).map(|_| 0.1 + rnd()).collect();
            let g: Vec<f64> = (0..mr).map(|i| if i % 2 == 0 { 0.0 } else { rnd() }).collect();
            let mut cols = vec![];
            let mut vals = vec![];
            for _ in 0..nr {
                let mut c = vec![];
                for r in 0..mr {
                    if rnd() < dens {
                        c.push((r, vals.len()));
                        vals.push(2.0 * rnd() - 1.0);
                        // a duplicate entry now and then (summed)
                        if rnd() < 0.1 {
                            c.push((r, vals.len()));
                            vals.push(2.0 * rnd() - 1.0);
                        }
                    }
                }
                cols.push(c);
            }
            let mut s = SchurLDL::new(nr, mr, &cols, 1e-13, 2e-7);
            modes[s.is_dense() as usize] = true;
            assert!(s.factor(&d, &g, &vals));
            let b: Vec<[f64; 2]> = (0..nr + mr).map(|_| [rnd() - 0.5, rnd() - 0.5]).collect();
            let mut v = b.clone();
            let mut t = vec![[0.0; 2]; mr];
            s.solve::<2>(&mut v, &mut t, 2);
            if s.regularize_count > 0 {
                continue; // a structurally singular S; not an exact solve
            }
            // residual of [D A'; A -G]
            for l in 0..2 {
                let mut e: Vec<f64> = b.iter().map(|x| x[l]).collect();
                for j in 0..nr {
                    e[j] -= d[j] * v[j][l];
                }
                for r in 0..mr {
                    e[nr + r] += g[r] * v[nr + r][l];
                }
                for (j, c) in cols.iter().enumerate() {
                    for &(r, k) in c {
                        e[j] -= vals[k] * v[nr + r][l];
                        e[nr + r] -= vals[k] * v[j][l];
                    }
                }
                let err = e.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                assert!(err < 1e-10, "nr {nr} mr {mr}: residual {err:e}");
            }
        }
        assert!(modes[0] && modes[1], "both the sparse and the dense mode");
    }
}
