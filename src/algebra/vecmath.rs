use super::{FloatT, ScalarMath, VectorMath};
use itertools::izip;
use std::borrow::Borrow;
use std::iter::zip;

impl<T: FloatT> VectorMath<T> for [T] {
    fn copy_from(&mut self, src: &[T]) -> &mut Self {
        self.copy_from_slice(src);
        self
    }

    fn select(&self, index: &[bool]) -> Vec<T> {
        assert_eq!(self.len(), index.len());
        zip(self, index)
            .filter(|(_x, &b)| b)
            .map(|(&x, _b)| x)
            .collect()
    }

    fn scalarop(&mut self, op: impl Fn(T) -> T) -> &mut Self {
        for x in &mut *self {
            *x = op(*x);
        }
        self
    }

    fn scalarop_from(&mut self, op: impl Fn(T) -> T, v: &[T]) -> &mut Self {
        for (x, v) in zip(&mut *self, v) {
            *x = op(*v);
        }
        self
    }

    fn translate(&mut self, c: T) -> &mut Self {
        //NB: translate is a scalar shift of all variables and is
        //used only in the NN cone to force vectors into R^n_+
        self.scalarop(|x| x + c)
    }

    fn set(&mut self, c: T) -> &mut Self {
        self.fill(c);
        self
    }

    fn scale(&mut self, c: T) -> &mut Self {
        self.scalarop(|x| x * c)
    }

    fn recip(&mut self) -> &mut Self {
        self.scalarop(T::recip)
    }

    fn sqrt(&mut self) -> &mut Self {
        self.scalarop(T::sqrt)
    }

    fn rsqrt(&mut self) -> &mut Self {
        self.scalarop(|x| T::recip(T::sqrt(x)))
    }

    fn negate(&mut self) -> &mut Self {
        self.scalarop(|x| -x)
    }

    fn hadamard(&mut self, y: &[T]) -> &mut Self {
        zip(&mut *self, y).for_each(|(x, y)| *x *= *y);
        self
    }

    fn clip(&mut self, min_thresh: T, max_thresh: T) -> &mut Self {
        self.scalarop(|x| x.clip(min_thresh, max_thresh))
    }

    fn normalize(&mut self) -> T {
        let norm = self.norm();
        if norm.is_zero() {
            return T::zero();
        }
        self.scale(norm.recip());
        norm
    }

    fn dot(&self, y: &[T]) -> T {
        let n = self.len().min(y.len());
        let (x, y) = (&self[..n], &y[..n]);
        // SAFETY: i < n for both slices
        sum8(n, |i| unsafe { *x.get_unchecked(i) * *y.get_unchecked(i) })
    }

    fn dot_shifted(z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        assert_eq!(z.len(), s.len());
        assert_eq!(z.len(), dz.len());
        assert_eq!(s.len(), ds.len());

        // SAFETY: all lengths are asserted equal above
        sum8(z.len(), |i| unsafe {
            let si = *s.get_unchecked(i) + α * *ds.get_unchecked(i);
            let zi = *z.get_unchecked(i) + α * *dz.get_unchecked(i);
            si * zi
        })
    }

    fn dist(&self, y: &Self) -> T {
        let dist2 = zip(self, y).fold(T::zero(), |acc, (&x, &y)| acc + T::powi(x - y, 2));
        T::sqrt(dist2)
    }

    fn sum(&self) -> T {
        // SAFETY: i < len
        sum8(self.len(), |i| unsafe { *self.get_unchecked(i) })
    }

    fn sumsq(&self) -> T {
        self.dot(self)
    }

    // 2-norm
    fn norm(&self) -> T {
        // fast path: plain sum of squares when no over/underflow is possible
        // SAFETY: i < len
        let (amax, sumsq) = max_sumsq8(self.len(), |i| unsafe { *self.get_unchecked(i) });
        let lo = T::min_positive_value().sqrt() / T::epsilon();
        if amax > lo && sumsq.is_finite() {
            return sumsq.sqrt();
        }
        if sumsq.is_nan() {
            return T::nan();
        }
        if amax == T::zero() {
            return T::zero();
        }
        stable_norm(self.iter())
    }

    //2-norm of elementwise product self.*v
    fn norm_scaled(&self, v: &[T]) -> T {
        assert_eq!(self.len(), v.len());
        // fast path: plain sum of squares when no over/underflow is possible
        // SAFETY: lengths asserted equal
        let (amax, sumsq) = max_sumsq8(self.len(), |i| unsafe { *self.get_unchecked(i) * *v.get_unchecked(i) });
        let lo = T::min_positive_value().sqrt() / T::epsilon();
        if amax > lo && sumsq.is_finite() {
            return sumsq.sqrt();
        }
        if sumsq.is_nan() {
            return T::nan();
        }
        if amax == T::zero() {
            return T::zero();
        }
        stable_norm(izip!(self, v).map(|(&yi, &vi)| yi * vi))
    }

    // 2-norm of (self + α.dz)
    fn norm_shifted(&self, dz: &[T], α: T) -> T {
        stable_norm(izip!(self, dz).map(|(&zi, &dzi)| zi + α * dzi))
    }

    // Returns infinity norm
    fn norm_inf(&self) -> T {
        // SAFETY: i < len
        let out = max8(self.len(), |i| unsafe { self.get_unchecked(i).abs() });
        // (x - x) is NaN iff x is NaN or infinite, so this probe flags
        // the need for an exact NaN check without a branchy main loop
        let probe = sum8(self.len(), |i| unsafe {
            *self.get_unchecked(i) - *self.get_unchecked(i)
        });
        if probe.is_nan() && self.iter().any(|v| v.is_nan()) {
            return T::nan();
        }
        out
    }

    // Returns one norm
    fn norm_one(&self) -> T {
        // SAFETY: i < len
        sum8(self.len(), |i| unsafe { self.get_unchecked(i).abs() })
    }

    //inf-norm of elementwise product self.*v
    fn norm_inf_scaled(&self, v: &Self) -> T {
        assert_eq!(self.len(), v.len());
        // SAFETY: lengths asserted equal
        max8(self.len(), |i| unsafe { T::abs(*self.get_unchecked(i) * *v.get_unchecked(i)) })
    }

    //
    fn norm_one_scaled(&self, v: &Self) -> T {
        let n = self.len().min(v.len());
        // SAFETY: i < n for both slices
        sum8(n, |i| unsafe { T::abs(*self.get_unchecked(i) * *v.get_unchecked(i)) })
    }

    // max absolute difference (used for unit testing)
    fn norm_inf_diff(&self, b: &[T]) -> T {
        zip(self, b).fold(T::zero(), |acc, (x, y)| T::max(acc, T::abs(*x - *y)))
    }

    fn minimum(&self) -> T {
        // SAFETY: i < len
        -max8_from(self.len(), -T::infinity(), |i| unsafe { -*self.get_unchecked(i) })
    }

    fn maximum(&self) -> T {
        // SAFETY: i < len
        max8_from(self.len(), -T::infinity(), |i| unsafe { *self.get_unchecked(i) })
    }

    fn mean(&self) -> T {
        let mean = if self.is_empty() {
            T::zero()
        } else {
            let num = self.sum();
            let den = T::from_usize(self.len()).unwrap();
            num / den
        };
        mean
    }

    fn is_finite(&self) -> bool {
        self.iter().all(|&x| T::is_finite(x))
    }

    fn axpby(&mut self, a: T, x: &[T], b: T) -> &mut Self {
        assert_eq!(self.len(), x.len());

        zip(&mut *self, x).for_each(|(y, x)| *y = a * (*x) + b * (*y));
        self
    }

    fn waxpby(&mut self, a: T, x: &[T], b: T, y: &[T]) -> &mut Self {
        assert_eq!(self.len(), x.len());
        assert_eq!(self.len(), y.len());

        for (w, (x, y)) in zip(&mut *self, zip(x, y)) {
            *w = a * (*x) + b * (*y);
        }
        self
    }
}

// numerically more stable 2-norm that avoids overflow/underflow
fn stable_norm<T, I, B>(x: I) -> T
where
    T: FloatT,
    I: Iterator<Item = B>,
    B: Borrow<T>,
{
    let (scale, sumsq) =
        x.filter(|b| *b.borrow() != T::zero())
            .fold((T::zero(), T::one()), |(scale, sumsq), b| {
                let xi = *b.borrow();
                let absxi = xi.abs();
                if scale < absxi {
                    let r = scale / absxi;
                    (absxi, T::one() + sumsq * r * r)
                } else {
                    let r = absxi / scale;
                    (scale, sumsq + r * r)
                }
            });
    scale * sumsq.sqrt()
}

// Reductions with 8 independent accumulators.  A single sequential
// accumulator makes every reduction latency bound (one dependent add per
// element); independent lanes let the compiler vectorize the loop.

#[inline(always)]
fn sum8<T: FloatT>(n: usize, f: impl Fn(usize) -> T) -> T {
    let mut acc = [T::zero(); 8];
    let n8 = n - n % 8;
    let mut i = 0;
    while i < n8 {
        for (k, a) in acc.iter_mut().enumerate() {
            *a += f(i + k);
        }
        i += 8;
    }
    let mut out = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for j in n8..n {
        out += f(j);
    }
    out
}

// max over f(i) and the initial value.  NaN values are ignored, as with
// T::max, so callers needing NaN propagation must check separately.
#[inline(always)]
fn max8_from<T: FloatT>(n: usize, init: T, f: impl Fn(usize) -> T) -> T {
    let mut acc = [init; 8];
    let n8 = n - n % 8;
    let mut i = 0;
    while i < n8 {
        for (k, a) in acc.iter_mut().enumerate() {
            let v = f(i + k);
            *a = if v > *a { v } else { *a };
        }
        i += 8;
    }
    let mut out = init;
    for a in acc {
        out = if a > out { a } else { out };
    }
    for j in n8..n {
        let v = f(j);
        out = if v > out { v } else { out };
    }
    out
}

#[inline(always)]
fn max8<T: FloatT>(n: usize, f: impl Fn(usize) -> T) -> T {
    max8_from(n, T::zero(), f)
}

// max |f(i)| and sum f(i)² in a single pass, 8 lanes
#[inline(always)]
fn max_sumsq8<T: FloatT>(n: usize, f: impl Fn(usize) -> T) -> (T, T) {
    let mut mx = [T::zero(); 8];
    let mut sq = [T::zero(); 8];
    let n8 = n - n % 8;
    let mut i = 0;
    while i < n8 {
        for k in 0..8 {
            let t = f(i + k);
            let a = t.abs();
            mx[k] = if a > mx[k] { a } else { mx[k] };
            sq[k] += t * t;
        }
        i += 8;
    }
    let mut m = T::zero();
    for a in mx {
        m = if a > m { a } else { m };
    }
    let mut s = ((sq[0] + sq[4]) + (sq[1] + sq[5])) + ((sq[2] + sq[6]) + (sq[3] + sq[7]));
    for j in n8..n {
        let t = f(j);
        let a = t.abs();
        m = if a > m { a } else { m };
        s += t * t;
    }
    (m, s)
}
