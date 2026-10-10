use crate::algebra::*;
use std::iter::zip;

impl<T: FloatT> MatrixVectorMultiply<T> for CscMatrix<T> {
    fn gemv(&self, y: &mut [T], x: &[T], a: T, b: T) {
        _csc_axpby_N(self, y, x, a, b);
    }
}

impl<T: FloatT> MatrixVectorMultiply<T> for Adjoint<'_, CscMatrix<T>> {
    fn gemv(&self, y: &mut [T], x: &[T], a: T, b: T) {
        _csc_axpby_T(self.src, y, x, a, b);
    }
}

impl<T: FloatT> SymMatrixMath<T> for Symmetric<'_, CscMatrix<T>> {
    fn symv(&self, y: &mut [T], x: &[T], a: T, b: T) {
        _csc_symv_unsafe(self.src, self.uplo, y, x, a, b)
    }

    fn quad_form(&self, y: &[T], x: &[T]) -> T {
        _csc_quad_form(self.src, self.uplo, y, x)
    }
}

impl<T: FloatT> MatrixMath<T> for CscMatrix<T> {
    fn col_sums(&self, sums: &mut [T]) {
        assert_eq!(self.n, sums.len());
        for (col, sum) in sums.iter_mut().enumerate() {
            let rng = self.colptr[col]..self.colptr[col + 1];
            *sum = self.nzval[rng].sum();
        }
    }

    fn row_sums(&self, sums: &mut [T]) {
        assert_eq!(self.m, sums.len());
        sums.fill(T::zero());
        for (&row, &val) in zip(&self.rowval, &self.nzval) {
            sums[row] += val;
        }
    }

    fn col_norms(&self, norms: &mut [T]) {
        norms.fill(T::zero());
        self.col_norms_no_reset(norms);
    }

    fn col_norms_no_reset(&self, norms: &mut [T]) {
        assert_eq!(norms.len(), self.colptr.len() - 1);

        for (i, v) in norms.iter_mut().enumerate() {
            let mut m = *v;
            for &nzval in &self.nzval[self.colptr[i]..self.colptr[i + 1]] {
                m = max_ignore_nan(m, T::abs(nzval));
            }
            *v = m;
        }
    }

    fn col_norms_sym(&self, norms: &mut [T]) {
        norms.fill(T::zero());
        self.col_norms_sym_no_reset(norms);
    }

    fn col_norms_sym_no_reset(&self, norms: &mut [T]) {
        assert_eq!(norms.len(), self.colptr.len() - 1);

        let n = norms.len();
        for i in 0..n {
            let (first, last) = (self.colptr[i], self.colptr[i + 1]);
            let mut m = norms[i];
            for (&r, &v) in zip(&self.rowval[first..last], &self.nzval[first..last]) {
                let tmp = T::abs(v);
                m = max_ignore_nan(m, tmp);
                if r != i {
                    norms[r] = max_ignore_nan(norms[r], tmp);
                }
            }
            norms[i] = m;
        }
    }

    fn row_norms(&self, norms: &mut [T]) {
        norms.fill(T::zero());
        self.row_norms_no_reset(norms);
    }

    fn row_norms_no_reset(&self, norms: &mut [T]) {
        assert_eq!(self.rowval.len(), *self.colptr.last().unwrap());

        let m = norms.len();
        for (&row, val) in zip(&self.rowval, &self.nzval) {
            assert!(row < m);
            // SAFETY: row < m checked above
            let nr = unsafe { norms.get_unchecked_mut(row) };
            *nr = max_ignore_nan(*nr, T::abs(*val));
        }
    }
}

impl<T: FloatT> MatrixMathMut<T> for CscMatrix<T> {
    //scalar mut operations
    fn scale(&mut self, c: T) {
        self.nzval.scale(c);
    }

    fn negate(&mut self) {
        self.nzval.negate();
    }

    fn lscale(&mut self, l: &[T]) {
        let m = l.len();
        for (val, &row) in zip(&mut self.nzval, &self.rowval) {
            assert!(row < m);
            // SAFETY: row < m checked above
            *val *= unsafe { *l.get_unchecked(row) };
        }
    }

    fn rscale(&mut self, r: &[T]) {
        let colptr = &self.colptr;
        let vals = &mut self.nzval;

        assert_eq!(vals.len(), *colptr.last().unwrap());
        for i in 0..self.n {
            vals[colptr[i]..colptr[i + 1]].scale(r[i]);
        }
    }

    fn lrscale(&mut self, l: &[T], r: &[T]) {
        assert_eq!(self.nzval.len(), *self.colptr.last().unwrap());

        for (col, &ri) in r.iter().enumerate() {
            let (first, last) = (self.colptr[col], self.colptr[col + 1]);
            let vals = &mut self.nzval[first..last];
            let rows = &self.rowval[first..last];

            let m = l.len();
            for (val, &row) in zip(vals, rows) {
                assert!(row < m);
                // SAFETY: row < m checked above
                *val *= unsafe { *l.get_unchecked(row) } * ri;
            }
        }
    }
}

#[allow(non_snake_case)]
#[allow(dead_code)]
fn _csc_symv_safe<T: FloatT>(
    A: &CscMatrix<T>,
    _uplo: MatrixTriangle,
    y: &mut [T],
    x: &[T],
    a: T,
    b: T,
) {
    // NB: this function works identically regardless of uplo.  Implemented
    // this for for consistency with _csc_quad_form interface

    y.scale(b);

    assert!(x.len() == A.n);
    assert!(y.len() == A.n);
    assert!(A.n == A.m);

    for (col, &xcol) in x.iter().enumerate() {
        let first = A.colptr[col];
        let last = A.colptr[col + 1];
        let rows = &A.rowval[first..last];
        let nzvals = &A.nzval[first..last];

        for (&row, &Aij) in zip(rows, nzvals) {
            y[row] += a * Aij * xcol;

            if row != col {
                //don't double up on the diagonal
                y[col] += a * Aij * x[row];
            }
        }
    }
}

#[allow(non_snake_case)]
fn _csc_symv_unsafe<T: FloatT>(
    A: &CscMatrix<T>,
    _uplo: MatrixTriangle,
    y: &mut [T],
    x: &[T],
    a: T,
    b: T,
) {
    // NB: this function works identically regardless of uplo.  Implemented
    // this for for consistency with _csc_quad_form interface

    y.scale(b);

    assert!(x.len() == A.n);
    assert!(y.len() == A.n);
    assert!(A.n == A.m);
    assert!(A.colptr.len() == A.n + 1);
    assert!(A.rowval.len() >= A.colptr[A.n] && A.nzval.len() >= A.colptr[A.n]);
    // The column's contribution through the transpose is accumulated in a
    // register rather than through memory, and the diagonal is masked
    // branch-free so it is not counted twice.
    unsafe {
        for col in 0..A.n {
            let first = *A.colptr.get_unchecked(col);
            let last = *A.colptr.get_unchecked(col + 1);
            let xcol = a * *x.get_unchecked(col);
            let mut acc = T::zero();
            for k in first..last {
                let row = *A.rowval.get_unchecked(k);
                let Aij = *A.nzval.get_unchecked(k);
                debug_assert!(row < A.n);
                *y.get_unchecked_mut(row) += Aij * xcol;
                let t = Aij * *x.get_unchecked(row);
                acc += if row != col { t } else { T::zero() };
            }
            *y.get_unchecked_mut(col) += a * acc;
        }
    }
}

#[allow(non_snake_case)]
#[allow(clippy::comparison_chain)]
fn _csc_quad_form<T: FloatT>(M: &CscMatrix<T>, uplo: MatrixTriangle, y: &[T], x: &[T]) -> T {
    assert_eq!(M.n, M.m);
    assert_eq!(x.len(), M.n);
    assert_eq!(y.len(), M.n);
    assert!(M.colptr.len() == M.n + 1);
    assert!(M.nzval.len() == M.rowval.len());

    if M.n == 0 {
        return T::zero();
    }

    let mut out = T::zero();

    let cmp = match uplo {
        MatrixTriangle::Triu => usize::lt,
        MatrixTriangle::Tril => usize::gt,
    };

    for col in 0..M.n {
        //column number

        let mut tmp1 = T::zero();
        let mut tmp2 = T::zero();

        //start / stop indices for the current column
        let first = M.colptr[col];
        let last = M.colptr[col + 1];

        let values = &M.nzval[first..last];
        let rows = &M.rowval[first..last];

        for (&Mv, &row) in zip(values, rows) {
            if cmp(&row, &col) {
                //triangular terms only
                tmp1 += Mv * x[row];
                tmp2 += Mv * y[row];
            } else if row == col {
                out += Mv * x[col] * y[col];
            } else {
                panic!("Input matrix should be in triangular form.");
            }
        }
        out += tmp1 * y[col] + tmp2 * x[col];
    }
    out
}

// sparse matrix-vector multiply, no transpose
#[allow(non_snake_case)]
fn _csc_axpby_N<T: FloatT>(A: &CscMatrix<T>, y: &mut [T], x: &[T], a: T, b: T) {
    //first do the b*y part
    if b.is_zero() {
        y.fill(T::zero());
    } else if b == T::one() {
    } else if b == -T::one() {
        y.negate();
    } else {
        y.scale(b);
    }

    // if a is zero, we're done
    if a.is_zero() {
        return;
    }

    assert_eq!(A.nzval.len(), *A.colptr.last().unwrap());
    assert_eq!(x.len(), A.n);

    //y += A*x
    assert_eq!(y.len(), A.m);
    for (j, &xj) in x.iter().enumerate().take(A.n) {
        let xj = a * xj;
        let (first, last) = (A.colptr[j], A.colptr[j + 1]);
        for (&row, &v) in zip(&A.rowval[first..last], &A.nzval[first..last]) {
            y[row] += v * xj;
        }
    }
}

// sparse matrix-vector multiply, transposed
#[allow(non_snake_case)]
fn _csc_axpby_T<T: FloatT>(A: &CscMatrix<T>, y: &mut [T], x: &[T], a: T, b: T) {
    //first do the b*y part
    if b.is_zero() {
        y.fill(T::zero());
    } else if b == T::one() {
    } else if b == -T::one() {
        y.negate();
    } else {
        y.scale(b);
    }

    // if a is zero, we're done
    if a.is_zero() {
        return;
    }

    assert_eq!(A.nzval.len(), *A.colptr.last().unwrap());
    assert_eq!(x.len(), A.m);

    //y += A'*x
    assert_eq!(y.len(), A.n);
    for (j, yj) in y.iter_mut().enumerate() {
        let (first, last) = (A.colptr[j], A.colptr[j + 1]);
        let mut acc = T::zero();
        for (&row, &v) in zip(&A.rowval[first..last], &A.nzval[first..last]) {
            acc += v * x[row];
        }
        *yj += a * acc;
    }
}

#[test]
fn test_symv_safe_and_unsafe() {
    let Aup = CscMatrix::from(&[
        [4.0, -3.0, 7.0, 0.0],
        [0.0, 8.0, -1.0, 0.0],
        [0.0, 0.0, 2.0, -3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);

    let Alo: CscMatrix = Aup.t().into();

    let x = vec![1., 2., -3., -4.];
    let a = -2.;
    let b = 3.;

    let mut y = vec![0., 1., -1., 2.];
    _csc_symv_unsafe(&Aup, MatrixTriangle::Triu, &mut y, &x, a, b);
    assert_eq!(y, vec![46.0, -29.0, -25.0, -4.0]);

    let mut y = vec![0., 1., -1., 2.];
    _csc_symv_unsafe(&Alo, MatrixTriangle::Tril, &mut y, &x, a, b);
    assert_eq!(y, vec![46.0, -29.0, -25.0, -4.0]);

    let mut y = vec![0., 1., -1., 2.];
    _csc_symv_safe(&Aup, MatrixTriangle::Triu, &mut y, &x, a, b);
    assert_eq!(y, vec![46.0, -29.0, -25.0, -4.0]);

    let mut y = vec![0., 1., -1., 2.];
    _csc_symv_safe(&Alo, MatrixTriangle::Tril, &mut y, &x, a, b);
    assert_eq!(y, vec![46.0, -29.0, -25.0, -4.0]);
}

// max(m, v) for an accumulator m that is never NaN: as T::max, a NaN v is
// ignored, but this compiles to a single compare-select
#[inline(always)]
fn max_ignore_nan<T: FloatT>(m: T, v: T) -> T {
    if v > m {
        v
    } else {
        m
    }
}
