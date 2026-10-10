#![allow(non_snake_case)]

use super::array_py::{extract_f64_vec, extract_usize_vec};
use crate::algebra::CscMatrix;
use core::ops::Deref;
use pyo3::prelude::*;

//We can't implement the foreign trait FromPyObject directly on CscMatrix
//since it is outside the crate, so put a dummy wrapper around it here.
#[pyclass]
pub struct PyCscMatrix(CscMatrix<f64>);

impl Deref for PyCscMatrix {
    type Target = CscMatrix<f64>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl From<PyCscMatrix> for CscMatrix<f64> {
    fn from(mat: PyCscMatrix) -> Self {
        mat.0
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyCscMatrix {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, pyo3::PyAny>) -> PyResult<Self> {
        let obj: &Bound<'py, pyo3::PyAny> = &obj;
        let nzval = extract_f64_vec(&obj.getattr("data")?)?;
        let rowval = extract_usize_vec(&obj.getattr("indices")?)?;
        let colptr = extract_usize_vec(&obj.getattr("indptr")?)?;
        let shape: Vec<usize> = obj.getattr("shape")?.extract()?;

        let mut mat = CscMatrix::new(shape[0], shape[1], colptr, rowval, nzval);

        // if the python object was not in standard format, force the rust
        // object to still be nicely formatted
        let is_canonical: bool = obj.getattr("has_canonical_format")?.extract()?;

        if !is_canonical {
            let _ = mat.canonicalize();
        }

        Ok(PyCscMatrix(mat))
    }
}
