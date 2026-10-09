//! Fast extraction of numeric vectors from Python objects.
//!
//! Extracting a `Vec<f64>` from a numpy array with pyo3's generic sequence
//! conversion visits every element as a Python object, which dominates the
//! cost of passing large problems in.  Here numpy converts the input to a
//! contiguous little-endian array once, and its bytes are copied in a single
//! pass.  This uses only the stable ABI (no buffer protocol, which abi3
//! lacks before Python 3.11).  Inputs that numpy cannot convert to a 1-D
//! array of the requested type fall back to the generic conversion, so the
//! accepted inputs and the errors raised are as before.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

// numpy.ascontiguousarray(obj, dtype=dtype).tobytes(), for 1-D arrays only
fn contiguous_bytes<'py>(obj: &Bound<'py, PyAny>, dtype: &str) -> Option<Bound<'py, PyBytes>> {
    let py = obj.py();
    let np = py.import("numpy").ok()?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("dtype", dtype).ok()?;
    let arr = np.call_method("ascontiguousarray", (obj,), Some(&kwargs)).ok()?;
    let ndim: usize = arr.getattr("ndim").ok()?.extract().ok()?;
    if ndim != 1 {
        return None;
    }
    arr.call_method0("tobytes").ok()?.downcast_into::<PyBytes>().ok()
}

/// A vector of f64 from any sequence or numpy array of numbers
pub(crate) fn extract_f64_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    match contiguous_bytes(obj, "<f8") {
        Some(bytes) => Ok(bytes
            .as_bytes()
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()),
        None => obj.extract(),
    }
}

/// A vector of usize from any sequence or numpy array of integers
pub(crate) fn extract_usize_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    // only integer inputs take the fast path: a float array would be
    // truncated by numpy, where the generic conversion raises an error
    let is_int = obj
        .getattr("dtype")
        .and_then(|d| d.getattr("kind"))
        .and_then(|k| k.extract::<String>())
        .map(|k| k == "i" || k == "u")
        .unwrap_or(false);
    if is_int {
        if let Some(bytes) = contiguous_bytes(obj, "<i8") {
            return bytes
                .as_bytes()
                .chunks_exact(8)
                .map(|c| {
                    usize::try_from(i64::from_le_bytes(c.try_into().unwrap())).map_err(|_| {
                        pyo3::exceptions::PyOverflowError::new_err("index out of range for usize")
                    })
                })
                .collect();
        }
    }
    obj.extract()
}

/// Wrapper type so that vector arguments can use the fast conversion
pub(crate) struct PyF64Vec(pub Vec<f64>);

impl<'a> FromPyObject<'a> for PyF64Vec {
    fn extract_bound(obj: &Bound<'a, PyAny>) -> PyResult<Self> {
        extract_f64_vec(obj).map(PyF64Vec)
    }
}
