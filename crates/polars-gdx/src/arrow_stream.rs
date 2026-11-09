//! Retain a Python execution lease for an exported Arrow stream's lifetime.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use std::ffi::{c_char, c_int};

use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow::ffi_stream::FFI_ArrowArrayStream;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

struct CallbackState {
    disabled: bool,
    active: usize,
}

static CALLBACKS: Mutex<CallbackState> = Mutex::new(CallbackState {
    disabled: false,
    active: 0,
});
static CALLBACKS_FINISHED: Condvar = Condvar::new();

struct CallbackRegistration;

impl CallbackRegistration {
    fn enter() -> Option<Self> {
        let mut state = CALLBACKS.lock().unwrap_or_else(|error| error.into_inner());
        if state.disabled {
            return None;
        }
        state.active += 1;
        Some(Self)
    }
}

impl Drop for CallbackRegistration {
    fn drop(&mut self) {
        let mut state = CALLBACKS.lock().unwrap_or_else(|error| error.into_inner());
        state.active -= 1;
        CALLBACKS_FINISHED.notify_all();
    }
}

struct StreamLease {
    owner: Py<PyAny>,
    disarmed: Arc<AtomicBool>,
}

impl Drop for StreamLease {
    fn drop(&mut self) {
        // Release may run on a foreign thread without the GIL. Executions
        // disarm their guards when closing, and the final atexit hook forbids
        // any remaining callback from attaching to a finalizing interpreter.
        if !self.disarmed.swap(true, Ordering::AcqRel) {
            let Some(_registration) = CallbackRegistration::enter() else {
                return;
            };
            Python::with_gil(|py| {
                if let Err(error) = self.owner.call_method0(py, "close") {
                    // Arrow's release callback returns void. Explicit close
                    // is fallible; release failures use Python's error hook.
                    error.write_unraisable(py, Some(self.owner.bind(py)));
                }
            });
        }
    }
}

struct StreamState {
    // Field order releases the upstream stream before closing its execution.
    inner: FFI_ArrowArrayStream,
    _lease: StreamLease,
}

// These callbacks forward the C ABI without importing/re-exporting schemas or
// arrays. Arrow 56's schema importer discards ordered-dictionary flags; opaque
// forwarding also preserves extension metadata and future Arrow formats.
//
// Safety invariant: private_data points to our uniquely owned, boxed StreamState
// until release. The Arrow C Stream protocol forbids concurrent calls or use
// after release. Output pointers have the producer's usual ABI requirements.
unsafe extern "C" fn get_schema(
    stream: *mut FFI_ArrowArrayStream,
    out: *mut FFI_ArrowSchema,
) -> c_int {
    // SAFETY: the live exported stream owns this StreamState for this call.
    let inner = unsafe { &mut (*((*stream).private_data.cast::<StreamState>())).inner };
    match inner.get_schema {
        // SAFETY: forwarding the caller's output pointer to the live producer.
        Some(callback) => unsafe { callback(inner, out) },
        None => 22, // EINVAL: upstream stream was released.
    }
}

unsafe extern "C" fn get_next(
    stream: *mut FFI_ArrowArrayStream,
    out: *mut FFI_ArrowArray,
) -> c_int {
    // SAFETY: the live exported stream owns this StreamState for this call.
    let inner = unsafe { &mut (*((*stream).private_data.cast::<StreamState>())).inner };
    match inner.get_next {
        // SAFETY: forwarding the caller's output pointer to the live producer.
        Some(callback) => unsafe { callback(inner, out) },
        None => 22,
    }
}

unsafe extern "C" fn get_last_error(stream: *mut FFI_ArrowArrayStream) -> *const c_char {
    // SAFETY: the live exported stream owns this StreamState for this call.
    let inner = unsafe { &mut (*((*stream).private_data.cast::<StreamState>())).inner };
    match inner.get_last_error {
        // SAFETY: the producer's error string stays owned by the upstream stream.
        Some(callback) => unsafe { callback(inner) },
        None => std::ptr::null(),
    }
}

unsafe extern "C" fn release(stream: *mut FFI_ArrowArrayStream) {
    if stream.is_null() {
        return;
    }
    // SAFETY: the consumer passes the live, uniquely owned exported C stream.
    let stream = unsafe { &mut *stream };
    let state = stream.private_data.cast::<StreamState>();
    stream.release = None;
    stream.get_schema = None;
    stream.get_next = None;
    stream.get_last_error = None;
    stream.private_data = std::ptr::null_mut();
    if !state.is_null() {
        // SAFETY: private_data came from Box::into_raw exactly once below;
        // clearing the header before destruction makes release idempotent.
        drop(unsafe { Box::from_raw(state) });
    }
}

#[pyclass(name = "_ArrowStreamGuard")]
pub struct ArrowStreamGuard(Arc<AtomicBool>);

#[pymethods]
impl ArrowStreamGuard {
    fn disarm(&self) {
        self.0.store(true, Ordering::Release);
    }
}

// Arrow already implements Send for the FFI stream. A mutex permits the Python
// holder to be shared while preserving the protocol's single ownership transfer.
#[pyclass(name = "_RetainedArrowStream")]
pub struct RetainedArrowStream(Mutex<Option<FFI_ArrowArrayStream>>);

#[pymethods]
impl RetainedArrowStream {
    #[pyo3(signature = (requested_schema=None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        if requested_schema.is_some() {
            return Err(PyValueError::new_err(
                "the upstream Arrow stream has already negotiated its schema",
            ));
        }
        let stream = self
            .0
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Arrow stream state poisoned"))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Arrow stream already consumed"))?;
        // Export the repr(C) FFI value itself, with no wrapper-layout cast.
        PyCapsule::new_with_destructor(
            py,
            stream,
            Some(std::ffi::CString::new("arrow_array_stream").expect("valid capsule name")),
            |_stream: FFI_ArrowArrayStream, _context| {},
        )
    }
}

#[pyfunction]
pub fn _retain_arrow_stream(
    source: &Bound<'_, PyCapsule>,
    owner: Py<PyAny>,
) -> PyResult<(RetainedArrowStream, ArrowStreamGuard)> {
    if !source.is_valid() || source.name()? != Some(c"arrow_array_stream") {
        return Err(PyValueError::new_err(
            "expected an arrow_array_stream capsule",
        ));
    }
    // SAFETY: a validated Arrow PyCapsule carries a well-formed, aligned C
    // stream. from_raw moves the value and empties the capsule's payload,
    // preventing its destructor from releasing the transferred stream twice.
    let inner = unsafe { FFI_ArrowArrayStream::from_raw(source.pointer().cast()) };
    if inner.release.is_none() || inner.get_schema.is_none() || inner.get_next.is_none() {
        return Err(PyValueError::new_err(
            "Arrow stream is released or incomplete",
        ));
    }
    let disarmed = Arc::new(AtomicBool::new(false));
    let state = Box::new(StreamState {
        inner,
        _lease: StreamLease {
            owner,
            disarmed: Arc::clone(&disarmed),
        },
    });
    let stream = FFI_ArrowArrayStream {
        get_schema: Some(get_schema),
        get_next: Some(get_next),
        get_last_error: Some(get_last_error),
        release: Some(release),
        private_data: Box::into_raw(state).cast(),
    };
    Ok((
        RetainedArrowStream(Mutex::new(Some(stream))),
        ArrowStreamGuard(disarmed),
    ))
}

#[pyfunction]
pub fn _disable_arrow_callbacks(py: Python<'_>) {
    {
        let mut state = CALLBACKS.lock().unwrap_or_else(|error| error.into_inner());
        state.disabled = true;
    }
    // A callback may already be waiting to acquire the GIL. Stop admission
    // atomically, then release the GIL until every admitted callback finishes.
    py.allow_threads(|| {
        let mut state = CALLBACKS.lock().unwrap_or_else(|error| error.into_inner());
        while state.active != 0 {
            state = CALLBACKS_FINISHED
                .wait(state)
                .unwrap_or_else(|error| error.into_inner());
        }
    });
}
