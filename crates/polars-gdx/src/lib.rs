//! `polars_gdx._core` Python extension module.
//!
//! Two selectable backends (M7 task 042, docs/m7-adapter-contract.md §4):
//! `legacy` (vendored native GDX C library) and `gdxrs` (pinned pure-Rust
//! core). Both are compiled by default so differential tests can compare
//! them; a gdxrs-only build (`--no-default-features --features gdxrs`)
//! works without GAMS, C++ or a native GDX library. Backend selection is
//! explicit in the Python layer; a gdxrs failure is never retried through
//! the legacy backend.

mod arrow_stream;
#[cfg(feature = "gdxrs")]
pub mod gdxrs_backend;
#[cfg(feature = "legacy")]
pub mod legacy;

use pyo3::prelude::*;

/// Python entry point: `polars_gdx._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("HAS_LEGACY", cfg!(feature = "legacy"))?;
    m.add("HAS_GDXRS", cfg!(feature = "gdxrs"))?;
    m.add_function(wrap_pyfunction!(arrow_stream::_retain_arrow_stream, m)?)?;
    m.add_function(wrap_pyfunction!(arrow_stream::_disable_arrow_callbacks, m)?)?;
    m.add_class::<arrow_stream::RetainedArrowStream>()?;
    m.add_class::<arrow_stream::ArrowStreamGuard>()?;
    #[cfg(feature = "legacy")]
    m.add_class::<legacy::Reader>()?;
    #[cfg(feature = "gdxrs")]
    m.add_class::<gdxrs_backend::GdxrsReader>()?;
    #[cfg(feature = "gdxrs")]
    m.add_class::<gdxrs_backend::GdxrsBatchCursor>()?;
    #[cfg(feature = "gdxrs")]
    m.add_class::<gdxrs_backend::GdxrsArrowStream>()?;
    Ok(())
}
