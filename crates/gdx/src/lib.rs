//! Safe, idiomatic Rust wrapper for reading and writing GAMS GDX files.
//!
//! Backed by the vendored, MIT-licensed GAMS GDX C library (via [`gdx_sys`]);
//! requires no GAMS installation at runtime.
//!
//! ```no_run
//! use gdx::GdxFile;
//! let file = GdxFile::open("scenario.gdx")?;
//! for sym in file.symbols() {
//!     println!("{} ({}, dim {})", sym.name, sym.kind.as_str(), sym.dim);
//! }
//! let records = file.read("c")?;
//! # Ok::<(), gdx::GdxError>(())
//! ```

mod error;
mod lock;
mod reader;
mod types;
mod writer;

pub use error::{GdxError, Result};
pub use reader::{
    read_symbol_raw_parallel_pos, ActionPred, GdxFile, IndexPred, RawSymbolData, RecordAction,
};
pub use types::{Record, SymbolInfo, SymbolType, ValueField};
pub use writer::GdxWriter;
