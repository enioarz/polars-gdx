//! Raw FFI bindings to the GAMS GDX C library (`libgdxcclib64`).
//!
//! These bind directly to the library's exported entry points: the
//! object-management functions (`gdxcreate`/`gdxfree`) and the `c__gdx*`
//! "explicit object" wrappers, which take the GDX object pointer as their
//! first argument. This avoids the dynamic-loading wrapper (`gdxcc.c`) and its
//! global function-pointer table entirely.
//!
//! Symbol names are the lowercase forms exported on Linux/macOS (the upstream
//! header lowercases them via macros for the no-leading-underscore convention).
//!
//! Everything here is `unsafe`; see the `gdx` crate for the safe wrapper.
#![allow(non_snake_case)]

use std::os::raw::{c_char, c_double, c_int, c_void};

/// Short-string buffer size used throughout the GDX API.
pub const GMS_SSSIZE: usize = 256;
/// Maximum number of index dimensions for a symbol.
pub const GMS_MAX_INDEX_DIM: usize = 20;
/// Number of value fields per record (level, marginal, lower, upper, scale).
pub const GMS_VAL_MAX: usize = 5;

pub const GMS_VAL_LEVEL: usize = 0;
pub const GMS_VAL_MARGINAL: usize = 1;
pub const GMS_VAL_LOWER: usize = 2;
pub const GMS_VAL_UPPER: usize = 3;
pub const GMS_VAL_SCALE: usize = 4;

pub const GMS_DT_SET: c_int = 0;
pub const GMS_DT_PAR: c_int = 1;
pub const GMS_DT_VAR: c_int = 2;
pub const GMS_DT_EQU: c_int = 3;
pub const GMS_DT_ALIAS: c_int = 4;

/// Length of the special-value array returned by [`c__gdxgetspecialvalues`].
pub const GMS_SVIDX_MAX: usize = 7;
pub const GMS_SVIDX_UNDEF: usize = 0;
pub const GMS_SVIDX_NA: usize = 1;
pub const GMS_SVIDX_PINF: usize = 2;
pub const GMS_SVIDX_MINF: usize = 3;
pub const GMS_SVIDX_EPS: usize = 4;

/// Opaque GDX object handle (`TGXFileRec_t`).
pub type GdxObj = c_void;

extern "C" {
    /// Create a GDX object. Returns 1 on success and writes a message into `msg`.
    pub fn gdxcreate(pobj: *mut *mut GdxObj, msg: *mut c_char, msglen: c_int) -> c_int;
    /// Destroy a GDX object created by [`gdxcreate`].
    pub fn gdxfree(pobj: *mut *mut GdxObj) -> c_int;

    /// Open a GDX file for reading. Returns 1 on success; otherwise `errnr` is set.
    pub fn c__gdxopenread(obj: *mut GdxObj, filename: *const c_char, errnr: *mut c_int) -> c_int;
    /// Open a GDX file for writing. Returns 1 on success; otherwise `errnr` is set.
    pub fn c__gdxopenwrite(
        obj: *mut GdxObj,
        filename: *const c_char,
        producer: *const c_char,
        errnr: *mut c_int,
    ) -> c_int;
    /// Close the currently open GDX file.
    pub fn c__gdxclose(obj: *mut GdxObj) -> c_int;

    /// Number of symbols and unique elements in the open file.
    pub fn c__gdxsysteminfo(obj: *mut GdxObj, symcnt: *mut c_int, uelcnt: *mut c_int) -> c_int;
    /// Name, dimension and type (`GMS_DT_*`) of symbol number `synr` (1-based).
    pub fn c__gdxsymbolinfo(
        obj: *mut GdxObj,
        synr: c_int,
        syid: *mut c_char,
        dim: *mut c_int,
        typ: *mut c_int,
    ) -> c_int;
    /// Record count, user info (subtype) and explanatory text of symbol `synr`.
    pub fn c__gdxsymbolinfox(
        obj: *mut GdxObj,
        synr: c_int,
        reccnt: *mut c_int,
        userinfo: *mut c_int,
        expl: *mut c_char,
    ) -> c_int;
    /// Domain identifiers for symbol `synr`. `domainids` must hold `dim` `char*`.
    pub fn c__gdxsymbolgetdomainx(
        obj: *mut GdxObj,
        synr: c_int,
        domainids: *mut *mut c_char,
    ) -> c_int;

    /// Relaxed domain definition for symbol `synr`: `domainids` holds `dim`
    /// identifiers (not checked against known sets, no domain checking).
    pub fn c__gdxsymbolsetdomainx(
        obj: *mut GdxObj,
        synr: c_int,
        domainids: *const *const c_char,
    ) -> c_int;

    /// Case-insensitive symbol lookup by name; sets `synr` (0 universe,
    /// -1 not found). Returns non-zero when found.
    pub fn c__gdxfindsymbol(obj: *mut GdxObj, syid: *const c_char, synr: *mut c_int) -> c_int;

    /// Begin reading symbol `synr` in string mode; sets `nrecs`.
    pub fn c__gdxdatareadstrstart(obj: *mut GdxObj, synr: c_int, nrecs: *mut c_int) -> c_int;
    /// Read one record: fills `keystr` (array of `dim` C strings) and `values[5]`.
    /// Returns 1 while records remain, 0 when exhausted.
    pub fn c__gdxdatareadstr(
        obj: *mut GdxObj,
        keystr: *mut *mut c_char,
        values: *mut c_double,
        dimfrst: *mut c_int,
    ) -> c_int;
    /// Finish the current read.
    pub fn c__gdxdatareaddone(obj: *mut GdxObj) -> c_int;

    /// Begin writing a symbol in string mode.
    pub fn c__gdxdatawritestrstart(
        obj: *mut GdxObj,
        syid: *const c_char,
        expltxt: *const c_char,
        dimen: c_int,
        typ: c_int,
        userinfo: c_int,
    ) -> c_int;
    /// Write one record: `keystr` (array of `dim` C strings) and `values[5]`.
    pub fn c__gdxdatawritestr(
        obj: *mut GdxObj,
        keystr: *const *const c_char,
        values: *const c_double,
    ) -> c_int;
    /// Finish the current write.
    pub fn c__gdxdatawritedone(obj: *mut GdxObj) -> c_int;

    /// Special-value array (EPS/NA/+Inf/-Inf/Undef) used in records.
    pub fn c__gdxgetspecialvalues(obj: *mut GdxObj, avals: *mut c_double) -> c_int;
    /// Number of the last error, or 0.
    pub fn c__gdxgetlasterror(obj: *mut GdxObj) -> c_int;
    /// Human-readable message for error number `errnr`.
    pub fn c__gdxerrorstr(obj: *mut GdxObj, errnr: c_int, errmsg: *mut c_char) -> c_int;

    /// Begin reading symbol `synr` in raw (integer-key) mode; sets `nrecs`.
    /// Keys arrive as 1-based global UEL indices; use [`c__gdxumuelget`] to resolve them.
    pub fn c__gdxdatareadrawstart(obj: *mut GdxObj, synr: c_int, nrecs: *mut c_int) -> c_int;

    /// Bulk raw read: calls `dp(indx, vals)` once per record from a single
    /// FFI crossing. Signature: `void (*)(const int*, const double*)`.
    pub fn c__gdxdatareadrawfast(
        obj: *mut GdxObj,
        synr: c_int,
        dp: extern "C" fn(*const c_int, *const c_double),
        nrecs: *mut c_int,
    ) -> c_int;
    /// Read one record in raw mode: fills `keyint` (array of `dim` UEL indices) and `values[5]`.
    /// Returns 1 while records remain, 0 when exhausted.
    pub fn c__gdxdatareadraw(
        obj: *mut GdxObj,
        keyint: *mut c_int,
        values: *mut c_double,
        dimfrst: *mut c_int,
    ) -> c_int;

    /// UEL count and highest mapped index for the open file.
    pub fn c__gdxumuelinfo(obj: *mut GdxObj, uelcnt: *mut c_int, highmap: *mut c_int) -> c_int;
    /// Bulk raw read with early termination and a user-data pointer: calls
    /// `dp(indx, vals, afdim, uptr)` once per record from a single FFI
    /// crossing; the read stops when `dp` returns 0.
    /// Signature: `int (*)(const int*, const double*, int, void*)`.
    pub fn c__gdxdatareadrawfastex(
        obj: *mut GdxObj,
        synr: c_int,
        dp: extern "C" fn(*const c_int, *const c_double, c_int, *mut c_void) -> c_int,
        nrecs: *mut c_int,
        uptr: *mut c_void,
    ) -> c_int;

    /// (polars-gdx extension) Checkpointed positional raw read. Delivers
    /// records of symbol `synr` to `dp`, starting at the checkpoint
    /// (`start_pos`, `start_offset`) (0 = start of the symbol data) and
    /// stopping once the stream reaches the checkpoint (`end_pos`,
    /// `end_offset`) (exclusive; `i64::MAX` = to the end of the data).
    /// For uncompressed data the offsets are ignored; for block-compressed
    /// data `start_pos` is the physical start of the compressed block
    /// holding the resume record and `start_offset` the record's offset
    /// within the decompressed block. On return `*next_pos`/`*next_offset`
    /// hold the checkpoint just past the last record consumed (0 at end of
    /// data).
    pub fn c__gdxdatareadrawrange(
        obj: *mut GdxObj,
        synr: c_int,
        start_pos: i64,
        start_offset: u32,
        end_pos: i64,
        end_offset: u32,
        dp: extern "C" fn(*const c_int, *const c_double, c_int, *mut c_void) -> c_int,
        nrecs: *mut c_int,
        uptr: *mut c_void,
        next_pos: *mut i64,
        next_offset: *mut u32,
    ) -> c_int;
    /// (polars-gdx extension) Whether symbol `synr`'s data section is
    /// block-compressed. Returns 0 on failure (treated as uncompressed).
    pub fn c__gdxsymboliscompressed(obj: *mut GdxObj, synr: c_int) -> c_int;
    /// (polars-gdx extension) Byte span [start, end) of symbol `synr`'s
    /// data section (physical, uncompressed). Returns 0 on failure.
    pub fn c__gdxsymboldataspan(
        obj: *mut GdxObj,
        synr: c_int,
        start_pos: *mut i64,
        end_pos: *mut i64,
    ) -> c_int;
    /// (polars-gdx extension) Collect the exact physical start positions of
    /// every restart record (first-changed dimension 1) of symbol `synr`.
    /// `dp` receives `(lo32(pos), hi32(pos), rec_nr)` in its index argument;
    /// the values argument is null. Returns 0 on failure.
    pub fn c__gdxcollectrestartpositions(
        obj: *mut GdxObj,
        synr: c_int,
        dp: extern "C" fn(*const c_int, *const c_double, c_int, *mut c_void) -> c_int,
        uptr: *mut c_void,
    ) -> c_int;
    /// List the unique elements actually used by one dimension of a symbol:
    /// calls `dp(rawindex, mappedindex, uptr)` once per unique element
    /// (in UEL order). `filternr` must be `DOMC_EXPAND` (-1, no filter).
    /// Signature: `void (*)(int, int, void*)`.
    pub fn c__gdxgetdomainelements(
        obj: *mut GdxObj,
        synr: c_int,
        dimpos: c_int,
        filternr: c_int,
        dp: extern "C" fn(c_int, c_int, *mut c_void),
        nrelem: *mut c_int,
        uptr: *mut c_void,
    ) -> c_int;
    /// Resolve UEL number `uelnr` to its label string.
    /// Writes into `uel` (caller provides [`GMS_SSSIZE`] buffer); sets `uelmap`.
    /// Returns 1 on success, 0 if `uelnr` is out of range.
    pub fn c__gdxumuelget(
        obj: *mut GdxObj,
        uelnr: c_int,
        uel: *mut c_char,
        uelmap: *mut c_int,
    ) -> c_int;
}
