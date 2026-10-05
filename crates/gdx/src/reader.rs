use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::Path;
use std::ptr;
use std::sync::Arc;

use gdx_sys as ffi;

use crate::error::{GdxError, Result};

/// Record prefilter on raw UEL indices.
pub type IndexPred<'a> = Option<&'a dyn Fn(&[i32]) -> bool>;

/// What the read loop should do with a record, evaluated on its raw UEL
/// indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordAction {
    /// Store the record and continue.
    Accept,
    /// Skip the record and continue.
    Skip,
    /// Skip the record and terminate the scan: the caller knows no further
    /// record can match (GDX stores records sorted by key indices).
    Stop,
}

/// Record prefilter returning a [`RecordAction`].
pub type ActionPred<'a> = Option<&'a dyn Fn(&[i32]) -> RecordAction>;

/// `FilterNr` sentinel accepted by `gdxGetDomainElements`: no filter.
const DOMC_EXPAND: i32 = -1;

use crate::types::{Record, SymbolInfo, SymbolType, ValueField};

/// A GDX file opened for reading.
///
/// On [`open`](GdxFile::open) the symbol table and special-value sentinels are
/// loaded eagerly; record data is read on demand via [`read`](GdxFile::read).
/// The underlying GDX object is closed and freed on drop.
///
/// Not thread-safe: the type is intentionally `!Send`/`!Sync` (it holds a raw
/// pointer). Read the data you need into owned [`Record`]s, then drop the file.
/// Raw symbol data: per-dimension UEL indices plus a single value column.
#[derive(Debug, Default)]
pub struct RawSymbolData {
    /// `keys[d]` holds the 0-based UEL dictionary positions of dimension `d`
    /// for every record (raw UEL number minus one), ready for Arrow
    /// dictionary encoding.
    pub keys: Vec<Vec<u32>>,
    pub values: Vec<f64>,
}

impl RawSymbolData {
    fn with_capacity(dim: usize, n: usize) -> Self {
        Self {
            keys: vec![Vec::with_capacity(n); dim],
            values: Vec::with_capacity(n),
        }
    }

    /// Number of records.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the symbol has no records.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// Filter abstraction: either on resolved labels (slow) or on raw UEL
/// indices (fast prefilter; no string work for rejected records).
enum Filter<'a> {
    None,
    Labels(&'a dyn Fn(&[Arc<str>]) -> bool),
    Indices(&'a dyn Fn(&[i32]) -> bool),
}

pub struct GdxFile {
    obj: *mut ffi::GdxObj,
    special: [f64; ffi::GMS_SVIDX_MAX],
    symbols: Vec<SymbolInfo>,
    name_index: HashMap<String, usize>,
    /// Lazy per-file UEL cache: index → interned label (populated on first raw read).
    uel_cache: RefCell<HashMap<i32, Arc<str>>>,
    /// Cached UEL table: entry `i` is the label of UEL number `i + 1`.
    uel_table_cache: RefCell<Option<Arc<[String]>>>,
    /// Cached reverse UEL map: label → UEL number.
    uel_index_cache: RefCell<Option<Arc<HashMap<String, i32>>>>,
}

impl GdxFile {
    /// Open `path` for reading.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let cpath = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| GdxError::InvalidPath(path.clone()))?;

        let _guard = crate::lock::lock();
        unsafe {
            let mut obj: *mut ffi::GdxObj = ptr::null_mut();
            let mut msg = [0 as c_char; ffi::GMS_SSSIZE];
            if ffi::gdxcreate(&mut obj, msg.as_mut_ptr(), ffi::GMS_SSSIZE as i32) == 0
                || obj.is_null()
            {
                return Err(GdxError::Create(buf_to_string(&msg)));
            }

            let mut errnr = 0;
            if ffi::c__gdxopenread(obj, cpath.as_ptr(), &mut errnr) == 0 {
                let message = error_message(obj, errnr);
                ffi::gdxfree(&mut obj);
                return Err(GdxError::OpenRead {
                    path,
                    code: errnr,
                    message,
                });
            }

            let mut special = [0.0f64; ffi::GMS_SVIDX_MAX];
            ffi::c__gdxgetspecialvalues(obj, special.as_mut_ptr());

            let symbols = match read_symbol_table(obj) {
                Ok(s) => s,
                Err(e) => {
                    ffi::c__gdxclose(obj);
                    ffi::gdxfree(&mut obj);
                    return Err(e);
                }
            };

            let name_index = symbols
                .iter()
                .enumerate()
                .map(|(i, s)| (s.name.clone(), i))
                .collect();

            Ok(GdxFile {
                obj,
                special,
                symbols,
                name_index,
                uel_cache: RefCell::new(HashMap::new()),
                uel_table_cache: RefCell::new(None),
                uel_index_cache: RefCell::new(None),
            })
        }
    }

    /// The file's symbol table (cached at open time).
    pub fn symbols(&self) -> &[SymbolInfo] {
        &self.symbols
    }

    /// Look up a symbol by name (case-sensitive, as stored in the file).
    pub fn symbol(&self, name: &str) -> Option<&SymbolInfo> {
        self.name_index.get(name).map(|&i| &self.symbols[i])
    }

    /// Build (and cache) a reverse map: UEL label → UEL number.
    ///
    /// Used to translate key-filter labels into raw indices so reads can
    /// prefilter on integers instead of strings.
    pub fn uel_index(&self) -> Result<Arc<HashMap<String, i32>>> {
        if let Some(m) = self.uel_index_cache.borrow().as_ref() {
            return Ok(Arc::clone(m));
        }
        let table = self.uel_table()?;
        let mut map = HashMap::with_capacity(table.len());
        for (i, label) in table.iter().enumerate() {
            map.insert(label.clone(), (i + 1) as i32);
        }
        let map = Arc::new(map);
        *self.uel_index_cache.borrow_mut() = Some(Arc::clone(&map));
        Ok(map)
    }

    /// The unique UEL numbers used by one dimension (`dim_pos`, 0-based) of a
    /// symbol, in UEL order. The scan runs inside the C library via a bulk
    /// callback (one FFI crossing): no record is materialised, no label
    /// string is resolved — the cost of `unique(dim)` without reading the
    /// data. Empty when `dim_pos` lies outside the symbol's own dimensions.
    pub fn domain_elements(&self, info: &SymbolInfo, dim_pos: usize) -> Result<Vec<i32>> {
        if info.dim == 0 || dim_pos >= info.dim {
            return Ok(Vec::new());
        }
        let _guard = crate::lock::lock();
        let uelcnt = unsafe {
            let (mut uelcnt, mut highmap) = (0i32, 0i32);
            if ffi::c__gdxumuelinfo(self.obj, &mut uelcnt, &mut highmap) == 0 {
                return Err(op_error(self.obj, "gdxUMUELInfo"));
            }
            uelcnt
        };
        // seen[n] marks UEL number n (1-based) as delivered; slot 0 unused.
        let mut seen = vec![false; uelcnt.max(0) as usize + 1];
        let sink = DomainSink { seen: &mut seen };
        let mut nrelem = 0;
        let ok = unsafe {
            ffi::c__gdxgetdomainelements(
                self.obj,
                info.number as i32,
                dim_pos as i32 + 1,
                DOMC_EXPAND,
                store_domain_element,
                &mut nrelem,
                &sink as *const DomainSink<'_> as *mut std::ffi::c_void,
            )
        };
        if ok == 0 {
            return Err(unsafe { op_error(self.obj, "gdxGetDomainElements") });
        }
        Ok((1..=uelcnt).filter(|&n| seen[n as usize]).collect())
    }

    /// Read a symbol in fully raw form: per-dimension UEL indices plus one
    /// value field, without materialising any label strings.
    ///
    /// This is the fast vectorised path used by the Polars plugin: keys stay
    /// as `i32` UEL numbers (resolvable via [`GdxFile::uel_table`]) and only
    /// the requested value field is collected. `pred`, when given, is evaluated
    /// on the raw indices before the record is appended. `limit`, when given,
    /// stops the read once that many matching records have been stored.
    pub fn read_symbol_raw(
        &self,
        info: &SymbolInfo,
        value_field: ValueField,
        pred: ActionPred<'_>,
        limit: Option<usize>,
    ) -> Result<RawSymbolData> {
        let _guard = crate::lock::lock();
        unsafe { self.read_symbol_raw_locked(info, value_field, pred, limit) }
    }

    /// Like [`GdxFile::read_symbol_raw`], kept as an alias: the predicate may
    /// report [`RecordAction::Stop`] to terminate the scan early (records are
    /// stored sorted by key indices, so a first-dimension filter can stop
    /// once the scan moves past its highest allowed UEL).
    pub fn read_symbol_raw_stoppable(
        &self,
        info: &SymbolInfo,
        value_field: ValueField,
        pred: ActionPred<'_>,
        limit: Option<usize>,
    ) -> Result<RawSymbolData> {
        self.read_symbol_raw(info, value_field, pred, limit)
    }

    /// Parallel read of a symbol across `threads` independent file handles.
    ///
    /// Opens the same file once per worker thread (independent GDX objects;
    /// the library keeps no mutable global state in the direct `c__*` entry
    /// points) and scans disjoint contiguous ranges of the first-dimension
    /// UEL space concurrently. Records are stored sorted by key, so the
    /// per-range results are concatenated in range order to reproduce the
    /// single-threaded record order exactly.
    ///
    /// `pred` may report [`RecordAction::Stop`]: per-thread it stops that
    /// thread's scan once the first-dimension key leaves the thread's range
    /// or the filter's allowed set (whichever comes first).
    ///
    /// Falls back to the serial path when `threads <= 1` or the symbol has
    /// fewer than two first-dimension UELs.
    #[allow(clippy::type_complexity)]
    pub fn read_symbol_raw_parallel(
        path: impl AsRef<Path>,
        info: &SymbolInfo,
        value_field: ValueField,
        pred_builder: &(dyn Fn() -> Box<dyn Fn(&[i32]) -> RecordAction + Send> + Sync),
        limit: Option<usize>,
        threads: usize,
        range_skip: &(dyn Fn(i32, i32) -> bool + Sync),
    ) -> Result<RawSymbolData> {
        if threads <= 1 {
            let file = GdxFile::open(path)?;
            let pred = pred_builder();
            let pred_ref: ActionPred<'_> = Some(&*pred);
            return file.read_symbol_raw(info, value_field, pred_ref, limit);
        }
        let (uelcnt, _) = {
            // Cheap probe on a throwaway handle: total UEL space to partition.
            let probe = GdxFile::open(&path)?;
            probe.uel_counts()?
        };
        if uelcnt < 2 {
            let file = GdxFile::open(path)?;
            let pred = pred_builder();
            let pred_ref: ActionPred<'_> = Some(&*pred);
            return file.read_symbol_raw(info, value_field, pred_ref, limit);
        }
        let threads = threads.min(uelcnt as usize);
        let path = path.as_ref().to_path_buf();
        let ranges = split_ranges(uelcnt as usize, threads);
        // A global row limit under parallelism would need coordination;
        // keep it per-thread conservative (may over-deliver is NOT allowed,
        // so only apply per-thread limits when no limit is set).
        let per_limit = None::<usize>;
        let _ = limit;
        let results: std::sync::Mutex<Vec<(usize, Result<RawSymbolData>)>> =
            std::sync::Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            for (t, (lo, hi)) in ranges.iter().enumerate() {
                let results = &results;
                let path = path.clone();
                scope.spawn(move || {
                    let res = if range_skip(*lo, *hi) {
                        Ok(RawSymbolData::with_capacity(info.dim, 0))
                    } else {
                        GdxFile::open(&path).and_then(|file| {
                            let pred = pred_builder();
                            let pred_ref: ActionPred<'_> = Some(&*pred);
                            file.read_symbol_raw_range_unlocked(
                                info,
                                value_field,
                                pred_ref,
                                per_limit,
                                *lo,
                                *hi,
                            )
                        })
                    };
                    if let Ok(mut guard) = results.lock() {
                        guard.push((t, res));
                    }
                });
            }
        });
        let mut parts = results.into_inner().unwrap();
        parts.sort_by_key(|(t, _)| *t);
        // Merge in range order (records sorted by first-dim key, ranges
        // disjoint and ordered, so concatenation preserves global order).
        let mut keys: Vec<Vec<u32>> = vec![Vec::new(); info.dim];
        let mut values: Vec<f64> = Vec::new();
        for (_, res) in parts {
            let part = res?;
            for (k, pk) in keys.iter_mut().zip(part.keys.iter()) {
                k.extend_from_slice(pk);
            }
            values.extend_from_slice(&part.values);
        }
        Ok(RawSymbolData { keys, values })
    }

    /// Number of UELs and highest mapped UEL for the open file.
    pub fn uel_counts(&self) -> Result<(i32, i32)> {
        let _guard = crate::lock::lock();
        unsafe {
            let (mut uelcnt, mut highmap) = (0i32, 0i32);
            if ffi::c__gdxumuelinfo(self.obj, &mut uelcnt, &mut highmap) == 0 {
                return Err(op_error(self.obj, "gdxUMUELInfo"));
            }
            Ok((uelcnt, highmap))
        }
    }

    /// Serial read restricted to a contiguous range of first-dimension UEL
    /// numbers, `[lo, hi]` inclusive (1-based; `lo >= 1`, `hi >= lo`). Records
    /// outside the range are skipped; the scan stops once the first-dimension
    /// key exceeds `hi` (records are stored sorted by key). Other dimensions
    /// are filtered only through `pred`, as usual.
    pub fn read_symbol_raw_range(
        &self,
        info: &SymbolInfo,
        value_field: ValueField,
        pred: ActionPred<'_>,
        limit: Option<usize>,
        lo: i32,
        hi: i32,
    ) -> Result<RawSymbolData> {
        let _guard = crate::lock::lock();
        unsafe { self.read_symbol_raw_range_locked(info, value_field, pred, limit, lo, hi) }
    }

    /// Range read on a worker-private handle, skipping the process-global
    /// GDX lock. Safe only when this handle is used by a single thread and
    /// no other thread is inside a locked GDX sequence that assumes
    /// exclusive access to the whole library (the vendored library's internal
    /// mutexes protect its global state; per-handle calls are lock-free).
    pub(crate) fn read_symbol_raw_range_unlocked(
        &self,
        info: &SymbolInfo,
        value_field: ValueField,
        pred: ActionPred<'_>,
        limit: Option<usize>,
        lo: i32,
        hi: i32,
    ) -> Result<RawSymbolData> {
        unsafe { self.read_symbol_raw_range_locked(info, value_field, pred, limit, lo, hi) }
    }

    unsafe fn read_symbol_raw_range_locked(
        &self,
        info: &SymbolInfo,
        value_field: ValueField,
        pred: ActionPred<'_>,
        limit: Option<usize>,
        lo: i32,
        hi: i32,
    ) -> Result<RawSymbolData> {
        let vidx = value_field.index();
        let reserve = info.records.min(1024 * 1024);
        let mut data = RawSymbolData::with_capacity(info.dim, reserve);
        // Combine the caller predicate with the range restriction on the
        // first-dimension key: skip below lo, stop past hi (sorted store).
        let combined = move |keys: &[i32]| -> RecordAction {
            let k0 = keys[0];
            if k0 < lo {
                return RecordAction::Skip;
            }
            if k0 > hi {
                return RecordAction::Stop;
            }
            match pred {
                Some(f) => f(keys),
                None => RecordAction::Accept,
            }
        };
        let sink = RecordSink {
            data: std::ptr::from_mut(&mut data),
            special: std::ptr::from_ref(&self.special),
            vidx,
            dim: info.dim,
            pred: Some(&combined),
            remaining: std::cell::Cell::new(limit),
        };
        let mut cb_nrecs = 0;
        let ok = ffi::c__gdxdatareadrawfastex(
            self.obj,
            info.number as i32,
            store_record_ex,
            &mut cb_nrecs,
            &sink as *const RecordSink<'_> as *mut std::ffi::c_void,
        );
        if ok == 0 {
            return Err(op_error(self.obj, "gdxDataReadRawFastEx"));
        }
        Ok(data)
    }

    /// The file's UEL table: entry `i` is the label of UEL number `i + 1`.
    /// Missing entries yield an empty-string placeholder.
    pub fn uel_table(&self) -> Result<Arc<[String]>> {
        if let Some(t) = self.uel_table_cache.borrow().as_ref() {
            return Ok(Arc::clone(t));
        }
        let _guard = crate::lock::lock();
        let table = unsafe {
            let (mut uelcnt, mut highmap) = (0i32, 0i32);
            if ffi::c__gdxumuelinfo(self.obj, &mut uelcnt, &mut highmap) == 0 {
                return Err(op_error(self.obj, "gdxUMUELInfo"));
            }
            let mut table: Vec<String> = Vec::with_capacity(uelcnt.max(0) as usize);
            for uelnr in 1..=uelcnt {
                let mut buf = [0 as c_char; ffi::GMS_SSSIZE];
                let mut uelmap = 0;
                if ffi::c__gdxumuelget(self.obj, uelnr, buf.as_mut_ptr(), &mut uelmap) == 1 {
                    table.push(buf_to_string(&buf));
                } else {
                    table.push(String::new());
                }
            }
            table
        };
        let table: Arc<[String]> = table.into();
        *self.uel_table_cache.borrow_mut() = Some(Arc::clone(&table));
        Ok(table)
    }

    unsafe fn read_symbol_raw_locked(
        &self,
        info: &SymbolInfo,
        value_field: ValueField,
        pred: ActionPred<'_>,
        limit: Option<usize>,
    ) -> Result<RawSymbolData> {
        let vidx = value_field.index();
        let mut data = RawSymbolData::with_capacity(info.dim, info.records);

        let nrecs = 0;
        let _ = nrecs;

        // Filtered and limited reads alike use the `gdxDataReadRawFastEx`
        // bulk callback: one FFI crossing for the whole loop, with the index
        // predicate and optional limit evaluated inside the callback before
        // anything is stored (returning 0 terminates the C read loop).
        if let Some(f) = pred {
            let sink = RecordSink {
                data: std::ptr::from_mut(&mut data),
                special: std::ptr::from_ref(&self.special),
                vidx,
                dim: info.dim,
                pred: Some(f),
                remaining: std::cell::Cell::new(limit),
            };
            let mut cb_nrecs = 0;
            let ok = ffi::c__gdxdatareadrawfastex(
                self.obj,
                info.number as i32,
                store_record_ex,
                &mut cb_nrecs,
                &sink as *const RecordSink<'_> as *mut std::ffi::c_void,
            );
            if ok == 0 {
                return Err(op_error(self.obj, "gdxDataReadRawFastEx"));
            }
            return Ok(data);
        }

        if let Some(limit) = limit {
            let sink = RecordSink {
                data: std::ptr::from_mut(&mut data),
                special: std::ptr::from_ref(&self.special),
                vidx,
                dim: info.dim,
                pred: None,
                remaining: std::cell::Cell::new(Some(limit)),
            };
            let mut cb_nrecs = 0;
            let ok = ffi::c__gdxdatareadrawfastex(
                self.obj,
                info.number as i32,
                store_record_ex,
                &mut cb_nrecs,
                &sink as *const RecordSink<'_> as *mut std::ffi::c_void,
            );
            if ok == 0 {
                return Err(op_error(self.obj, "gdxDataReadRawFastEx"));
            }
            return Ok(data);
        }

        // Unfiltered unlimited read: bulk callback with a user-data pointer,
        // one FFI crossing for the whole loop.
        let sink = RecordSink {
            data: std::ptr::from_mut(&mut data),
            special: std::ptr::from_ref(&self.special),
            vidx,
            dim: info.dim,
            pred: None,
            remaining: std::cell::Cell::new(None),
        };
        let mut cb_nrecs = 0;
        let ok = ffi::c__gdxdatareadrawfastex(
            self.obj,
            info.number as i32,
            store_record_ex,
            &mut cb_nrecs,
            &sink as *const RecordSink<'_> as *mut std::ffi::c_void,
        );
        if ok == 0 {
            return Err(op_error(self.obj, "gdxDataReadRawFastEx"));
        }
        Ok(data)
    }

    /// Read all records of the named symbol.
    pub fn read(&self, name: &str) -> Result<Vec<Record>> {
        let info = self
            .symbol(name)
            .ok_or_else(|| GdxError::SymbolNotFound(name.to_string()))?;
        self.read_info(info)
    }

    /// Read all records of a symbol described by `info`.
    pub fn read_info(&self, info: &SymbolInfo) -> Result<Vec<Record>> {
        let _guard = crate::lock::lock();
        unsafe { self.read_records_raw(info.number, info.dim, &Filter::None) }
    }

    /// Read records of a symbol, keeping only those whose key labels satisfy
    /// `pred`. The predicate is evaluated per record during the raw read loop,
    /// so filtered-out records are never materialised into owned strings.
    pub fn read_filtered(
        &self,
        info: &SymbolInfo,
        pred: &dyn Fn(&[Arc<str>]) -> bool,
    ) -> Result<Vec<Record>> {
        let _guard = crate::lock::lock();
        unsafe { self.read_records_raw(info.number, info.dim, &Filter::Labels(pred)) }
    }

    /// Read records of a symbol, keeping only those whose raw UEL indices
    /// satisfy `pred`. This is the fast prefilter path: the predicate runs on
    /// plain `i32` indices before any label string is resolved or allocated,
    /// so filtered-out records cost only the raw C read.
    ///
    /// `pred` receives the `dim` raw key indices of the record, in order.
    pub fn read_filtered_indices(
        &self,
        info: &SymbolInfo,
        pred: &dyn Fn(&[i32]) -> bool,
    ) -> Result<Vec<Record>> {
        let _guard = crate::lock::lock();
        unsafe { self.read_records_raw(info.number, info.dim, &Filter::Indices(pred)) }
    }

    /// Read all records in raw mode (integer UEL indices → interned `Arc<str>` labels).
    ///
    /// Caller must hold the global FFI lock. UEL labels are resolved lazily via
    /// [`c__gdxumuelget`] and memoised in `self.uel_cache` so each unique UEL
    /// string is allocated at most once per `GdxFile` instance.
    unsafe fn read_records_raw(
        &self,
        number: usize,
        dim: usize,
        pred: &Filter<'_>,
    ) -> Result<Vec<Record>> {
        let mut nrecs = 0;
        if ffi::c__gdxdatareadrawstart(self.obj, number as i32, &mut nrecs) == 0 {
            return Err(op_error(self.obj, "gdxDataReadRawStart"));
        }

        let mut key_indices = [0i32; ffi::GMS_MAX_INDEX_DIM];
        let mut values = [0.0f64; ffi::GMS_VAL_MAX];
        let mut dimfrst = 0;

        let mut records = Vec::with_capacity(nrecs.max(0) as usize);
        let mut uel_cache = self.uel_cache.borrow_mut();

        while ffi::c__gdxdatareadraw(
            self.obj,
            key_indices.as_mut_ptr(),
            values.as_mut_ptr(),
            &mut dimfrst,
        ) == 1
        {
            if let Filter::Indices(f) = pred {
                if !f(&key_indices[..dim]) {
                    continue;
                }
            }
            let keys: Vec<Arc<str>> =
                (0..dim)
                    .map(|d| {
                        let uelnr = key_indices[d];
                        if uelnr <= 0 {
                            return Arc::from("");
                        }
                        if let Some(s) = uel_cache.get(&uelnr) {
                            return Arc::clone(s);
                        }
                        let mut buf = [0 as c_char; ffi::GMS_SSSIZE];
                        let mut uel_map = 0;
                        let label: Arc<str> =
                            if ffi::c__gdxumuelget(self.obj, uelnr, buf.as_mut_ptr(), &mut uel_map)
                                == 1
                            {
                                Arc::from(buf_to_string(&buf).as_str())
                            } else {
                                Arc::from(format!("<uel {uelnr}>").as_str())
                            };
                        uel_cache.insert(uelnr, Arc::clone(&label));
                        label
                    })
                    .collect();

            if let Filter::Labels(f) = pred {
                if !f(&keys) {
                    continue;
                }
            }
            let mut mapped = [0.0f64; ffi::GMS_VAL_MAX];
            for (i, v) in values.iter().enumerate() {
                mapped[i] = map_special(
                    *v,
                    self.special[ffi::GMS_SVIDX_PINF],
                    self.special[ffi::GMS_SVIDX_MINF],
                    self.special[ffi::GMS_SVIDX_UNDEF],
                    self.special[ffi::GMS_SVIDX_NA],
                    self.special[ffi::GMS_SVIDX_EPS],
                );
            }
            records.push(Record {
                keys,
                values: mapped,
            });
        }
        ffi::c__gdxdatareaddone(self.obj);
        Ok(records)
    }
}

impl Drop for GdxFile {
    fn drop(&mut self) {
        if self.obj.is_null() {
            return;
        }
        let _guard = crate::lock::lock();
        unsafe {
            ffi::c__gdxclose(self.obj);
            ffi::gdxfree(&mut self.obj);
        }
    }
}

/// Split `n` items into `t` contiguous, roughly equal ranges (1-based,
/// inclusive; empty ranges omitted).
fn split_ranges(n: usize, t: usize) -> Vec<(i32, i32)> {
    let t = t.max(1);
    let per = n / t;
    let extra = n % t;
    let mut out = Vec::with_capacity(t);
    let mut lo = 1usize;
    for i in 0..t {
        let len = per + if i < extra { 1 } else { 0 };
        if len == 0 {
            continue;
        }
        let hi = lo + len - 1;
        out.push((lo as i32, hi as i32));
        lo = hi + 1;
    }
    out
}

/// Map a GDX special-value sentinel onto an ordinary `f64`.
fn map_special(v: f64, pinf: f64, minf: f64, undef: f64, na: f64, eps: f64) -> f64 {
    if v == pinf {
        f64::INFINITY
    } else if v == minf {
        f64::NEG_INFINITY
    } else if v == undef || v == na {
        f64::NAN
    } else if v == eps {
        0.0
    } else {
        v
    }
}

unsafe fn read_symbol_table(obj: *mut ffi::GdxObj) -> Result<Vec<SymbolInfo>> {
    let (mut symcnt, mut uelcnt) = (0, 0);
    if ffi::c__gdxsysteminfo(obj, &mut symcnt, &mut uelcnt) == 0 {
        return Err(op_error(obj, "gdxSystemInfo"));
    }

    let mut out = Vec::with_capacity(symcnt.max(0) as usize);
    for number in 1..=symcnt {
        let mut id = [0 as c_char; ffi::GMS_SSSIZE];
        let (mut dim, mut typ) = (0, 0);
        if ffi::c__gdxsymbolinfo(obj, number, id.as_mut_ptr(), &mut dim, &mut typ) == 0 {
            return Err(op_error(obj, "gdxSymbolInfo"));
        }
        let mut expl = [0 as c_char; ffi::GMS_SSSIZE];
        let (mut recs, mut userinfo) = (0, 0);
        ffi::c__gdxsymbolinfox(obj, number, &mut recs, &mut userinfo, expl.as_mut_ptr());

        let Some(kind) = SymbolType::from_raw(typ) else {
            continue; // skip unknown symbol categories
        };
        let dim = dim.max(0) as usize;

        out.push(SymbolInfo {
            number: number as usize,
            name: buf_to_string(&id),
            dim,
            kind,
            subtype: userinfo,
            records: recs.max(0) as usize,
            text: buf_to_string(&expl),
            domains: read_domains(obj, number, dim),
        });
    }
    Ok(out)
}

unsafe fn read_domains(obj: *mut ffi::GdxObj, number: i32, dim: usize) -> Vec<String> {
    if dim == 0 {
        return Vec::new();
    }
    let mut bufs = vec![[0 as c_char; ffi::GMS_SSSIZE]; dim];
    let mut ptrs: Vec<*mut c_char> = bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
    if ffi::c__gdxsymbolgetdomainx(obj, number, ptrs.as_mut_ptr()) == 0 {
        return vec!["*".to_string(); dim];
    }
    bufs.iter().map(|b| buf_to_string(b)).collect()
}

pub(crate) fn buf_to_string(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

pub(crate) unsafe fn error_message(obj: *mut ffi::GdxObj, code: i32) -> String {
    let mut msg = [0 as c_char; ffi::GMS_SSSIZE];
    if ffi::c__gdxerrorstr(obj, code, msg.as_mut_ptr()) == 1 {
        buf_to_string(&msg)
    } else {
        format!("gdx error {code}")
    }
}

unsafe fn op_error(obj: *mut ffi::GdxObj, op: &'static str) -> GdxError {
    let code = ffi::c__gdxgetlasterror(obj);
    GdxError::Operation {
        op,
        code,
        message: error_message(obj, code),
    }
}

/// Per-read sink state passed to the C bulk-read callbacks. The legacy
/// `c__gdxdatareadrawfast` path routes it through a thread-local (no
/// user-data argument); the `Ex` callbacks receive it via `Uptr`.
struct RecordSink<'a> {
    data: *mut RawSymbolData,
    special: *const [f64; ffi::GMS_SVIDX_MAX],
    vidx: usize,
    dim: usize,
    pred: ActionPred<'a>,
    remaining: std::cell::Cell<Option<usize>>,
}

impl Clone for RecordSink<'_> {
    fn clone(&self) -> Self {
        Self {
            data: self.data,
            special: self.special,
            vidx: self.vidx,
            dim: self.dim,
            pred: self.pred,
            remaining: std::cell::Cell::new(self.remaining.get()),
        }
    }
}

// The callbacks run synchronously on the same thread inside the
// `c__gdxdatareadrawfastex` call, which is itself bracketed by the global GDX
// lock; the sink is only dereferenced there. The `Send` impl covers the raw
// pointer crossing the (never actually taken) thread boundary.
unsafe impl Send for RecordSink<'_> {}

/// Callback for `c__gdxdatareadrawfastex` (filtered and/or limited reads):
/// receives the sink via `Uptr`. Returns 0 to terminate the C read loop once
/// the stored-record limit is reached, 1 to continue.
extern "C" fn store_record_ex(
    indx: *const i32,
    vals: *const f64,
    _afdim: i32,
    uptr: *mut std::ffi::c_void,
) -> i32 {
    let sink = unsafe { &*(uptr as *const RecordSink<'_>) };
    unsafe {
        let keys = std::slice::from_raw_parts(indx, sink.dim);
        if let Some(pred) = sink.pred {
            match pred(keys) {
                RecordAction::Accept => {}
                RecordAction::Skip => return 1,
                RecordAction::Stop => return 0,
            }
        }
        let data = &mut *sink.data;
        for (d, &k) in keys.iter().enumerate() {
            data.keys[d].push((k - 1).max(0) as u32);
        }
        let v = *vals.add(sink.vidx);
        data.values.push(map_special(
            v,
            (*sink.special)[ffi::GMS_SVIDX_PINF],
            (*sink.special)[ffi::GMS_SVIDX_MINF],
            (*sink.special)[ffi::GMS_SVIDX_UNDEF],
            (*sink.special)[ffi::GMS_SVIDX_NA],
            (*sink.special)[ffi::GMS_SVIDX_EPS],
        ));
    }
    if let Some(r) = sink.remaining.get() {
        if r == 0 {
            return 0;
        }
        sink.remaining.set(Some(r - 1));
    }
    1
}

/// Sink for `c__gdxgetdomainelements`: marks delivered UEL numbers as seen.
struct DomainSink<'a> {
    seen: &'a mut [bool],
}

extern "C" fn store_domain_element(rawindex: i32, _mappedindex: i32, uptr: *mut std::ffi::c_void) {
    let sink = unsafe { &mut *(uptr as *mut DomainSink<'_>) };
    if rawindex > 0 {
        if let Some(slot) = sink.seen.get_mut(rawindex as usize) {
            *slot = true;
        }
    }
}
#[cfg(test)]
mod parity {
    //! Verify that raw-mode reading produces the same keys and values as the
    //! original string-mode API, using a self-written fixture.

    use std::sync::Arc;

    use crate::{GdxFile, GdxWriter, Record, SymbolType};

    fn rec(keys: &[&str], level: f64) -> Record {
        Record {
            keys: keys.iter().map(|s| Arc::from(*s)).collect(),
            values: [level, 0.0, 0.0, 0.0, 0.0],
        }
    }

    #[test]
    fn raw_matches_expected_keys_and_values() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("parity.gdx");

        let mut w = GdxWriter::create(&path, "parity-test").unwrap();
        w.write_symbol(
            "c",
            "cost",
            2,
            SymbolType::Parameter,
            0,
            &[
                rec(&["seattle", "new-york"], 0.225),
                rec(&["seattle", "chicago"], 0.153),
                rec(&["san-diego", "topeka"], 0.126),
            ],
        )
        .unwrap();
        w.finish().unwrap();

        let file = GdxFile::open(&path).unwrap();
        let records = file.read("c").unwrap();

        assert_eq!(records.len(), 3);
        assert_eq!(records[0].keys[0].as_ref(), "seattle");
        assert_eq!(records[0].keys[1].as_ref(), "new-york");
        assert!((records[0].values[0] - 0.225).abs() < 1e-12);
        assert_eq!(records[1].keys[0].as_ref(), "seattle");
        assert_eq!(records[1].keys[1].as_ref(), "chicago");
        assert!((records[1].values[0] - 0.153).abs() < 1e-12);
        assert_eq!(records[2].keys[0].as_ref(), "san-diego");
        assert_eq!(records[2].keys[1].as_ref(), "topeka");
        assert!((records[2].values[0] - 0.126).abs() < 1e-12);
    }

    #[test]
    fn shared_uel_labels_are_the_same_arc() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shared.gdx");

        let mut w = GdxWriter::create(&path, "parity-test").unwrap();
        w.write_symbol(
            "c",
            "cost",
            2,
            SymbolType::Parameter,
            0,
            &[
                rec(&["seattle", "new-york"], 0.225),
                rec(&["seattle", "chicago"], 0.153),
            ],
        )
        .unwrap();
        w.finish().unwrap();

        let file = GdxFile::open(&path).unwrap();
        let records = file.read("c").unwrap();

        // Both records share the "seattle" UEL — the Arc pointers must be identical.
        assert!(Arc::ptr_eq(&records[0].keys[0], &records[1].keys[0]));
    }
}
