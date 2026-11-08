//! gdxrs backend (M7 task 042).
//!
//! Pure-Rust selectable backend over the pinned gdxrs core revision
//! (docs/m7-adapter-contract.md §1). Mirrors the legacy `Reader` Python
//! surface: symbol metadata, UEL table, per-dimension used-UEL enumeration,
//! exact label resolution and Arrow record access with dictionary-encoded
//! key columns over the full UEL table (mode A admission).
//!
//! Deliberate Float64 projection (contract §3): typed core values are
//! matched and the adapter result constructed directly — `EPS` maps to
//! positive zero, UNDEF/NA map to `null` (not the legacy NaN), ±INF map to
//! ±inf and acronyms map to `null` under `Preserve`. EPS and acronym
//! projections are counted per scan and reported via `scan_statistics`;
//! they are never silently coerced.
//!
//! A gdxrs source/decode/resource failure is an error; this module never
//! falls back to the legacy backend.

use std::sync::{Arc, Mutex};

use arrow::array::{
    ArrayRef, BinaryArray, DictionaryArray, Float64Array, Int16Array, Int32Array, StringArray,
    UInt32Array,
};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_pyarrow::IntoPyArrow;
use gdxrs::{
    DimensionSelection, KeySelection, OpenOptions, Reader as Core, ScanOptions, SpecialValue,
    SymbolType, UelId, Value, ValueField, ValueProjection,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

fn to_py_err(e: gdxrs::Error) -> PyErr {
    PyRuntimeError::new_err(format!("gdxrs backend: {e}"))
}

fn closed_err() -> PyErr {
    PyRuntimeError::new_err("reader is closed")
}

fn not_found(name: &str) -> PyErr {
    PyRuntimeError::new_err(format!("symbol {name:?} not found"))
}

/// Adapter-side admission for the mode A full-dictionary materialization
/// (UEL strings plus the shared Arrow dictionary values array), charged
/// against the combined adapter envelope of the adapter contract §6
/// (256 MiB). The gdxrs core's own budget does not cover adapter
/// conversions, so this is checked before any allocation: a dictionary
/// that does not fit is rejected clearly instead of silently changing the
/// planned schema or ballooning process memory.
const DEFAULT_DICTIONARY_BUDGET_BYTES: usize = 256 * 1024 * 1024;

fn dictionary_admission_err(labels: usize, estimate: usize, budget: usize) -> PyErr {
    PyRuntimeError::new_err(format!(
        "gdxrs backend: full-dictionary admission of {labels} labels needs ~{estimate} \
         bytes, exceeding the adapter dictionary budget of {budget} bytes; raise \
         dictionary_budget_bytes or use the bounded ID mode (task 043)"
    ))
}

/// Per-scan projection counters surfaced through `scan_statistics`.
#[derive(Clone, Copy, Debug, Default)]
struct ProjectionStats {
    decoded_rows: u64,
    eps_to_zero: u64,
    acronyms_to_null: u64,
}

/// Read-only gdxrs-backed GDX file handle. The pinned core enforces the
/// root memory budget (`OpenOptions`, default 64 MiB); metadata is paged
/// lazily and records are decoded in core-sized batches. The handle is
/// released on drop or earlier via `close()`; every method raises
/// `RuntimeError("reader is closed")` afterwards.
#[pyclass]
pub struct GdxrsReader {
    core: Option<Core>,
    /// Adapter envelope share admitted for the full UEL dictionary.
    dictionary_budget_bytes: usize,
    /// Full UEL table (mode A dictionary admission), decoded once per file.
    uel_strings: Mutex<Option<Arc<Vec<String>>>>,
    uel_array: Mutex<Option<Arc<StringArray>>>,
    statistics: Mutex<ProjectionStats>,
}

type SymbolTuple = (String, String, usize, usize, Vec<String>, String);

fn kind_str(kind: SymbolType) -> &'static str {
    match kind {
        SymbolType::Set => "Set",
        SymbolType::Parameter => "Parameter",
        SymbolType::Variable => "Variable",
        SymbolType::Equation => "Equation",
        SymbolType::Alias => "Alias",
    }
}

fn utf8(bytes: &[u8], what: &str) -> PyResult<String> {
    String::from_utf8(bytes.to_vec()).map_err(|_| {
        PyRuntimeError::new_err(format!(
            "gdxrs backend: {what} is not valid UTF-8; mode A requires valid UTF-8 labels"
        ))
    })
}

impl GdxrsReader {
    fn core(&self) -> PyResult<&Core> {
        self.core.as_ref().ok_or_else(closed_err)
    }

    /// The file's UEL table (cached), strictly UTF-8; entry `i` is the label
    /// of 1-based UEL number `i + 1`.
    fn uel_strings(&self) -> PyResult<Arc<Vec<String>>> {
        if let Some(t) = self.uel_strings.lock().unwrap().as_ref() {
            return Ok(Arc::clone(t));
        }
        let core = self.core()?;
        // Admission pass: count labels and bytes before allocating anything.
        let mut labels = 0usize;
        let mut label_bytes = 0usize;
        {
            let mut cursor = core.uels().map_err(to_py_err)?;
            while let Some(entry) = cursor.next().map_err(to_py_err)? {
                labels += 1;
                label_bytes = label_bytes.saturating_add(entry.label.len() + 1);
            }
        }
        // The dictionary is retained in two simultaneous copies — the
        // cached Vec<String> table and the shared Arrow StringArray — and
        // building the Arrow array needs a further temporary values copy
        // plus a borrowed-slice vector. All are charged together, since
        // they are live at the same time:
        //   strings cache: labels * sizeof(String) + label payload
        //   Arrow array:   i32 offsets + values payload + validity bitmap
        //   temporaries:   labels * sizeof(&str) + one values-buffer copy
        let estimate = labels
            .checked_mul(std::mem::size_of::<String>())
            // String cache payload.
            .and_then(|n| n.checked_add(label_bytes))
            // Arrow i32 offsets and validity bitmap.
            .and_then(|n| n.checked_add(4 * labels.saturating_add(1)))
            .and_then(|n| n.checked_add(labels.saturating_add(7) / 8))
            // Arrow values payload (second live copy of the labels).
            .and_then(|n| n.checked_add(label_bytes))
            // Conversion temporaries: borrowed-slice vector + one more
            // transient values-buffer copy while the array is built.
            .and_then(|n| n.checked_add(labels * std::mem::size_of::<&str>()))
            .and_then(|n| n.checked_add(label_bytes))
            .ok_or_else(|| {
                dictionary_admission_err(labels, usize::MAX, self.dictionary_budget_bytes)
            })?;
        if estimate > self.dictionary_budget_bytes {
            return Err(dictionary_admission_err(
                labels,
                estimate,
                self.dictionary_budget_bytes,
            ));
        }
        let mut table: Vec<String> = Vec::with_capacity(labels);
        let mut cursor = core.uels().map_err(to_py_err)?;
        while let Some(entry) = cursor.next().map_err(to_py_err)? {
            let index = entry.id.get() as usize;
            if index == 0 || index > core.information().map_err(to_py_err)?.uel_count as usize {
                return Err(PyRuntimeError::new_err(
                    "gdxrs backend: UEL ID outside the file table",
                ));
            }
            if table.len() < index {
                table.resize(index, String::new());
            }
            table[index - 1] = utf8(entry.label, "UEL label")?;
        }
        let arc = Arc::new(table);
        *self.uel_strings.lock().unwrap() = Some(Arc::clone(&arc));
        Ok(arc)
    }

    fn uel_array(&self) -> PyResult<Arc<StringArray>> {
        if let Some(a) = self.uel_array.lock().unwrap().as_ref() {
            return Ok(Arc::clone(a));
        }
        let table = self.uel_strings()?;
        let array = Arc::new(StringArray::from(
            table.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ));
        *self.uel_array.lock().unwrap() = Some(Arc::clone(&array));
        Ok(array)
    }

    /// Resolve filter labels to UEL IDs with exact (case-sensitive,
    /// byte-for-byte) matching. Unknown labels resolve to no ID; a filter
    /// whose labels all resolve to nothing stays present with an empty ID
    /// set, meaning zero rows — it is never dropped.
    fn resolve_filters(
        &self,
        filters: &[(usize, Vec<String>)],
    ) -> PyResult<Vec<(usize, Vec<UelId>)>> {
        let core = self.core()?;
        let mut out = Vec::with_capacity(filters.len());
        for (dim, labels) in filters {
            let encoded: Vec<Vec<u8>> = labels.iter().map(|l| l.as_bytes().to_vec()).collect();
            let refs: Vec<&[u8]> = encoded.iter().map(|b| b.as_slice()).collect();
            let matches = core.lookup_labels(&refs).map_err(to_py_err)?;
            let mut ids = Vec::with_capacity(labels.len());
            for ordinal in 0..matches.len() {
                if let Some(resolved) = matches.get(ordinal) {
                    if let Some(id) = resolved.id {
                        ids.push(id);
                    }
                }
            }
            ids.sort_unstable_by_key(|id: &UelId| id.get());
            ids.dedup();
            out.push((*dim, ids));
        }
        Ok(out)
    }

    /// Unique UEL IDs used by one dimension of a symbol, in UEL order, via
    /// the core's spill-backed bounded domain enumeration. No records are
    /// materialized and no full dictionary is retained.
    fn used_uels(&self, core: &Core, name: &str, dim: usize) -> PyResult<Vec<i32>> {
        let handle = core
            .find_symbol(name.as_bytes())
            .map_err(to_py_err)?
            .ok_or_else(|| not_found(name))?;
        let metadata = handle.metadata().map_err(to_py_err)?;
        if dim >= usize::from(metadata.dimension) {
            return Err(PyValueError::new_err(format!(
                "gdxrs backend: dimension {dim} out of range (symbol has {} dimensions)",
                metadata.dimension
            )));
        }
        let base = core.selection(&handle, &[]).map_err(to_py_err)?;
        let scratch = std::env::temp_dir().join(format!(
            "polars-gdx-gdxrs-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&scratch).map_err(|e| {
            PyRuntimeError::new_err(format!("gdxrs backend: scratch directory: {e}"))
        })?;
        let result = (|| {
            let spill = gdxrs::SpillOptions::new(&scratch);
            let mut cursor = core
                .unique_domains(&handle, &base, &[dim as u8], core.scan_options(), &spill)
                .map_err(to_py_err)?;
            let mut ids = Vec::new();
            while let Some(entry) = cursor.next().map_err(to_py_err)? {
                if entry.dimension == dim as u8 {
                    ids.push(entry.id.get() as i32);
                }
            }
            Ok(ids)
        })();
        drop(handle);
        let _ = std::fs::remove_dir_all(&scratch);
        result
    }
}

#[pymethods]
impl GdxrsReader {
    #[new]
    #[pyo3(signature = (path, dictionary_budget_bytes=None))]
    fn new(path: &str, dictionary_budget_bytes: Option<usize>) -> PyResult<Self> {
        let core = Core::open(path, OpenOptions::default()).map_err(to_py_err)?;
        let budget = dictionary_budget_bytes.unwrap_or(DEFAULT_DICTIONARY_BUDGET_BYTES);
        if budget == 0 {
            return Err(PyValueError::new_err(
                "gdxrs backend: dictionary_budget_bytes must be positive",
            ));
        }
        Ok(Self {
            core: Some(core),
            dictionary_budget_bytes: budget,
            uel_strings: Mutex::new(None),
            uel_array: Mutex::new(None),
            statistics: Mutex::new(ProjectionStats::default()),
        })
    }

    /// Release the underlying file handle immediately (idempotent). Any
    /// later use of this reader raises `RuntimeError`.
    fn close(&mut self) {
        self.core = None;
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        Ok(slf)
    }

    fn __exit__(
        &mut self,
        _exc_type: PyObject,
        _exc_value: PyObject,
        _traceback: PyObject,
    ) -> PyResult<()> {
        self.close();
        Ok(())
    }

    /// The file's UEL table: entry `i` is the label of 1-based UEL number
    /// `i + 1`. Labels are strictly UTF-8; invalid bytes are rejected.
    fn uel_table(&self) -> PyResult<Vec<String>> {
        if self.core.is_none() {
            return Err(closed_err());
        }
        Ok(self.uel_strings()?.iter().cloned().collect())
    }

    /// List all symbols: (name, type, dim, record_count, domains, text).
    fn symbols(&self) -> PyResult<Vec<SymbolTuple>> {
        let core = self.core()?;
        let mut out = Vec::new();
        let mut cursor = core.symbols().map_err(to_py_err)?;
        while let Some(meta) = cursor.next().map_err(to_py_err)? {
            let handle = core.symbol(meta.id).map_err(to_py_err)?;
            let mut domains = Vec::with_capacity(usize::from(meta.dimension));
            let mut domain_cursor = core.domains(&handle).map_err(to_py_err)?;
            while let Some(view) = domain_cursor.next().map_err(to_py_err)? {
                domains.push(utf8(view.name, "domain name")?);
            }
            drop(domain_cursor);
            drop(handle);
            // `records` matches the legacy backend's public row count:
            // physical stored records, except 0-dimensional symbols report
            // their synthesized default row (raw count 1).
            let records = if meta.dimension == 0 {
                meta.raw_record_count as usize
            } else {
                meta.physical_record_count as usize
            };
            out.push((
                utf8(meta.name, "symbol name")?,
                kind_str(meta.kind).to_string(),
                usize::from(meta.dimension),
                records,
                domains,
                utf8(meta.description, "symbol text")?,
            ));
        }
        Ok(out)
    }

    /// The unique UEL numbers (1-based) used by one dimension of a symbol,
    /// in UEL order — the raw-index equivalent of `unique(dim)`.
    fn domain_elements(&self, name: &str, dim_pos: usize) -> PyResult<Vec<i32>> {
        if self.core.is_none() {
            return Err(closed_err());
        }
        let core = self.core()?;
        self.used_uels(core, name, dim_pos)
    }

    /// Resolve key-filter labels to UEL IDs (exact match), then restrict
    /// each dimension's ID set to the UELs actually used by that dimension
    /// of the symbol. A label that exists globally but is never used in
    /// the dimension drops to an empty filter — a guaranteed-empty read
    /// detected before touching the data.
    fn resolve_uel_indices_used(
        &self,
        filters: Vec<(usize, Vec<String>)>,
        symbol: &str,
    ) -> PyResult<Vec<(usize, Vec<i32>)>> {
        if self.core.is_none() {
            return Err(closed_err());
        }
        let core = self.core()?;
        let resolved = self.resolve_filters(&filters)?;
        let mut out = Vec::with_capacity(resolved.len());
        for (dim, ids) in resolved {
            if ids.is_empty() {
                out.push((dim, Vec::new()));
                continue;
            }
            let used = self.used_uels(core, symbol, dim)?;
            let used: std::collections::HashSet<i32> = used.into_iter().collect();
            let mut retained: Vec<i32> = ids
                .iter()
                .map(|id| id.get() as i32)
                .filter(|id| used.contains(id))
                .collect();
            retained.sort_unstable();
            retained.dedup();
            out.push((dim, retained));
        }
        Ok(out)
    }

    /// Per-scan projection counters for the last `read_arrow` call:
    /// core decoded rows plus the deliberate EPS-to-zero and
    /// acronym-to-null projections (contract §3).
    fn scan_statistics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let stats = *self.statistics.lock().unwrap();
        let dict = PyDict::new(py);
        dict.set_item("decoded_rows", stats.decoded_rows)?;
        dict.set_item("eps_to_zero", stats.eps_to_zero)?;
        dict.set_item("acronyms_to_null", stats.acronyms_to_null)?;
        Ok(dict)
    }

    /// Resolve original UEL IDs to their exact label bytes (bounded mode
    /// B resolver, adapter contract §5). Returns an Arrow RecordBatch
    /// with columns `uel_id: UInt32` and `label: Binary`, preserving
    /// request order and duplicate requests and returning the original
    /// bytes without any decoding. Zero or out-of-range IDs reject the
    /// whole request with ValueError before any result; a request whose
    /// conversion allocation exceeds `dictionary_budget_bytes` raises a
    /// resource error before returning a partial table. Labels are
    /// fetched through the core's paged label cursor — no full
    /// dictionary is materialized.
    fn resolve_labels(&self, ids: &Bound<'_, PyAny>, py: Python<'_>) -> PyResult<PyObject> {
        let core = self.core()?;
        let uel_count = core.information().map_err(to_py_err)?.uel_count;
        // Preflight from the BORROWED sequence before PyO3 extracts it:
        // accepting `Vec<u32>` would materialize the whole request
        // (4 bytes per ID) before admission. The fixed cursor costs plus
        // the incoming extraction, the validation vector (one UelId per
        // request) and the collected buffer's Vec headers must fit the
        // budget first, so a huge request against a tiny budget is
        // rejected before any request-sized allocation exists.
        const RESOLVER_FIXED_BYTES: usize = 64 * 1024;
        let len = ids.len().map_err(|_| {
            PyValueError::new_err(
                "gdxrs backend: resolve_labels expects a sized sequence of UEL IDs",
            )
        })?;
        let request_sized = RESOLVER_FIXED_BYTES
            .checked_add(len * std::mem::size_of::<u32>())
            .and_then(|n| n.checked_add(len * std::mem::size_of::<UelId>()))
            .and_then(|n| n.checked_add(len * std::mem::size_of::<Vec<u8>>()))
            .ok_or_else(|| {
                PyRuntimeError::new_err("gdxrs backend: label request layout overflow")
            })?;
        if request_sized > self.dictionary_budget_bytes {
            return Err(PyRuntimeError::new_err(format!(
                "gdxrs backend: label resolution of {len} IDs needs at least ~{request_sized} \
                 bytes of fixed and request-sized storage, exceeding the adapter \
                 dictionary budget of {} bytes",
                self.dictionary_budget_bytes
            )));
        }
        let ids: Vec<u32> = ids.extract()?;
        let mut validated: Vec<UelId> = Vec::with_capacity(ids.len());
        for id in &ids {
            if *id == 0 || *id > uel_count {
                return Err(PyValueError::new_err(format!(
                    "gdxrs backend: UEL ID {id} is zero or outside the file table (1..={uel_count})"
                )));
            }
            validated.push(
                UelId::new(*id)
                    .map_err(|e| PyValueError::new_err(format!("gdxrs backend: {e}")))?,
            );
        }
        // Admission pass: sum the label bytes before building anything.
        let mut label_bytes = 0usize;
        {
            let mut cursor = core.uel_labels().map_err(to_py_err)?;
            for id in &validated {
                let label = cursor.get(*id).map_err(to_py_err)?;
                label_bytes = label_bytes.saturating_add(label.len());
            }
        }
        // Every simultaneously live allocation is charged, including
        // validation storage, borrowed slices and conversion temporaries.
        // Per-label heap allocations carry allocator granularity.
        let granularity = |bytes: usize, count: usize| bytes.saturating_add(16 * count);
        let heap = granularity(label_bytes, ids.len());
        let estimate = RESOLVER_FIXED_BYTES
            // Validation storage: one UelId (u32) per request.
            .checked_add(4 * ids.len())
            // Collected Vec<Vec<u8>>: headers + heap payload.
            .and_then(|n| n.checked_add(ids.len() * std::mem::size_of::<Vec<u8>>()))
            .and_then(|n| n.checked_add(heap))
            // Arrow BinaryArray: i32 offsets + values payload + validity.
            .and_then(|n| n.checked_add(4 * ids.len().saturating_add(1)))
            .and_then(|n| n.checked_add(granularity(label_bytes, ids.len())))
            .and_then(|n| n.checked_add(ids.len().saturating_add(7) / 8))
            // Borrowed-slice vector used to build the array.
            .and_then(|n| n.checked_add(ids.len() * std::mem::size_of::<&[u8]>()))
            // One transient values-buffer copy during the array build.
            .and_then(|n| n.checked_add(granularity(label_bytes, ids.len())))
            // The returned UInt32 id column.
            .and_then(|n| n.checked_add(4 * ids.len()))
            .ok_or_else(|| {
                PyRuntimeError::new_err("gdxrs backend: label request layout overflow")
            })?;
        if estimate > self.dictionary_budget_bytes {
            return Err(PyRuntimeError::new_err(format!(
                "gdxrs backend: label resolution of {} IDs needs ~{estimate} bytes, \
                 exceeding the adapter dictionary budget of {} bytes",
                ids.len(),
                self.dictionary_budget_bytes
            )));
        }
        // Collection pass: exact original bytes, request order preserved.
        let mut labels: Vec<Vec<u8>> = Vec::with_capacity(validated.len());
        {
            let mut cursor = core.uel_labels().map_err(to_py_err)?;
            for id in &validated {
                let label = cursor.get(*id).map_err(to_py_err)?;
                labels.push(label.to_vec());
            }
        }
        let label_column: ArrayRef = Arc::new(BinaryArray::from(
            labels.iter().map(|b| b.as_slice()).collect::<Vec<_>>(),
        ));
        let id_column: ArrayRef = Arc::new(UInt32Array::from(ids));
        let schema = Schema::new(vec![
            Field::new("uel_id", arrow::datatypes::DataType::UInt32, false),
            Field::new("label", arrow::datatypes::DataType::Binary, false),
        ]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![id_column, label_column])
            .map_err(|e| PyRuntimeError::new_err(format!("record batch error: {e}")))?;
        Ok(batch.into_pyarrow(py)?.into_any())
    }

    /// Resolve label strings to their original one-based UEL IDs with
    /// exact (case-sensitive, byte-for-byte after UTF-8 encoding)
    /// matching, without materializing the dictionary. Entry `i` is the
    /// ID of `labels[i]`, or None when the label is not in the file.
    fn label_ids(&self, labels: &Bound<'_, PyAny>) -> PyResult<Vec<Option<u32>>> {
        let core = self.core()?;
        // Preflight from the borrowed list before extraction, mirroring
        // resolve_labels: the UTF-8 payload bytes are counted by walking
        // the borrowed list (no string is extracted yet), and every
        // simultaneously live copy is charged — the extracted Vec<String>
        // heap payloads and the core's lookup buffer, each with per-string
        // allocation granularity.
        const LOOKUP_FIXED_BYTES: usize = 64 * 1024;
        let list = labels.downcast::<pyo3::types::PyList>().map_err(|_| {
            PyValueError::new_err("gdxrs backend: label_ids expects a list of labels")
        })?;
        let len = list.len();
        let mut payload_bytes = 0usize;
        for item in list {
            let string = item.downcast::<pyo3::types::PyString>().map_err(|_| {
                PyValueError::new_err("gdxrs backend: label_ids expects labels to be strings")
            })?;
            let text = string
                .to_str()
                .map_err(|_| {
                    PyValueError::new_err("gdxrs backend: label_ids expects labels to be strings")
                })?
                .len();
            payload_bytes = payload_bytes.saturating_add(text + 1);
        }
        let granularity = |bytes: usize| bytes.saturating_add(16 * len);
        let request_sized = LOOKUP_FIXED_BYTES
            .checked_add(len * (std::mem::size_of::<String>() + std::mem::size_of::<usize>()))
            // Two simultaneous payload copies: the extracted Vec<String>
            // heap and the core's lookup buffer.
            .and_then(|n| n.checked_add(granularity(payload_bytes)))
            .and_then(|n| n.checked_add(granularity(payload_bytes)))
            .ok_or_else(|| {
                PyRuntimeError::new_err("gdxrs backend: label request layout overflow")
            })?;
        if request_sized > self.dictionary_budget_bytes {
            return Err(PyRuntimeError::new_err(format!(
                "gdxrs backend: label lookup of {len} labels needs at least ~{request_sized} \
                 bytes of fixed and request-sized storage, exceeding the adapter \
                 dictionary budget of {} bytes",
                self.dictionary_budget_bytes
            )));
        }
        let labels: Vec<String> = labels.extract()?;
        let encoded: Vec<Vec<u8>> = labels.iter().map(|l| l.as_bytes().to_vec()).collect();
        let refs: Vec<&[u8]> = encoded.iter().map(|b| b.as_slice()).collect();
        let matches = core.lookup_labels(&refs).map_err(to_py_err)?;
        let mut out = Vec::with_capacity(labels.len());
        for ordinal in 0..matches.len() {
            let id = matches
                .get(ordinal)
                .ok_or_else(|| PyRuntimeError::new_err("gdxrs backend: missing lookup result"))?
                .id;
            out.push(id.map(UelId::get));
        }
        Ok(out)
    }

    /// Read one symbol as an Arrow RecordBatch, mirroring the legacy
    /// backend's `read_arrow`. `key_mode="enum"` (default) emits Arrow
    /// dictionary key arrays over the full UEL table with 0-based raw UEL
    /// indices, admitted against `dictionary_budget_bytes` before any
    /// decoding; `key_mode="uel_id"` emits the original one-based UEL IDs
    /// as plain UInt32 key columns and never decodes or materializes any
    /// label. The value column is Float64 with the deliberate projection
    /// documented in the adapter contract. `key_filter` labels match
    /// exactly (UTF-8 encoded, byte-for-byte) in either mode; an empty
    /// resolved ID set yields zero rows with the planned schema. `n_rows`
    /// is a global accepted-row limit. Parallel reads are not implemented
    /// for this backend yet (task 046); `threads` must be None/0/1.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, key_names=None, value_field=None, key_filter=None, n_rows=None, threads=None, key_mode=None))]
    fn read_arrow(
        &self,
        name: &str,
        key_names: Option<Vec<String>>,
        value_field: Option<String>,
        key_filter: Option<Vec<(usize, Vec<String>)>>,
        n_rows: Option<usize>,
        threads: Option<usize>,
        key_mode: Option<String>,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let core = self.core()?;
        if let Some(n) = threads {
            if n > 1 {
                return Err(PyValueError::new_err(
                    "gdxrs backend: parallel reads are not implemented yet (task 046); \
                     use threads=None",
                ));
            }
        }
        let handle = core
            .find_symbol(name.as_bytes())
            .map_err(to_py_err)?
            .ok_or_else(|| not_found(name))?;
        let metadata = handle.metadata().map_err(to_py_err)?;
        let dim = usize::from(metadata.dimension);
        let kind = metadata.kind;
        let field = match value_field.as_deref() {
            Some(s) => parse_field(s)?,
            None => ValueField::Level,
        };
        let mut filters = Vec::new();
        for (dim_idx, labels) in key_filter.unwrap_or_default() {
            if dim_idx >= dim {
                return Err(PyRuntimeError::new_err(format!(
                    "key filter dimension {dim_idx} out of range (symbol has {dim} dimensions)"
                )));
            }
            filters.push((dim_idx, labels));
        }
        let id_mode = match key_mode.as_deref() {
            None | Some("enum") => false,
            Some("uel_id") => true,
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "key_mode must be 'enum' or 'uel_id', got {other:?}"
                )))
            }
        };
        // Reset the per-scan counters, then admit the mode A dictionary
        // before decoding or accumulating any record: a rejected admission
        // must not have done the work first (rejected reads report zero
        // decoded rows). The bounded ID mode decodes no labels and needs
        // no dictionary admission.
        *self.statistics.lock().unwrap() = ProjectionStats::default();
        if !id_mode {
            self.uel_strings()?;
        }
        let resolved = self.resolve_filters(&filters)?;
        // A conjunction containing an empty allowed set cannot match a
        // record; a zero limit likewise reads nothing. Both still return
        // the planned schema.
        let empty = n_rows == Some(0) || resolved.iter().any(|(_, ids)| ids.is_empty());
        // Sets, parameters and aliases store only the level field; a
        // non-level request reads the records and projects zero values,
        // matching the legacy backend's field defaults.
        let fieldless = !matches!(kind, SymbolType::Variable | SymbolType::Equation)
            && field != ValueField::Level;
        let mut keys: Vec<Vec<i32>> = vec![Vec::new(); dim];
        let mut values: Vec<Option<f64>> = Vec::new();
        if !empty {
            let clauses: Vec<DimensionSelection<'_>> = resolved
                .iter()
                .map(|(d, ids)| DimensionSelection {
                    dimension: *d as u8,
                    keys: KeySelection::AnyOf(ids),
                })
                .collect();
            let selection = core.selection(&handle, &clauses).map_err(to_py_err)?;
            let options = ScanOptions {
                limit: n_rows.map(|n| n as u64),
                value_projection: match kind {
                    SymbolType::Variable | SymbolType::Equation => {
                        ValueProjection::Mask(1u8 << field.index())
                    }
                    _ => ValueProjection::All,
                },
                ..core.scan_options()
            };
            let mut cursor = core
                .scan_selected(&handle, &selection, options)
                .map_err(to_py_err)?;
            let mut stats = ProjectionStats::default();
            loop {
                let batch = match cursor.next_batch().map_err(to_py_err)? {
                    Some(batch) => batch,
                    None => break,
                };
                for (d, key_column) in keys.iter_mut().enumerate().take(dim) {
                    let column = batch.key_column(d as u8).ok_or_else(|| {
                        PyRuntimeError::new_err("gdxrs backend: missing key column")
                    })?;
                    key_column.extend(column.iter().map(|id| id.get() as i32));
                }
                if fieldless {
                    // Zero-valued projection; do not touch the level column.
                    values.resize(values.len() + batch.len(), Some(0.0));
                } else if let Some(column) = batch.value_column(field) {
                    for value in column {
                        values.push(project_value(*value, &mut stats)?);
                    }
                } else if !batch.is_empty() {
                    return Err(PyRuntimeError::new_err(
                        "gdxrs backend: missing value column",
                    ));
                }
            }
            stats.decoded_rows = cursor.statistics().decoded_rows;
            cursor.close().map_err(to_py_err)?;
            *self.statistics.lock().unwrap() = stats;
        }
        drop(handle);
        let n_uels = if id_mode {
            0
        } else {
            self.uel_strings()?.len()
        };
        let batch = to_record_batch(
            self,
            kind,
            dim,
            &mut keys,
            &mut values,
            key_names,
            field,
            n_uels,
            id_mode,
        )?;
        Ok(batch.into_pyarrow(py)?.into_any())
    }
}

/// Construct the adapter Float64 result directly from the typed core value
/// (contract §3). EPS becomes positive zero (bits 0), losing EPS identity by
/// design and counted; UNDEF/NA and acronyms (Preserve) become `null`,
/// acronyms counted; ±INF become ±inf; ordinary numbers convert exactly.
fn project_value(value: Value, stats: &mut ProjectionStats) -> PyResult<Option<f64>> {
    match value {
        Value::NumberBits(bits) => Ok(Some(f64::from_bits(bits))),
        Value::Special(SpecialValue::Epsilon) => {
            stats.eps_to_zero += 1;
            Ok(Some(0.0))
        }
        Value::Special(SpecialValue::Undefined) | Value::Special(SpecialValue::NotAvailable) => {
            Ok(None)
        }
        Value::Special(SpecialValue::PositiveInfinity) => Ok(Some(f64::INFINITY)),
        Value::Special(SpecialValue::NegativeInfinity) => Ok(Some(f64::NEG_INFINITY)),
        Value::Acronym { .. } => {
            stats.acronyms_to_null += 1;
            Ok(None)
        }
    }
}

fn field_name(field: ValueField) -> String {
    match field {
        ValueField::Level => "level",
        ValueField::Marginal => "marginal",
        ValueField::Lower => "lower",
        ValueField::Upper => "upper",
        ValueField::Scale => "scale",
    }
    .to_string()
}

fn parse_field(s: &str) -> PyResult<ValueField> {
    match s.to_ascii_lowercase().as_str() {
        "level" => Ok(ValueField::Level),
        "marginal" => Ok(ValueField::Marginal),
        "lower" => Ok(ValueField::Lower),
        "upper" => Ok(ValueField::Upper),
        "scale" => Ok(ValueField::Scale),
        other => Err(PyRuntimeError::new_err(format!(
            "unknown value field {other:?} (expected level/marginal/lower/upper/scale)"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn to_record_batch(
    reader: &GdxrsReader,
    kind: SymbolType,
    dim: usize,
    keys: &mut [Vec<i32>],
    values: &mut Vec<Option<f64>>,
    key_names: Option<Vec<String>>,
    field: ValueField,
    n_uels: usize,
    id_mode: bool,
) -> PyResult<RecordBatch> {
    let mut names: Vec<String> = match &key_names {
        Some(names) if names.len() == dim => names.clone(),
        _ => (0..dim).map(|i| format!("dim_{i}")).collect(),
    };
    let value_name = match kind {
        SymbolType::Variable | SymbolType::Equation => field_name(field),
        _ => "value".to_string(),
    };
    if names.contains(&value_name) {
        return Err(PyRuntimeError::new_err(format!(
            "column name collision: key column {value_name:?} conflicts with the value column"
        )));
    }
    names.push(value_name.clone());

    let dict_values = if id_mode {
        None
    } else {
        Some(reader.uel_array()?)
    };
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(dim + 1);
    for key_column in keys.iter_mut().take(dim) {
        let column = std::mem::take(key_column);
        if id_mode {
            // Bounded ID mode: the original one-based UEL IDs as plain
            // UInt32. No labels are decoded and no dictionary exists, so
            // batch-local category codes can never be reinterpreted.
            columns.push(Arc::new(UInt32Array::from(
                column
                    .into_iter()
                    .map(|k| u32::try_from(k).ok())
                    .collect::<Vec<_>>(),
            )) as ArrayRef);
            continue;
        }
        let dict_values = dict_values.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("gdxrs backend: dictionary missing in enum mode")
        })?;
        let dict: ArrayRef = if n_uels > 0 && n_uels <= i16::MAX as usize + 1 {
            let indices = Int16Array::from(
                column
                    .into_iter()
                    .map(|k| (k - 1) as i16)
                    .collect::<Vec<_>>(),
            );
            Arc::new(DictionaryArray::new(
                indices,
                dict_values.clone() as ArrayRef,
            ))
        } else if n_uels <= i32::MAX as usize {
            let indices = Int32Array::from(column.into_iter().map(|k| k - 1).collect::<Vec<_>>());
            Arc::new(DictionaryArray::new(
                indices,
                dict_values.clone() as ArrayRef,
            ))
        } else {
            let indices = UInt32Array::from(
                column
                    .into_iter()
                    .map(|k| (k - 1) as u32)
                    .collect::<Vec<_>>(),
            );
            Arc::new(DictionaryArray::new(
                indices,
                dict_values.clone() as ArrayRef,
            ))
        };
        columns.push(dict);
    }
    columns.push(Arc::new(Float64Array::from(std::mem::take(values))) as ArrayRef);
    let fields: Vec<Field> = names
        .iter()
        .zip(&columns)
        .enumerate()
        .map(|(index, (name, col))| Field::new(name.clone(), col.data_type().clone(), index == dim))
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| PyRuntimeError::new_err(format!("record batch error: {e}")))
}
