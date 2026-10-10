use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use arrow::array::{
    ArrayRef, DictionaryArray, Float64Array, Int16Array, Int32Array, StringArray, UInt32Array,
};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_pyarrow::IntoPyArrow;
use gdx::{GdxFile, SymbolInfo, SymbolType, ValueField};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn to_py_err(e: gdx::GdxError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Cold-cache dim-0 filters only build the restart index (one sequential
/// pass ≈ a serial read) when the filter's span sits far enough into the
/// file that the UEL-range path would decode nearly everything anyway:
/// the filter's highest UEL number over the file's UEL count approximates
/// how far into the data that span starts. Below this threshold the
/// UEL-range path is already cheap; above it the index pass pays for
/// itself by letting every later filtered read seek straight to the span.
const LATE_SPAN_MIN_KEY_FRACTION: f64 = 0.75;

fn closed_err() -> PyErr {
    PyRuntimeError::new_err("reader is closed")
}

/// `GdxFile` holds a raw FFI pointer, so it is structurally `!Send`/`!Sync`.
/// Python entry points retain the GIL, serializing access to the inner
/// `RefCell` caches. The GDX mutex protects native operations, but does not
/// cover every cache access. Do not release the GIL or declare free-threaded
/// support without adding synchronization for the whole handle and its caches.
/// The `Option` wrapper allows releasing the
/// file handle early via `close()`; dropping it runs `GdxFile`'s own `Drop`
/// (close + free under the global lock), so no extra cleanup is needed here.
struct SendGdxFile(GdxFile);
unsafe impl Send for SendGdxFile {}
unsafe impl Sync for SendGdxFile {}

/// Read-only GDX file handle.
///
/// The symbol table is parsed eagerly; record data is read on demand per
/// symbol. Reads are serialized by the global GDX lock; nothing is read
/// beyond what is requested. The underlying GDX object is released when the
/// reader is dropped, or earlier via [`close`](Reader::close); every method
/// raises `RuntimeError("reader is closed")` afterwards.
#[pyclass]
struct Reader {
    file: Option<SendGdxFile>,
    /// Original path; parallel reads open independent handles from it.
    path: String,
    /// Cached dictionary values (the UEL table) and its Arrow form, built once per file.
    uel_strings: Mutex<Option<Arc<[String]>>>,
    uel_array: Mutex<Option<Arc<StringArray>>>,
}

/// Symbol metadata: (name, type_str, dim, records, domains, text).
type SymbolTuple = (String, String, usize, usize, Vec<String>, String);

#[pymethods]
impl Reader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let file = GdxFile::open(path).map_err(to_py_err)?;
        Ok(Self {
            file: Some(SendGdxFile(file)),
            path: path.to_string(),
            uel_strings: Mutex::new(None),
            uel_array: Mutex::new(None),
        })
    }

    /// Release the underlying GDX file handle immediately (idempotent).
    /// Any later use of this reader raises `RuntimeError`.
    fn close(&mut self) {
        self.file = None;
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
    /// `i + 1`. Used by the Python layer to translate Polars predicates into
    /// native key filters.
    fn uel_table(&self) -> PyResult<Vec<String>> {
        if self.file.is_none() {
            return Err(closed_err());
        }
        Ok(self.uel_strings()?.iter().cloned().collect())
    }

    /// List all symbols: (name, type, dim, record_count, domains, text).
    fn symbols(&self) -> PyResult<Vec<SymbolTuple>> {
        let Some(file) = self.file.as_ref() else {
            return Err(closed_err());
        };
        Ok(file.0.symbols().iter().map(symbol_tuple).collect())
    }

    /// The unique UEL numbers (1-based) used by one dimension of a symbol,
    /// in UEL order — the raw-index equivalent of `unique(dim)`, obtained
    /// with a single bulk C scan that materialises no records.
    fn domain_elements(&self, name: &str, dim_pos: usize) -> PyResult<Vec<i32>> {
        let Some(file) = self.file.as_ref() else {
            return Err(closed_err());
        };
        let info = file
            .0
            .symbol(name)
            .ok_or_else(|| PyRuntimeError::new_err(format!("symbol {name:?} not found")))?;
        file.0.domain_elements(info, dim_pos).map_err(to_py_err)
    }

    /// Resolve key-filter labels to raw UEL indices (exact match), then
    /// restrict each dimension's index set to the UELs actually used by that
    /// dimension of the symbol (bulk C domain scan, no record read). This is
    /// `resolve_uel_indices` plus a used-UELS intersection: predicates whose
    /// label exists globally but never in that dimension drop to an empty
    /// filter — a guaranteed-empty read detected before touching the data.
    fn resolve_uel_indices_used(
        &self,
        filters: Vec<(usize, Vec<String>)>,
        symbol: &str,
    ) -> PyResult<Vec<(usize, Vec<i32>)>> {
        let Some(file) = self.file.as_ref() else {
            return Err(closed_err());
        };
        let info = file
            .0
            .symbol(symbol)
            .ok_or_else(|| PyRuntimeError::new_err(format!("symbol {symbol:?} not found")))?;
        let as_sets: Vec<(usize, std::collections::HashSet<String>)> = filters
            .into_iter()
            .map(|(d, labels)| (d, labels.into_iter().collect()))
            .collect();
        let mut resolved = self.resolve_uel_indices(&as_sets)?;
        for (d, idxs) in resolved.iter_mut() {
            if idxs.is_empty() {
                continue;
            }
            let used = file.0.domain_elements(info, *d).map_err(to_py_err)?;
            if used.is_empty() {
                continue;
            }
            let used_set: std::collections::HashSet<i32> = used.into_iter().collect();
            idxs.retain(|&i| used_set.contains(&i));
            if idxs.is_empty() {
                // Label never used in this dimension: no record can match.
                continue;
            }
            if idxs.len() == 1 {
                // Single-index fast path in the read loop.
                continue;
            }
        }
        Ok(resolved)
    }

    /// Read one symbol as Arrow IPC (Feather V2) bytes, with optional key
    /// prefiltering applied inside the native read loop.
    ///
    /// - `key_names`: optional per-dimension column names (default `dim_0..`).
    /// - `value_field`: which of level/marginal/lower/upper/scale to emit for
    ///   Variables/Equations (default `level`).
    /// - `key_filter`: list of `(dim_index, allowed_labels)` pairs; records
    ///   with a key outside the allowed set are skipped before materialising.
    ///   Labels are matched exactly (case-sensitively).
    ///
    /// Returns an `arrow` RecordBatch (zero-copy into Python via PyCapsule).
    ///
    /// - `n_rows`: optional row limit; the native read stops early once that
    ///   many (post-prefilter) records have been stored.
    /// - `threads`: when > 1 and no row limit is set, the symbol is read by
    ///   that many independent file handles in parallel, each scanning a
    ///   contiguous range of the first-dimension UEL space. Results are
    ///   concatenated in range order, so record order matches the serial
    ///   read exactly. None/0/1 means serial.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (name, key_names=None, value_field=None, key_filter=None, n_rows=None, threads=None))]
    fn read_arrow(
        &self,
        name: &str,
        key_names: Option<Vec<String>>,
        value_field: Option<String>,
        key_filter: Option<Vec<(usize, Vec<String>)>>,
        n_rows: Option<usize>,
        threads: Option<usize>,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let Some(file) = self.file.as_ref() else {
            return Err(closed_err());
        };
        let info = file
            .0
            .symbol(name)
            .ok_or_else(|| PyRuntimeError::new_err(format!("symbol {name:?} not found")))?;
        let filters = parse_filters(key_filter, info.dim)?;
        let field = match value_field.as_deref() {
            Some(s) => Some(parse_field(s)?),
            None => None,
        };
        // Fast vectorised path: raw UEL indices (no label strings) with an
        // optional index-based prefilter; keys become Arrow dictionary arrays.
        // Note: a filter whose labels resolve to no UEL indices (label not
        // in file) stays present with an empty index set, which matches no
        // records — the correct zero-row semantics.
        let index_filters = if filters.is_empty() {
            Vec::new()
        } else {
            self.resolve_uel_indices(&filters)?
        };
        let vfield = field.unwrap_or(ValueField::Level);
        // A conjunction containing an empty allowed set cannot match a record.
        // Skip both serial decoding and parallel restart-index construction.
        if n_rows == Some(0) || index_filters.iter().any(|(_, allowed)| allowed.is_empty()) {
            let mut data = gdx::RawSymbolData {
                keys: vec![Vec::new(); info.dim],
                values: Vec::new(),
            };
            let batch = to_record_batch(self, info, &mut data, key_names, vfield)?;
            return Ok(batch.into_pyarrow(py)?.into_any());
        }

        // Membership bitmaps per filtered dimension: one bit per UEL number,
        // so the hot-path check per record is a single load instead of a
        // binary search. Records are stored sorted by key indices, so when
        // dimension 0 is filtered the scan can stop outright once it moves
        // past that dimension's highest allowed UEL.
        let n_uels = self.uel_strings()?.len();
        let bitmaps = index_filters
            .iter()
            .map(|(d, allowed)| {
                let mut bm = vec![0u64; n_uels / 64 + 1];
                for &i in allowed {
                    if i > 0 {
                        bm[(i - 1) as usize / 64] |= 1u64 << ((i - 1) % 64);
                    }
                }
                (*d, bm)
            })
            .collect::<Vec<_>>();
        // Highest allowed UEL in dimension 0 (0 = no dim-0 filter): the scan
        // terminates once it passes this key, because records are sorted.
        let stop_limit = index_filters
            .iter()
            .find(|(d, _)| *d == 0)
            .and_then(|(_, allowed)| allowed.last().copied())
            .unwrap_or(0);
        let n_filters = bitmaps.len();
        let bitmaps = std::sync::Arc::new(bitmaps);
        let make_pred = move || {
            let bitmaps = std::sync::Arc::clone(&bitmaps);
            move |idx_slice: &[i32]| -> gdx::RecordAction {
                for &(d, ref bm) in bitmaps.iter() {
                    let k = idx_slice[d];
                    if k <= 0 {
                        return gdx::RecordAction::Skip;
                    }
                    let u = (k - 1) as usize;
                    if !bm.get(u / 64).is_some_and(|w| w & (1u64 << (u % 64)) != 0) {
                        // First-dimension key beyond the filter's highest
                        // allowed UEL: records are stored sorted by key, so
                        // every remaining record is also beyond the filter.
                        if d == 0 && k > stop_limit {
                            return gdx::RecordAction::Stop;
                        }
                        return gdx::RecordAction::Skip;
                    }
                }
                gdx::RecordAction::Accept
            }
        };
        let mut data = match (threads.unwrap_or(0) > 1, n_rows) {
            (true, None) => {
                let builder = move || {
                    Box::new(make_pred()) as Box<dyn Fn(&[i32]) -> gdx::RecordAction + Send>
                };
                // Dim-0 filter span: [min, max] of allowed UEL numbers.
                let span = index_filters.iter().find(|(d, _)| *d == 0).map(|(_, a)| {
                    (
                        a.first().copied().unwrap_or(0),
                        a.last().copied().unwrap_or(0),
                    )
                });
                if let Some((min, max)) = span {
                    // A first-dimension filter constrains records to a
                    // contiguous key range. Prefer the span-seek path: the
                    // cached restart index knows the exact byte window of
                    // the key span, so workers decode only the relevant
                    // bytes instead of the whole prefix before the span.
                    // Building the index costs one sequential pass, so on a
                    // cold cache only do it for late spans (where the
                    // range path decodes nearly the whole file anyway);
                    // once cached (any parallel read of the symbol) the
                    // span path is never slower.
                    // UEL numbers are assigned sequentially at write time
                    // and records are sorted by dim-0 key, so the filter's
                    // highest UEL over the UEL count approximates how far
                    // into the file the range path would have to decode.
                    let late_span = match file.0.uel_counts() {
                        Ok((uelcnt, _)) => {
                            uelcnt > 0
                                && f64::from(max) / f64::from(uelcnt) > LATE_SPAN_MIN_KEY_FRACTION
                        }
                        Err(_) => false,
                    };
                    if late_span || gdx::restart_positions_cached(&self.path, info) {
                        if let Ok(data) = gdx::read_symbol_raw_span_parallel(
                            &self.path,
                            info,
                            vfield,
                            &builder,
                            threads.unwrap(),
                            min,
                            max,
                        ) {
                            data
                        } else {
                            uel_range_parallel(
                                &self.path,
                                info,
                                vfield,
                                &builder,
                                threads.unwrap(),
                                min,
                                max,
                            )
                            .map_err(to_py_err)?
                        }
                    } else {
                        uel_range_parallel(
                            &self.path,
                            info,
                            vfield,
                            &builder,
                            threads.unwrap(),
                            min,
                            max,
                        )
                        .map_err(to_py_err)?
                    }
                } else {
                    // No dim-0 filter: split by file position instead. A
                    // cached restart-position index (one cheap sequential
                    // pass, built once per file+symbol) provides exact
                    // record boundaries, so each worker decodes only its
                    // own byte range and non-leading-dimension filters no
                    // longer pay the sequential-decode wall-time floor.
                    // Falls back to a verified serial read when the
                    // per-range record counts do not sum up exactly.
                    gdx::read_symbol_raw_parallel_pos(
                        &self.path,
                        info,
                        vfield,
                        &builder,
                        threads.unwrap(),
                    )
                    .map_err(to_py_err)?
                }
            }
            _ => {
                let pred_obj = make_pred();
                let pred: gdx::ActionPred<'_> = if n_filters == 0 {
                    None
                } else {
                    Some(&pred_obj)
                };
                file.0
                    .read_symbol_raw(info, vfield, pred, n_rows)
                    .map_err(to_py_err)?
            }
        };
        let batch = to_record_batch(self, info, &mut data, key_names, vfield)?;
        Ok(batch.into_pyarrow(py)?.into_any())
    }
}

/// Dim-0 UEL-range parallel read (the original range-split path): each
/// worker scans from the symbol start and stops past its range's highest
/// UEL, so a worker whose range contains no allowed UEL returns empty
/// without touching the file.
#[allow(clippy::type_complexity)]
fn uel_range_parallel(
    path: &str,
    info: &gdx::SymbolInfo,
    vfield: gdx::ValueField,
    builder: &(dyn Fn() -> Box<dyn Fn(&[i32]) -> gdx::RecordAction + Send> + Sync),
    threads: usize,
    min: i32,
    max: i32,
) -> gdx::Result<gdx::RawSymbolData> {
    let range_skip = move |lo: i32, hi: i32| hi < min || lo > max;
    gdx::GdxFile::read_symbol_raw_parallel(path, info, vfield, builder, None, threads, &range_skip)
}

impl Reader {
    /// The file's UEL table (cached), as `Arc<[String]>`.
    fn uel_strings(&self) -> PyResult<Arc<[String]>> {
        if let Some(t) = self.uel_strings.lock().unwrap().as_ref() {
            return Ok(Arc::clone(t));
        }
        let Some(file) = self.file.as_ref() else {
            return Err(closed_err());
        };
        let table = file.0.uel_table().map_err(to_py_err)?;
        *self.uel_strings.lock().unwrap() = Some(Arc::clone(&table));
        Ok(table)
    }

    /// The file's UEL table (cached) as an Arrow `StringArray`, shared as the
    /// dictionary-values array of every key column.
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

    /// Resolve filter labels to their raw UEL indices for the open file,
    /// matching exactly (case-sensitively).
    ///
    /// Labels not present in the file are simply never matched; they map to
    /// no index and the filter stays empty for that label (no error). The
    /// index vectors are sorted so the hot-path predicate can use binary
    /// search.
    fn resolve_uel_indices(
        &self,
        filters: &[(usize, HashSet<String>)],
    ) -> PyResult<Vec<(usize, Vec<i32>)>> {
        let Some(file) = self.file.as_ref() else {
            return Err(closed_err());
        };
        let label_to_index = file.0.uel_index().map_err(to_py_err)?;
        let mut out = Vec::with_capacity(filters.len());
        for (d, labels) in filters {
            let mut idxs: Vec<i32> = labels
                .iter()
                .filter_map(|l| label_to_index.get(l.as_str()).copied())
                .collect();
            idxs.sort_unstable();
            idxs.dedup();
            out.push((*d, idxs));
        }
        Ok(out)
    }
}

fn symbol_tuple(s: &SymbolInfo) -> SymbolTuple {
    (
        s.name.clone(),
        s.kind.as_str().to_string(),
        s.dim,
        s.records,
        s.domains.clone(),
        s.text.clone(),
    )
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

fn parse_filters(
    key_filter: Option<Vec<(usize, Vec<String>)>>,
    dim: usize,
) -> PyResult<Vec<(usize, HashSet<String>)>> {
    let mut out = Vec::new();
    for (dim_idx, labels) in key_filter.unwrap_or_default() {
        if dim_idx >= dim {
            return Err(PyRuntimeError::new_err(format!(
                "key filter dimension {dim_idx} out of range (symbol has {dim} dimensions)"
            )));
        }
        out.push((dim_idx, labels.into_iter().collect::<HashSet<_>>()));
    }
    Ok(out)
}

fn to_record_batch(
    reader: &Reader,
    info: &SymbolInfo,
    data: &mut gdx::RawSymbolData,
    key_names: Option<Vec<String>>,
    field: ValueField,
) -> PyResult<RecordBatch> {
    let dim = info.dim;
    let mut names: Vec<String> = match &key_names {
        Some(names) if names.len() == dim => names.clone(),
        _ => (0..dim).map(|i| format!("dim_{i}")).collect(),
    };
    let value_name = match info.kind {
        SymbolType::Variable | SymbolType::Equation => field.as_str().to_ascii_lowercase(),
        _ => "value".to_string(),
    };
    if names.contains(&value_name) {
        return Err(PyRuntimeError::new_err(format!(
            "column name collision: key column {value_name:?} conflicts with the value column"
        )));
    }
    names.push(value_name.clone());

    let uels = reader.uel_strings()?;
    let dict_values = reader.uel_array()?;
    let n_uels = uels.len();

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(dim + 1);
    for d in 0..dim {
        let keys = std::mem::take(&mut data.keys[d]);
        let dict: ArrayRef = if n_uels > 0 && n_uels <= i16::MAX as usize + 1 {
            let indices = Int16Array::from(keys.into_iter().map(|k| k as i16).collect::<Vec<_>>());
            Arc::new(DictionaryArray::new(
                indices,
                dict_values.clone() as ArrayRef,
            ))
        } else if n_uels <= i32::MAX as usize {
            let indices = Int32Array::from(keys.into_iter().map(|k| k as i32).collect::<Vec<_>>());
            Arc::new(DictionaryArray::new(
                indices,
                dict_values.clone() as ArrayRef,
            ))
        } else {
            let indices = UInt32Array::from(keys);
            Arc::new(DictionaryArray::new(
                indices,
                dict_values.clone() as ArrayRef,
            ))
        };
        columns.push(dict);
    }
    columns.push(Arc::new(Float64Array::from(std::mem::take(&mut data.values))) as ArrayRef);
    let fields: Vec<Field> = names
        .iter()
        .zip(&columns)
        .map(|(name, col)| Field::new(name.clone(), col.data_type().clone(), false))
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| PyRuntimeError::new_err(format!("record batch error: {e}")))
}

/// Python entry point: `polars_gdx._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Reader>()?;
    Ok(())
}
