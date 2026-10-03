use std::collections::{HashMap, HashSet};
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

fn closed_err() -> PyErr {
    PyRuntimeError::new_err("reader is closed")
}

/// `GdxFile` holds a raw FFI pointer, so it is structurally `!Send`/`!Sync`.
/// The `gdx` crate serializes every FFI call behind a process-global mutex,
/// which makes cross-thread access sound; all methods here take `&self` and
/// are safe under concurrent calls. The `Option` wrapper allows releasing the
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

    /// Read one symbol as Arrow IPC (Feather V2) bytes, with optional key
    /// prefiltering applied inside the native read loop.
    ///
    /// - `key_names`: optional per-dimension column names (default `dim_0..`).
    /// - `value_field`: which of level/marginal/lower/upper/scale to emit for
    ///   Variables/Equations (default `level`).
    /// - `key_filter`: list of `(dim_index, allowed_labels)` pairs; records
    ///   with a key outside the allowed set are skipped before materialising.
    ///   Labels are matched case-insensitively (GAMS semantics).
    ///
    /// Returns an `arrow` RecordBatch (zero-copy into Python via PyCapsule).
    ///
    /// - `n_rows`: optional row limit; the native read stops early once that
    ///   many (post-prefilter) records have been stored.
    #[pyo3(signature = (name, key_names=None, value_field=None, key_filter=None, n_rows=None))]
    fn read_arrow(
        &self,
        name: &str,
        key_names: Option<Vec<String>>,
        value_field: Option<String>,
        key_filter: Option<Vec<(usize, Vec<String>)>>,
        n_rows: Option<usize>,
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
        let pred: gdx::IndexPred<'_> = match index_filters.as_slice() {
            [] => None,
            [(d, allowed)] if allowed.len() == 1 => {
                let idx = allowed[0];
                let d = *d;
                Some(&move |idx_slice: &[i32]| idx_slice[d] == idx)
            }
            many => Some(&|idx_slice: &[i32]| {
                many.iter()
                    .all(|&(d, ref allowed)| allowed.binary_search(&idx_slice[d]).is_ok())
            }),
        };
        let mut data = file
            .0
            .read_symbol_raw(info, vfield, pred, n_rows)
            .map_err(to_py_err)?;
        let batch = to_record_batch(self, info, &mut data, key_names, vfield)?;
        Ok(batch.into_pyarrow(py)?.into_any())
    }
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
    /// matching case-insensitively (ASCII-folded) the way GAMS does.
    ///
    /// Labels not present in the file are simply never matched; they map to
    /// no index and the filter stays empty for that label (no error). When
    /// several UELs differ only by case, the lowest UEL number wins. The
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
        let mut folded: HashMap<String, i32> = HashMap::with_capacity(label_to_index.len());
        for (label, &idx) in label_to_index.iter() {
            let entry = folded.entry(label.to_ascii_uppercase()).or_insert(idx);
            if idx < *entry {
                *entry = idx;
            }
        }
        let mut out = Vec::with_capacity(filters.len());
        for (d, labels) in filters {
            let mut idxs: Vec<i32> = labels
                .iter()
                .filter_map(|l| {
                    label_to_index
                        .get(l.as_str())
                        .copied()
                        .or_else(|| folded.get(&l.to_ascii_uppercase()).copied())
                })
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
