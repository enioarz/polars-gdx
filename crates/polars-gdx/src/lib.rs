use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{ArrayRef, DictionaryArray, Float64Array, StringArray, UInt32Array};
use arrow::datatypes::{Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow_pyarrow::IntoPyArrow;
use gdx::{GdxFile, SymbolInfo, SymbolType, ValueField};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

fn to_py_err(e: gdx::GdxError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// `GdxFile` holds a raw FFI pointer, so it is structurally `!Send`/`!Sync`.
/// The `gdx` crate serializes every FFI call behind a process-global mutex,
/// which makes cross-thread access sound; all methods here take `&self` and
/// are safe under concurrent calls.
struct SendGdxFile(GdxFile);
unsafe impl Send for SendGdxFile {}
unsafe impl Sync for SendGdxFile {}

/// Read-only GDX file handle.
///
/// The symbol table is parsed eagerly; record data is read on demand per
/// symbol. Reads are serialized by the global GDX lock; nothing is read
/// beyond what is requested.
#[pyclass]
struct Reader {
    file: SendGdxFile,
}

/// Symbol metadata: (name, type_str, dim, records, domains, text).
type SymbolTuple = (String, String, usize, usize, Vec<String>, String);

#[pymethods]
impl Reader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let file = GdxFile::open(path).map_err(to_py_err)?;
        Ok(Self {
            file: SendGdxFile(file),
        })
    }

    /// The file's UEL table: entry `i` is the label of 1-based UEL number
    /// `i + 1`. Used by the Python layer to translate Polars predicates into
    /// native key filters.
    fn uel_table(&self) -> PyResult<Vec<String>> {
        self.file.0.uel_table().map_err(to_py_err)
    }

    /// List all symbols: (name, type, dim, record_count, domains, text).
    fn symbols(&self) -> Vec<SymbolTuple> {
        self.file.0.symbols().iter().map(symbol_tuple).collect()
    }

    /// Read one symbol as Arrow IPC (Feather V2) bytes, with optional key
    /// prefiltering applied inside the native read loop.
    ///
    /// - `key_names`: optional per-dimension column names (default `dim_0..`).
    /// - `value_field`: which of level/marginal/lower/upper/scale to emit for
    ///   Variables/Equations (default `level`).
    /// - `key_filter`: list of `(dim_index, allowed_labels)` pairs; records
    ///   with a key outside the allowed set are skipped before materialising.
    ///
    /// Returns an `arrow` RecordBatch (zero-copy into Python via PyCapsule).
    #[pyo3(signature = (name, key_names=None, value_field=None, key_filter=None))]
    fn read_arrow(
        &self,
        name: &str,
        key_names: Option<Vec<String>>,
        value_field: Option<String>,
        key_filter: Option<Vec<(usize, Vec<String>)>>,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        let info = self
            .file
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
        let pred: gdx::IndexPred<'_> = if index_filters.is_empty() {
            None
        } else {
            Some(&|idx: &[i32]| {
                index_filters
                    .iter()
                    .all(|&(d, ref allowed)| allowed.contains(&idx[d]))
            })
        };
        let mut data = self
            .file
            .0
            .read_symbol_raw(info, vfield, pred)
            .map_err(to_py_err)?;

        let batch = to_record_batch(self, info, &mut data, key_names, vfield)?;
        Ok(batch.into_pyarrow(py)?.into_any())
    }
}

impl Reader {
    /// Resolve filter labels to their raw UEL indices for the open file.
    ///
    /// Labels not present in the file are simply never matched; they map to
    /// no index and the filter stays empty for that label (no error).
    fn resolve_uel_indices(
        &self,
        filters: &[(usize, HashSet<String>)],
    ) -> PyResult<Vec<(usize, HashSet<i32>)>> {
        let label_to_index = self.file.0.uel_index().map_err(to_py_err)?;
        let mut out = Vec::with_capacity(filters.len());
        for (d, labels) in filters {
            let idxs: HashSet<i32> = labels
                .iter()
                .filter_map(|l| label_to_index.get(l.as_str()).copied())
                .collect();
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
    names.push(value_name.clone());

    let uels = reader.file.0.uel_table().map_err(to_py_err)?;
    let dict_values = Arc::new(StringArray::from(
        uels.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
    ));

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(dim + 1);
    for d in 0..dim {
        let indices = UInt32Array::from(std::mem::take(&mut data.keys[d]));
        let dict = DictionaryArray::new(indices, dict_values.clone() as ArrayRef);
        columns.push(Arc::new(dict) as ArrayRef);
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
