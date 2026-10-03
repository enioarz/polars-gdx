use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, StringArray};
use arrow::datatypes::{Field, Schema};
use arrow::ipc::writer::FileWriter;
use arrow::record_batch::RecordBatch;
use gdx::{GdxFile, SymbolInfo, SymbolType, ValueField};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

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

        let records = if filters.is_empty() {
            self.file.0.read_info(info).map_err(to_py_err)?
        } else {
            self.file
                .0
                .read_filtered(info, &|keys| {
                    filters
                        .iter()
                        .all(|&(d, ref allowed)| allowed.contains(keys[d].as_ref()))
                })
                .map_err(to_py_err)?
        };

        let bytes = to_arrow_ipc(info, &records, key_names, field)?;
        let bytes = PyBytes::new(py, &bytes);
        Ok(bytes.into_any().unbind())
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

fn to_arrow_ipc(
    info: &SymbolInfo,
    records: &[gdx::Record],
    key_names: Option<Vec<String>>,
    field: Option<ValueField>,
) -> PyResult<Vec<u8>> {
    let dim = info.dim;

    let mut names: Vec<String> = match &key_names {
        Some(names) if names.len() == dim => names.clone(),
        _ => (0..dim).map(|i| format!("dim_{i}")).collect(),
    };

    let value_name = match info.kind {
        SymbolType::Variable | SymbolType::Equation => match field {
            Some(f) => f.as_str().to_ascii_lowercase(),
            None => ValueField::Level.as_str().to_ascii_lowercase(),
        },
        _ => "value".to_string(),
    };
    names.push(value_name);

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(dim + 1);
    for d in 0..dim {
        let values: Vec<&str> = records.iter().map(|r| r.keys[d].as_ref()).collect();
        columns.push(Arc::new(StringArray::from(values)) as ArrayRef);
    }
    let idx = field
        .map(|f| f.index())
        .unwrap_or(ValueField::Level.index());
    let values: Vec<f64> = records.iter().map(|r| r.values[idx]).collect();
    columns.push(Arc::new(Float64Array::from(values)) as ArrayRef);

    let fields: Vec<Field> = names
        .iter()
        .zip(&columns)
        .map(|(name, col)| Field::new(name.clone(), col.data_type().clone(), false))
        .collect();
    let schema = Arc::new(Schema::new(fields));

    let batch = RecordBatch::try_new(schema.clone(), columns)
        .map_err(|e| PyRuntimeError::new_err(format!("record batch error: {e}")))?;
    let mut buf: Vec<u8> = Vec::new();
    let mut writer = FileWriter::try_new(&mut buf, schema.as_ref())
        .map_err(|e| PyRuntimeError::new_err(format!("arrow writer error: {e}")))?;
    writer
        .write(&batch)
        .map_err(|e| PyRuntimeError::new_err(format!("arrow write error: {e}")))?;
    writer
        .finish()
        .map_err(|e| PyRuntimeError::new_err(format!("arrow finish error: {e}")))?;
    Ok(buf)
}

/// Python entry point: `polars_gdx._core`.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Reader>()?;
    Ok(())
}
