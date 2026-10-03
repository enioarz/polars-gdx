# API reference

The public Python API is re-exported from `polars_gdx`.

## `list_symbols(path) -> pl.DataFrame`

Lists every symbol in a GDX file. Columns:

| Column    | Type             | Meaning                                   |
|-----------|------------------|-------------------------------------------|
| `name`    | `String`         | Symbol name                               |
| `type`    | `String`         | `Set`, `Parameter`, `Variable`, `Equation`, `Alias` |
| `dim`     | `UInt32`         | Number of dimensions                      |
| `records` | `UInt64`         | Number of records                         |
| `domains` | `List(String)`   | Domain set names (`*` for relaxed dims)   |
| `text`    | `String`         | Explanatory text                           |

## `scan_gdx(path, *, symbol, value_field=None, key_filter=None) -> pl.LazyFrame`

Returns a `LazyFrame` over one symbol. Nothing is read until an action such as
`collect()` runs.

**Parameters**

- `path` — path to the `.gdx` file.
- `symbol` — symbol name (case-sensitive, as stored in the file). A
  `ValueError` listing available symbols is raised when it is not found.
- `value_field` — for Variable/Equation symbols: which field to load, one of
  `level`, `marginal`, `lower`, `upper`, `scale`. Defaults to `level`. Ignored
  for Set/Parameter/Alias symbols, which expose a `value` column.
- `key_filter` — optional `{dim_index: allowed_labels}` prefilter, applied
  natively inside the Rust read loop, before any record is materialised.
  Labels that do not appear in the file's UEL table are dropped; a dimension
  with no remaining labels yields zero rows.

**Schema**

Key columns are strings (Polars may materialise them as `Categorical`); the
value column is `Float64`.

**Pushdown behavior**

The LazyFrame is registered through Polars' IO-source interface, so:

- Column projection is pushed down: only selected columns are materialised.
- Predicates that reduce to conjunctions of `key_column == "label"` on key
  columns are additionally folded into the native prefilter and skipped during
  the raw read. The predicate is still re-applied by Polars for exact
  semantics. Other predicates are evaluated by Polars after the read.
- `n_rows` (e.g. `head`) is applied before column selection.

## `Reader`

The `polars_gdx.Reader` class is the underlying Rust extension type. It is
re-exported for introspection but is not part of the stable API surface;
prefer `scan_gdx` and `list_symbols`.
