# Plan 01 — Cache UEL data on the Reader (single build per file)

## Problem

Every `collect()` walks the file's full UEL table **three times**:

1. Python `_predicate_key_filter` builds `set(reader.uel_table())` — full FFI
   loop + `Vec<String>` (python/polars_gdx/lazy.py:170-171).
2. Rust `Reader.read_arrow` → `resolve_uel_indices` → `GdxFile::uel_index()` —
   second full FFI loop, plus `Ok(m.clone())` clones the whole map
   (crates/gdx/src/reader.rs:145-173).
3. Rust `to_record_batch` → `GdxFile::uel_table()` — third full FFI loop,
   plus a fresh `StringArray` allocation per read
   (crates/polars-gdx/src/lib.rs:179-187).

On files with 10^5+ UELs this dominates small reads and wastes large
allocations.

## Goal

Build the UEL table (labels, in UEL-number order) **at most once per
`GdxFile`**, derive the reverse map from it, and reuse both for all
subsequent reads.

## Changes

### crates/gdx/src/reader.rs

- Change `GdxFile`'s caches to a single forward table:

  ```rust
  uel_table: RefCell<Option<Arc<[String]>>>,       // entry i = label of UEL i+1
  uel_index: RefCell<Option<Arc<HashMap<String, i32>>>> // reverse map
  ```

- Add `GdxFile::uel_table(&self) -> Result<Arc<[String]>>`:
  - if cached, return the `Arc` clone (cheap);
  - otherwise do the existing `gdxUMUELInfo`/`gdxUMUELGet` loop once, store
    `Arc<[String]>`, and derive `uel_index` in the same pass
    (label → UEL number `i+1`; note: for duplicate labels keep the *first*
    occurrence to match current `HashMap::insert` semantics — check whether
    current code inserts later or earlier and preserve that behavior).
- `uel_index()` returns `Arc<HashMap<String, i32>>` (clone of the Arc, not of
  the map).
- Keep behavior for missing UELs (`gdxUMUELGet` != 1 → empty-string
  placeholder) exactly as today.
- The existing `uel_cache` (per-UEL interning used by the `Record`-based
  string path) can stay as-is; do not touch `read_records_raw`.

### crates/polars-gdx/src/lib.rs

- `Reader` caches `Arc<[String]>` (table) and `Arc<StringArray>` after first
  build, e.g. in `RefCell` fields; `to_record_batch` reuses the
  `StringArray` across reads.
- `resolve_uel_indices` uses the returned `Arc<HashMap<..>>` without
  cloning it.

### python/polars_gdx/lazy.py

- `Reader.uel_table()` now returns the same cached data; no Python-side
  change strictly required, but `_predicate_key_filter` should avoid the
  `set(...)` rebuild if a cheap `contains`-style helper is exposed
  (optional; only do it if trivial via a new `#[pyo3]` method
  `uel_exists(label) -> bool` or by keeping a Python-side cached frozenset
  keyed on the Reader instance).

## Verification

- `cargo test --workspace`
- `cargo clippy --workspace --all-targets -- -D warnings`, `cargo fmt --check`
- Build (`maturin develop --release`, see AGENTS.md env vars) and
  `pytest tests/`
- Benchmark sanity (optional): `make_bench` fixture + `bench_vs_gamsapi.py`
  should show full-read time unchanged or slightly better; the 1000-row
  filtered scenario should improve the most.
- Add a pytest that reads the same symbol twice and asserts the second read
  is correct (guards against a broken shared cache).

## Gotchas

- All FFI calls must stay under the global `lock()` (crates/gdx/src/lock.rs).
- `GdxFile` is `!Send`; caches are `RefCell` — fine, no new thread-safety
  surface. The Python `Reader` is wrapped in `SendGdxFile`; keep `&self`
  methods.
- Do not change record-data semantics: empty filters still mean "zero rows".
