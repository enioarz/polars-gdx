# Plan 07 — Smaller dictionary indices (i16/i32) for key columns

## Problem

Key columns are always emitted as `Dictionary(UInt32, Utf8)`
(crates/polars-gdx/src/lib.rs:188-196), regardless of the file's UEL count.
GDX files with fewer than 32k/65k unique labels (the vast majority) could
use `Dictionary(Int16, Utf8)` / `Dictionary(Int32, Utf8)`, halving (or better)
key-column memory and speeding Polars' string-dictionary ops
(group-by, join, unique on keys).

`uelcnt` is known before any record is read (`gdxUMUELInfo`, already used in
`uel_table`), so the choice is cheap and exact.

## Goal

Pick the narrowest index type that can represent the file's UEL count:
`i16` when `uelcnt <= i16 range` (note: indices are 0-based, UEL numbers
1-based, and the `store` path currently stores `u32`), else `i32`, else
`u32` as today.

## Changes

### crates/gdx/src/reader.rs

- `RawSymbolData.keys` is `Vec<Vec<u32>>` — the storage type inside the read
  loop. Two options:
  1. **Keep `u32` internally, narrow at Arrow-build time**
     (recommended, minimal change): `to_record_batch` converts each
     `Vec<u32>` to `Int16Array`/`Int32Array` when `uelcnt` fits. This pays a
     conversion pass but keeps the hot read loop untouched and the gdx
     crate's public API stable.
  2. Make the index width a parameter of the read (template the sink): more
     invasive, saves one conversion pass; only do this if benchmarks show
     the conversion is visible (it is O(records × dim) integer copies —
     cheap vs. the C read floor).
- Whatever option: do **not** change `RawSymbolData`'s public shape unless
  option 2 is taken; option 1 keeps the diff inside crates/polars-gdx.

### crates/polars-gdx/src/lib.rs — `to_record_batch`

- Determine `uelcnt` (from the cached UEL table length after Plan 01, or a
  direct `gdxUMUELInfo` call).
- Build the dictionary's value field and index array accordingly:

  ```rust
  let (indices, index_type): (ArrayRef, DataType) = if uelcnt <= 0x7FFF {
      (Arc::new(Int16Array::from(keys_i16)), DataType::Int16)
  } else if uelcnt <= 0x7FFF_FFFF {
      (Arc::new(Int32Array::from(keys_i32)), DataType::Int32)
  } else {
      (Arc::new(UInt32Array::from(keys_u32)), DataType::UInt32)
  };
  ```

  (Exact conversion mechanics depend on option 1 vs 2 above; a `cast` or a
  per-element `map` is fine.)
- The schema advertised to Polars (`scan_gdx`'s `schema=` argument,
  python/polars_gdx/lazy.py:110-112) currently says `pl.String` for keys —
  Polars casts the incoming dictionary to whatever the registered schema
  promises, so **check** whether a narrower dictionary still flows through
  `pl.from_arrow` + `register_io_source` without a silent copy or a schema
  mismatch error. If the registered schema forces String materialization,
  the win evaporates; in that case consider registering keys as
  `pl.Categorical`/`pl.Enum` instead — which is arguably the better
  representation of GDX keys anyway (fixed UEL universe). Any dtype change
  is user-visible: gate it behind a keyword (`key_dtype="auto"|"string"|"categorical"`,
  default `"auto"`) and document it.

### Fallback / interaction

- If schema constraints make narrowing impossible without changing the
  registered dtype, the fallback deliverable of this plan is: register keys
  as `pl.Categorical` (or `pl.Enum` with the UEL list as categories) and skip
  the index-width work — benchmark both and keep whichever actually helps
  Polars-side workloads (filter, group_by, join on keys). Report findings.

## Verification

- pytest suite green (the tests use `set(df["dim_0"])` etc. — dtype-agnostic,
  but `test_scan_projection_pushdown` asserts column *names*, fine).
- New pytest asserting the frame's key dtypes match the documented default.
- Memory check (optional): `pl.DataFrame.estimated_size()` before/after on
  the 2M-record fixture.
- `cargo test --workspace`, clippy, fmt.

## Gotchas

- The `(k - 1).max(0) as u32` index computation (reader.rs store path and
  filtered path) must not be "simplified" — the `max(0)` guards UEL number 0
  / negative sentinels. Keep it intact under either option.
- `register_io_source` schema mismatch produces confusing Polars errors —
  verify the round trip eagerly rather than trusting zero-copy claims.
- User-visible dtype changes need a README note and a changelog entry.
