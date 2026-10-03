# Plan 03 — True `n_rows` pushdown: stop the native read early

## Problem

`scan_gdx`'s IO-source callback applies `n_rows` **after** the whole symbol
has been read (python/polars_gdx/lazy.py:150):

```python
if n_rows is not None:
    df = df.head(n_rows)
```

So `scan_gdx(big, symbol="x").head(10).collect()` still walks all 2M records
in the C loop, materializes keys and values, then throws 99.999% away. This
is the biggest remaining gap versus the library's "read only what you need"
promise (see README lazy/prefilter claims).

The Rust read loop is fully under our control (crates/gdx/src/reader.rs,
`read_symbol_raw_locked`), so it can stop as soon as `n_rows` matching
records have been stored.

## Goal

`head(n)` / `n_rows` terminates the native read after n records (n *after*
prefiltering), so cost scales with n, not with symbol size.

## Changes

### crates/gdx — API

- Extend `read_symbol_raw` with a row limit. Preferred signature (keep the
  old one as a thin wrapper or change callers — there are few):

  ```rust
  pub fn read_symbol_raw(
      &self,
      info: &SymbolInfo,
      value_field: ValueField,
      pred: IndexPred<'_>,
      limit: Option<usize>,           // NEW
  ) -> Result<RawSymbolData>
  ```

- In `read_symbol_raw_locked`:
  - filtered path: after `pred` passes, decrement a counter; when it hits
    zero, break out of the `while` loop (still call `gdxDataReadDone`).
  - unfiltered bulk path: the `store_record` callback must be able to signal
    stop. Options:
    1. `gdxDataReadRawFast`'s callback contract — check the vendored header
       (crates/gdx-sys/third_party/gdx/src/apigenerator/include/) for the
       documented stop mechanism. If the callback returns non-zero/has a
       documented "stop" return value, use it and add that constant to
       crates/gdx-sys/src/lib.rs.
    2. If no stop mechanism exists, fall back to the per-record
       `gdxDataReadRaw` loop for limited reads (same code path as filtered;
       the bulk path is only used when there is neither pred nor limit).
  - Whichever option: `gdxDataReadDone` must still run (the probe-read code
    and all existing paths always pair start with done).

### crates/polars-gdx — exposure

- `Reader.read_arrow` gains an `n_rows: Option<usize>` parameter (add to the
  `#[pyo3(signature = ...)]` list; keep the old call shape working by giving
  it a default `None` so Python callers that don't pass it are unaffected —
  pyo3 supports defaults in the signature macro).
- Pass `limit` through to `read_symbol_raw`.

### python/polars_gdx/lazy.py

- In `_read`, pass `n_rows=n_rows` to `reader.read_arrow(...)`.
- **Keep** the `df.head(n_rows)` afterwards — it is cheap now (rows <= n) and
  guards exact semantics (e.g. filters merged from predicates still reapply
  `pred_expr`, which can only shrink, never grow).
- Careful with interaction: `pred_expr` re-filter happens *after* the read.
  If the native prefilter is the exact same predicate, head-after-filter is
  correct. If the predicate was NOT natively translatable (falls back to
  Polars-side only), applying the native limit before `pred_expr` could
  under-fill `head(n)`. Fix: only pass `n_rows` down when the predicate was
  fully folded into the native prefilter (`native is not None`) **or** when
  there is no predicate at all; otherwise read unlimited and let
  `df.filter(...).head(n)` stand. Document this choice in a comment-free way
  via a test.

## Verification

- New pytest: on `tests/data/trnsport.gdx`, `scan_gdx(...).head(2).collect()`
  equals `read_gdx(...).head(2)` row-for-row (order and values).
- New pytest with a predicate + head: `filter(dim_0 == "seattle").head(1)`
  gives exactly 1 seattle row.
- Existing pytest suite green; `cargo test --workspace`, clippy, fmt.
- Benchmark: 2M-record fixture, `head(100)` should drop from ~full-read time
  to near-instant; worth adding to README's benchmark table.

## Gotchas

- GDX record order is deterministic for a given file; head-pushdown relies
  on that (fine — the eager path also yields file order).
- The `SINK` thread-local bracketing (set/take) must stay balanced even on
  early exit.
