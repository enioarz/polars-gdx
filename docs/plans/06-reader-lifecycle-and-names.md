# Plan 06 — Reader lifecycle (file handle) + key/value name-collision fix

Two small, independent robustness fixes; they only share the `Reader`/
`scan_gdx` surface, so they ship as one plan.

## Part A — GDX file handle stays open for the LazyFrame's lifetime

### Problem

`scan_gdx` (python/polars_gdx/lazy.py:121) captures a `Reader` in the
IO-source closure. Nothing closes the underlying GDX file object until the
LazyFrame is collected **and garbage-collected**. Consequences:

- Long-lived LazyFrames hold open file handles indefinitely.
- On Windows the file stays locked: users cannot overwrite/delete the .gdx
  while a scan object exists.
- Many scans → many simultaneously open handles (GDX objects also hold
  memory for the parsed symbol table).

### Changes

1. Expose `Reader.close()` in crates/polars-gdx/src/lib.rs:
   - Move `GdxFile` out: change `Reader { file: SendGdxFile }` to hold
     `Option<SendGdxFile>` internally, or add
     `GdxFile::close(&self) -> Result<()>` in crates/gdx/src/reader.rs that
     performs the `c__gdxclose`/`gdxfree` pair (under the global lock) and
     makes the object inert; `Drop` then no-ops. Preferred: `Option` in the
     pyclass — `GdxFile`'s `Drop` impl already handles the cleanup, so
     `self.file = None` is sufficient and reuses existing code.
   - Methods on a closed Reader must return a clear error (new
     `PyRuntimeError("reader is closed")`), not panic or segfault — check
     every `#[pymethods]` entry (`uel_table`, `symbols`, `read_arrow`).
2. Python `scan_gdx`:
   - In `_read`, after the final `yield`, best-effort close is **not** safe:
     Polars may call `_read` more than once (e.g. after a predicate isn't
     fully pushable, or for `collect_all` re-tries). Instead:
     - add `Reader.__enter__/__exit__` or a `close()` and document that
       users may close explicitly, **and**
     - register the Reader with `weakref.finalize` on the closure object
       is unreliable (the closure itself is referenced by the LazyFrame).
   - Pragmatic minimal fix (recommended): keep the current lifetime but
     document it, add `close()` for opt-in eager release, and make the
     docstring of `scan_gdx` mention: "the underlying file stays open until
     the LazyFrame is collected and released; use `read_gdx` or explicit
     `Reader` + `close()` if you need the handle released."
   - If you go further: an explicit `scan_gdx(..., eager_close=True)` is
     **wrong** (breaks re-collection). Do not auto-close inside `_read`.

### Verification

- pytest: `scan_gdx(...)` twice from the same returned LazyFrame works
  (collect, then collect again) — regression guard against an
  over-aggressive close.
- pytest (if close() added): `r = Reader(path); r.close();` then
  `r.symbols()` raises the documented error, and `Reader(path)` on the same
  file works again afterwards (handle released; on Windows, deletion of the
  file between close and reopen must succeed — skip on other platforms).
- Existing suite green.

## Part B — key/value column name collision

### Problem

`_key_names` (python/polars_gdx/lazy.py:227-242) de-duplicates key names
against each other but not against `value_name`. If a parameter has a domain
set literally named `value` (or a variable with a domain named `level`), the
schema dict `{name: ...} | {value_name: ...}` silently overwrites the key
column (Python side: the key disappears from the schema), and/or the Arrow
`RecordBatch::try_new` fails on duplicate field names
(crates/polars-gdx/src/lib.rs:196-206), surfacing as a raw internal error.

### Changes

- Python `_key_names` gains the value name as a second parameter and skips
  it in the de-duplication loop:

  ```python
  def _key_names(symbol, domains, dim, reserved: set[str]) -> list[str]:
      names = []
      for i, d in enumerate(domains[:dim]):
          candidate = d if d and d != "*" else f"dim_{i}"
          if candidate in names or candidate in reserved:
              candidate = f"{candidate}_{i}"
          names.append(candidate)
  ```

  Call it after `value_name` is computed (requires reordering: compute
  `value_name` before `key_names` in `scan_gdx`). Also handle the (extremely
  unlikely) case where the suffixed name still collides — the current
  `f"{candidate}_{i}"` with the positional index is already unique against
  `names`, and `reserved` names get the same suffix treatment.
- Rust `to_record_batch` already names columns from the passed
  `key_names`; no Rust change needed once Python sends collision-free
  names. But add a defensive check on the Rust side: if
  `names[..dim]` contains `value_name`, return a clear
  `PyRuntimeError("column name collision ...")` instead of relying on the
  Arrow error (cheap, one `contains`).

### Verification

- New pytest with a handcrafted fixture: build a small GDX via
  `crates/gdx`'s `GdxWriter` (see `crates/gdx/src/reader.rs` parity tests or
  `make_fixture` example for the pattern) with a domain named `value`, then
  `scan_gdx` it and assert the columns are `value_0` (or whichever disambiguated
  name) plus `value`.
- Existing suite green (trnsport names never collide).

## Gotchas

- Both parts touch `python/polars_gdx/lazy.py`; Part A's `scan_gdx`
  docstring change and Part B's signature change are adjacent — fine as one
  PR.
- Do not change the public `scan_gdx`/`read_gdx` signatures.
- If Part A's `close()` lands, keep `SendGdxFile`'s unsafe impls' soundness
  comments accurate (crates/polars-gdx/src/lib.rs:14-22).
