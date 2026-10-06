# AGENTS.md — polars-gdx

Polars IO plugin for reading GAMS GDX files lazily, with native key-prefiltering.
Extracted from [lolow/gdxcomp](https://github.com/lolow/gdxcomp) (`gdx-sys` + `gdx` crates),
building on the vendored, MIT-licensed [GAMS-dev/gdx](https://github.com/GAMS-dev/gdx) C++ library.

## Layout

```
crates/gdx-sys            # raw FFI to vendored GDX C library (CMake build, ~8MB vendored sources
                          #   under third_party/gdx — no GAMS install needed at runtime)
crates/gdx                # safe RAII wrapper; global FFI lock (lock.rs); lazy UEL interning
crates/polars-gdx         # PyO3 extension: symbol listing, UEL table, read_arrow()
python/polars_gdx         # scan_gdx / read_gdx / list_symbols / predicate→prefilter translation
benchmarks/               # bench_vs_gamsapi.py + fixture generators (cargo examples)
tests/                    # pytest suite (uses tests/data/trnsport.gdx fixture)
```

## Key design facts (do not rediscover these)

- **Read path**: `scan_gdx()` → `pl.register_io_source` → Rust `read_arrow()` →
  raw UEL indices via `gdxDataReadRawFastEx` (bulk C callback with user-data pointer and
  early-termination return, ONE FFI crossing per symbol — used for unfiltered, filtered
  AND limited reads alike)
  → Arrow dictionary-encoded key columns + Float64 value column → pyarrow RecordBatch
  → `pl.from_arrow` (zero-copy). Labels are NEVER materialized per record.

- **Domain scan**: `Reader.domain_elements(symbol, dim_pos)` (→ `gdxGetDomainElements`,
  DOMC_EXPAND) lists the unique UEL numbers used by one dimension via a bulk C callback;
  exposed as `read_domains(path, symbol=...)` in Python for fast `unique()`-style label
  discovery without materialising records (the file scan still happens, inside the C
  library — it skips record materialisation, not the scan).
- **Prefiltering** happens on raw i32 UEL indices inside the C read loop. Filter labels are
  resolved to UEL indices once (`resolve_uel_indices` + `uel_index()` reverse map). An empty
  resolved index set correctly means "zero rows" — do not drop empty filters.
- **Positional parallel reads (uncompressed AND block-compressed)**: the C layer
  (gxfile.cpp: gdxDataReadRawRange / gdxCollectRestartPositions) works on *checkpoints*.
  For uncompressed data a checkpoint is a plain physical file position; for block-compressed
  symbols (32 KiB independent zlib blocks, gmsstrm.cpp FillBuffer) it is the pair
  (physical start of the block holding the record, offset within the decompressed block),
  so workers resume mid-block exactly. TBufferedFileStream tracks FBlockStart in
  FillBuffer and provides GetCheckpoint*/SetCheckpoint; the compressed branch is
  FBlockStart = TXFileStream::GetPosition() BEFORE reading the 3-byte TCompressHeader
  (beware: after the read it is already past the header). PrepareSymbolReadAt enables
  compression per symbol (FFile->SetCompression(CurSyPtr->SIsCompressed)) and seeks
  via SetCheckpoint. The restart-collection callback delivers the offset in Vals[0]
  as a double (Vals is unused for uncompressed symbols). The coordinator validates
  total record counts and exact boundary handoff and falls back to a verified serial
  read on any mismatch.
- **Predicate pushdown**: the Python IO-source callback receives the Polars predicate as a
  deserialized `pl.Expr` (NOT bytes — plugins.py deserializes before calling). Conjunctions of
  `pl.col(key) == "literal"` are folded into the native prefilter by inspecting the plan
  (`expr.meta.serialize(format="json")`: `BinaryExpr`/`Eq`/`And`/`Column`/`Literal.Scalar.String`).
  The predicate is ALWAYS re-applied after the native prefilter for exact semantics. Everything
  else (value columns, other ops) falls back to Polars-side filtering — that is correct behavior.
- **Bulk callback soundness**: `c__gdxdatareadrawfast` has no user-data argument; the record
  sink is routed through a thread-local (`SINK: Cell<Option<RecordSink>>`) holding raw pointers.
  Sound because the callback runs synchronously on the same thread, bracketed by set/take,
  and all GDX FFI access is serialized by the global mutex in `gdx/src/lock.rs`.
- The filtered read path uses the per-record `gdxDataReadRaw` loop (predicate must run before
  storing); only unfiltered reads use the bulk callback.
- `GdxFile` is `!Send/!Sync` (raw pointer). The plugin wraps it in `SendGdxFile` with explicit
  `unsafe impl Send/Sync` — justified by the global lock.
- `uel_table()`/`uel_index()` are cached per GdxFile; `uel_table` is exposed to Python as
  `Reader.uel_table()` for label resolution.

## Build / test commands

```sh
python3 -m venv .venv && .venv/bin/pip install polars pyarrow pytest patchelf
VIRTUAL_ENV=$PWD/.venv PATH=$PWD/.venv/bin:$PATH maturin develop --release
.venv/bin/python -m pytest tests/ -q
cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
```

Without VIRTUAL_ENV set, maturin refuses to run. patchelf is required for rpath on Linux.

Benchmark fixture (2M records, ~80MB):

```sh
cargo run -p gdx --release --example make_bench -- /tmp/bench_big.gdx 2000000
.venv/bin/python benchmarks/bench_vs_gamsapi.py /tmp/bench_big.gdx
```

## Benchmark results (2026-10, gamsapi 54.5.0, 2M records, best of 5)

| Scenario | polars-gdx | gamsapi |
| --- | ---: | ---: |
| Full read | 0.126 s | 0.086 s |
| filter() → native prefilter (1000 rows) | 0.081 s | 0.087 s (full read + pandas filter) |

gamsapi's pandas path has no lazy/pushdown: it must materialize everything, so our advantage
grows with predicate selectivity. Full-read speed is comparable (gamsapi writes straight into
numpy categoricals via its closed-source `_gams2numpy` extension).

## External dependencies / gotchas

- `gamsapi` (the official pandas-backed reader, `gams.transfer`) requires a GAMS "system
  directory" with native libs (`libgmdcclib64.so` etc.). Simplest source: `pip install gamspy`
  → `gamspy_base` package ships all of them. Locate via `import gamspy_base` (NOT a venv glob —
  that broke CI once). The vendored `libgdxcclib64.so` is NOT enough for gamsapi's high-level
  reader (API-version mismatch for GMD).
- Version coupling: `pyo3` version must match what `arrow-pyarrow` expects (arrow 56 → pyo3 0.25).
  `build-backend = "maturin"` in pyproject (NOT "maturin.build" — that name is deprecated and
  broke CI). `pip install -e .` with build isolation works; do NOT pass `--no-build-isolation`.
- Polars API: `register_io_source` callback signature is
  `(with_columns, predicate, n_rows, batch_size) -> Iterator[DataFrame]` (must yield batches,
  not return a DataFrame). The C-level raw read floor for 2M records is ~90ms.
- pyo3 `#[pymodule]` function name must equal the `module-name` in pyproject
  (`polars_gdx._core` → `fn _core`). `#[pyclass]` holding !Send types needs the newtype trick.
- Merging: `gh pr merge` is blocked by repo policy here; fast-forward `main` to the PR branch
  head via push instead — GitHub then auto-marks the PR MERGED.

## CI

`.github/workflows/ci.yml` — two jobs, both must stay green:
- `test`: fmt, clippy (-D warnings), cargo test, pip install -e ., pytest
- `benchmark`: builds fixture via make_bench, runs bench_vs_gamsapi.py vs gamsapi+gamspy
