# Architecture

polars-gdx is a Rust extension module exposed to Python via PyO3/maturin, built
on three workspace crates:

```
crates/gdx-sys        raw FFI bindings to the vendored GDX C library
crates/gdx            safe RAII wrapper (global lock, lazy UEL interning)
crates/polars-gdx     PyO3 extension: symbol listing + Arrow IPC streaming
python/polars_gdx     Python layer: scan_gdx / list_symbols
```

## Data flow

1. `scan_gdx(path, symbol=...)` opens the file via `gdx-sys`, reads the symbol
   table, and derives a schema (key columns + one value column).
2. The LazyFrame is registered with Polars through `register_io_source`, so
   reads happen only on collection.
3. On collection, the extension reads the symbol through `gdx` into Arrow
   RecordBatch buffers, streamed to Python as Arrow IPC bytes and handed to
   Polars zero-copy via pyarrow.
4. Key filters (explicit `key_filter` and pushdown-extracted key equality
   predicates) are resolved to UEL intern indexes and applied **inside the raw
   read loop**: non-matching records never reach the Arrow buffers.

## GDX library provenance

The vendored C sources under `crates/gdx-sys/third_party/gdx` come from
[lolow/gdxcomp](https://github.com/lolow/gdxcomp), which in turn builds on the
official [GAMS-dev/gdx](https://github.com/GAMS-dev/gdx) (MIT licensed). The
build uses CMake; no submodule or GAMS installation is needed.

## Compatibility and correctness

- `crates/gdx/tests` includes an oracle test (values compared against a
  reference reader) and a roundtrip test.
- `tests/test_lazy.py` exercises the Python layer against a real GDX fixture
  (`tests/data/trnsport.gdx`).
- `benchmarks/bench_vs_gamsapi.py` compares read performance against the
  official `gamsapi` package on a generated multi-million-record fixture.
