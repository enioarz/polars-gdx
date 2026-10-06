# polars-gdx

Lazy, prefiltered Polars access to [GAMS GDX](https://github.com/GAMS-dev/gdx) files — no GAMS installation required.

- `scan_gdx(path, symbol=...)` returns a `pl.LazyFrame`; nothing is read until `collect()`, and column projection is pushed down via Polars' IO-source interface. `read_gdx(...)` is the eager convenience wrapper (`scan_gdx(...).collect()`, same arguments).
- **Key columns are Polars `Enum` typed over the file's UEL table** (requires Polars ≥ 2.0). The Rust layer emits Arrow dictionary arrays whose values are the full UEL table in file order and whose indices are the raw 0-based UEL numbers, so Polars reinterprets the buffers **zero-copy**: no per-record string materialisation, and `==`/`is_in`/`join`/`group_by` on keys run as integer category comparisons. Use `.cast(pl.String)` when you need string operations on keys.
- Predicates on key columns (`key_filter={dim_index: allowed_labels}`, `filter(pl.col(key) == label)` or `filter(pl.col(key).is_in(labels))`) are folded into a **native, index-based prefilter inside the Rust read loop**, so unwanted records are never materialized — unlike the official `gamsapi` reader, which must load everything into pandas first. The prefilter runs inside a bulk C callback (`gdxDataReadRawFastEx`): one FFI crossing for the whole symbol, with early termination for `head(n)`.
- `threads=` (on `scan_gdx`/`read_gdx`) parallelises the raw read across independent file handles. With a first-dimension filter, each worker scans a contiguous range of the first key dimension and whole non-matching ranges are skipped without touching the file. Without one, the data section is split by *file position* at exact record boundaries: a cached restart-position index (one cheap sequential pass, built once per file+symbol) provides the boundaries, and each worker decodes only its own byte range. In both cases results are concatenated in range order so record order matches the serial read exactly, and any boundary/count mismatch falls back to a verified serial read. `threads="auto"` (recommended) uses all cores for symbols of 5M+ records and stays serial for smaller ones. Filters on non-leading dimensions still decode every record within each worker's range, but the decode is now spread across all workers instead of the single-threaded wall-time floor. Block-compressed GDX files (written with GAMS compression) are fully supported: record checkpoints become (compressed-block start, offset-in-block) pairs, so workers resume mid-block exactly and the span-seek path skips the decompression of the whole prefix before a first-dimension filter's window.
- `read_domains(path, symbol=...)` lists the unique labels actually used per index dimension in a single bulk C scan (`gdxGetDomainElements`) — the cost of `unique()` over a key column without reading or materialising any records, valuable on very large symbols.
- Under the hood: hand-written FFI to the vendored MIT-licensed GDX C library, extracted from [lolow/gdxcomp](https://github.com/lolow/gdxcomp), building on [GAMS-dev/gdx](https://github.com/GAMS-dev/gdx).

## Usage

```python
import polars as pl
from polars_gdx import scan_gdx, list_symbols

list_symbols("trnsport.gdx").filter(pl.col("type") == "Param")

# lazy: nothing read yet
x = scan_gdx("trnsport.gdx", symbol="x")

# read only the seattle rows, skipping all others natively
x.filter(pl.col("dim_0") == "seattle").collect()

# eager convenience: read_gdx = scan_gdx(...).collect()
from polars_gdx import read_gdx
df = read_gdx("trnsport.gdx", symbol="x", key_filter={0: ["seattle"]})

# large symbols: parallel raw read across all cores
big = scan_gdx("huge.gdx", symbol="x", threads="auto")

# unique labels per index dimension without reading the records:
from polars_gdx import read_domains
read_domains("trnsport.gdx", symbol="x")
```

### Label matching

Key labels (`key_filter` and `filter(pl.col(key) == label)` predicates) are matched **exactly** (case-sensitively, byte-for-byte as stored in the file). The labels in the produced frame always keep the file's original casing.

### File handle lifetime

`scan_gdx` keeps the GDX file open until the returned `LazyFrame` is collected and released (Polars may read the source more than once). Use `read_gdx`, or an explicit `Reader` with its `close()` / context-manager support, to release the handle eagerly — relevant on Windows, where an open handle locks the file.

## Benchmark

vs [`gamsapi`](https://pypi.org/project/gamsapi/) 54.5.0 (pandas-backed), on a 2M-record parameter (80 MB GDX); best of 5 runs. See `benchmarks/bench_vs_gamsapi.py` (run as a CI job on every push).

| Scenario | polars-gdx | gamsapi (pandas) |
| --- | ---: | ---: |
| Full read (2M rows) | 0.117 s | 0.086 s |
| `filter(dim_0 == label)` — 1000 of 2M rows | **0.006 s** | 0.096 s¹ |
| Explicit `key_filter` — 1000 rows | **0.005 s** | — |
| `head(1000)` of 2M rows | **0.004 s** | 0.101 s¹ |

¹ gamsapi has no lazy loading or pushdown: it must materialize the whole symbol, then filter. The advantage of polars-gdx grows with predicate selectivity; `head(n)`/`n_rows` terminates the native read after n records.

## Build

```sh
maturin develop --release   # builds the vendored GDX C library + extension
pytest
```

Requires a C++17 compiler and CMake ≥ 3.5 at build time; runtime has no GAMS dependency.

## Layout

```text
crates/gdx-sys        # raw FFI; builds vendored GDX-dev/gdx via CMake
crates/gdx            # safe RAII wrapper (global lock; lazy UEL interning)
crates/polars-gdx     # PyO3 extension: symbol listing + Arrow streaming with prefiltering
python/polars_gdx     # Python layer: scan_gdx / list_symbols
```
