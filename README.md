# polars-gdx

Lazy, prefiltered Polars access to [GAMS GDX](https://github.com/GAMS-dev/gdx) files — no GAMS installation required.

- `scan_gdx(path, symbol=...)` returns a `pl.LazyFrame`; nothing is read until `collect()`, and column projection is pushed down via Polars' IO-source interface. `read_gdx(...)` is the eager convenience wrapper (`scan_gdx(...).collect()`, same arguments).
- Predicates on key columns (`key_filter={dim_index: allowed_labels}`, `filter(pl.col(key) == label)` or `filter(pl.col(key).is_in(labels))`) are folded into a **native, index-based prefilter inside the Rust read loop**, so unwanted records are never materialized — unlike the official `gamsapi` reader, which must load everything into pandas first. The prefilter runs inside a bulk C callback (`gdxDataReadRawFastEx`): one FFI crossing for the whole symbol, with early termination for `head(n)`.
- `threads=n` (on `scan_gdx`/`read_gdx`) parallelises the raw read across n independent file handles: each worker scans a contiguous range of the first key dimension, results are concatenated in range order so record order matches the serial read exactly, and a first-dimension filter lets whole non-matching ranges be skipped without touching the file. Parallel scanning requires rebuilding the vendored GDX C library with its internal mutexes enabled (the default in this repo).
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

# unique labels per index dimension without reading the records:
from polars_gdx import read_domains
read_domains("trnsport.gdx", symbol="x")
```

### Label matching

Key labels (`key_filter` and `filter(pl.col(key) == label)` predicates) are matched **case-insensitively** (ASCII-folded), the way GAMS does — `"SEATTLE"` matches a stored `"seattle"`. The labels in the produced frame always keep the file's original casing. For predicate filters whose label only differs in case from the stored one, the plugin deliberately skips the native prefilter and re-applies a case-folded predicate on the full read, because Polars' `==` is case-sensitive and would otherwise silently drop the rows.

### File handle lifetime

`scan_gdx` keeps the GDX file open until the returned `LazyFrame` is collected and released (Polars may read the source more than once). Use `read_gdx`, or an explicit `Reader` with its `close()` / context-manager support, to release the handle eagerly — relevant on Windows, where an open handle locks the file.

## Benchmark

vs [`gamsapi`](https://pypi.org/project/gamsapi/) 54.5.0 (pandas-backed), on a 2M-record parameter (80 MB GDX); best of 5 runs. See `benchmarks/bench_vs_gamsapi.py` (run as a CI job on every push).

| Scenario | polars-gdx | gamsapi (pandas) |
| --- | ---: | ---: |
| Full read (2M rows) | 0.128 s | 0.086 s |
| `filter(dim_0 == label)` — 1000 of 2M rows | **0.059 s** | 0.096 s¹ |
| Explicit `key_filter` — 1000 rows | **0.058 s** | — |
| `head(1000)` of 2M rows | **0.001 s** | 0.101 s¹ |

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
