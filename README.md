# polars-gdx

Lazy, prefILTERed Polars access to [GAMS GDX](https://github.com/GAMS-dev/gdx) files — no GAMS installation required.

- `scan_gdx(path, symbol=...)` returns a `pl.LazyFrame`; reads happen only on `collect()`.
- Column projection is pushed down via Polars' IO-source interface.
- `key_filter={dim_index: allowed_labels}` is applied **natively inside the Rust read loop**, so filtered records are skipped before materialisation — typically much faster than the official `gamsapi` reader, which materialises everything into Python objects.
- Under the hood: hand-written FFI to the vendored MIT-licensed GDX C library, extracted from [lolow/gdxcomp](https://github.com/lolow/gdxcomp) (`gdx-sys` + `gdx` crates), building on [GAMS-dev/gdx](https://github.com/GAMS-dev/gdx).

## Layout

```
crates/gdx-sys        # raw FFI; builds vendored GAMS-dev/gdx via CMake (submodule-free, sources vendored)
crates/gdx            # safe RAII wrapper (global lock; lazy UEL interning)
crates/polars-gdx     # PyO3 extension: symbol listing + Arrow IPC streaming with prefiltering
python/polars_gdx     # Python layer: scan_gdx / list_symbols
```

## Usage

```python
import polars as pl
from polars_gdx import scan_gdx, list_symbols

list_symbols("trnsport.gdx").filter(pl.col("type") == "Param")

# lazy: nothing read yet
x = scan_gdx("trnsport.gdx", symbol="x")
# read only the seattle rows, skipping all others natively
x.filter(pl.col("dim_0") == "seattle").collect()
```

## Benchmark

Against [`gamsapi`](https://pypi.org/project/gamsapi/) 54.5.0 (`gams.transfer`, pandas-backed), on a 2M-record 2-dimensional parameter (80 MB GDX). Best of 5 runs; see `benchmarks/bench_vs_gamsapi.py` (also run as a CI job on every push).

| Scenario | polars-gdx | gamsapi (pandas) |
| --- | ---: | ---: |
| Full read (2M rows) | 0.126 s | 0.086 s |
| `filter(dim_0 == label)` — 1000 of 2M rows | **0.081 s** | 0.087 s¹ |
| Explicit `key_filter` prefilter — 1000 rows | 0.080 s | — |

¹ gamsapi has no lazy loading or pushdown: its only option is materializing the whole symbol into pandas, then filtering. polars-gdx folds `filter()` predicates on key columns into a native index-based prefilter inside the GDX read loop, so the unwanted 1,999,000 records are never materialized at all — while still handing you a lazily composable Polars frame (categorical keys, zero-copy Arrow handoff).

Speedup grows with selectivity: the more records your predicate excludes, the larger the advantage over a full materialization.

## Build

```sh
maturin develop --release   # builds the vendored GDX C library + extension
pytest
```

Requires a C++17 compiler and CMake ≥ 3.5 at build time; runtime has no GAMS dependency.
