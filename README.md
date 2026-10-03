# polars-gdx

Lazy, prefiltered Polars access to [GAMS GDX](https://github.com/GAMS-dev/gdx) files — no GAMS installation required.

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

## Build

```sh
maturin develop --release   # builds the vendored GDX C library + extension
pytest
```

Requires a C++17 compiler and CMake ≥ 3.5 at build time; runtime has no GAMS dependency.

## Documentation & community

- [Documentation](docs/README.md) — quickstart, API reference, architecture
- [Contributing](CONTRIBUTING.md) — development setup and guidelines
- [Security](SECURITY.md) — how to report vulnerabilities
- [Support](SUPPORT.md) — where to ask questions
