# Quickstart

## Installation

From a source checkout (a wheel release is planned):

```sh
pip install -e .
```

Build requirements: Rust ≥ 1.80, a C++17 compiler, and CMake ≥ 3.5. The
vendored GDX C library is compiled automatically during the build; at runtime
there is no GAMS dependency.

For development:

```sh
python -m venv .venv && source .venv/bin/activate
pip install maturin patchelf polars pyarrow pytest
maturin develop --release
```

## Reading a GDX file

```python
import polars as pl
from polars_gdx import scan_gdx, list_symbols

# Inspect what is in the file
symbols = list_symbols("trnsport.gdx")
print(symbols.filter(pl.col("type") == "Param"))

# Lazy: nothing is read yet
x = scan_gdx("trnsport.gdx", symbol="x")

# Read only the seattle rows; the key constraint is pushed into the
# native read loop, so other records are skipped before materialisation
seattle = x.filter(pl.col("dim_0") == "seattle").collect()
```

## Reading variables and equations

Variable and equation symbols carry five fields. By default `scan_gdx` loads
`level`; select another with `value_field`:

```python
marginals = scan_gdx(
    "trnsport.gdx", symbol="demand", value_field="marginal"
).collect()
```

Sets and parameters always expose a single `value` column and ignore
`value_field`.

## Prefiltering explicitly

Pass `key_filter` to skip records natively regardless of what Polars does with
the plan. Keys are dimension indexes, labels are UEL strings:

```python
lf = scan_gdx(
    "trnsport.gdx",
    symbol="x",
    key_filter={0: ["seattle", "san-diego"]},
)
```

## Column naming

Key columns are named after the symbol's domain sets when available, and
`dim_<i>` otherwise (for example `dim_0`, `dim_1` for a symbol with `*`
domains). The value column is `value` for sets/parameters and the selected
field name for variables/equations.
