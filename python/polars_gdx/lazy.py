"""Lazy GDX scanning via Polars' IO source interface.

The Rust extension streams Arrow IPC bytes per symbol; Polars drives reading
through `pl.register_io_source`, so column projection is pushed down and only
the columns actually selected are materialised by Polars. Key prefiltering is
additionally applied inside the native GDX read loop, before any record is
materialised.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Iterator, Sequence

import polars as pl
from polars.io.plugins import register_io_source

from polars_gdx._core import Reader

if TYPE_CHECKING:
    from pathlib import Path

_VALUE_FIELDS = ("level", "marginal", "lower", "upper", "scale")


def list_symbols(path: str | Path) -> pl.DataFrame:
    """List the symbols in a GDX file as a DataFrame."""
    reader = Reader(str(path))
    rows = reader.symbols()
    return pl.DataFrame(
        {
            "name": [r[0] for r in rows],
            "type": [r[1] for r in rows],
            "dim": [r[2] for r in rows],
            "records": [r[3] for r in rows],
            "domains": [r[4] for r in rows],
            "text": [r[5] for r in rows],
        },
        schema={
            "name": pl.String,
            "type": pl.String,
            "dim": pl.UInt32,
            "records": pl.UInt64,
            "domains": pl.List(pl.String),
            "text": pl.String,
        },
    )


def scan_gdx(
    path: str | Path,
    *,
    symbol: str,
    value_field: str | None = None,
    key_filter: dict[int, Sequence[str]] | None = None,
) -> pl.LazyFrame:
    """Lazily scan one symbol of a GDX file.

    Parameters
    ----------
    path
        Path to the ``.gdx`` file.
    symbol
        Symbol name (case-sensitive, as stored in the file).
    value_field
        For Variables/Equations: which field to load
        (``level``, ``marginal``, ``lower``, ``upper``, ``scale``).
        Defaults to ``level``. Ignored for Sets/Parameters, which always
        expose a single ``value`` column.
    key_filter
        Optional prefilter ``{dim_index: allowed_labels}`` applied natively
        while reading, so records with non-matching keys are skipped before
        materialisation. This runs in the Rust read loop, in addition to any
        predicate pushdown Polars performs on the produced frame.

    Example
    -------
    >>> lf = scan_gdx("trnsport.gdx", symbol="x")  # doctest: +SKIP
    >>> lf.filter(pl.col("dim_0") == "seattle").collect()  # doctest: +SKIP
    """
    reader = Reader(str(path))
    info = {r[0]: r for r in reader.symbols()}
    if symbol not in info:
        raise ValueError(
            f"symbol {symbol!r} not found in {path!r}; available: {sorted(info)}"
        )
    _, type_str, dim, _n, domains, _text = info[symbol]

    key_names = _key_names(symbol, domains, dim)
    if value_field is not None and value_field not in _VALUE_FIELDS:
        raise ValueError(f"value_field must be one of {_VALUE_FIELDS}, got {value_field!r}")
    value_name = "value" if type_str in ("Set", "Parameter", "Alias") else (value_field or "level")
    schema = {name: pl.String for name in key_names} | {value_name: pl.Float64}

    filters = None
    if key_filter:
        filters = [(d, list(labels)) for d, labels in key_filter.items()]

    def _read(
        with_columns: list[str] | None,
        predicate: None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        del predicate, batch_size
        data = reader.read_arrow(
            symbol,
            key_names=list(key_names),
            value_field=(None if value_name == "value" else value_name),
            key_filter=filters,
        )
        df = pl.read_ipc(io.BytesIO(data))
        if n_rows is not None:
            df = df.head(n_rows)
        if with_columns is not None:
            df = df.select(with_columns)
        yield df

    return register_io_source(_read, schema=schema)


def _key_names(symbol: str, domains: list[str], dim: int) -> list[str]:
    names = []
    for i, d in enumerate(domains[:dim]):
        if d and d != "*":
            candidate = d
        else:
            candidate = f"dim_{i}"
        # Column names must be unique; de-duplicate with a positional suffix.
        if candidate in names:
            candidate = f"{candidate}_{i}"
        names.append(candidate)
    return names
