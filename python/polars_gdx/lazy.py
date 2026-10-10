"""Lazy GDX scanning via Polars' IO source interface.

The Rust extension returns an Arrow RecordBatch per symbol; Polars drives reading
through `pl.register_io_source`, so column projection is pushed down and only
the columns actually selected are materialised by Polars. Key prefiltering is
additionally applied inside the native GDX read loop, before any record is
materialised.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Iterator, Sequence

import polars as pl
from polars.io.plugins import register_io_source

from polars_gdx._core import Reader

if TYPE_CHECKING:
    from pathlib import Path

_VALUE_FIELDS = ("level", "marginal", "lower", "upper", "scale")

#: Smallest symbol (in records) for which ``threads="auto"`` parallelises
#: the raw read. Below this, open/coordination overhead outweighs the gain.
_AUTO_THREADS_MIN_RECORDS = 5_000_000


def _resolve_threads(threads: int | str | None, n_records: int) -> int | None:
    """``"auto"`` -> CPU count for large symbols, else the given value."""
    if threads == "auto":
        if n_records < _AUTO_THREADS_MIN_RECORDS:
            return None
        import os

        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover - non-Linux
            return os.cpu_count()
    if threads == 0:
        return None
    return threads


def read_domains(path: str | Path, *, symbol: str) -> pl.DataFrame:
    """List the labels actually used per index dimension of a symbol.

    Returns a DataFrame with one column per dimension (named like
    ``scan_gdx``'s key columns), each holding the unique labels used by that
    dimension, in file order.

    This is the fast equivalent of reading the symbol and taking
    ``unique()`` per key column: the scan runs inside the GDX library via a
    bulk callback — no records are materialised and no value column is read —
    so it costs one pass over the raw indices regardless of the number of
    records. (Columns are padded to equal length with ``null``.)
    """
    with Reader(str(path)) as reader:
        info = {r[0]: r for r in reader.symbols()}
        if symbol not in info:
            raise ValueError(
                f"symbol {symbol!r} not found in {path!r}; available: {sorted(info)}"
            )
        _, _type_str, dim, _n, domains, _text = info[symbol]
        uels = reader.uel_table()
        names = _key_names(domains, dim, reserved=set())
        series = []
        for d in range(dim):
            used = reader.domain_elements(symbol, d)
            labels = [uels[i - 1] if 0 < i <= len(uels) else None for i in used]
            series.append(pl.Series(names[d], labels, dtype=pl.String))
    if not series:
        return pl.DataFrame()
    longest = max(len(s) for s in series)
    series = [s.rechunk() for s in series]
    return pl.DataFrame(
        {s.name: s.extend_constant(None, longest - len(s)) for s in series}
    )


def list_symbols(path: str | Path) -> pl.DataFrame:
    """List the symbols in a GDX file as a DataFrame."""
    with Reader(str(path)) as reader:
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


def read_gdx(
    path: str | Path,
    *,
    symbol: str,
    value_field: str | None = None,
    key_filter: dict[int, Sequence[str]] | None = None,
    threads: int | str | None = None,
) -> pl.DataFrame:
    """Eagerly read one symbol of a GDX file into a DataFrame.

    Convenience wrapper around :func:`scan_gdx`: same arguments, but collects
    the frame immediately. The lazy path (and thus the native prefilter and
    Polars predicate pushdown) is still used under the hood.
    """
    return scan_gdx(
        path,
        symbol=symbol,
        value_field=value_field,
        key_filter=key_filter,
        threads=threads,
    ).collect()


def scan_gdx(
    path: str | Path,
    *,
    symbol: str,
    value_field: str | None = None,
    key_filter: dict[int, Sequence[str]] | None = None,
    threads: int | str | None = None,
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
    threads
        Number of worker threads for the raw read. When > 1 and no row
        limit applies, the read is split across that many independent file
        handles: first-dimension filters seek straight into the matching
        byte range (via a cached restart index), other reads split by file
        position. Record order always matches the serial read exactly.
        ``"auto"`` (recommended for large symbols)
        parallelises reads of symbols with at least
        ``_AUTO_THREADS_MIN_RECORDS`` records using all available cores;
        smaller symbols stay serial. ``None``/0/1 means serial.

    Notes
    -----
    **Label matching**: key labels are matched **exactly** (case-sensitive,
    byte-for-byte as stored in the file) for both ``key_filter`` and
    ``filter(pl.col(key) == label)`` predicates.

    **File handle lifetime**: the underlying GDX file stays open until the
    returned LazyFrame is collected and released, because Polars may read the
    source more than once. Use :func:`read_gdx` (or an explicit
    ``Reader`` plus its ``close()``/context-manager support) if you need the
    handle released eagerly.

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
    if value_field is not None and value_field not in _VALUE_FIELDS:
        raise ValueError(f"value_field must be one of {_VALUE_FIELDS}, got {value_field!r}")
    threads = _resolve_threads(threads, _n)
    value_name = "value" if type_str in ("Set", "Parameter", "Alias") else (value_field or "level")
    # Key column names must never collide with the value column (a domain
    # set literally named ``value`` would otherwise overwrite it in the
    # schema dict and break the Arrow record batch).
    key_names = _key_names(domains, dim, reserved={value_name})
    # Key columns are declared as Polars Enums over the file's UEL table.
    # The Rust layer emits Arrow dictionary arrays whose values are exactly
    # the UEL table (file order) and whose indices are the raw 0-based UEL
    # numbers, so Polars reinterprets the buffers zero-copy: no string
    # materialisation, and `==`/`is_in`/join/group-by on keys run as integer
    # category comparisons. Polars 2.0 maps Arrow dictionaries to Categorical
    # when no schema is given and enforces the schema declared to
    # register_io_source, so this must be declared explicitly.
    key_dtype = pl.Enum(reader.uel_table())
    schema = {name: key_dtype for name in key_names} | {value_name: pl.Float64}

    explicit_filters = None
    if key_filter:
        explicit_filters = [(d, list(labels)) for d, labels in key_filter.items()]

    def _read(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        del batch_size
        filters = explicit_filters
        pred_expr = None
        native_limit_safe = True
        if predicate is not None:
            pred_expr = predicate
            native = _predicate_key_filter(pred_expr, key_names, reader)
            if native is None:
                # Predicate not fully translatable: a native row limit could
                # under-fill head(n), so read unlimited and let Polars-side
                # filter().head(n) apply.
                native_limit_safe = False
            else:
                # The predicate depends only on key columns: fold it into the
                # native prefilter so non-matching records are skipped during
                # the raw read, and re-apply it exactly on the produced frame.
                pred_expr = _exact_expr(native, key_names)
                merged = {d: list(labels) for d, labels in explicit_filters or []}
                for d, labels in native:
                    if d in merged:
                        allowed = set(labels)
                        merged[d] = [l for l in merged[d] if l in allowed]
                    else:
                        merged[d] = list(labels)
                filters = list(merged.items())

        batch = reader.read_arrow(
            symbol,
            key_names=list(key_names),
            value_field=(None if value_name == "value" else value_name),
            key_filter=filters,
            n_rows=(n_rows if native_limit_safe else None),
            threads=threads,
        )
        # pyarrow RecordBatch -> polars: zero-copy over the Arrow buffers.
        # Passing the Enum schema makes Polars reinterpret the dictionary
        # indices as enum physicals instead of decoding to strings.
        df = pl.from_arrow(batch, schema=schema)
        if pred_expr is not None:
            df = df.filter(pred_expr)
        if n_rows is not None:
            df = df.head(n_rows)
        if with_columns is not None:
            df = df.select(with_columns)
        yield df

    # `is_pure` lets the optimizer de-duplicate repeated occurrences of the
    # same scan within one plan (the source is a deterministic file read).
    return register_io_source(_read, schema=schema, is_pure=True)


def _predicate_key_filter(
    pred: pl.Expr, key_names: list[str], reader: Reader
) -> list[tuple[int, list[str]]] | None:
    """Extract a native key filter from a predicate, if possible.

    Recognises conjunctions of `pl.col(k) == "label"` (either operand order)
    and `col.is_in([labels])` where `k` is a key column, by inspecting the
    predicate's serialized plan. Returns a list of
    ``(dim_index, allowed_labels)`` pairs, or None when the predicate cannot
    be translated (the caller falls back to Polars-side filtering).

    Labels are matched exactly (case-sensitively) against the file's UEL
    table; labels not present in the file can never match, so they are
    dropped, and an empty set means no rows at all.
    """
    try:
        plan = json.loads(pred.meta.serialize(format="json"))
    except Exception:
        return None
    filters: list[tuple[int, list[str]]] = []
    if not _collect_eq_filters(plan, key_names, filters):
        return None
    if not filters:
        return None
    uels = set(reader.uel_table())
    resolved = []
    for d, labels in filters:
        existing = [l for l in labels if l in uels]
        if not existing:
            # Label never used in this dimension: no record can match.
            return [(d, [])]
        resolved.append((d, existing))
    return resolved


def _exact_expr(filters: list[tuple[int, list[str]]], key_names: list[str]) -> pl.Expr:
    """Rebuild the predicate with exact (stored) labels.

    Within a dimension, labels are alternatives (OR) — a dimension's
    constraint is a membership test; across dimensions they conjoin (AND),
    matching the native prefilter semantics for both `==` and `is_in`.
    All labels must resolve to at least one stored label (the caller
    checked); an empty per-dimension label set means no rows at all.
    """
    expr: pl.Expr | None = None
    for d, labels in filters:
        if not labels:
            return pl.lit(False)
        # No `.cast(pl.String)`: key columns are Enums, so `==`/`is_in` on
        # the stored labels resolve to integer category comparisons.
        col = pl.col(key_names[d])
        dim_expr = col == labels[0] if len(labels) == 1 else col.is_in(labels)
        expr = dim_expr if expr is None else expr & dim_expr
    return expr if expr is not None else pl.lit(False)


def _collect_eq_filters(
    node: dict, key_names: list[str], out: list[tuple[int, list[str]]]
) -> bool:
    """Walk a serialized predicate plan; collect key-column constraints.

    Recognises `col == "label"` and `col.is_in([labels])` (plus conjunctions
    thereof). Returns True when the whole node is translatable, else False.
    """
    if "BinaryExpr" in node:
        b = node["BinaryExpr"]
        if b["op"] == "And":
            return _collect_eq_filters(b["left"], key_names, out) and (
                _collect_eq_filters(b["right"], key_names, out)
            )
        if b["op"] != "Eq":
            return False
        for col_side, lit_side in ((b["left"], b["right"]), (b["right"], b["left"])):
            if "Column" in col_side and col_side["Column"] in key_names:
                label = _string_literal(lit_side)
                if label is not None:
                    out.append((key_names.index(col_side["Column"]), [label]))
                    return True
                return False
        return False
    if "Function" in node:
        f = node["Function"]
        opts = f.get("function", {}).get("Boolean", {})
        if "IsIn" not in opts:
            return False
        inputs = f.get("input", [])
        if len(inputs) != 2 or "Column" not in inputs[0]:
            return False
        col = inputs[0]["Column"]
        if col not in key_names:
            return False
        labels = _list_literal_strings(inputs[1])
        if labels is None:
            return False
        out.append((key_names.index(col), labels))
        return True
    return False


def _list_literal_strings(node: dict) -> list[str] | None:
    """String values of an Arrow-IPC list literal, else None.

    Polars serializes `is_in` list literals as a continuation marker
    (0xFFFFFFFF) followed by an Arrow IPC stream of a one-column table.
    """
    try:
        raw = bytes(node["Literal"]["Scalar"]["List"])
    except (KeyError, TypeError):
        return None
    try:
        import pyarrow as pa

        reader = pa.ipc.open_stream(pa.BufferReader(raw))
        table = pa.Table.from_batches(list(reader))
    except Exception:
        return None
    col = table.column(0)
    if pa.types.is_string(col.type) or pa.types.is_large_string(col.type):
        return [v for v in col.to_pylist() if v is not None]
    # string_view (pyarrow >= 16): not covered by is_string
    if "string_view" in str(col.type):
        return [v for v in col.to_pylist() if v is not None]
    return None


def _string_literal(node: dict) -> str | None:
    """The node's string value if it is a plain string literal, else None."""
    try:
        lit = node["Literal"]["Scalar"]["String"]
        return lit
    except (KeyError, TypeError):
        return None


def _key_names(domains: list[str], dim: int, reserved: set[str]) -> list[str]:
    """Unique key-column names, also avoiding the (reserved) value column."""
    names = []
    for i, d in enumerate(domains[:dim]):
        if d and d != "*":
            candidate = d
        else:
            candidate = f"dim_{i}"
        # Column names must be unique and must never collide with the value
        # column; de-duplicate with a positional suffix.
        while candidate in names or candidate in reserved:
            candidate = f"{candidate}_{i}"
        names.append(candidate)
    return names


