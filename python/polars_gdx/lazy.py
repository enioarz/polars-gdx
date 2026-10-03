"""Lazy GDX scanning via Polars' IO source interface.

The Rust extension streams Arrow IPC bytes per symbol; Polars drives reading
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


def read_gdx(
    path: str | Path,
    *,
    symbol: str,
    value_field: str | None = None,
    key_filter: dict[int, Sequence[str]] | None = None,
) -> pl.DataFrame:
    """Eagerly read one symbol of a GDX file into a DataFrame.

    Convenience wrapper around :func:`scan_gdx`: same arguments, but collects
    the frame immediately. The lazy path (and thus the native prefilter and
    Polars predicate pushdown) is still used under the hood.
    """
    return scan_gdx(
        path, symbol=symbol, value_field=value_field, key_filter=key_filter
    ).collect()


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

    Notes
    -----
    **Label matching**: key labels are matched case-insensitively
    (ASCII-folded), the way GAMS does, for both ``key_filter`` and
    ``filter(pl.col(key) == label)`` predicates. The labels stored in the
    produced frame keep the file's original casing.

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
    value_name = "value" if type_str in ("Set", "Parameter", "Alias") else (value_field or "level")
    # Key column names must never collide with the value column (a domain
    # set literally named ``value`` would otherwise overwrite it in the
    # schema dict and break the Arrow record batch).
    key_names = _key_names(domains, dim, reserved={value_name})
    schema = {name: pl.String for name in key_names} | {value_name: pl.Float64}

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
            native, folded = _predicate_key_filter(pred_expr, key_names, reader)
            if folded:
                # Some labels only match case-insensitively: Polars' `==` is
                # case-sensitive, so the native prefilter would admit rows
                # that the re-applied predicate would then drop silently.
                # Fall back to a case-folded predicate on the full read; the
                # predicate re-apply makes a native row limit unsafe.
                pred_expr = _case_insensitive_expr(native, key_names)
                native_limit_safe = False
                if explicit_filters is None:
                    filters = None
            elif native is None:
                # Predicate not fully translatable: a native row limit could
                # under-fill head(n), so read unlimited and let Polars-side
                # filter().head(n) apply.
                native_limit_safe = False
            else:
                # The predicate depends only on key columns: fold it into the
                # native prefilter so non-matching records are skipped during
                # the raw read. The predicate is re-applied below for exact
                # semantics (no-op on the already-prefiltered rows).
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
        )
        # pyarrow RecordBatch -> polars: zero-copy over the Arrow buffers.
        df = pl.from_arrow(batch)
        if pred_expr is not None:
            df = df.filter(pred_expr)
        if n_rows is not None:
            df = df.head(n_rows)
        if with_columns is not None:
            df = df.select(with_columns)
        yield df

    return register_io_source(_read, schema=schema)


def _predicate_key_filter(
    pred: pl.Expr, key_names: list[str], reader: Reader
) -> tuple[list[tuple[int, list[str]]] | None, bool]:
    """Extract a native key filter from a predicate, if possible.

    Recognises conjunctions of `pl.col(k) == "label"` (either operand order)
    where `k` is a key column, by inspecting the predicate's serialized plan.
    Returns ``(filters, folded)``:

    - ``filters``: list of ``(dim_index, allowed_labels)``, or None when the
      predicate cannot be translated (caller falls back to Polars-side
      filtering).
    - ``folded``: True when some labels exist in the file only under a
      different letter case (matched via ASCII case folding, GAMS-style).
      The caller must then re-apply a case-insensitive predicate itself
      instead of folding the constraint into the native prefilter, because
      the re-applied Polars ``==`` would otherwise silently drop those rows.
      When ``folded`` is True, ``filters`` carries the full extracted
      constraint set (with the user-supplied labels) for that rebuild.
    """
    try:
        plan = json.loads(pred.meta.serialize(format="json"))
    except Exception:
        return None, False
    filters: list[tuple[int, list[str]]] = []
    if not _collect_eq_filters(plan, key_names, filters):
        return None, False
    if not filters:
        return None, False
    # Resolve labels against the file's UEL table: labels not present in the
    # file can never match, so drop them; an empty set means no rows at all.
    # Matching is case-insensitive (ASCII-folded) like GAMS: a label that only
    # exists in the file under different casing still matches, but is flagged
    # so the caller can keep Polars-side semantics correct.
    uels = set(reader.uel_table())
    folded_uels = {l.upper(): l for l in uels}
    resolved = []
    folded = False
    for d, labels in filters:
        existing = []
        for l in labels:
            if l in uels:
                existing.append(l)
            elif l.upper() in folded_uels:
                existing.append(l)
                folded = True
        if not existing:
            if folded:
                return filters, True
            return [(d, [])], False
        resolved.append((d, existing))
    return resolved, folded


def _case_insensitive_expr(filters: list[tuple[int, list[str]]], key_names: list[str]) -> pl.Expr:
    """Rebuild an eq-conjunction predicate with case-insensitive equality."""
    expr: pl.Expr | None = None
    for d, labels in filters:
        for l in labels:
            eq = pl.col(key_names[d]).cast(pl.String).str.to_uppercase() == l.upper()
            expr = eq if expr is None else expr & eq
    assert expr is not None
    return expr


def _collect_eq_filters(
    node: dict, key_names: list[str], out: list[tuple[int, list[str]]]
) -> bool:
    """Walk a serialized predicate plan; collect key-column constraints.

    Returns True when the whole node is translatable, False otherwise.
    """
    if "BinaryExpr" not in node:
        return False
    b = node["BinaryExpr"]
    if b["op"] == "And":
        return _collect_eq_filters(b["left"], key_names, out) and _collect_eq_filters(
            b["right"], key_names, out
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
        if candidate in names or candidate in reserved:
            candidate = f"{candidate}_{i}"
        names.append(candidate)
    return names


