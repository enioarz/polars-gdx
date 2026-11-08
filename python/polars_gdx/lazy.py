"""Lazy GDX scanning via Polars' IO source interface.

Polars drives reading through `register_io_source`. The gdxrs backend streams
owned Arrow batches with projection and key-filter pushdown; the legacy backend
returns one batch per symbol. Residual predicates run before the final projection
and global accepted-row limit.
"""

from __future__ import annotations

import atexit
import io
import json
import os
import sys
import threading
import time
from typing import TYPE_CHECKING, Iterator, Sequence

import polars as pl
from polars.io.plugins import register_io_source

from polars_gdx import _core

if TYPE_CHECKING:
    from pathlib import Path

Reader = getattr(_core, "Reader", None)
GdxrsReader = getattr(_core, "GdxrsReader", None)

_VALUE_FIELDS = ("level", "marginal", "lower", "upper", "scale")


def _validate_backend_name(backend: str) -> str:
    """Validate a backend name against the compiled backends and return it.

    ``"legacy"`` selects the vendored native GDX C library backend (default
    in 0.6.x); ``"gdxrs"`` selects the pinned pure-Rust gdxrs core (M7 task
    042). Selection is explicit and final: a gdxrs source, decode or
    resource failure is raised, never retried through the legacy backend.
    """
    if backend == "legacy":
        if Reader is None:
            raise RuntimeError(
                "the legacy backend is not compiled into this build "
                "(cargo feature 'legacy')"
            )
        return "legacy"
    if backend == "gdxrs":
        if GdxrsReader is None:
            raise RuntimeError(
                "the gdxrs backend is not compiled into this build "
                "(cargo feature 'gdxrs')"
            )
        return "gdxrs"
    raise ValueError(f"backend must be 'legacy' or 'gdxrs', got {backend!r}")


def _default_backend() -> str:
    """The default backend: ``POLARS_GDX_BACKEND`` when it names an
    available backend, else ``"legacy"`` when compiled in, else
    ``"gdxrs"``."""
    override = os.environ.get("POLARS_GDX_BACKEND")
    if override is not None:
        return _validate_backend_name(override)
    if Reader is not None:
        return "legacy"
    if GdxrsReader is not None:
        return "gdxrs"
    raise RuntimeError(
        "no GDX backend is compiled in; build with the 'legacy' and/or "
        "'gdxrs' cargo features"
    )


def _reader_class(backend: str | None):
    """Resolve a backend name (or the default) to its reader class."""
    name = backend if backend is not None else _default_backend()
    return Reader if _validate_backend_name(name) == "legacy" else GdxrsReader


#: Backends compiled into this extension.
AVAILABLE_BACKENDS = tuple(
    name
    for name, cls in (("legacy", Reader), ("gdxrs", GdxrsReader))
    if cls is not None
)

#: Default backend (docs/m7-adapter-contract.md §4: legacy stays the 0.6.x
#: default; ``POLARS_GDX_BACKEND`` overrides it for differential test runs).
DEFAULT_BACKEND = _default_backend()

#: Live streaming cursors (M7 task 044). A noncollecting-sink consumer can
#: drop a suspended source without closing it, and Polars may finalize the
#: suspended generator on an engine thread during interpreter shutdown,
#: where destroying extension objects can abort the process. The registry
#: retains cursors until their workers have joined. The atexit hook closes
#: remaining cursors and drains registered callbacks before teardown.
_OPEN_CURSORS: set = set()


#: Set once the interpreter begins shutting down: streaming sources
#: check it before every extension call so a Polars engine thread that
#: resumes a suspended generator during teardown executes only pure
#: Python and never touches the finalizing interpreter through PyO3.
_SHUTTING_DOWN = threading.Event()

#: Registered setup and production steps, including predicate planning,
#: Arrow conversion and residual evaluation. Yielded batches hold no registration;
#: their cursor stays in the registry until close completes.
_IN_FLIGHT_LOCK = threading.Lock()
_IN_FLIGHT = 0


def _enter_callback() -> bool:
    """Atomically gate on shutdown and register an in-flight callback.

    The shutdown check and the registration happen under the same lock
    the shutdown hook uses to set the flag, so no callback can slip
    between the check and the increment (and no shutdown can complete
    its gate while a callback is registering). Returns False once
    shutdown has begun — the caller must not touch the extension; every
    True is matched by exactly one :func:`_exit_callback`.
    """
    global _IN_FLIGHT
    with _IN_FLIGHT_LOCK:
        if _SHUTTING_DOWN.is_set():
            return False
        _IN_FLIGHT += 1
        return True


def _exit_callback() -> None:
    """Release a registration acquired by :func:`_enter_callback`."""
    global _IN_FLIGHT
    with _IN_FLIGHT_LOCK:
        _IN_FLIGHT -= 1


def _close_open_cursors() -> None:
    # 1. Forbid new source steps: the flag is set under the same lock the
    #    callbacks gate and register on, so no callback can enter
    #    production after this point and in-flight/teardown-resumed
    #    generators break out of their loops before any extension call.
    with _IN_FLIGHT_LOCK:
        _SHUTTING_DOWN.set()
    # 2. Close every published cursor (stopping and joining production
    #    while the interpreter is still alive) and wait for CONFIRMED
    #    completion of every registered callback: a callback suspended
    #    inside next_stream() must reacquire the GIL and return before
    #    teardown, or the finalizing interpreter can abort the process.
    #    The wait has no deadline — blocked synchronous OS reads cannot
    #    be forcibly interrupted (documented), so shutdown waits for
    #    them by design. Sleeping releases the GIL so the callbacks can
    #    finish. Only the PROCESSED snapshot is removed from the
    #    registry: a cursor published while this loop runs (setup stays
    #    registered through publication) is closed in a later iteration
    #    — never silently erased, and shutdown can never miss a
    #    published cursor with unjoined workers.
    while True:
        with _IN_FLIGHT_LOCK:
            pending = list(_OPEN_CURSORS)
            if _IN_FLIGHT == 0 and not pending:
                break
        for cursor in pending:
            try:
                cursor.close()
            except Exception:
                pass
        if pending:
            with _IN_FLIGHT_LOCK:
                _OPEN_CURSORS.difference_update(pending)
        time.sleep(0.001)


atexit.register(_close_open_cursors)

#: Smallest symbol (in records) for which ``threads="auto"`` parallelises
#: the raw read. Below this, open/coordination overhead outweighs the gain.
_AUTO_THREADS_MIN_RECORDS = 5_000_000

#: Admitted core-worker concurrency for the gdxrs backend (adapter
#: contract §6: at most four core workers).
_GDXRS_MAX_WORKERS = 4


def _resolve_threads(threads: int | str | None, n_records: int) -> int | None:
    """``"auto"`` -> CPU count for large symbols, else the given value."""
    if threads == "auto":
        if n_records < _AUTO_THREADS_MIN_RECORDS:
            return None
        try:
            return len(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover - non-Linux
            return os.cpu_count()
    if threads == 0:
        return None
    return threads


def read_domains(
    path: str | Path, *, symbol: str, backend: str | None = None
) -> pl.DataFrame:
    """List the labels actually used per index dimension of a symbol.

    Returns a DataFrame with one column per dimension (named like
    ``scan_gdx``'s key columns), each holding the unique labels used by that
    dimension, in file order.

    This is the fast equivalent of reading the symbol and taking
    ``unique()`` per key column: the scan runs inside the GDX library via a
    bulk callback — no records are materialised and no value column is read —
    so it costs one pass over the raw indices regardless of the number of
    records. (Columns are padded to equal length with ``null``.)

    ``backend`` selects the reader implementation: ``"legacy"`` (vendored
    native GDX C library, default in 0.6.x) or ``"gdxrs"`` (pinned pure-Rust
    core, M7 task 042). The default is ``POLARS_GDX_BACKEND`` when set,
    else ``"legacy"`` when compiled in.
    """
    with _reader_class(backend)(str(path)) as reader:
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


def list_symbols(path: str | Path, backend: str | None = None) -> pl.DataFrame:
    """List the symbols in a GDX file as a DataFrame.

    ``backend`` selects the reader implementation (see :func:`scan_gdx`).
    """
    with _reader_class(backend)(str(path)) as reader:
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
    backend: str | None = None,
    key_mode: str = "enum",
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
        backend=backend,
        key_mode=key_mode,
    ).collect()


def scan_gdx(
    path: str | Path,
    *,
    symbol: str,
    value_field: str | None = None,
    key_filter: dict[int, Sequence[str]] | None = None,
    threads: int | str | None = None,
    backend: str | None = None,
    key_mode: str = "enum",
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
        The ``gdxrs`` backend admits 2..=4 workers (M7 task 046): the read
        runs on the core's parallel scan over an explicit checkpoint
        index, built once per reader and symbol at a one-pass build cost
        and reused for repeated scans; results are byte-identical to the
        serial read. Requests above the admission raise ``ValueError``,
        and ``"auto"`` never exceeds it. Blocked synchronous OS reads
        inside an in-flight worker step cannot be forcibly interrupted;
        cancellation joins the current step.
    backend
        ``"legacy"`` (vendored native GDX C library; default in 0.6.x) or
        ``"gdxrs"`` (pinned pure-Rust core; M7 task 042). The default is
        ``POLARS_GDX_BACKEND`` when set to an available backend, else
        ``"legacy"`` when compiled in. Backend selection is explicit and
        final: a gdxrs source, decode or resource failure is raised, never
        retried through the legacy backend.
    key_mode
        Key column representation (gdxrs backend only, M7 task 043).
        ``"enum"`` (default) declares key columns as ``pl.Enum`` over the
        full UEL table in file order, admitted against the adapter's
        dictionary budget before any output; insufficient admission is
        rejected clearly, never by silently changing the schema. Labels
        are strictly UTF-8; invalid bytes are rejected before output.
        ``"uel_id"`` emits the original one-based UEL IDs as plain
        ``UInt32`` key columns and never decodes or materializes any
        label — the bounded mode for files whose dictionary exceeds the
        budget. String filters (``key_filter`` and ``==``/``is_in``
        predicates) still match exactly by encoding as UTF-8 and
        resolving to IDs. Batch-local category codes are never
        reinterpreted as global IDs; the schema is chosen before
        execution and preserved across empty batches, partitions and
        repeated scans. Use ``Reader.resolve_labels(ids)`` to resolve
        original IDs to exact label bytes. In ``"uel_id"`` mode, string
        ``key_filter`` labels and ``col(key) == "label"`` predicates match
        exactly (labels are encoded as UTF-8 and resolved to their IDs);
        ``is_in`` predicates must use the integer IDs, since Polars does
        not plan String lists against UInt32 columns. Label-equality translation
        requires predicate pushdown; when disabled, use integer ID predicates or
        explicit string ``key_filter``. Mixing label and numeric predicates on
        the same key column is rejected; use integer IDs consistently there.

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
    reader_cls = _reader_class(backend)
    backend_is_gdxrs = reader_cls is GdxrsReader
    if key_mode not in ("enum", "uel_id"):
        raise ValueError(f"key_mode must be 'enum' or 'uel_id', got {key_mode!r}")
    if key_mode == "uel_id" and not backend_is_gdxrs:
        raise ValueError(
            "key_mode='uel_id' is only supported by the gdxrs backend"
        )
    reader = reader_cls(str(path))
    info = {r[0]: r for r in reader.symbols()}
    if symbol not in info:
        raise ValueError(
            f"symbol {symbol!r} not found in {path!r}; available: {sorted(info)}"
        )
    _, type_str, dim, _n, domains, _text = info[symbol]
    if value_field is not None and value_field not in _VALUE_FIELDS:
        raise ValueError(f"value_field must be one of {_VALUE_FIELDS}, got {value_field!r}")
    if backend_is_gdxrs:
        if threads == "auto":
            # ``"auto"`` never exceeds the worker admission (contract §6);
            # small symbols stay serial.
            auto = _resolve_threads("auto", _n)
            threads = auto if auto is None else min(auto, _GDXRS_MAX_WORKERS)
        else:
            threads = _resolve_threads(threads, _n)
        if threads is not None and threads > _GDXRS_MAX_WORKERS:
            # Explicit requests above the configured worker admission are
            # rejected, never silently clamped (contract §6).
            raise ValueError(
                f"gdxrs backend: threads={threads} exceeds the configured "
                f"worker admission of {_GDXRS_MAX_WORKERS}"
            )
    else:
        threads = _resolve_threads(threads, _n)
    value_name = "value" if type_str in ("Set", "Parameter", "Alias") else (value_field or "level")
    # Key column names must never collide with the value column (a domain
    # set literally named ``value`` would otherwise overwrite it in the
    # schema dict and break the Arrow record batch).
    key_names = _key_names(domains, dim, reserved={value_name})
    # The schema is chosen now, before execution, and preserved across
    # empty batches, partitions and repeated scans.
    if key_mode == "uel_id":
        # Bounded ID mode: original one-based UEL IDs as plain UInt32.
        # No labels are decoded and no dictionary is materialized, so
        # files whose dictionary exceeds the budget still stream.
        key_dtype = pl.UInt32
    else:
        # Key columns are declared as Polars Enums over the file's UEL
        # table. The Rust layer emits Arrow dictionary arrays whose values
        # are exactly the UEL table (file order) and whose indices are the
        # raw 0-based UEL numbers, so Polars reinterprets the buffers
        # zero-copy: no string materialisation, and `==`/`is_in`/join/
        # group-by on keys run as integer category comparisons. Polars 2.0
        # maps Arrow dictionaries to Categorical when no schema is given
        # and enforces the schema declared to register_io_source, so this
        # must be declared explicitly. The full dictionary is admitted
        # against the adapter budget before any output.
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
        if not backend_is_gdxrs:
            # Only the gdxrs backend streams batches (task 044); the legacy
            # backend still ignores batch_size and emits one batch.
            del batch_size
        # The gdxrs callback is gated and registered BEFORE its first
        # extension call: predicate planning itself calls the extension
        # (reader.label_ids() for ID-mode label predicates), so shutdown
        # must not be able to complete while a paused preamble resumes
        # into it. A shutdown that has already begun ends the source
        # before any extension call.
        if backend_is_gdxrs and not _enter_callback():
            return
        registered = backend_is_gdxrs
        cursor = None
        try:
            # CURSOR CLEANUP (the outer finally below) owns everything
            # from cursor creation on: setup, registration release, the
            # handoff between them and the whole streaming loop — a
            # cancellation anywhere in that range discards and closes
            # the published cursor instead of retaining it with unjoined
            # workers.
            try:
                filters = explicit_filters
                pred_expr = None
                shadow_mask = None
                native_limit_safe = True
                if predicate is not None:
                    pred_expr = predicate
                    native = _predicate_key_filter(
                        pred_expr, key_names, reader, id_mode=(key_mode == "uel_id")
                    )
                    if native is None:
                        # Predicate not fully translatable: a native row
                        # limit could under-fill head(n), so read unlimited
                        # and let Polars-side filter().head(n) apply.
                        native_limit_safe = False
                        if key_mode == "uel_id" and _references_key_labels(
                            pred_expr, key_names
                        ):
                            # The predicate compares key columns against
                            # string labels but also contains conditions the
                            # adapter cannot translate (residuals). Polars
                            # cannot compare String literals against UInt32
                            # columns. Translate label comparisons to exact
                            # original IDs and evaluate the residual on the ID
                            # frame, preserving other expressions and types.
                            pred_expr = None
                            shadow_mask = (predicate, key_names)
                        else:
                            shadow_mask = None
                    else:
                        shadow_mask = None
                        # The predicate depends only on key columns: fold it
                        # into the native prefilter so non-matching records
                        # are skipped during the raw read, and re-apply it
                        # exactly on the produced frame.
                        pred_expr = _exact_expr(
                            native,
                            key_names,
                            reader=reader,
                            id_mode=(key_mode == "uel_id"),
                        )
                        merged = {
                            d: list(labels) for d, labels in explicit_filters or []
                        }
                        for d, labels in native:
                            if d in merged:
                                allowed = set(labels)
                                merged[d] = [l for l in merged[d] if l in allowed]
                            else:
                                merged[d] = list(labels)
                        filters = list(merged.items())

                read_kwargs = {
                    "key_names": list(key_names),
                    "value_field": (None if value_name == "value" else value_name),
                    "key_filter": filters,
                    "n_rows": (n_rows if native_limit_safe else None),
                    "threads": threads,
                }
                # Projection pushdown (task 045): the columns the plan
                # needs. A residual predicate must find its required
                # columns in the frame until it has been evaluated, so
                # every predicate-referenced column is retained
                # defensively (Polars already includes them in
                # with_columns, but the source does not rely on that).
                needed_columns = None
                if backend_is_gdxrs and with_columns is not None:
                    needed = list(dict.fromkeys(with_columns))
                    if predicate is not None:
                        try:
                            plan = json.loads(
                                predicate.meta.serialize(format="json")
                            )
                            referenced = [
                                name
                                for name in _referenced_columns(plan)
                                if name in schema
                            ]
                            needed = list(dict.fromkeys([*needed, *referenced]))
                        except Exception:
                            needed = None
                    needed_columns = needed
                if backend_is_gdxrs:
                    # M7 task 044: cursor-backed streaming batches. The
                    # source emits incrementally without accumulating the
                    # whole symbol; `batch_size` is honored as the maximum
                    # rows per emitted batch (subject to the core's
                    # separate byte budget for wide rows). Each yielded
                    # batch owns valid data after the lending core cursor
                    # advances or closes.
                    read_kwargs["key_mode"] = key_mode
                    read_kwargs["columns"] = needed_columns
                    # Source setup is itself an extension call (index
                    # build, worker spawn, admission) and stays REGISTERED
                    # through cursor publication: the cursor joins the
                    # registry before the registration is released, so
                    # shutdown either closes it in its sweep or picks it
                    # up during the drain — it can never miss a published
                    # cursor with unjoined workers.
                    cursor = reader.scan_batches(
                        symbol, batch_rows=batch_size, **read_kwargs
                    )
                    # Keep a registry reference so a disposed-but-unclosed
                    # source is finalized on the main thread, not on a
                    # Polars engine thread during interpreter shutdown.
                    _OPEN_CURSORS.add(cursor)
                else:
                    batch = reader.read_arrow(symbol, **read_kwargs)
                    # pyarrow RecordBatch -> polars: zero-copy over the
                    # Arrow buffers. Passing the Enum schema makes Polars
                    # reinterpret the dictionary indices as enum physicals
                    # instead of decoding to strings.
                    df = pl.from_arrow(batch, schema=schema)
                    if shadow_mask is not None:
                        df = _apply_key_label_predicate(df, reader, *shadow_mask)
                    elif pred_expr is not None:
                        df = df.filter(pred_expr)
                    if n_rows is not None:
                        df = df.head(n_rows)
                    if with_columns is not None:
                        df = df.select(with_columns)
                    yield df
                    return
            finally:
                # The single registration cleanup path: runs exactly once
                # on every exit from the registered preamble (normal end,
                # exception, cancellation), so the in-flight counter can
                # never be double-decremented.
                if registered:
                    _exit_callback()
            if cursor is not None:
                # Global accepted-row limit for residual predicates that
                # the core could not apply: track the remainder across
                # batches.
                remaining = n_rows if not native_limit_safe else None
                while True:
                    if not _enter_callback():
                        # Shutdown has begun: no extension call may
                        # happen; end the source immediately.
                        break
                    try:
                        # Consume the Arrow C Data Interface stream
                        # capsule: no pyarrow object is created on this
                        # (possibly Polars engine) thread — pyarrow objects
                        # created there can abort the interpreter at
                        # shutdown. Polars maps the dictionary keys to
                        # Categorical, so the declared schema (Enum keys)
                        # is re-applied by an exact cast; the dictionary
                        # is the full UEL table in file order, so the cast
                        # preserves the values. The in-flight count covers
                        # the WHOLE callback body so the shutdown drain
                        # waits until the callback is fully back at its
                        # suspension point.
                        holder = cursor.next_stream()
                        if holder is None:
                            break
                        df = pl.from_arrow(holder).struct.unnest()
                        df = df.select(
                            [
                                pl.col(name).cast(schema[name])
                                for name in df.columns
                            ]
                        )
                        if shadow_mask is not None:
                            df = _apply_key_label_predicate(
                                df, reader, *shadow_mask
                            )
                        elif pred_expr is not None:
                            df = df.filter(pred_expr)
                        if remaining is not None:
                            df = df.head(remaining)
                            remaining -= df.height
                        elif n_rows is not None:
                            # The core already enforced the global limit;
                            # per-batch head is a no-op except on the final
                            # batch.
                            df = df.head(n_rows)
                        if with_columns is not None:
                            df = df.select(with_columns)
                    finally:
                        _exit_callback()
                    yield df
                    if remaining is not None and remaining <= 0:
                        break
        finally:
            # CURSOR CLEANUP — the outer path, covering setup, the
            # registration release, the handoff between them and the
            # whole streaming loop. Every exit (normal end of stream,
            # early break, exception, cancellation, consumer disposal via
            # GeneratorExit) discards the published cursor and closes it
            # (stopping and joining core production) while the
            # interpreter is still alive. If the generator is only
            # finalized during interpreter shutdown (a sink that dropped
            # it without closing), skip the extension call: the atexit
            # hook already stopped production, and extension calls during
            # finalization can abort.
            if cursor is not None:
                # Close BEFORE unregistering: the cursor stays in the
                # registry until close() completes, so a concurrent
                # shutdown cannot observe an empty registry while this
                # cursor's workers are still unjoined — it waits for the
                # in-flight close instead of returning.
                if not sys.is_finalizing():
                    cursor.close()
                _OPEN_CURSORS.discard(cursor)

    # `is_pure` lets the optimizer de-duplicate repeated occurrences of the
    # same scan within one plan (the source is a deterministic file read).
    return register_io_source(_read, schema=schema, is_pure=True)


def _predicate_key_filter(
    pred: pl.Expr,
    key_names: list[str],
    reader: Reader,
    id_mode: bool = False,
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
    if id_mode:
        # Bounded ID mode: no dictionary may be materialized, so label
        # existence is checked through the bounded lookup instead of the
        # full UEL table. Filters stay in label space; the IDs are
        # resolved when the exact predicate is re-applied.
        known = set()
        batch = []
        for d, labels in filters:
            batch.extend(labels)
        ids = reader.label_ids(batch)
        for label, id_ in zip(batch, ids):
            if id_ is not None:
                known.add(label)
        resolved = []
        for d, labels in filters:
            existing = [l for l in labels if l in known]
            if not existing:
                # Label never used in this dimension: no record can match.
                return [(d, [])]
            resolved.append((d, existing))
        return resolved
    uels = set(reader.uel_table())
    resolved = []
    for d, labels in filters:
        existing = [l for l in labels if l in uels]
        if not existing:
            # Label never used in this dimension: no record can match.
            return [(d, [])]
        resolved.append((d, existing))
    return resolved


def _references_key_labels(pred: pl.Expr, key_names: list[str]) -> bool:
    """True when the serialized predicate plan compares a key column
    against string label literals (== either side, or is_in), which Polars
    cannot evaluate against UInt32 ID columns."""
    return bool(_classify_label_conjuncts(pred, key_names)[0])


def _split_conjuncts(node: dict) -> list[dict]:
    """Split a serialized plan into its top-level AND conjuncts."""
    if "BinaryExpr" in node and node["BinaryExpr"]["op"] == "And":
        b = node["BinaryExpr"]
        return _split_conjuncts(b["left"]) + _split_conjuncts(b["right"])
    return [node]


def _label_comparison(node: dict, key_names: list[str]) -> dict[str, list[str]]:
    """If the node is a recognized key-label comparison (== or is_in
    against string literals), return {column: [labels]}; else {}."""
    if "BinaryExpr" in node:
        b = node["BinaryExpr"]
        if b["op"] == "Eq":
            for col_side, lit_side in ((b["left"], b["right"]), (b["right"], b["left"])):
                if "Column" in col_side and col_side["Column"] in key_names:
                    label = _string_literal(lit_side)
                    if label is not None:
                        return {col_side["Column"]: [label]}
            return {}
    if "Function" in node:
        f = node["Function"]
        opts = f.get("function", {}).get("Boolean", {})
        if "IsIn" in opts:
            inputs = f.get("input", [])
            if len(inputs) == 2 and "Column" in inputs[0] and inputs[0]["Column"] in key_names:
                labels = _list_literal_strings(inputs[1])
                if labels is not None:
                    return {inputs[0]["Column"]: labels}
    return {}


def _referenced_columns(node: dict) -> set[str]:
    """Every column name referenced anywhere in a serialized subtree."""
    if not isinstance(node, dict):
        return set()
    if "Column" in node:
        return {node["Column"]}
    names: set[str] = set()
    for value in node.values():
        if isinstance(value, dict):
            names |= _referenced_columns(value)
        elif isinstance(value, list):
            for item in value:
                names |= _referenced_columns(item)
    return names


def _classify_label_conjuncts(
    pred: pl.Expr, key_names: list[str]
) -> tuple[dict[str, list[str]], set[str]]:
    """Classify the predicate's top-level AND conjuncts.

    Returns ({column: [label literals]}, {key columns referenced by other
    conjuncts}). A key column that appears in both sets mixes string-label
    and other (e.g. numeric) comparisons and cannot be evaluated against a
    single frame."""
    try:
        plan = json.loads(pred.meta.serialize(format="json"))
    except Exception:
        return {}, set()
    labels: dict[str, list[str]] = {}
    other: set[str] = set()
    for conjunct in _split_conjuncts(plan):
        comparison = _label_comparison(conjunct, key_names)
        if comparison:
            for column, literals in comparison.items():
                labels.setdefault(column, []).extend(literals)
        else:
            other |= _referenced_columns(conjunct) & set(key_names)
    return labels, other


def _apply_key_label_predicate(
    df: pl.DataFrame, reader, predicate: pl.Expr, key_names: list[str]
) -> pl.DataFrame:
    """Resolve only label literals, preserving all other expressions and ID types.

    JSON round-tripping uses the same local Polars expression representation as
    predicate extraction. No file-provided serialized expression is loaded.
    """
    labels, other = _classify_label_conjuncts(predicate, key_names)
    mixed = sorted(set(labels) & other)
    if mixed:
        raise ValueError(
            f"the predicate mixes string-label and numeric comparisons on the "
            f"same uel_id key column(s) {mixed}; compare integer IDs "
            "consistently (labels resolve to their original UEL IDs)"
        )
    plan = json.loads(predicate.meta.serialize(format="json"))
    expressions = []
    for conjunct in _split_conjuncts(plan):
        comparison = _label_comparison(conjunct, key_names)
        if comparison:
            name, literals = next(iter(comparison.items()))
            resolved = reader.label_ids(sorted(set(literals)))
            ids = sorted({value for value in resolved if value is not None})
            expression = pl.col(name).is_in(ids) if ids else pl.lit(False)
        else:
            expression = pl.Expr.deserialize(io.StringIO(json.dumps(conjunct)), format="json")
        expressions.append(expression)
    mask = pl.lit(True)
    for expression in expressions:
        mask &= expression
    return df.filter(mask)


def _exact_expr(
    filters: list[tuple[int, list[str]]],
    key_names: list[str],
    reader: Reader | None = None,
    id_mode: bool = False,
) -> pl.Expr:
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
        col = pl.col(key_names[d])
        if id_mode:
            # Bounded ID mode: key columns hold the original one-based
            # UEL IDs, so resolve the exact labels to their IDs through
            # the bounded lookup and compare integers. The full dictionary
            # is never materialized.
            ids = reader.label_ids(list(dict.fromkeys(labels)))
            resolved = sorted({i for i in ids if i is not None})
            if not resolved:
                return pl.lit(False)
            dim_expr = (
                col == resolved[0] if len(resolved) == 1 else col.is_in(resolved)
            )
        else:
            # No `.cast(pl.String)`: key columns are Enums, so `==`/`is_in`
            # on the stored labels resolve to integer category comparisons.
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
    if "BinaryExpr" in node and node["BinaryExpr"]["op"] == "And":
        binary = node["BinaryExpr"]
        return _collect_eq_filters(binary["left"], key_names, out) and _collect_eq_filters(
            binary["right"], key_names, out
        )
    comparison = _label_comparison(node, key_names)
    out.extend((key_names.index(name), labels) for name, labels in comparison.items())
    return bool(comparison)


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


