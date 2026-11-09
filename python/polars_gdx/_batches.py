"""Execution ownership for GDX batch consumers.

Polars 2.1 can retain a Python source generator after dropping its batch
consumer. Bind sources to an execution before starting it, so consumer disposal
can close cursors independently of that generator's eventual finalization.
"""

from __future__ import annotations

import atexit
import contextvars
import io
import threading
import uuid
import weakref

import polars as pl

from polars_gdx import _core

# Registered before lazy.py's cursor drain and weakref's consumer finalizers;
# LIFO atexit ordering disables foreign-thread callbacks after those cleanups.
atexit.register(_core._disable_arrow_callbacks)

_BINDING = contextvars.ContextVar("gdx_batch_execution", default=None)
_SOURCES = weakref.WeakValueDictionary()


class _Execution:
    def __init__(self):
        self._condition = threading.Condition()
        self._closed = False
        self._setups = 0
        self._cursors = {}
        self._stream_guards = []
        self.sources = {}

    @property
    def closed(self):
        with self._condition:
            return self._closed

    def enter_setup(self):
        with self._condition:
            if self._closed:
                return False
            self._setups += 1
            return True

    def leave_setup(self):
        with self._condition:
            self._setups -= 1
            self._condition.notify_all()

    def attach(self, cursor, registry):
        # Setup remains registered until publication, including when close
        # races with cursor construction.
        with self._condition:
            self._cursors[cursor] = registry
            self._condition.notify_all()

    def detach(self, cursor):
        with self._condition:
            self._cursors.pop(cursor, None)
            self._condition.notify_all()

    def retain_stream_guard(self, guard):
        with self._condition:
            if self._closed:
                guard.disarm()
            else:
                self._stream_guards.append(guard)

    def close(self):
        with self._condition:
            self._closed = True
            for guard in self._stream_guards:
                guard.disarm()
            self._stream_guards.clear()
        while True:
            with self._condition:
                pending = list(self._cursors.items())
                if not pending and not self._setups:
                    return
            error = None
            for cursor, registry in pending:
                try:
                    cursor.close()
                except BaseException as exc:
                    if error is None:
                        error = exc
                else:
                    registry.discard(cursor)
                    self.detach(cursor)
            if error is not None:
                # Keep failed owners registered for an explicit retry or
                # process shutdown; explicit close surfaces the failure.
                raise error
            with self._condition:
                if self._setups and not self._cursors:
                    self._condition.wait()


class _Source:
    def __init__(self, read):
        self.read = read
        self.key = uuid.uuid4().hex
        _SOURCES[self.key] = self


class _ScanFunction:
    def __init__(self, source, execution=None):
        self.source = source
        self.execution = execution

    def io_source(self, columns, predicate, n_rows, batch_size):
        return self.source.read(
            columns, predicate, n_rows, batch_size, _execution=self.execution
        )

    def __call__(self, columns, predicate, n_rows, batch_size):
        parsed = None
        applied = True
        if predicate:
            try:
                parsed = pl.Expr.deserialize(predicate)
            except Exception:
                # Match register_io_source: Polars evaluates a predicate
                # which this runtime cannot deserialize.
                applied = False
        return self.io_source(columns, parsed, n_rows, batch_size), applied

    def __reduce__(self):
        # Only a live in-process source reference is serialized, never the
        # Reader or its open handles. No cloudpickle dependency is needed.
        return _restore_scan, (self.source.key,)


def _restore_scan(key):
    source = _SOURCES.get(key)
    if source is None:
        raise RuntimeError("the GDX plan's in-process source is no longer alive")
    execution = _BINDING.get()
    if execution is None:
        return _ScanFunction(source)
    if key not in execution.sources:
        execution.sources[key] = _ScanFunction(source, execution)
    return execution.sources[key]


def register_source(read, schema, *, owned_batches):
    frame = pl.LazyFrame._scan_python_function(
        schema=schema,
        scan_fn=_ScanFunction(_Source(read)),
        pyarrow=False,
        is_pure=True,
    )
    if owned_batches:
        return _GdxLazyFrame._from_pyldf(frame._ldf)
    return frame


class _ArrowExecutionLease:
    def __init__(self, execution):
        self.execution = execution
        # Also close an unreleased exported stream before interpreter teardown.
        self.finalizer = weakref.finalize(self, execution.close)

    def close(self):
        self.execution.close()
        self.finalizer.detach()


class _OwnedBatches:
    def __init__(self, inner, execution):
        self._inner = inner
        self._execution = execution
        self._exported = False
        # The callback retains the execution, not this consumer. A source
        # generator retained by Polars cannot keep the consumer alive.
        self._finalizer = weakref.finalize(self, execution.close)

    def __iter__(self):
        return self

    def __next__(self):
        if self._inner is None:
            raise StopIteration
        try:
            return next(self._inner)
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                raise primary from cleanup
            raise

    def close(self):
        if self._exported:
            # Ownership moved to the Arrow stream, including across context
            # exit or disposal of this original iterator.
            return
        try:
            self._execution.close()
        finally:
            self._inner = None
        self._finalizer.detach()

    def __arrow_c_stream__(self, requested_schema=None):
        """Transfer iteration and execution ownership to an Arrow C stream."""
        if self._inner is None:
            raise RuntimeError("batch consumer is closed or already exported")
        # Let Polars negotiate the requested schema before moving ownership.
        capsule = self._inner.__arrow_c_stream__(requested_schema)
        try:
            lease = _ArrowExecutionLease(self._execution)
            stream, guard = _core._retain_arrow_stream(capsule, lease)
            self._execution.retain_stream_guard(guard)
            exported = stream.__arrow_c_stream__()
            self._finalizer.detach()
            self._exported = True
            self._inner = None
            return exported
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                raise primary from cleanup
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.close()
        except BaseException as cleanup:
            if exc is not None:
                raise exc from cleanup
            raise


def collect_batches(frame: pl.LazyFrame, **options):
    """Collect batches with GDX cancellation on close, context exit or disposal.

    Accepts Polars ``collect_batches`` keyword arguments. Use this helper for
    composed plans such as ``pl.concat`` that return a plain Polars LazyFrame.
    Gdxrs frames returned by ``scan_gdx`` use it automatically. Keep the iterator
    alive while consuming; ``with`` or explicit ``close()`` also reports cleanup
    errors. Previously yielded frames retain their own buffers.

    Arrow export transfers ownership to the exported stream. The original
    iterator is exhausted and its close becomes a no-op; releasing the stream
    closes the execution, even when the original iterator is still alive.

    Binding uses a local Polars plan clone: custom UDF captures must be
    serializable, and in-memory plan inputs are copied. GDX records and open
    readers are not serialized; live-source bindings are process-local.
    """
    execution = _Execution()
    try:
        # Clone the local plan with execution-bound scan callbacks. Normal
        # collect() keeps its existing path; concurrent consumers get distinct
        # bindings even when they execute the very same LazyFrame.
        payload = frame.serialize()
        token = _BINDING.set(execution)
        try:
            bound = pl.LazyFrame.deserialize(io.BytesIO(payload))
        finally:
            _BINDING.reset(token)
            execution.sources.clear()
        del payload
        inner = pl.LazyFrame.collect_batches(bound, **options)
        return _OwnedBatches(inner, execution)
    except BaseException as primary:
        try:
            execution.close()
        except BaseException as cleanup:
            raise primary from cleanup
        raise


class _GdxLazyFrame(pl.LazyFrame):
    def collect_batches(self, **options):
        return collect_batches(self, **options)
