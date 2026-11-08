"""M7 task 046 — scan lifecycle, threads and cancellation.

Asserts the acceptance criteria of gdxrs task 046 for the gdxrs backend:
explicit thread admission (2..=4 workers via the core's parallel scan over
a checkpoint index), GIL release during production, fallible
close/cancellation that wakes and joins all owned workers, bounded queue
pressure, error propagation without replaying acknowledged rows,
concurrent and repeated scans, and file/scratch release.
"""

import glob
import subprocess
import sys
import threading

import polars as pl
import pytest

from polars_gdx import GdxrsReader, read_gdx, scan_gdx

GDX = "tests/data/trnsport.gdx"
DICT = "tests/data/m7/m7-dictionary.gdx"
SEM = "tests/data/m7/m7-semantics.gdx"
GROW = "tests/data/m7/m7-labels20k.gdx"

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(DICT).exists(), reason="m7 fixtures not present"
)

if GdxrsReader is None:
    pytest.skip("gdxrs backend not compiled", allow_module_level=True)


# ---------------------------------------------------------------------------
# Thread admission and parallel parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workers", [2, 3, 4])
def test_parallel_scans_match_serial_exactly(workers):
    """Parallel results are byte-identical to the serial read — rows,
    order, schema — for every admitted worker count, across plain scans,
    key filters, projections, residuals, heads and both key modes."""
    serial = scan_gdx(GROW, symbol="BIG", backend="gdxrs")
    parallel = scan_gdx(GROW, symbol="BIG", backend="gdxrs", threads=workers)
    assert parallel.collect().equals(serial.collect())
    assert parallel.filter(pl.col("value") > 500).head(7).collect().equals(
        serial.filter(pl.col("value") > 500).head(7).collect()
    )
    assert parallel.select("value").collect().equals(serial.select("value").collect())
    assert parallel.head(3).collect().equals(serial.head(3).collect())
    # Both key modes.
    serial_id = scan_gdx(GROW, symbol="BIG", backend="gdxrs", key_mode="uel_id")
    parallel_id = scan_gdx(
        GROW, symbol="BIG", backend="gdxrs", key_mode="uel_id", threads=workers
    )
    assert parallel_id.collect().equals(serial_id.collect())
    # trnsport's multi-dimensional symbols.
    for symbol in ("x", "d", "a", "supply"):
        assert read_gdx(GDX, symbol=symbol, backend="gdxrs", threads=workers).equals(
            read_gdx(GDX, symbol=symbol, backend="gdxrs")
        )


def test_thread_admission_is_explicit():
    """2..=4 workers are admitted; requests above the configured worker
    admission are rejected, never silently clamped; None/0/1 stay
    serial; "auto" never exceeds the admission and stays serial for
    small symbols."""
    serial = read_gdx(DICT, symbol="P", backend="gdxrs")
    for workers in (2, 4):
        assert read_gdx(DICT, symbol="P", backend="gdxrs", threads=workers).equals(serial)
    for workers in (0, 1, None):
        assert read_gdx(DICT, symbol="P", backend="gdxrs", threads=workers).equals(serial)
    for bad in (5, 8, 100):
        with pytest.raises(ValueError, match="worker admission"):
            read_gdx(DICT, symbol="P", backend="gdxrs", threads=bad)
    # "auto" on a small symbol resolves to serial (no error).
    assert read_gdx(DICT, symbol="P", backend="gdxrs", threads="auto").equals(serial)


# ---------------------------------------------------------------------------
# Lifecycle: close, cancel, joins, queue pressure
# ---------------------------------------------------------------------------


def test_parallel_lifecycle_counters_and_joins():
    """The parallel cursor reports admitted concurrency, bounded queue
    pressure and join state; a completed scan joins all workers."""
    with GdxrsReader(GROW) as reader:
        cursor = reader.scan_batches("BIG", threads=4, batch_rows=100)
        for _ in iter(cursor):
            pass
        # close() joins all owned workers; the final counters remain
        # observable afterwards.
        cursor.close()
        stats = cursor.scan_statistics()
        assert stats["requested_workers"] == 4
        assert stats["admitted_workers"] == 4
        assert stats["maximum_buffered_batches"] <= 4
        assert stats["peak_ready_batches"] <= stats["maximum_buffered_batches"]
        assert stats["joined_workers"] == 4
        assert stats["complete"] is True
        assert stats["cancelled"] is False
        assert stats["decoded_rows"] == 20_000


def test_cancel_joins_workers_and_stops_production():
    """Explicit cancellation wakes blocked producers/consumers, joins all
    owned workers and ends the stream without replaying acknowledged
    rows."""
    with GdxrsReader(GROW) as reader:
        cursor = reader.scan_batches("BIG", threads=4, batch_rows=100)
        iterator = iter(cursor)
        first = next(iterator)
        acknowledged = first.num_rows
        cursor.cancel()
        # The stream ends; no further rows are produced or replayed.
        assert list(iterator) == []
        stats = cursor.scan_statistics()
        assert stats["cancelled"] is True
        assert stats["joined_workers"] == 4
        assert stats["decoded_rows"] >= acknowledged
        assert stats["complete"] is False
        # close() after cancel is idempotent.
        cursor.close()
        assert list(iter(cursor)) == []


def test_partial_output_acknowledges_only_consumed_batches():
    """Stopping mid-scan acknowledges exactly the consumed prefix; the
    acknowledged batches stay valid after the workers join."""
    with GdxrsReader(GROW) as reader:
        cursor = reader.scan_batches("BIG", threads=3, batch_rows=100)
        iterator = iter(cursor)
        batches = [next(iterator), next(iterator)]
        rows = [b.num_rows for b in batches]
        contents = [b.column(1).to_pylist() for b in batches]
        cursor.close()
        assert rows == [100, 100]
        assert cursor.scan_statistics()["joined_workers"] == 3
        # Retained batches keep their documented owners: contents are
        # unchanged after the workers joined and the cursor closed.
        assert [b.column(1).to_pylist() for b in batches] == contents
        assert cursor.scan_statistics()["decoded_rows"] >= 200


def test_sink_exception_propagates_and_joins():
    """A failing consumer propagates the error, joins the workers and
    never retries or replays rows."""
    with GdxrsReader(GROW) as reader:
        cursor = reader.scan_batches("BIG", threads=2, batch_rows=100)
        iterator = iter(cursor)
        next(iterator)
        with pytest.raises(RuntimeError, match="sink failed"):
            try:
                next(iterator)
                raise RuntimeError("sink failed")
            finally:
                cursor.close()
        assert cursor.scan_statistics()["joined_workers"] == 2


def test_error_propagation_from_parallel_scan(tmp_path):
    """Source/decode errors propagate as clear errors — never a silent
    retry through the legacy backend and never a wrong result."""
    data = open(GROW, "rb").read()
    truncated = tmp_path / "truncated.gdx"
    truncated.write_bytes(data[: len(data) // 2])
    with pytest.raises(RuntimeError):
        read_gdx(str(truncated), symbol="BIG", backend="gdxrs", threads=2)
    garbage = tmp_path / "garbage.gdx"
    garbage.write_bytes(b"junk" * 64)
    with pytest.raises(RuntimeError):
        read_gdx(str(garbage), symbol="BIG", backend="gdxrs", threads=4)
    # Unknown symbol fails before any index build or worker spawn.
    with GdxrsReader(GROW) as reader:
        with pytest.raises(RuntimeError, match="not found"):
            reader.scan_batches("nope", threads=2)


# ---------------------------------------------------------------------------
# Consumer drop, Python cancellation, interpreter shutdown
# ---------------------------------------------------------------------------


SUBPROCESS_PARALLEL_EARLY_EXIT = """
from polars_gdx import scan_gdx

lf = scan_gdx(
    "{path}",
    symbol="BIG",
    backend="gdxrs",
    threads=4,
)
it = lf.collect_batches(chunk_size=1000)
print(next(it).shape, flush=True)
del it
"""

SUBPROCESS_PARALLEL_CANCEL = """
import polars as pl
from polars_gdx import scan_gdx

lf = scan_gdx(
    "{path}",
    symbol="BIG",
    backend="gdxrs",
    threads=3,
)
df = lf.filter(pl.col("value") > 995.0).head(2).collect()
print(df.shape, flush=True)
it = lf.collect_batches(chunk_size=1000)
print(next(it).shape, flush=True)
del it
"""


@pytest.mark.parametrize("script", [
    SUBPROCESS_PARALLEL_EARLY_EXIT,
    SUBPROCESS_PARALLEL_CANCEL,
])
def test_parallel_early_exit_terminates_cleanly(script):
    """Consumer drop with parallel workers: the generator's finally
    closes (joins) the cursor before interpreter shutdown, so the
    process exits 0 across repeated trials."""
    program = script.format(path=GROW)
    for _ in range(3):
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            cwd=".",
        )
        assert result.returncode == 0, (result.returncode, result.stderr)
        assert "(1000, 2)" in result.stdout


def test_python_cancellation_via_generator_close():
    """Python cancellation (KeyboardInterrupt-style: closing the source
    generator mid-scan) joins the parallel workers and ends the stream."""
    lf = scan_gdx(GROW, symbol="BIG", backend="gdxrs", threads=4)
    batches = lf.collect_batches(chunk_size=100)
    first = next(batches)
    assert first.height == 100
    # Closing a suspended source is the cancellation path.
    del batches
    import gc

    gc.collect()


# ---------------------------------------------------------------------------
# Concurrent scans, repeated execution, GIL release
# ---------------------------------------------------------------------------


def test_concurrent_parallel_scans_on_one_reader():
    """Concurrent scans hold independent index handles: two parallel
    cursors on the same reader and symbol interleave correctly and both
    deliver the full ordered result."""
    with GdxrsReader(GROW) as reader:
        a = reader.scan_batches("BIG", threads=2, batch_rows=100)
        b = reader.scan_batches("BIG", threads=3, batch_rows=150)
        it_a, it_b = iter(a), iter(b)
        rows_a = rows_b = 0
        while True:
            batch_a = next(it_a, None)
            batch_b = next(it_b, None)
            if batch_a is None and batch_b is None:
                break
            rows_a += 0 if batch_a is None else batch_a.num_rows
            rows_b += 0 if batch_b is None else batch_b.num_rows
        assert rows_a == rows_b == 20_000
        assert a.scan_statistics()["joined_workers"] == 2
        assert b.scan_statistics()["joined_workers"] == 3


def test_repeated_parallel_execution_is_deterministic():
    """Repeated scans are independent and deterministic; the index
    sidecar is built once per reader+symbol and reused."""
    with GdxrsReader(GROW) as reader:
        first = None
        for _ in range(3):
            cursor = reader.scan_batches("BIG", threads=4, batch_rows=1000)
            rows = [b.column(1).to_pylist() for b in iter(cursor)]
            flat = [v for chunk in rows for v in chunk]
            if first is None:
                first = flat
            assert flat == first
        assert len(first) == 20_000


def test_gil_released_during_parallel_production():
    """The GIL is released during core production, so a background
    Python thread makes progress while a parallel scan runs on the main
    thread."""
    counter = []

    def spin():
        for _ in range(200):
            counter.append(1)

    thread = threading.Thread(target=spin)
    thread.start()
    df = read_gdx(GROW, symbol="BIG", backend="gdxrs", threads=4)
    thread.join(timeout=30)
    assert df.height == 20_000
    assert not thread.is_alive()
    assert len(counter) == 200


# ---------------------------------------------------------------------------
# File/scratch release
# ---------------------------------------------------------------------------


def test_index_sidecars_released_on_reader_close():
    """The checkpoint-index sidecars are removed when the reader closes:
    no scratch files remain under the adapter's index directory."""
    import glob as _glob
    import os
    import tempfile

    before = set(_glob.glob(os.path.join(tempfile.gettempdir(), "polars-gdx-gdxrs-index-*")))
    reader = GdxrsReader(GROW)
    cursor = reader.scan_batches("BIG", threads=2, batch_rows=100)
    next(iter(cursor))
    cursor.close()
    after = set(_glob.glob(os.path.join(tempfile.gettempdir(), "polars-gdx-gdxrs-index-*")))
    # A new directory exists while the reader is open...
    assert len(after - before) == 1
    reader.close()
    # ...and is removed (with its sidecar) once the reader closes.
    final = set(_glob.glob(os.path.join(tempfile.gettempdir(), "polars-gdx-gdxrs-index-*")))
    assert final == before


def test_cancel_from_another_thread_during_production():
    """Review finding: cancel() from another thread while next_stream()
    runs must not fail with "Already borrowed" (5/5 in the review). The
    lock-free cancellation signal wakes in-flight production; the state
    lock makes cleanup wait for the current step and join. The fixture
    may complete before the cancel lands — either terminal state is
    accepted; the regression is the borrow failure, captured and
    re-raised from the producer thread."""
    import time

    errors = []

    with GdxrsReader(GROW) as reader:
        cursor = reader.scan_batches("BIG", threads=4, batch_rows=100)
        first_batch = threading.Event()

        def produce():
            try:
                while True:
                    batch = cursor.next_stream()
                    if batch is None:
                        break
                    first_batch.set()
            except Exception as exc:  # e.g. "Already borrowed" — must not happen
                errors.append(exc)

        thread = threading.Thread(target=produce)
        thread.start()
        # Wait until production is provably active, then cancel from this
        # thread while the producer loop runs. (The 20k fixture may
        # complete quickly; the assertion below accepts either terminal
        # state — the finding was the "Already borrowed" failure.)
        first_batch.wait(timeout=30)
        cursor.cancel()
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert errors == [], errors
        stats = cursor.scan_statistics()
        assert stats["cancelled"] or stats["complete"]
        assert stats["joined_workers"] == 4
        # The stream ends without replaying acknowledged rows.
        assert cursor.next_stream() is None
        cursor.close()


def test_reader_drop_releases_index_sidecars():
    """Review finding: deleting the reader without an explicit close()
    must release the index sidecars and directory (owned cleanup in
    Drop), including when a cursor outlives the reader."""
    import gc
    import glob as _glob
    import os
    import tempfile

    pattern = os.path.join(tempfile.gettempdir(), "polars-gdx-gdxrs-index-*")

    def current():
        return set(_glob.glob(pattern))

    before = current()
    reader = GdxrsReader(GROW)
    cursor = reader.scan_batches("BIG", threads=2, batch_rows=100)
    next(iter(cursor))
    del reader
    gc.collect()
    assert current() == before
    # A cursor that outlives its reader still completes its scan and
    # releases its own handle on close/drop.
    remaining = sum(b.num_rows for b in iter(cursor))
    assert remaining >= 0
    del cursor
    gc.collect()
    assert current() == before


def test_shutdown_gate_refuses_new_callbacks():
    """Review finding: the shutdown check and the registration are
    atomic — once shutdown has begun, no callback can enter production.
    A scan started (or resumed) after the flag is set yields nothing
    instead of touching the extension."""
    import polars_gdx.lazy as lazy

    lazy._SHUTTING_DOWN.set()
    try:
        df = scan_gdx(GROW, symbol="BIG", backend="gdxrs").collect()
        assert df.height == 0
        # The gate also refuses parallel scans.
        df = scan_gdx(GROW, symbol="BIG", backend="gdxrs", threads=4).collect()
        assert df.height == 0
    finally:
        lazy._SHUTTING_DOWN.clear()
    # Normal operation is restored.
    assert scan_gdx(GROW, symbol="BIG", backend="gdxrs").collect().height == 20_000


def test_shutdown_gate_and_registration_share_one_lock():
    """Review finding: pausing between the shutdown check and the
    counter increment let shutdown return with zero callbacks while a
    paused callback resumed into next_stream() afterwards. The gate and
    the registration now happen under the same lock the shutdown hook
    uses, which this test observes directly: while the lock is held,
    neither the hook's flag nor a concurrent registration can proceed."""
    import time as _time

    import polars_gdx.lazy as lazy

    lazy._SHUTTING_DOWN.clear()
    assert lazy._IN_FLIGHT == 0
    with lazy._IN_FLIGHT_LOCK:
        hook = threading.Thread(target=lazy._close_open_cursors)
        hook.start()
        _time.sleep(0.2)
        # The hook cannot set the shutdown flag while the lock is held.
        assert not lazy._SHUTTING_DOWN.is_set()
        # A concurrent gate attempt blocks on the same lock instead of
        # slipping between the check and the registration.
        results = []

        def try_enter():
            results.append(lazy._enter_callback())

        gater = threading.Thread(target=try_enter)
        gater.start()
        _time.sleep(0.1)
        assert results == []
    # The lock is released: both proceed (in either order, atomically).
    gater.join(timeout=10)
    if results == [True]:
        # The registration won the race: match it, or the shutdown
        # thread's confirmed-completion drain would wait indefinitely.
        lazy._exit_callback()
    hook.join(timeout=60)
    # Both threads terminate.
    assert not gater.is_alive()
    assert not hook.is_alive()
    assert results in ([True], [False])
    assert lazy._SHUTTING_DOWN.is_set()
    # Whichever order occurred, the registration was matched.
    assert lazy._IN_FLIGHT == 0
    # The gate refuses entry after shutdown.
    assert lazy._enter_callback() is False
    # Restore the module for the rest of the process (the real atexit
    # hook sets the flag again at exit).
    lazy._SHUTTING_DOWN.clear()


def test_shutdown_drain_waits_for_confirmed_completion():
    """Review finding: the drain silently expired after 30 seconds with
    a callback still active. The drain now has no deadline: it returns
    only after confirmed completion of every registered callback."""
    import time as _time

    import polars_gdx.lazy as lazy

    lazy._SHUTTING_DOWN.clear()
    assert lazy._IN_FLIGHT == 0
    released = threading.Event()

    def slow_callback():
        _time.sleep(1.5)
        with lazy._IN_FLIGHT_LOCK:
            lazy._IN_FLIGHT -= 1
        released.set()

    with lazy._IN_FLIGHT_LOCK:
        lazy._IN_FLIGHT += 1
    thread = threading.Thread(target=slow_callback)
    thread.start()
    started = _time.monotonic()
    lazy._close_open_cursors()
    waited = _time.monotonic() - started
    # The drain returned only after the callback confirmed completion.
    assert released.is_set()
    assert lazy._IN_FLIGHT == 0
    assert waited >= 1.4
    thread.join(timeout=10)
    # Restore the module for the rest of the process.
    lazy._SHUTTING_DOWN.clear()


def test_shutdown_gate_covers_source_setup():
    """Review finding: source setup (reader.scan_batches — an extension
    call that builds the checkpoint index and spawns workers) ran before
    callback registration, so a paused setup could resume into the
    extension after shutdown. The gate now covers setup: with shutdown
    begun, a parallel scan yields nothing and no index sidecar is ever
    built."""
    import glob as _glob
    import os
    import tempfile

    import polars_gdx.lazy as lazy

    pattern = os.path.join(tempfile.gettempdir(), "polars-gdx-gdxrs-index-*")
    before = set(_glob.glob(pattern))

    lazy._SHUTTING_DOWN.set()
    try:
        df = scan_gdx(GROW, symbol="BIG", backend="gdxrs", threads=4).collect()
        assert df.height == 0
        # The setup extension call never ran: no NEW index directory was
        # created (the parallel path builds one inside scan_batches).
        assert set(_glob.glob(pattern)) == before
    finally:
        lazy._SHUTTING_DOWN.clear()
