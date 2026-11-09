"""Arrow stream interoperability and ownership of exported batch consumers."""

import gc
import subprocess
import sys
import threading
from pathlib import Path

import polars as pl
import pyarrow as pa
import pytest

from polars_gdx import AVAILABLE_BACKENDS, scan_gdx
import polars_gdx.lazy as lazy

DICT = "tests/data/m7/m7-dictionary.gdx"
GROW = "tests/data/m7/m7-labels20k.gdx"

pytestmark = pytest.mark.skipif(
    "gdxrs" not in AVAILABLE_BACKENDS or not Path(DICT).exists(),
    reason="gdxrs backend and M7 fixtures required",
)


@pytest.mark.parametrize("workers", [None, 4])
@pytest.mark.parametrize("key_mode", ["enum", "uel_id"])
@pytest.mark.parametrize("start_lazy", [False, True])
def test_arrow_export_matches_polars(workers, key_mode, start_lazy):
    lf = scan_gdx(DICT, symbol="P", backend="gdxrs", threads=workers, key_mode=key_mode)
    native = pl.LazyFrame.collect_batches(lf, chunk_size=2, lazy=start_lazy)
    with pa.RecordBatchReader.from_stream(native) as reader:
        expected = reader.read_all()
    with pa.RecordBatchReader.from_stream(
        lf.collect_batches(chunk_size=2, lazy=start_lazy)
    ) as reader:
        actual = reader.read_all()
    assert actual.equals(expected, check_metadata=True)


@pytest.mark.parametrize("dispose_original", ["drop", "close", "keep"])
@pytest.mark.parametrize("release_stream", ["close", "drop", "thread"])
def test_arrow_stream_owns_execution(dispose_original, release_stream):
    before = set(lazy._OPEN_CURSORS)
    lf = scan_gdx(GROW, symbol="BIG", backend="gdxrs", threads=4)
    batches = lf.collect_batches(chunk_size=100)
    execution = batches._execution
    reader = pa.RecordBatchReader.from_stream(batches)
    first = reader.read_next_batch()
    snapshot = first.to_pydict()
    owned = set(lazy._OPEN_CURSORS) - before
    assert owned
    if dispose_original == "close":
        batches.close()
    if dispose_original != "keep":
        del batches
    del lf
    gc.collect()
    assert not execution.closed
    assert reader.read_next_batch().num_rows == 100
    if release_stream == "close":
        reader.close()
    elif release_stream == "drop":
        del reader
        gc.collect()
    else:
        thread = threading.Thread(target=reader.close)
        thread.start()
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert execution.closed
    assert not owned & lazy._OPEN_CURSORS
    assert all(c.scan_statistics()["joined_workers"] == 4 for c in owned)
    assert first.to_pydict() == snapshot


def test_unconsumed_arrow_capsule_releases_execution():
    lf = scan_gdx(GROW, symbol="BIG", backend="gdxrs", threads=4)
    batches = lf.collect_batches(chunk_size=100, lazy=True)
    execution = batches._execution
    capsule = batches.__arrow_c_stream__(requested_schema=None)
    del batches, lf
    gc.collect()
    assert not execution.closed
    del capsule
    gc.collect()
    assert execution.closed


def test_requested_schema_matches_native_protocol():
    lf = scan_gdx(DICT, symbol="P", backend="gdxrs").select("value")
    requested = pa.schema([("value", pa.float64())])
    native = pl.LazyFrame.collect_batches(lf, chunk_size=2)
    batches = lf.collect_batches(chunk_size=2)
    try:
        try:
            with pa.RecordBatchReader.from_stream(native, schema=requested) as reader:
                expected = reader.read_all()
        except (ValueError, TypeError, NotImplementedError) as error:
            with pytest.raises(type(error)):
                pa.RecordBatchReader.from_stream(batches, schema=requested)
            # Drain the native reference iterator if negotiation was rejected.
            list(native)
        else:
            with pa.RecordBatchReader.from_stream(batches, schema=requested) as reader:
                assert reader.read_all().equals(expected, check_metadata=True)
    finally:
        batches.close()


def test_arrow_source_error_preserves_diagnostics_and_cleanup(tmp_path):
    path = tmp_path / "changed.gdx"
    path.write_bytes(Path(GROW).read_bytes())
    lf = scan_gdx(path, symbol="BIG", backend="gdxrs", threads=4)
    batches = lf.collect_batches(lazy=True)
    execution = batches._execution
    with path.open("ab") as file:
        file.write(b"changed after opening")
    try:
        with pytest.raises(Exception, match="[Ss]ource|changed"):
            with pa.RecordBatchReader.from_stream(batches) as reader:
                reader.read_all()
    finally:
        batches.close()
    assert execution.closed


@pytest.mark.parametrize("arrow_first", [False, True])
def test_unreleased_arrow_stream_exits_cleanly(arrow_first):
    imports = "import pyarrow as pa\nfrom polars_gdx import scan_gdx\n"
    if not arrow_first:
        imports = "from polars_gdx import scan_gdx\nimport pyarrow as pa\n"
    program = imports + '''
lf = scan_gdx("tests/data/m7/m7-labels20k.gdx", symbol="BIG", backend="gdxrs", threads=4)
reader = pa.RecordBatchReader.from_stream(lf.collect_batches(chunk_size=100))
print(reader.read_next_batch().num_rows, flush=True)
'''
    for _ in range(3):
        result = subprocess.run(
            [sys.executable, "-c", program], capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "100"


def test_shutdown_drains_admitted_arrow_release_callback():
    # Isolate the native shutdown gate: it is intentionally irreversible.
    program = '''
import threading
import time
import polars as pl
import pyarrow as pa
from polars_gdx import _core
started = threading.Event()
finish = threading.Event()
closed = threading.Event()
class Owner:
    def close(self):
        started.set()
        assert finish.wait(10)
        closed.set()
native = pl.LazyFrame({"x": [1, 2]}).collect_batches()
holder, guard = _core._retain_arrow_stream(native.__arrow_c_stream__(), Owner())
reader = pa.RecordBatchReader.from_stream(holder)
releaser = threading.Thread(target=reader.close)
releaser.start()
assert started.wait(10)
def unblock():
    time.sleep(0.2)
    finish.set()
unblocker = threading.Thread(target=unblock)
unblocker.start()
_core._disable_arrow_callbacks()
assert closed.is_set()
releaser.join(10)
unblocker.join(10)
assert not releaser.is_alive()
assert not unblocker.is_alive()
'''
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
