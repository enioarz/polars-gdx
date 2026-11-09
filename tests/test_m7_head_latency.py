"""M7 task 048 — head latency: finite-limit early-stop mode resolution.

Asserts the adapter fix for the head-latency investigation: a finite
accepted-row limit pushed to the gdxrs source resolves to the serial
early-stop cursor — for ``threads="auto"`` and explicit worker requests
alike — without building or strictly reopening a checkpoint index and
without admitting any worker. ``"auto"`` stays a per-execution policy and
resolves to serial on an unprepared source, even for symbols above the
legacy 5M-record auto threshold. Explicit unlimited scans keep the
parallel path. Results are byte-identical to the serial read everywhere.

The frozen prefix expectation for ``m7-labels20k.gdx`` (BIG: 20,000
records, keys ``u00000``..``u19999``, values ``i % 1000``) is authored from
the generator recipe (``gen_m7_fixtures.rs``), not derived from the
backend under test.
"""

import os
import tempfile
from pathlib import Path

import polars as pl
import pytest

import polars_gdx.lazy as lazy
from polars_gdx import GdxrsReader, collect_batches, read_gdx, scan_gdx

GROW = "tests/data/m7/m7-labels20k.gdx"
COMPRESSED = "tests/data/compressed.gdx"

pytestmark = pytest.mark.skipif(
    not Path(GROW).exists(), reason="m7 fixtures not present"
)

if GdxrsReader is None:
    pytest.skip("gdxrs backend not compiled", allow_module_level=True)

#: Frozen BIG prefix expectation (generator recipe, 20,000 records,
#: value = ordinal % 1000).
BIG_RECORDS = 20_000
BIG_KEYS = [f"u{i:05}" for i in range(BIG_RECORDS)]
BIG_VALUES = [float(i % 1000) for i in range(BIG_RECORDS)]

THREAD_POLICIES = [None, "auto", 2, 4]
HEADS = [0, 1, 5, 1024]


def _index_directories():
    """Only this process's scratch; other scans may run beside pytest."""
    return set(
        Path(tempfile.gettempdir()).glob(f"polars-gdx-gdxrs-index-{os.getpid()}-*")
    )


def _expected_prefix(n, key_mode):
    count = min(n, BIG_RECORDS)
    if key_mode == "uel_id":
        keys = list(range(1, count + 1))
    else:
        keys = BIG_KEYS[:count]
    return {"dim_0": keys, "value": BIG_VALUES[:count]}


def _assert_frozen_prefix(df, n, key_mode):
    expected = _expected_prefix(n, key_mode)
    assert df.height == len(expected["value"])
    assert df.get_column("value").to_list() == expected["value"]
    assert df.get_column("dim_0").to_list() == expected["dim_0"]


# ---------------------------------------------------------------------------
# Deterministic prefix regressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key_mode", ["enum", "uel_id"])
@pytest.mark.parametrize("threads", THREAD_POLICIES)
@pytest.mark.parametrize("n", HEADS)
def test_head_prefix_is_the_expected_serial_prefix(n, threads, key_mode):
    """head(0/1/5/1024) produces the independently expected frozen prefix
    for every threading policy — auto and explicit workers included —
    in both key modes, via the LazyFrame path."""
    df = (
        scan_gdx(GROW, symbol="BIG", backend="gdxrs", key_mode=key_mode, threads=threads)
        .head(n)
        .collect()
    )
    _assert_frozen_prefix(df, n, key_mode)
    # Identical to the plain serial read.
    serial = scan_gdx(GROW, symbol="BIG", backend="gdxrs", key_mode=key_mode).head(n)
    assert serial.collect().equals(df)


@pytest.mark.parametrize("threads", THREAD_POLICIES)
@pytest.mark.parametrize("n", HEADS)
def test_head_prefix_on_compressed_fixture(n, threads):
    """The same early-stop guarantees hold on the zlib-compressed
    fixture (legacy-written GDX read by the gdxrs core)."""
    if not Path(COMPRESSED).exists():
        pytest.skip("compressed fixture not present")
    serial = read_gdx(COMPRESSED, symbol="big", backend="gdxrs")
    df = (
        scan_gdx(COMPRESSED, symbol="big", backend="gdxrs", threads=threads)
        .head(n)
        .collect()
    )
    assert df.equals(serial.head(n))


@pytest.mark.parametrize("threads", [None, "auto", 2, 4])
@pytest.mark.parametrize("n", [0, 1, 5, 1024])
def test_direct_reader_head_is_serial_early_stop(n, threads):
    """The direct reader API applies the same resolution: a finite
    n_rows with requested workers still produces the frozen prefix."""
    with GdxrsReader(GROW) as reader:
        if threads == "auto":
            # "auto" is a Python-layer policy; the reader API takes the
            # resolved worker count. Exercise the equivalent explicit
            # worker request (the resolution under test is the Rust-side
            # limit guard) plus the serial default.
            requests = [None, 4]
        else:
            requests = [threads]
        for requested in requests:
            frames = [pl.from_arrow(b) for b in reader.scan_batches("BIG", n_rows=n, threads=requested)]
            if frames:
                df = pl.concat(frames)
                assert df.height == min(n, BIG_RECORDS)
                assert df.get_column("value").to_list() == BIG_VALUES[: df.height]
            else:
                assert n == 0


# ---------------------------------------------------------------------------
# No index build/reopen, zero admitted workers on finite early stop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threads", [None, 2, 3, 4])
def test_finite_limit_admits_no_workers_and_no_index(threads):
    """A finite accepted-row limit resolves to the serial early-stop
    cursor even when workers were explicitly requested: the cursor
    reports no parallel lifecycle counters (zero admitted workers) and
    no checkpoint-index sidecar directory appears for this process."""
    before = _index_directories()
    with GdxrsReader(GROW) as reader:
        cursor = reader.scan_batches("BIG", n_rows=5, threads=threads)
        for _ in cursor:
            pass
        stats = cursor.scan_statistics()
        assert "requested_workers" not in stats
        assert "admitted_workers" not in stats
        assert "joined_workers" not in stats
        assert stats["decoded_rows"] <= 5
        cursor.close()
    assert _index_directories() == before


def test_unlimited_explicit_workers_still_run_parallel():
    """Control for the assertions above: an unlimited scan with explicit
    workers still runs the parallel path (parallel lifecycle counters
    present) and stays byte-identical to the serial read."""
    with GdxrsReader(GROW) as reader:
        cursor = reader.scan_batches("BIG", threads=4, batch_rows=1000)
        seen = False
        for _ in cursor:
            stats = cursor.scan_statistics()
            assert stats["requested_workers"] == 4
            assert stats["admitted_workers"] == 4
            seen = True
        assert seen
        cursor.close()
    assert read_gdx(GROW, symbol="BIG", backend="gdxrs", threads=4).equals(
        read_gdx(GROW, symbol="BIG", backend="gdxrs")
    )


@pytest.mark.parametrize("threads", ["auto", 4])
def test_lazy_head_never_builds_an_index(threads):
    """The LazyFrame path with a pushed head limit creates no checkpoint
    index directory, for auto and explicit worker requests alike."""
    before = _index_directories()
    assert (
        scan_gdx(GROW, symbol="BIG", backend="gdxrs", threads=threads)
        .head(5)
        .collect()
        .get_column("value")
        .to_list()
        == BIG_VALUES[:5]
    )
    assert _index_directories() == before


# ---------------------------------------------------------------------------
# "auto" policy resolution
# ---------------------------------------------------------------------------


def test_auto_above_threshold_resolves_serial(monkeypatch):
    """auto is resolved per execution and stays serial even for a symbol
    above the legacy 5M-record auto threshold (threshold monkeypatched
    below the fixture's record count): no index is built and the results
    match the serial read, for full scans and heads."""
    monkeypatch.setattr(lazy, "_AUTO_THREADS_MIN_RECORDS", 1)
    assert BIG_RECORDS >= lazy._AUTO_THREADS_MIN_RECORDS
    before = _index_directories()
    serial = scan_gdx(GROW, symbol="BIG", backend="gdxrs")
    auto = scan_gdx(GROW, symbol="BIG", backend="gdxrs", threads="auto")
    assert auto.collect().equals(serial.collect())
    assert auto.head(5).collect().equals(serial.head(5).collect())
    # A full auto scan resolved to serial: no parallel preparation
    # (index build/strict reopen) happened for this process.
    assert _index_directories() == before


@pytest.mark.parametrize("big", [os.environ.get("POLARS_GDX_HEAD_LATENCY_BIG")])
def test_auto_head_on_genuinely_large_symbol(big):
    """Env-gated: auto and explicit worker requests on a genuinely
    large (>= 5M-record) fixture produce the serial prefix without
    parallel preparation. Point POLARS_GDX_HEAD_LATENCY_BIG at a
    generated large fixture (symbol BIG, value/dim_0 columns)."""
    if not big or not Path(big).exists():
        pytest.skip("POLARS_GDX_HEAD_LATENCY_BIG fixture not provided")
    before = _index_directories()
    serial_head = scan_gdx(big, symbol="BIG", backend="gdxrs").head(5).collect()
    for threads in ("auto", 4):
        df = (
            scan_gdx(big, symbol="BIG", backend="gdxrs", threads=threads)
            .head(5)
            .collect()
        )
        assert df.equals(serial_head)
        assert df.get_column("value").to_list() == BIG_VALUES[:5]
    assert _index_directories() == before


# ---------------------------------------------------------------------------
# Predicates, projections and consumers on the early-stop path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("threads", ["auto", 4])
def test_translatable_predicate_head(threads):
    """A fully translatable key predicate keeps its pushed limit: the
    early-stop path resolves serial and the exact filtered prefix is
    returned."""
    before = _index_directories()
    df = (
        scan_gdx(GROW, symbol="BIG", backend="gdxrs", threads=threads)
        .filter(pl.col("dim_0") == "u00000")
        .head(5)
        .collect()
    )
    assert df.height == 1
    assert df.get_column("dim_0").to_list() == ["u00000"]
    assert df.get_column("value").to_list() == [0.0]
    assert _index_directories() == before


@pytest.mark.parametrize("threads", ["auto", 4])
def test_residual_predicate_head_keeps_exact_semantics(threads):
    """A residual (non-translatable) predicate never has its output
    limit pushed ahead of filtering: the result matches the serial
    backend's filter-then-head exactly, whatever execution mode was
    resolved for the unlimited source read."""
    serial = scan_gdx(GROW, symbol="BIG", backend="gdxrs")
    expected = serial.filter(pl.col("value") > 500).head(7).collect()
    df = (
        scan_gdx(GROW, symbol="BIG", backend="gdxrs", threads=threads)
        .filter(pl.col("value") > 500)
        .head(7)
        .collect()
    )
    assert df.equals(expected)
    assert df.get_column("value").to_list() == [float(v) for v in range(501, 508)]


@pytest.mark.parametrize("threads", ["auto", 4])
@pytest.mark.parametrize("n", [1, 5, 1024])
def test_projection_head(threads, n):
    """Projection pushdown composes with the early-stop path: projected
    key-only and key+value heads match the serial read."""
    serial = scan_gdx(GROW, symbol="BIG", backend="gdxrs")
    for projection in (["value"], ["dim_0"], ["dim_0", "value"]):
        expected = serial.select(projection).head(n).collect()
        df = (
            scan_gdx(GROW, symbol="BIG", backend="gdxrs", threads=threads)
            .select(projection)
            .head(n)
            .collect()
        )
        assert df.equals(expected)


@pytest.mark.parametrize("threads", ["auto", 4])
def test_disposed_batch_consumer_on_early_stop(threads):
    """A disposed batch consumer on a head plan cancels and joins
    cleanly on the early-stop path."""
    plan = scan_gdx(GROW, symbol="BIG", backend="gdxrs", threads=threads).head(5)
    batches = collect_batches(plan)
    rows = 0
    for batch in batches:
        rows += batch.height
    batches.close()
    assert rows == 5
    # Disposal before exhaustion also joins cleanly.
    batches = collect_batches(plan)
    next(iter(batches))
    batches.close()
    assert lazy._IN_FLIGHT == 0


def test_explicit_worker_admission_still_validated():
    """The requested-admission validation is unchanged: explicit counts
    above the 2..=4 admission are rejected even when a limit would have
    resolved them to serial."""
    for bad in (5, 8):
        with pytest.raises(ValueError, match="worker admission"):
            scan_gdx(GROW, symbol="BIG", backend="gdxrs", threads=bad).head(5)
        with pytest.raises(ValueError, match="worker admission"):
            read_gdx(GROW, symbol="BIG", backend="gdxrs", threads=bad)
    with GdxrsReader(GROW) as reader:
        with pytest.raises(ValueError, match="worker admission"):
            reader.scan_batches("BIG", n_rows=5, threads=5)
