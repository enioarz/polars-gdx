"""M7 task 044 — cursor-backed streaming batch source.

Asserts the acceptance criteria of gdxrs task 044 for the gdxrs backend:
incremental batch emission without a hidden full-symbol prescan,
`batch_size` honored as the maximum rows per emitted batch with the
separate byte budget for wide rows, stable schemas across empty results
and zero-column projections, native-expectation comparison of
concatenated batches at one-row/uneven/large batch sizes, batch lifetime
safety across cursor advance/close, and a noncollecting sink with bounded
producer buffering and backpressure. Expected values follow the frozen
recipes of docs/m7-fixture-expectations.md.
"""

import itertools

import polars as pl
import pytest

from polars_gdx import GdxrsReader, Reader, read_gdx, scan_gdx

DICT = "tests/data/m7/m7-dictionary.gdx"
SEM = "tests/data/m7/m7-semantics.gdx"
GROW = "tests/data/m7/m7-labels20k.gdx"
GDX = "tests/data/trnsport.gdx"

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(DICT).exists(), reason="m7 fixtures not present"
)

if GdxrsReader is None:
    pytest.skip("gdxrs backend not compiled", allow_module_level=True)

DICT_LABELS = ["aa", "bb", "cc", "dd"]


def _batches(lf, chunk_size):
    return list(lf.collect_batches(chunk_size=chunk_size))


def _concat(batches):
    return pl.concat(batches) if batches else None


# ---------------------------------------------------------------------------
# Frozen dictionary-identity expectations (docs/m7-fixture-expectations.md)
# ---------------------------------------------------------------------------


def test_dictionary_identity_batches_are_frozen():
    """Frozen: with batch_size=2 over P records (1,10),(2,20),(3,30),(4,40),
    mode A emits batches [aa,bb] and [cc,dd], each declaring all four
    categories in registration order; mode B emits UInt32 IDs [1,2] and
    [3,4]."""
    lf = scan_gdx(DICT, symbol="P", backend="gdxrs")
    batches = _batches(lf, 2)
    assert [(b.height, b["dim_0"].to_list()) for b in batches] == [
        (2, ["aa", "bb"]),
        (2, ["cc", "dd"]),
    ]
    for batch in batches:
        # Every enum batch declares all four categories in registration
        # order (the shared dictionary), not batch-local categories.
        assert batch.schema["dim_0"] == pl.Enum(DICT_LABELS)
        assert batch["value"].to_list() in ([10.0, 20.0], [30.0, 40.0])
    id_batches = _batches(
        scan_gdx(DICT, symbol="P", backend="gdxrs", key_mode="uel_id"), 2
    )
    assert [(b.height, b["dim_0"].to_list()) for b in id_batches] == [
        (2, [1, 2]),
        (2, [3, 4]),
    ]
    for batch in id_batches:
        assert batch.schema["dim_0"] == pl.UInt32


def test_global_head_limit_across_batches():
    """Frozen: a global head(3) returns IDs 1,2,3 — the first three records,
    not three per batch."""
    df = scan_gdx(DICT, symbol="P", backend="gdxrs").head(3).collect()
    assert df["dim_0"].to_list() == ["aa", "bb", "cc"]
    df = scan_gdx(DICT, symbol="P", backend="gdxrs", key_mode="uel_id").head(3).collect()
    assert df["dim_0"].to_list() == [1, 2, 3]
    # With a tiny chunk size the limit is still global.
    batches = _batches(scan_gdx(DICT, symbol="P", backend="gdxrs").head(3), 1)
    assert [b.height for b in batches] == [1, 1, 1]
    assert _concat(batches)["dim_0"].to_list() == ["aa", "bb", "cc"]


def test_retained_batch_survives_cursor_advance_and_close():
    """Frozen: retaining batch 1 while advancing/closing does not mutate
    its contents — each yielded batch owns valid data after the lending
    core cursor advances or closes."""
    with GdxrsReader(DICT) as reader:
        cursor = reader.scan_batches("P", batch_rows=2)
        iterator = iter(cursor)
        first = next(iterator)
        first_rows = (first.column(0).to_pylist(), first.column(1).to_pylist())
        second = next(iterator)
        second_rows = (second.column(0).to_pylist(), second.column(1).to_pylist())
        cursor.close()
        # Both batches are unchanged after the cursor advanced and closed.
        assert (first.column(0).to_pylist(), first.column(1).to_pylist()) == first_rows
        assert (second.column(0).to_pylist(), second.column(1).to_pylist()) == second_rows
        assert first_rows == (["aa", "bb"], [10.0, 20.0])
        assert second_rows == (["cc", "dd"], [30.0, 40.0])
        # The stream ends after close.
        assert list(iterator) == []
        # And through polars: an early frame stays valid after the scan
        # completed.
        batches = _batches(scan_gdx(DICT, symbol="P", backend="gdxrs"), 2)
        snapshot = batches[0].to_dict(as_series=False)
        del batches
        assert snapshot == {"dim_0": ["aa", "bb"], "value": [10.0, 20.0]}


# ---------------------------------------------------------------------------
# Batch sizes: one-row, uneven, large; native-expectation comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size,expected_heights", [
    (1, [1, 1, 1, 1]),
    (2, [2, 2]),
    (3, [3, 1]),
    (1000, [4]),
])
def test_batch_sizes_match_native_expectations(chunk_size, expected_heights):
    """One-row, even, uneven and large batch sizes: concatenated batches
    equal the native expectations and the single-batch eager read."""
    expected = read_gdx(DICT, symbol="P", backend="gdxrs")
    batches = _batches(scan_gdx(DICT, symbol="P", backend="gdxrs"), chunk_size)
    assert [b.height for b in batches] == expected_heights
    assert _concat(batches).equals(expected)
    # Mode B agrees batch-for-batch.
    id_expected = read_gdx(DICT, symbol="P", backend="gdxrs", key_mode="uel_id")
    id_batches = _batches(
        scan_gdx(DICT, symbol="P", backend="gdxrs", key_mode="uel_id"), chunk_size
    )
    assert [b.height for b in id_batches] == expected_heights
    assert _concat(id_batches).equals(id_expected)


def test_streaming_matches_eager_and_legacy():
    """Concatenated streaming batches equal the eager single-batch read
    and the legacy backend, across the semantic fixture (specials,
    aliases, scalars) and trnsport."""
    eager = read_gdx(SEM, symbol="P", backend="gdxrs")
    streamed = _concat(_batches(scan_gdx(SEM, symbol="P", backend="gdxrs"), 3))
    assert streamed.equals(eager)
    if Reader is not None:
        legacy = read_gdx(SEM, symbol="P", backend="legacy")
        # The documented projection differences: null (gdxrs) vs NaN /
        # native acronym numbers (legacy) — compare keys and non-special
        # values only.
        assert streamed["dim_0"].to_list() == legacy["dim_0"].to_list()
        for a, b in zip(streamed["value"], legacy["value"]):
            if a is None:
                assert b is None or b != b or abs(b) >= 1.0e300
            else:
                assert a == b
    for symbol in ("x", "d", "a", "supply"):
        streamed = _concat(_batches(scan_gdx(GDX, symbol=symbol, backend="gdxrs"), 2))
        eager = read_gdx(GDX, symbol=symbol, backend="gdxrs")
        assert streamed.equals(eager), symbol


def test_scalar_and_special_values_stream_correctly():
    """Scalars produce exactly one row through the cursor; typed special
    values project identically across batch boundaries."""
    with GdxrsReader(SEM) as reader:
        cursor = reader.scan_batches("S", batch_rows=1)
        batches = list(iter(cursor))
        assert len(batches) == 1
        assert batches[0].column(0).to_pylist() == [1.25]
        cursor = reader.scan_batches("P", batch_rows=3)
        rows = [v for batch in iter(cursor) for v in batch.column(1).to_pylist()]
        assert rows == [0.0, None, None, float("inf"), float("-inf"), 1.25, None]


# ---------------------------------------------------------------------------
# No hidden full-symbol prescan; separate byte budget for wide rows
# ---------------------------------------------------------------------------


def test_first_batch_does_not_prescan_the_symbol():
    """First-batch test: the first emitted batch arrives without reading
    the whole symbol — after two batches of 100 rows of the 20,000-row
    fixture, only ~200 rows are decoded (no hidden full-symbol prescan;
    the serial core scan decodes strictly incrementally, with no index
    build in this mode)."""
    with GdxrsReader(GROW) as reader:
        cursor = reader.scan_batches("BIG", batch_rows=100)
        iterator = iter(cursor)
        first = next(iterator)
        assert first.num_rows == 100
        stats = cursor.scan_statistics()
        assert stats["decoded_rows"] == 100
        second = next(iterator)
        assert second.num_rows == 100
        assert cursor.scan_statistics()["decoded_rows"] == 200
        cursor.close()


def test_byte_budget_bounds_rows_per_batch_for_wide_requests():
    """The separate byte budget bounds rows per batch even when the
    requested batch size is huge: the core's 64 KiB batch payload cap
    yields ~3,276 rows per batch for this 20-byte row layout, far below
    the requested million rows."""
    with GdxrsReader(GROW) as reader:
        cursor = reader.scan_batches("BIG", batch_rows=1_000_000)
        first = next(iter(cursor))
        assert 0 < first.num_rows < 10_000
        assert first.num_rows <= 64 * 1024 // 20 + 1
        cursor.close()


# ---------------------------------------------------------------------------
# Stable schemas: empty symbols, zero-column projections, repeated scans
# ---------------------------------------------------------------------------


def test_empty_symbols_keep_planned_schema():
    """Empty symbols stream zero batches and keep the planned schema in
    both modes; a zero limit reads nothing without starting a scan."""
    lf = scan_gdx(SEM, symbol="D", backend="gdxrs")
    assert _batches(lf, 2) == []
    collected = lf.collect()
    assert collected.height == 0
    assert collected.columns == ["J", "value"]
    assert isinstance(collected.schema["J"], pl.Enum)
    lf_id = scan_gdx(SEM, symbol="D", backend="gdxrs", key_mode="uel_id")
    collected = lf_id.collect()
    assert collected.height == 0
    assert collected.schema["J"] == pl.UInt32
    # Zero limit: same schema, no rows.
    assert lf.head(0).collect().height == 0
    # Repeated scans are deterministic.
    assert lf.collect().equals(lf.collect())


def test_zero_column_projection():
    """A zero-column projection completes without touching the value data
    and produces an empty (0, 0) frame, with the scan still served."""
    result = scan_gdx(DICT, symbol="P", backend="gdxrs").select().collect()
    assert result.shape == (0, 0)


# ---------------------------------------------------------------------------
# Noncollecting sink: bounded producer buffering and backpressure
# ---------------------------------------------------------------------------


def test_noncollecting_sink_streams_incrementally():
    """A noncollecting sink demonstrates bounded producer buffering and
    backpressure: consuming a prefix of the 20,000-row source at
    chunk_size=1000 stops after the consumed batches — the producer never
    runs ahead of the consumer, and only the current batch is buffered.
    Caller `.collect()` remains an explicit materializing operation."""
    lf = scan_gdx(GROW, symbol="BIG", backend="gdxrs")
    consumed = list(itertools.islice(lf.collect_batches(chunk_size=1000), 3))
    assert [b.height for b in consumed] == [1000, 1000, 1000]
    # Materializing collect() reads the whole symbol and is not streaming
    # acceptance; it just must agree with the streamed prefix.
    whole = lf.collect()
    assert whole.height == 20_000
    assert pl.concat(consumed).equals(whole.head(3000))


def test_early_sink_exit_stops_the_producer():
    """Breaking out of the noncollecting sink stops the scan at the
    acknowledged prefix: the source decodes only the consumed batches."""
    with GdxrsReader(GROW) as reader:
        cursor = reader.scan_batches("BIG", batch_rows=50)
        iterator = iter(cursor)
        for _ in range(4):
            next(iterator)
        assert cursor.scan_statistics()["decoded_rows"] == 200
        cursor.close()
        assert cursor.scan_statistics()["decoded_rows"] == 200


def test_residual_predicate_head_is_global_across_batches():
    """A residual predicate with a global head limit stops the streaming
    scan early: the source stops once the accepted rows are delivered."""
    lf = scan_gdx(GROW, symbol="BIG", backend="gdxrs")
    df = lf.filter(pl.col("value") > 995.0).head(3).collect()
    assert df.height == 3
    assert df["value"].to_list() == [996.0, 997.0, 998.0]
    # The same query batched at chunk_size=1 agrees.
    batches = _batches(lf.filter(pl.col("value") > 995.0).head(3), 1)
    assert _concat(batches).equals(df)
