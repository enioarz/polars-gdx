"""Regression coverage for empty reads and native row limits."""

import polars as pl
import pytest

from polars_gdx import Reader, scan_gdx


@pytest.mark.parametrize(
    "path,symbol,label",
    [
        ("tests/data/trnsport.gdx", "x", "seattle"),
        ("tests/data/compressed.gdx", "big", None),
    ],
)
@pytest.mark.parametrize("threads", [None, 4])
def test_raw_limits_match_sliced_read(path, symbol, label, threads):
    with Reader(path) as reader:
        if label is None:
            label = reader.read_arrow(symbol).column(0)[0].as_py()
        for filters in [None, [(0, [label])]]:
            full = reader.read_arrow(symbol, key_filter=filters, threads=threads)
            for limit in [0, 1, 2, full.num_rows + 1]:
                limited = reader.read_arrow(
                    symbol, key_filter=filters, n_rows=limit, threads=threads
                )
                assert limited.equals(full.slice(0, limit))


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("labels", [[], ["not-a-stored-label"]])
@pytest.mark.parametrize("threads", [None, 4])
def test_empty_filter_preserves_schema(dim, labels, threads):
    path = "tests/data/trnsport.gdx"
    with Reader(path) as reader:
        expected = reader.read_arrow("x").slice(0, 0)
        actual = reader.read_arrow("x", key_filter=[(dim, labels)], threads=threads)
        assert actual.equals(expected)
        # An early return must leave subsequent reads usable.
        assert reader.read_arrow("x").num_rows == 6
    frame = scan_gdx(path, symbol="x", key_filter={dim: labels}, threads=threads).collect()
    assert frame.is_empty()
    assert frame.schema == scan_gdx(path, symbol="x").collect_schema()


def test_empty_read_still_validates_arguments():
    with Reader("tests/data/trnsport.gdx") as reader:
        with pytest.raises(RuntimeError, match="out of range"):
            reader.read_arrow("x", key_filter=[(2, [])], n_rows=0)
        with pytest.raises(RuntimeError, match="unknown value field"):
            reader.read_arrow("x", value_field="invalid", n_rows=0)
        with pytest.raises(RuntimeError, match="collision"):
            reader.read_arrow("x", key_names=["level", "j"], n_rows=0)


def test_compressed_span_within_one_block_after_parallel_read():
    with Reader("tests/data/compressed.gdx") as reader:
        full = reader.read_arrow("big")
        label = full.column(0)[0].as_py()
        expected = reader.read_arrow("big", key_filter=[(0, [label])])
        assert expected.num_rows > 0
        # Populate the restart cache so the next read selects the span path.
        assert reader.read_arrow("big", threads=4).equals(full)
        assert reader.read_arrow(
            "big", key_filter=[(0, [label])], threads=4
        ).equals(expected)


def test_head_after_value_predicate_still_applies_limit_last():
    frame = scan_gdx("tests/data/trnsport.gdx", symbol="x")
    assert frame.filter(pl.col("level") > 0).head(2).collect().equals(
        frame.collect().filter(pl.col("level") > 0).head(2)
    )
