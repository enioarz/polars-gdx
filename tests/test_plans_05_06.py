"""Tests for plan 05 (exact label matching) and plan 06 (reader lifecycle,
key/value name collision) from docs/plans/."""

import polars as pl
import pytest

from polars_gdx import Reader, read_gdx, scan_gdx
from polars_gdx.lazy import _key_names

GDX = "tests/data/trnsport.gdx"
EDGE = "tests/data/edgecase.gdx"

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(GDX).exists(), reason="test gdx file not present"
)


@pytest.fixture(scope="module")
def edge_available():
    return __import__("pathlib").Path(EDGE).exists()


# ---------------------------------------------------------------------------
# Plan 05 — exact label matching
# ---------------------------------------------------------------------------


def test_key_filter_exact_match():
    """key_filter matches labels exactly (case-sensitively)."""
    df = read_gdx(GDX, symbol="x", key_filter={0: ["seattle"]})
    assert df.height == 3
    assert set(df["dim_0"].unique()) == {"seattle"}


def test_key_filter_wrong_case_yields_no_rows():
    df = read_gdx(GDX, symbol="x", key_filter={0: ["SEATTLE"]})
    assert df.height == 0


def test_predicate_exact_match():
    """filter(pl.col(key) == label) matches the exact stored label."""
    df = scan_gdx(GDX, symbol="x").filter(pl.col("dim_0") == "seattle").collect()
    assert df.height == 3
    assert set(df["dim_0"].unique()) == {"seattle"}


def test_predicate_wrong_case_yields_no_rows():
    df = scan_gdx(GDX, symbol="x").filter(pl.col("dim_0") == "SEATTLE").collect()
    assert df.height == 0


# ---------------------------------------------------------------------------
# Plan 06 part A — reader lifecycle
# ---------------------------------------------------------------------------


def test_reader_close_raises_on_use():
    r = Reader(GDX)
    r.close()
    with pytest.raises(RuntimeError, match="reader is closed"):
        r.symbols()
    r.close()  # idempotent


def test_reader_context_manager():
    with Reader(GDX) as r:
        assert r.uel_table()
    with pytest.raises(RuntimeError, match="reader is closed"):
        r.uel_table()


def test_reader_reopen_after_close(tmp_path, edge_available):
    import shutil

    src = __import__("pathlib").Path(GDX)
    dst = tmp_path / "copy.gdx"
    shutil.copy(src, dst)
    r = Reader(str(dst))
    r.close()
    # Handle released: the file can be replaced on all platforms.
    shutil.copy(src, dst)
    r2 = Reader(str(dst))
    assert r2.symbols()
    r2.close()


def test_scan_twice_same_lazyframe():
    """Regression guard: no over-aggressive close; a scan stays re-collectable."""
    lf = scan_gdx(GDX, symbol="x")
    first = lf.collect()
    second = lf.collect()
    assert first.equals(second)


def test_scan_gdx_docstring_documents_handle_lifetime():
    assert "stays open" in scan_gdx.__doc__


# ---------------------------------------------------------------------------
# Plan 06 part B — key/value column name collision
# ---------------------------------------------------------------------------


def test_key_names_avoids_value_collision():
    names = _key_names(["value", "value"], 2, reserved={"value"})
    assert names == ["value_0", "value_1"]


def test_scan_symbol_with_value_domain(edge_available):
    if not edge_available:
        pytest.skip("edgecase fixture not present")
    df = read_gdx(EDGE, symbol="p")
    assert df.height == 4
    assert df.columns == ["value_0", "value_1", "value"]
    assert set(df["value_0"].unique()) <= {"alpha", "beta"}
    assert sorted(df["value"].to_list()) == [1.0, 2.0, 3.0, 4.0]


def test_rust_rejects_colliding_key_names(edge_available):
    if not edge_available:
        pytest.skip("edgecase fixture not present")
    r = Reader(EDGE)
    with pytest.raises(RuntimeError, match="collision"):
        r.read_arrow("p", key_names=["value", "value"])
    r.close()
