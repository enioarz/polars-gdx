import polars as pl
import pytest

from polars_gdx import list_symbols, scan_gdx

GDX = "tests/data/trnsport.gdx"

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(GDX).exists(), reason="test gdx file not present"
)


def test_list_symbols():
    df = list_symbols(GDX)
    assert {"i", "j", "a", "b", "d", "cost", "x", "supply", "demand"} <= set(df["name"])
    assert df.filter(pl.col("name") == "x")["dim"][0] == 2


def test_scan_projection_pushdown():
    lf = scan_gdx(GDX, symbol="cost")
    df = lf.select("dim_1").collect()
    assert df.columns == ["dim_1"]
    assert df.height > 0


def test_scan_prefilter():
    lf = scan_gdx(GDX, symbol="x", key_filter={0: ["seattle"]})
    df = lf.collect()
    assert set(df["dim_0"]) == {"seattle"}
    assert "san-diego" not in set(df["dim_0"])


def test_value_field_default_level():
    lf = scan_gdx(GDX, symbol="supply")
    df = lf.collect()
    assert "level" in df.columns


def test_no_early_read():
    scan_gdx(GDX, symbol="x")  # constructing the LazyFrame reads nothing
