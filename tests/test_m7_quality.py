"""Regressions for current M7 adapter resource and ownership contracts."""

from collections.abc import Sequence

import polars as pl
import pytest

from polars_gdx import GdxrsReader, read_gdx
from polars_gdx.lazy import _apply_key_label_predicate

pytestmark = pytest.mark.skipif(GdxrsReader is None, reason="gdxrs backend not compiled")
DICT = "tests/data/m7/m7-dictionary.gdx"
LABELS = "tests/data/m7/m7-labelbytes.gdx"


def test_resolver_layout_overflow_precedes_sequence_access():
    class ImpossibleSequence(Sequence):
        def __len__(self):
            return 1 << 62

        def __getitem__(self, index):
            raise AssertionError("request must reject before accessing items")

    with GdxrsReader(LABELS) as reader:
        with pytest.raises(RuntimeError, match="layout overflow"):
            reader.resolve_labels(ImpossibleSequence())
        assert reader.resolve_labels((2, 1, 2)).to_pydict() == {
            "uel_id": [2, 1, 2], "label": [b"\xffb", b"aa", b"\xffb"]
        }


def test_borrowed_label_lookup_preserves_utf8_and_duplicate_order():
    with GdxrsReader(LABELS) as reader:
        assert reader.label_ids(["aa", "é", "aa", "AA"]) == [1, None, 1, None]


@pytest.mark.parametrize("names,message", [
    (["same", "same"], "duplicate"),
    (["level", "other"], "collision"),
])
def test_column_name_errors_precede_decoding(names, message):
    with GdxrsReader("tests/data/trnsport.gdx") as reader:
        with pytest.raises(RuntimeError, match=message):
            reader.read_arrow("x", key_names=names)
        assert reader.scan_statistics()["decoded_rows"] == 0


def test_domain_filter_validates_even_when_no_labels_resolve():
    with GdxrsReader(DICT) as reader:
        with pytest.raises(RuntimeError, match="not found"):
            reader.resolve_uel_indices_used([(0, ["unknown"])], "missing")
        with pytest.raises(ValueError, match="out of range"):
            reader.resolve_uel_indices_used([(5, ["unknown"])], "P")
        assert reader.resolve_uel_indices_used([(0, ["bb", "unknown", "bb"])], "P") == [(0, [2])]


def test_owned_arrow_batch_remains_valid_after_close():
    reader = GdxrsReader(DICT)
    batch = reader.read_arrow("P")
    reader.close()
    reader.close()
    assert batch.to_pydict() == {"dim_0": ["aa", "bb", "cc", "dd"], "value": [10., 20., 30., 40.]}
    with pytest.raises(RuntimeError, match="closed"):
        reader.uel_table()


def test_label_residual_keeps_id_columns_without_python_row_copies(monkeypatch):
    df = read_gdx(DICT, symbol="P", backend="gdxrs", key_mode="uel_id")

    def forbid_rows(*args, **kwargs):
        raise AssertionError("predicate conversion must not copy key cells to Python")

    with GdxrsReader(DICT) as reader, monkeypatch.context() as patch:
        patch.setattr(pl.Series, "to_list", forbid_rows)
        result = _apply_key_label_predicate(
            df, reader,
            (pl.col("dim_0") == "cc") & (pl.col("value").abs() > 20),
            ["dim_0"],
        )
    assert result.schema == df.schema
    assert result.rows() == [(3, 30.)]
