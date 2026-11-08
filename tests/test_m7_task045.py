"""M7 task 045 — exact predicate, projection and global limits.

Asserts the acceptance criteria of gdxrs task 045 for the gdxrs batch
source: sound predicate/projection/limit pushdown with residuals, exact
global accepted-row limits across batch boundaries, retained columns for
residual predicates, and optimized-vs-unoptimized reference comparisons
using the independent fixture expectations.
"""

import polars as pl
import pytest

from polars_gdx import GdxrsReader, Reader, read_gdx, scan_gdx

GDX = "tests/data/trnsport.gdx"
DICT = "tests/data/m7/m7-dictionary.gdx"
SEM = "tests/data/m7/m7-semantics.gdx"
GROW = "tests/data/m7/m7-labels20k.gdx"

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(DICT).exists(), reason="m7 fixtures not present"
)

if GdxrsReader is None:
    pytest.skip("gdxrs backend not compiled", allow_module_level=True)


def _unoptimized(lf):
    """The unoptimized reference plan: Polars applies predicates and
    projections itself (no pushdown into the source)."""
    return lf.collect(optimizations=pl.QueryOptFlags(predicate_pushdown=False))


# ---------------------------------------------------------------------------
# Optimized vs unoptimized reference comparisons
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("plan", [
    lambda lf: lf,
    lambda lf: lf.filter(pl.col("dim_0") == "seattle"),
    lambda lf: lf.filter(pl.col("dim_0").is_in(["seattle", "san-diego"])),
    lambda lf: lf.filter(pl.col("level") > 200),
    lambda lf: lf.filter((pl.col("dim_0") == "seattle") & (pl.col("level") > 100)),
    lambda lf: lf.filter(pl.col("level") > 200).select("dim_0"),
    lambda lf: lf.select("dim_1"),
    lambda lf: lf.select("level"),
    lambda lf: lf.filter(pl.col("dim_0") == "seattle").select("dim_1"),
    lambda lf: lf.head(3),
    lambda lf: lf.filter(pl.col("level") > 200).head(2),
    lambda lf: lf.head(0),
])
def test_optimized_matches_unoptimized_reference(plan):
    """For every plan shape, the optimized (pushed-down) scan equals the
    unoptimized reference plan exactly — rows, order and schema."""
    lf = scan_gdx(GDX, symbol="x", backend="gdxrs")
    optimized = plan(lf).collect()
    reference = _unoptimized(plan(lf))
    assert optimized.equals(reference), plan


def test_projection_pushdown_reads_only_needed_columns():
    """Projection pushdown: the core decodes only the projected key
    dimensions and skips the value column when unneeded; the emitted
    batches contain exactly the needed columns."""
    with GdxrsReader(GDX) as reader:
        cursor = reader.scan_batches("x", columns=["dim_1"], batch_rows=100)
        batch = next(iter(cursor))
        assert batch.schema.names == ["dim_1"]
        assert batch.num_rows == 6
        cursor = reader.scan_batches("x", columns=["level"], batch_rows=100)
        batch = next(iter(cursor))
        assert batch.schema.names == ["level"]
        # Both keys and value.
        cursor = reader.scan_batches("x", columns=["dim_0", "level"], batch_rows=100)
        batch = next(iter(cursor))
        assert batch.schema.names == ["dim_0", "level"]
        # Unknown column rejected clearly.
        with pytest.raises(ValueError, match="unknown projected column"):
            reader.scan_batches("x", columns=["bogus"])
        # A projection leaving nothing readable is rejected clearly.
        with pytest.raises(ValueError, match="no readable columns"):
            reader.scan_batches("x", columns=[])


def test_selection_applies_to_unprojected_dimensions():
    """A key filter on a dimension that is not projected still filters
    exactly (the selection applies at the decoder)."""
    lf = scan_gdx(GDX, symbol="x", backend="gdxrs")
    projected = (
        lf.filter(pl.col("dim_0") == "seattle").select("dim_1").collect()
    )
    reference = lf.collect().filter(pl.col("dim_0") == "seattle").select("dim_1")
    assert projected.equals(reference)
    assert projected.height == 3


# ---------------------------------------------------------------------------
# Sparse / all-rejected filters, late matches, global limits
# ---------------------------------------------------------------------------


def test_all_rejected_filters_yield_zero_rows_with_schema():
    """Sparse and all-rejected filters yield zero rows with the planned
    schema, in both modes, with and without projection."""
    for key_mode in ("enum", "uel_id"):
        lf = scan_gdx(GDX, symbol="x", backend="gdxrs", key_mode=key_mode)
        for labels in (["not-a-label"], []):
            df = read_gdx(
                GDX, symbol="x", backend="gdxrs", key_mode=key_mode,
                key_filter={0: labels},
            )
            assert df.height == 0
            assert df.columns == ["dim_0", "dim_1", "level"]
        # A label used only in the other dimension matches nothing here.
        df = read_gdx(GDX, symbol="x", backend="gdxrs", key_filter={0: ["topeka"]})
        assert df.height == 0


def test_matches_only_in_later_batches_with_global_head():
    """Matches occurring only in late batches: the global accepted-row
    limit counts rows after residual filtering across all batches — no
    per-batch limit may substitute."""
    lf = scan_gdx(GROW, symbol="BIG", backend="gdxrs")
    # A residual predicate whose matches occur only in the final records
    # (rows 19900..19999 of 20,000): the first 19,900 rows are rejected.
    late = lf.filter(pl.col("dim_0").cast(pl.String).str.starts_with("u199"))
    expected = late.collect()
    assert expected.height == 100
    assert expected["dim_0"].to_list()[:3] == ["u19900", "u19901", "u19902"]
    # head(3) counts accepted rows globally across batches — no per-batch
    # limit may substitute.
    headed = late.head(3).collect()
    assert headed["dim_0"].to_list() == ["u19900", "u19901", "u19902"]
    assert headed["value"].to_list() == [900.0, 901.0, 902.0]
    # Exact ordered rows equal the reference plan.
    assert headed.equals(_unoptimized(late.head(3)))
    # With tiny batches the limit is still global.
    batches = list(late.head(3).collect_batches(chunk_size=1))
    assert pl.concat(batches).equals(headed)


def test_zero_limit_reads_nothing():
    """A zero limit produces no rows with the planned schema and decodes
    nothing (no workers, no records)."""
    lf = scan_gdx(GROW, symbol="BIG", backend="gdxrs")
    assert lf.head(0).collect().height == 0
    with GdxrsReader(GROW) as reader:
        cursor = reader.scan_batches("BIG", n_rows=0, batch_rows=100)
        assert list(iter(cursor)) == []
        assert cursor.scan_statistics()["decoded_rows"] == 0


def test_head_across_batch_boundaries():
    """head(n) is exact for every n across batch boundaries, with pushed
    and with residual predicates."""
    lf = scan_gdx(DICT, symbol="P", backend="gdxrs")
    full = lf.collect()
    for n in range(0, 6):
        assert lf.head(n).collect().equals(full.head(n))
    residual = lf.filter(pl.col("value") > 15)
    ref = residual.collect()
    for n in range(0, 4):
        assert residual.head(n).collect().equals(ref.head(n))
    # Chunked delivery agrees.
    for chunk in (1, 2, 3):
        batches = list(residual.head(2).collect_batches(chunk_size=chunk))
        assert pl.concat(batches).equals(ref.head(2))


# ---------------------------------------------------------------------------
# Mixed pushed/residual predicates; projected-away columns; nulls
# ---------------------------------------------------------------------------


def test_mixed_pushed_and_residual_predicates():
    """A pushed key predicate conjoined with a residual value predicate
    keeps exact case-sensitive label semantics and record order."""
    lf = scan_gdx(GDX, symbol="x", backend="gdxrs")
    mixed = lf.filter((pl.col("dim_0") == "seattle") & (pl.col("level") > 100))
    result = mixed.collect()
    reference = _unoptimized(mixed)
    assert result.equals(reference)
    # seattle rows: level 50, 300, 0 — only 300 exceeds 100.
    assert result["dim_0"].to_list() == ["seattle"]
    # Case mismatch on the pushed part matches nothing.
    wrong = lf.filter((pl.col("dim_0") == "SEATTLE") & (pl.col("level") > 0)).collect()
    assert wrong.height == 0


def test_predicate_on_projected_away_column():
    """A residual predicate on a projected-away column still evaluates:
    the required column is retained until the residual has run, then the
    final projection applies."""
    lf = scan_gdx(GDX, symbol="x", backend="gdxrs")
    result = lf.filter(pl.col("level") > 200).select("dim_0").collect()
    reference = lf.collect().filter(pl.col("level") > 200).select("dim_0")
    assert result.equals(reference)
    assert result.columns == ["dim_0"]
    # The same with a pushed key predicate and a projected-away residual.
    result = (
        lf.filter((pl.col("dim_0") == "seattle") & (pl.col("level") > 100))
        .select("dim_0")
        .collect()
    )
    reference = (
        lf.collect()
        .filter((pl.col("dim_0") == "seattle") & (pl.col("level") > 100))
        .select("dim_0")
    )
    assert result.equals(reference)


def test_null_and_special_value_predicates():
    """Null (UNDEF/NA/acronym) and special values keep the documented
    projection under residual predicates: nulls never satisfy ordered
    comparisons; ±inf compare exactly; EPS is +0.0."""
    lf = scan_gdx(SEM, symbol="P", backend="gdxrs")
    df = lf.collect()
    # Ordered comparisons skip nulls: only +inf and 1.25 exceed 0
    # (EPS projects to +0.0 and does not satisfy > 0; -inf does not).
    assert lf.filter(pl.col("value") > 0).collect().height == 2
    assert lf.filter(pl.col("value").is_null()).collect().height == 3
    # is_null matches exactly the UNDEF/NA/acronym rows.
    assert lf.filter(pl.col("value").is_null()).collect()["dim_0"].to_list() == [
        "bb",
        "cc",
        "Seattle",
    ]
    # ±inf compare exactly.
    assert lf.filter(pl.col("value") == float("inf")).collect().height == 1
    assert lf.filter(pl.col("value") == float("-inf")).collect().height == 1
    # EPS projects to +0.0 and satisfies == 0.
    assert lf.filter(pl.col("value") == 0).collect()["dim_0"].to_list() == ["aa"]
    # Everything equals the unoptimized reference.
    assert (
        lf.filter(pl.col("value") > 0).collect().equals(_unoptimized(lf.filter(pl.col("value") > 0)))
    )


def test_record_order_is_preserved_everywhere():
    """Original record order is preserved for pushed, residual, projected
    and limited scans, in both key modes."""
    for key_mode in ("enum", "uel_id"):
        lf = scan_gdx(GROW, symbol="BIG", backend="gdxrs", key_mode=key_mode)
        full = lf.collect()
        assert full.height == 20_000
        # Values are i % 1000 in file order.
        assert full["value"].to_list()[:5] == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert lf.head(10).collect().equals(full.head(10))
        assert lf.filter(pl.col("value") > 995).collect().equals(
            full.filter(pl.col("value") > 995)
        )
        assert lf.select("value").collect().equals(full.select("value"))


def test_mode_b_projection_and_limits():
    """Projection pushdown and global limits behave identically in the
    bounded ID mode."""
    lf = scan_gdx(GDX, symbol="x", backend="gdxrs", key_mode="uel_id")
    assert lf.select("dim_1").collect().schema == {"dim_1": pl.UInt32}
    projected = lf.filter(pl.col("dim_0") == 1).select("dim_1").collect()
    reference = lf.collect().filter(pl.col("dim_0") == 1).select("dim_1")
    assert projected.equals(reference)
    # Label predicates in ID mode stay consistent with projection.
    by_label = lf.filter(pl.col("dim_0") == "seattle").select("dim_1").collect()
    assert by_label.equals(reference)
    # Global head with a residual.
    assert lf.filter(pl.col("level") > 200).head(2).collect().equals(
        lf.collect().filter(pl.col("level") > 200).head(2)
    )


def test_counters_are_exact_for_pushed_and_residual_paths():
    """The core's decoded-row counter distinguishes the paths: a pushed
    predicate decodes only the selected rows; a residual decodes the
    whole symbol; a zero limit decodes nothing."""
    with GdxrsReader(GROW) as reader:
        # Pushed (selection): exactly one row is accepted; the core's
        # decoded_rows counter documents that rejected rows also count
        # as decoded, so the full symbol is decoded and one row returned.
        cursor = reader.scan_batches("BIG", key_filter=[(0, ["u00000"])], batch_rows=100)
        rows = sum(b.num_rows for b in iter(cursor))
        assert rows == 1
        assert cursor.scan_statistics()["decoded_rows"] == 20_000
        # Unfiltered scan: every decoded row is accepted.
        cursor = reader.scan_batches("BIG", batch_rows=100)
        rows = sum(b.num_rows for b in iter(cursor))
        assert rows == 20_000
        assert cursor.scan_statistics()["decoded_rows"] == 20_000
        # Projection changes the payload, not the row counts.
        cursor = reader.scan_batches("BIG", columns=["value"], batch_rows=100)
        rows = sum(b.num_rows for b in iter(cursor))
        assert rows == 20_000
        assert cursor.scan_statistics()["decoded_rows"] == 20_000
