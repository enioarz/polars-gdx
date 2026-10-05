import polars as pl
import pytest

from polars_gdx import list_symbols, read_domains, read_gdx, scan_gdx

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


def test_predicate_uses_native_prefilter(monkeypatch):
    """pl.col(key) == label must fold into the native GDX prefilter."""
    import polars_gdx.lazy as lazy

    calls = []
    orig = lazy.Reader.read_arrow

    def spy(
        self, name, key_names=None, value_field=None, key_filter=None, n_rows=None, threads=None
    ):
        calls.append(key_filter)
        return orig(self, name, key_names, value_field, key_filter, n_rows, threads)

    monkeypatch.setattr(lazy.Reader, "read_arrow", spy)
    lf = lazy.scan_gdx(GDX, symbol="x")
    df = lf.filter(pl.col("dim_0") == "seattle").collect()
    assert df.height == 3
    assert calls and calls[-1] is not None and calls[-1][0][0] == 0
    assert "seattle" in calls[-1][0][1]


def test_predicate_unknown_label_yields_no_rows():
    lf = scan_gdx(GDX, symbol="x")
    df = lf.filter(pl.col("dim_0") == "atlantis").collect()
    assert df.height == 0


def test_predicate_conjunction_prefilter(monkeypatch):
    import polars_gdx.lazy as lazy

    calls = []
    orig = lazy.Reader.read_arrow

    def spy(
        self, name, key_names=None, value_field=None, key_filter=None, n_rows=None, threads=None
    ):
        calls.append(key_filter)
        return orig(self, name, key_names, value_field, key_filter, n_rows, threads)

    monkeypatch.setattr(lazy.Reader, "read_arrow", spy)
    lf = lazy.scan_gdx(GDX, symbol="x")
    df = lf.filter((pl.col("dim_0") == "seattle") & (pl.col("dim_1") == "chicago")).collect()
    assert df.height == 1
    merged = dict(calls[-1])
    assert "seattle" in merged[0]
    assert "chicago" in merged[1]


def test_predicate_on_value_column_not_native():
    """Predicates on the value column must still work (Polars-side)."""
    lf = scan_gdx(GDX, symbol="x", value_field="level")
    df = lf.filter(pl.col("level") > 100).collect()
    assert df.height > 0
    assert (df["level"] > 100).all()


def test_read_gdx_eager():
    df = read_gdx(GDX, symbol="x")
    assert isinstance(df, pl.DataFrame)
    assert set(df.columns) == {"dim_0", "dim_1", "level"}
    assert df.height == scan_gdx(GDX, symbol="x").collect().height


def test_read_gdx_key_filter():
    full = read_gdx(GDX, symbol="x")
    filtered = read_gdx(GDX, symbol="x", key_filter={0: ["seattle"]})
    assert filtered.height < full.height
    assert set(filtered["dim_0"].unique()) == {"seattle"}


def test_repeated_reads_consistent():
    """The UEL table/dictionary caches must not corrupt repeated reads."""
    first = read_gdx(GDX, symbol="x")
    second = read_gdx(GDX, symbol="x")
    assert first.equals(second)
    with_prefilter = read_gdx(GDX, symbol="x", key_filter={0: ["seattle"]})
    assert with_prefilter.height > 0
    assert set(with_prefilter["dim_0"]) == {"seattle"}


def test_head_pushdown_matches_eager():
    lazy_head = scan_gdx(GDX, symbol="x").head(2).collect()
    eager_head = read_gdx(GDX, symbol="x").head(2)
    assert lazy_head.equals(eager_head)


def test_head_after_filter():
    lf = scan_gdx(GDX, symbol="x").filter(pl.col("dim_0") == "seattle").head(1)
    df = lf.collect()
    assert df.height == 1
    assert df["dim_0"][0] == "seattle"


def test_head_after_untranslatable_predicate():
    lf = scan_gdx(GDX, symbol="x").filter(pl.col("level") > 0).head(1)
    df = lf.collect()
    assert df.height == 1
    assert (df["level"] > 0).all()


def test_single_label_filter_absent_label_yields_no_rows():
    df = scan_gdx(GDX, symbol="x", key_filter={0: ["atlantis"]}).collect()
    assert df.height == 0


def test_read_domains_matches_unique():
    df = read_gdx(GDX, symbol="x")
    domains = read_domains(GDX, symbol="x")
    assert domains.columns == ["dim_0", "dim_1"]
    assert set(domains["dim_0"].drop_nulls()) == set(df["dim_0"].unique())
    assert set(domains["dim_1"].drop_nulls()) == set(df["dim_1"].unique())


def test_read_domains_unused_uel_excluded():
    # "seattle" appears in the UEL table of the whole file; the domain scan
    # must list only labels actually used by the requested symbol.
    domains = read_domains(GDX, symbol="d")
    used = set(read_gdx(GDX, symbol="d")["dim_1"].unique())
    assert set(domains["dim_1"].drop_nulls()) == used


def test_filter_label_unused_in_dimension_yields_no_rows():
    # "seattle" exists globally, but is never used in dim 1 of symbol "x":
    # the used-UEL intersection must detect the guaranteed-empty read without
    # scanning the data.
    df = scan_gdx(GDX, symbol="x", key_filter={1: ["seattle"]}).collect()
    assert df.height == 0


def test_filter_first_dim_early_stop_correct():
    # Records are sorted by first-dimension key: an early-label filter must
    # still produce exactly the matching rows (the scan stops early).
    df = scan_gdx(GDX, symbol="x").filter(pl.col("dim_0") == "seattle").collect()
    assert set(df["dim_0"]) == {"seattle"}


def test_filter_multi_dim_conjunction_bitmap():
    df = (
        scan_gdx(GDX, symbol="x")
        .filter((pl.col("dim_0") == "seattle") & (pl.col("dim_1") == "topeka"))
        .collect()
    )
    assert df.height == 1
    assert df["dim_0"][0] == "seattle" and df["dim_1"][0] == "topeka"


def test_key_filter_multiple_labels_dim0():
    df = read_gdx(GDX, symbol="x", key_filter={0: ["seattle", "san-diego"]})
    assert set(df["dim_0"]) == {"seattle", "san-diego"}


def test_parallel_read_parity():
    # threads>1 partitions dim-0 UEL space across workers; concatenated in
    # range order the result must equal the serial read exactly.
    serial = read_gdx(GDX, symbol="x")
    parallel = read_gdx(GDX, symbol="x", threads=4)
    assert serial.equals(parallel)


def test_parallel_read_all_symbols_parity():
    for sym in ("x", "d", "i", "j", "a", "b", "cost", "supply", "demand"):
        serial = read_gdx(GDX, symbol=sym)
        parallel = read_gdx(GDX, symbol=sym, threads=3)
        assert serial.equals(parallel), sym


def test_parallel_filtered_read_parity():
    serial = read_gdx(GDX, symbol="x", key_filter={0: ["seattle", "san-diego"]})
    parallel = read_gdx(
        GDX, symbol="x", key_filter={0: ["seattle", "san-diego"]}, threads=4
    )
    assert serial.equals(parallel)
    assert set(parallel["dim_0"]) == {"seattle", "san-diego"}


def test_parallel_single_thread_falls_back_to_serial():
    assert read_gdx(GDX, symbol="x", threads=1).equals(read_gdx(GDX, symbol="x"))


def test_is_in_uses_native_prefilter(monkeypatch):
    """pl.col(key).is_in([...]) must fold into the native GDX prefilter."""
    import polars_gdx.lazy as lazy

    calls = []
    orig = lazy.Reader.read_arrow

    def spy(
        self, name, key_names=None, value_field=None, key_filter=None, n_rows=None, threads=None
    ):
        calls.append(key_filter)
        return orig(self, name, key_names, value_field, key_filter, n_rows, threads)

    monkeypatch.setattr(lazy.Reader, "read_arrow", spy)
    lf = lazy.scan_gdx(GDX, symbol="x")
    df = lf.filter(pl.col("dim_0").is_in(["seattle", "san-diego"])).collect()
    assert set(df["dim_0"]) == {"seattle", "san-diego"}
    assert calls and calls[-1] is not None
    merged = dict(calls[-1])
    assert set(merged[0]) == {"seattle", "san-diego"}


def test_is_in_and_eq_conjunction_prefilter():
    df = (
        scan_gdx(GDX, symbol="x")
        .filter(pl.col("dim_0").is_in(["seattle", "san-diego"]) & (pl.col("dim_1") == "topeka"))
        .collect()
    )
    assert df.height == 2
    assert set(df["dim_0"]) == {"seattle", "san-diego"}
    assert set(df["dim_1"]) == {"topeka"}


def test_is_in_absent_label_yields_no_rows():
    df = scan_gdx(GDX, symbol="x").filter(pl.col("dim_0").is_in(["nowhere"])).collect()
    assert df.height == 0


def test_is_in_partial_match():
    df = scan_gdx(GDX, symbol="x").filter(pl.col("dim_0").is_in(["seattle", "nowhere"])).collect()
    assert set(df["dim_0"]) == {"seattle"}


def test_threads_auto_serial_for_small_symbols():
    # trnsport symbols are far below the auto threshold: same result as serial.
    serial = read_gdx(GDX, symbol="x")
    auto = read_gdx(GDX, symbol="x", threads="auto")
    assert serial.equals(auto)


def test_threads_auto_resolves_to_cpu_count_for_large_symbols(monkeypatch):
    import polars_gdx.lazy as lazy

    class FakeInfo:
        pass

    # large symbol: resolves to (mocked) cpu count
    monkeypatch.setattr(lazy, "_AUTO_THREADS_MIN_RECORDS", 1_000_000)
    import os

    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        cpus = os.cpu_count()
    assert lazy._resolve_threads("auto", 2_000_000) == cpus
    assert lazy._resolve_threads("auto", 500_000) is None
    assert lazy._resolve_threads(4, 2_000_000) == 4
    assert lazy._resolve_threads(None, 2_000_000) is None
    assert lazy._resolve_threads(0, 2_000_000) is None
