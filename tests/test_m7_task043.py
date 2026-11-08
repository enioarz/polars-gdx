"""M7 task 043 — stable schemas, explicit Enum admission and bounded modes.

Asserts the frozen expectations of docs/m7-fixture-expectations.md for the
schema/dictionary modes of the adapter contract §5: mode A (budget-admitted
full-dictionary Enum) and mode B (`key_mode="uel_id"`, original one-based
UEL IDs as UInt32, no label decoding) with its bounded `resolve_labels`
resolver. Expected values are authored independently of adapter output.
"""

import hashlib
import struct

import polars as pl
import pytest

from polars_gdx import GdxrsReader, Reader, read_gdx, scan_gdx

DICT = "tests/data/m7/m7-dictionary.gdx"
LABELS = "tests/data/m7/m7-labelbytes.gdx"
GROW = "tests/data/m7/m7-labels20k.gdx"

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(DICT).exists(), reason="m7 fixtures not present"
)

if GdxrsReader is None:
    pytest.skip("gdxrs backend not compiled", allow_module_level=True)

#: SHA-256 digests of the generated fixtures (generator:
#: crates/polars-gdx/examples/gen_m7_fixtures.rs, gdxrs core revision
#: 9a7411764f176adb613185fa1e45d64b06123683).
FIXTURE_SHA256 = {
    DICT: "9c49102186c8cc65a7a06d08d6e22c856ebdc26a929ef56e7b47cfdecdcbadba",
    LABELS: "22e8a413814dcc7acc6991a91ab9c83b7b73e11f75b5dcc18df4a03d962aa63f",
    GROW: "3749aa7c0eaa5fd7f35ebe8cdd1c985e9bd95e596c0587efa9e84f7221cd6d1e",
}

DICT_LABELS = ["aa", "bb", "cc", "dd"]


def test_fixture_hashes_match_generator_output():
    for path, digest in FIXTURE_SHA256.items():
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == digest, path


# ---------------------------------------------------------------------------
# Mode A: budget-admitted full-dictionary Enum (schema stability)
# ---------------------------------------------------------------------------


def test_mode_a_enum_schema_and_values_are_frozen():
    """Frozen: mode A key columns are Enum over the full UEL table in file
    order with 0-based physical codes; values project exactly."""
    df = read_gdx(DICT, symbol="P", backend="gdxrs")
    assert df.schema == pl.Schema(
        {"dim_0": pl.Enum(DICT_LABELS), "value": pl.Float64}
    )
    assert df["dim_0"].to_list() == DICT_LABELS
    assert df["value"].to_list() == [10.0, 20.0, 30.0, 40.0]
    # Category identity: the enum physical codes are the global 0-based UEL
    # numbers, never batch-local category codes.
    assert df["dim_0"].to_physical().to_list() == [0, 1, 2, 3]


def test_mode_a_schema_stable_across_empty_results_and_repeated_scans():
    lf = scan_gdx(DICT, symbol="P", backend="gdxrs")
    expected = read_gdx(DICT, symbol="P", backend="gdxrs")
    # Empty result keeps the planned schema.
    empty = read_gdx(DICT, symbol="P", backend="gdxrs", key_filter={0: ["nope"]})
    assert empty.height == 0
    assert empty.schema == expected.schema
    # Zero limit keeps the planned schema.
    assert lf.head(0).collect().schema == expected.schema
    # Partitions/repeated scans are deterministic and identical.
    assert lf.collect().equals(expected)
    assert lf.collect().equals(expected)
    assert lf.head(2).collect().equals(expected.head(2))


def test_mode_a_global_head_limit():
    """Frozen: a global head(3) returns IDs 1,2,3 — the first three records,
    not three per batch."""
    df = scan_gdx(DICT, symbol="P", backend="gdxrs").head(3).collect()
    assert df["dim_0"].to_list() == ["aa", "bb", "cc"]
    assert df["value"].to_list() == [10.0, 20.0, 30.0]


def test_mode_a_admits_under_budget_and_rejects_clearly():
    """Frozen: a tiny dictionary budget rejects mode A before output; the
    schema is never silently changed."""
    with GdxrsReader(DICT, dictionary_budget_bytes=128) as reader:
        with pytest.raises(RuntimeError, match="dictionary budget"):
            reader.read_arrow("P")
    # The default budget admits this small dictionary.
    with GdxrsReader(DICT) as reader:
        assert reader.uel_table() == DICT_LABELS


def test_signed_zero_conversion_expectation():
    """Frozen adapter conversion expectation: ordinary NumberBits inputs
    for +0 and -0 project to bits 0000000000000000 and 8000000000000000
    (the gdxrs writer's Preserve policy keeps the signed zero on disk).
    The returned values' packed bytes are compared against the expected
    bit patterns, so simulated [+0.0, +0.0] output fails."""
    df = read_gdx(DICT, symbol="ZEROS", backend="gdxrs")
    packed = [struct.pack("<d", v) for v in df["value"].to_list()]
    assert packed == [
        struct.pack("<Q", 0x0000000000000000),
        struct.pack("<Q", 0x8000000000000000),
    ]
    # And the sign is preserved through the values themselves.
    import math

    assert [math.copysign(1.0, v) for v in df["value"].to_list()] == [1.0, -1.0]
    assert df["dim_0"].to_list() == ["aa", "bb"]


# ---------------------------------------------------------------------------
# Mode B: bounded original-ID columns
# ---------------------------------------------------------------------------


def test_mode_b_uelp_id_columns_are_frozen():
    """Frozen: key_mode="uel_id" returns the original one-based UEL IDs as
    UInt32 key columns with the exact values, without decoding labels."""
    df = read_gdx(DICT, symbol="P", backend="gdxrs", key_mode="uel_id")
    assert df.schema == pl.Schema({"dim_0": pl.UInt32, "value": pl.Float64})
    assert df["dim_0"].to_list() == [1, 2, 3, 4]
    assert df["value"].to_list() == [10.0, 20.0, 30.0, 40.0]
    # Category identity across modes: mode B IDs are exactly the mode A
    # physical codes + 1 (global UEL numbers, not batch-local codes).
    enum_df = read_gdx(DICT, symbol="P", backend="gdxrs")
    assert df["dim_0"].to_list() == [
        c + 1 for c in enum_df["dim_0"].to_physical().to_list()
    ]


def test_mode_b_global_head_limit():
    """Frozen: a global head(3) in ID mode returns IDs 1,2,3."""
    df = scan_gdx(DICT, symbol="P", backend="gdxrs", key_mode="uel_id").head(3).collect()
    assert df["dim_0"].to_list() == [1, 2, 3]
    assert df["value"].to_list() == [10.0, 20.0, 30.0]


def test_mode_b_schema_stable_across_empty_results_and_repeated_scans():
    expected = read_gdx(DICT, symbol="P", backend="gdxrs", key_mode="uel_id")
    empty = read_gdx(
        DICT, symbol="P", backend="gdxrs", key_mode="uel_id", key_filter={0: ["nope"]}
    )
    assert empty.height == 0
    assert empty.schema == expected.schema
    lf = scan_gdx(DICT, symbol="P", backend="gdxrs", key_mode="uel_id")
    assert lf.head(0).collect().schema == expected.schema
    assert lf.collect().equals(expected)
    assert lf.collect().equals(expected)


def test_mode_b_string_filters_match_exactly():
    """Frozen: string filters encode as UTF-8 and retain exact byte matching
    in ID mode; unknown labels yield zero rows with the planned schema."""
    df = read_gdx(
        DICT, symbol="P", backend="gdxrs", key_mode="uel_id", key_filter={0: ["bb", "dd"]}
    )
    assert df["dim_0"].to_list() == [2, 4]
    assert df["value"].to_list() == [20.0, 40.0]
    # Case-sensitive: "BB" matches nothing.
    df = read_gdx(
        DICT, symbol="P", backend="gdxrs", key_mode="uel_id", key_filter={0: ["BB"]}
    )
    assert df.height == 0
    assert df.schema == expected_schema_b()
    # Predicates on key columns: string equality resolves labels to their
    # IDs and matches exactly (Polars coerces the literal; the adapter
    # re-applies the exact integer comparison). is_in predicates plan
    # against the UInt32 column with integer IDs (Polars does not accept
    # String lists against UInt32 columns); unknown IDs match nothing.
    lf = scan_gdx(DICT, symbol="P", backend="gdxrs", key_mode="uel_id")
    assert lf.filter(pl.col("dim_0") == "cc").collect()["dim_0"].to_list() == [3]
    assert lf.filter(pl.col("dim_0") == "nope").collect().height == 0
    assert lf.filter(pl.col("dim_0") == "AA").collect().height == 0
    assert lf.filter(pl.col("dim_0").is_in([1, 3])).collect()["dim_0"].to_list() == [1, 3]
    assert lf.filter(pl.col("dim_0").is_in([999])).collect().height == 0


def test_mode_b_key_label_predicates_consistent_with_residuals():
    """Finding 2: a key-label predicate means the same thing whether or not
    it can be fully pushed down. Mixing a key-label comparison with a
    residual condition evaluates against exact label strings and returns
    the same rows as the pushed-down path."""
    lf = scan_gdx(DICT, symbol="P", backend="gdxrs", key_mode="uel_id")
    pushed = lf.filter(pl.col("dim_0") == "cc").collect()
    # Key label + residual value condition: same meaning, same rows.
    mixed = lf.filter((pl.col("dim_0") == "cc") & (pl.col("value") > 0)).collect()
    assert mixed.equals(pushed)
    assert mixed["dim_0"].to_list() == [3]
    # A residual that excludes the row excludes it here too.
    none = lf.filter((pl.col("dim_0") == "cc") & (pl.col("value") > 35)).collect()
    assert none.height == 0
    # is_in with String lists cannot even plan against UInt32 columns
    # (Polars schema validation, before the adapter is consulted): the
    # behavior is the same with or without residuals — a type error —
    # and integer IDs are the documented form.
    with pytest.raises(pl.exceptions.InvalidOperationError):
        lf.filter(
            (pl.col("dim_0").is_in(["aa", "cc"])) & (pl.col("value") >= 30)
        ).collect()
    assert (
        lf.filter(
            (pl.col("dim_0").is_in([1, 3])) & (pl.col("value") >= 30)
        ).collect()["dim_0"].to_list()
        == [3]
    )
    # Unknown labels still match nothing alongside residuals.
    assert (
        lf.filter((pl.col("dim_0") == "nope") & (pl.col("value") > 0)).collect().height
        == 0
    )
    # Pure residual predicates on the value column are untouched.
    assert lf.filter(pl.col("value") >= 30).collect()["dim_0"].to_list() == [3, 4]
    # The labelbytes fixture: non-UTF-8 labels never match a string
    # literal, with or without residuals.
    lf_b = scan_gdx(LABELS, symbol="B", backend="gdxrs", key_mode="uel_id")
    assert lf_b.filter(pl.col("dim_0") == "aa").collect()["dim_0"].to_list() == [1]
    assert (
        lf_b.filter((pl.col("dim_0") == "aa") & (pl.col("value") > 15)).collect().height
        == 0
    )


def test_mode_b_label_fallback_stays_bounded():
    """Finding 1: the residual fallback resolves only the predicate's label
    literals, never the output's labels. A label filter plus an
    always-true value condition succeeds on the 20,000-label growing
    fixture even under a 1 MiB reader budget that could never admit the
    full label resolution."""
    from polars_gdx.lazy import _apply_key_label_predicate

    lf = scan_gdx(GROW, symbol="BIG", backend="gdxrs", key_mode="uel_id")
    pushed = lf.filter(pl.col("dim_0") == "u00042").collect()
    assert pushed.height == 1
    # Same filter plus an always-true residual: identical rows.
    mixed = lf.filter(
        (pl.col("dim_0") == "u00042") & (pl.col("value") >= 0)
    ).collect()
    assert mixed.equals(pushed)
    # Directly against a 1 MiB-budget reader: the fallback resolves only
    # the single predicate literal, so no admission failure occurs (the
    # previous design resolved all 20,000 output labels here).
    df = read_gdx(GROW, symbol="BIG", backend="gdxrs", key_mode="uel_id")
    with GdxrsReader(GROW, dictionary_budget_bytes=1024 * 1024) as reader:
        out = _apply_key_label_predicate(
            df,
            reader,
            (pl.col("dim_0") == "u00042") & (pl.col("value") >= 0),
            ["dim_0"],
        )
        assert out.height == 1
        assert out["dim_0"].to_list() == [43]


def test_mode_b_numeric_key_predicates():
    """Finding 2: numeric ID predicates keep working alongside label
    predicates on other key columns; mixing string-label and numeric
    comparisons on the same key column is rejected clearly; with
    predicate pushdown disabled, string comparisons against UInt32 are
    the documented Polars type error."""
    # Numeric-only key predicates evaluate directly against the IDs.
    lf = scan_gdx(DICT, symbol="P", backend="gdxrs", key_mode="uel_id")
    assert lf.filter(pl.col("dim_0") == 3).collect()["dim_0"].to_list() == [3]
    assert lf.filter(pl.col("dim_0") < 3).collect()["dim_0"].to_list() == [1, 2]
    # Label and numeric comparisons on different key columns coexist.
    lf2 = scan_gdx("tests/data/trnsport.gdx", symbol="x", backend="gdxrs", key_mode="uel_id")
    both = lf2.filter((pl.col("dim_0") == "seattle") & (pl.col("dim_1") == 3)).collect()
    assert both.height == 1
    assert both["dim_1"].to_list() == [3]
    # Mixing on the SAME key column is rejected clearly, not with a
    # confusing Polars type error.
    with pytest.raises(ValueError, match="mixes string-label and numeric"):
        lf.filter((pl.col("dim_0") == "cc") & (pl.col("dim_0") == 3)).collect()
    # With pushdown disabled (QueryOptFlags(predicate_pushdown=False)),
    # Polars evaluates string comparisons against the UInt32 column
    # directly: the documented limitation, asserted as the specific
    # ComputeError Polars raises for String-vs-UInt32 comparisons.
    with pytest.raises(pl.exceptions.ComputeError, match="cannot compare"):
        lf.filter(pl.col("dim_0") == "cc").collect(
            optimizations=pl.QueryOptFlags(predicate_pushdown=False)
        )
    # The same plan with integer IDs works with pushdown disabled, since
    # the UInt32 comparison is then valid.
    no_pushdown = lf.filter(pl.col("dim_0") == 3).collect(
        optimizations=pl.QueryOptFlags(predicate_pushdown=False)
    )
    assert no_pushdown["dim_0"].to_list() == [3]


def test_resolve_labels_preflights_before_extraction():
    """Review finding: the request-sized preflight runs on the borrowed
    Python sequence before PyO3 extracts it, so a multi-million-ID request
    against a tiny budget is rejected without materializing the request."""
    with GdxrsReader(LABELS, dictionary_budget_bytes=128) as reader:
        with pytest.raises(RuntimeError, match="request-sized storage"):
            reader.resolve_labels([1] * 4_000_000)
        # A generator is not a sized sequence and is rejected clearly.
        with pytest.raises(ValueError, match="sized sequence"):
            reader.resolve_labels(i for i in [1, 2])


def expected_schema_b():
    return pl.Schema({"dim_0": pl.UInt32, "value": pl.Float64})


def test_mode_b_streams_without_materializing_categories():
    """Frozen: a tiny dictionary budget rejects mode A before output, while
    mode B still streams without collecting all labels — on a growing
    20,000-label table."""
    # Mode A on the growing table with a tiny admission: rejected before
    # any output (no records decoded).
    reader = GdxrsReader(GROW, dictionary_budget_bytes=1024)
    with pytest.raises(RuntimeError, match="dictionary budget"):
        reader.read_arrow("BIG")
    assert reader.scan_statistics()["decoded_rows"] == 0
    # Mode B on the same reader: no dictionary is materialized, the read
    # succeeds and reports the full table.
    batch = reader.read_arrow("BIG", key_mode="uel_id")
    assert batch.num_rows == 20_000
    assert batch.column(0).type == "uint32"
    # The default-budget mode A read also works on this table and agrees.
    enum_df = read_gdx(GROW, symbol="BIG", backend="gdxrs")
    id_df = read_gdx(GROW, symbol="BIG", backend="gdxrs", key_mode="uel_id")
    assert id_df.height == enum_df.height == 20_000
    assert id_df["dim_0"].to_list() == [
        c + 1 for c in enum_df["dim_0"].to_physical().to_list()
    ]
    assert id_df["value"].to_list() == enum_df["value"].to_list()
    reader.close()


def test_key_mode_validation_and_legacy_rejection():
    with pytest.raises(ValueError, match="key_mode"):
        read_gdx(DICT, symbol="P", backend="gdxrs", key_mode="bogus")
    if Reader is not None:
        with pytest.raises(ValueError, match="only supported by the gdxrs"):
            read_gdx(DICT, symbol="P", backend="legacy", key_mode="uel_id")


# ---------------------------------------------------------------------------
# Label bytes: strict UTF-8 in mode A, exact bytes in mode B
# ---------------------------------------------------------------------------


def test_mode_a_rejects_non_utf8_labels_before_output():
    """Frozen: raw label bytes 61 61 and ff 62 — mode A fails with a
    label-encoding error before emitting output."""
    with pytest.raises(RuntimeError, match="UTF-8"):
        read_gdx(LABELS, symbol="B", backend="gdxrs")
    with GdxrsReader(LABELS) as reader:
        with pytest.raises(RuntimeError, match="UTF-8"):
            reader.read_arrow("B")
        with pytest.raises(RuntimeError, match="UTF-8"):
            reader.uel_table()


def test_mode_b_returns_ids_without_decoding_labels():
    """Frozen: mode B returns IDs [1,2] and values [10,20] without decoding
    labels; the UTF-8 string filter 'aa' matches only ID 1."""
    df = read_gdx(LABELS, symbol="B", backend="gdxrs", key_mode="uel_id")
    assert df.schema == pl.Schema({"dim_0": pl.UInt32, "value": pl.Float64})
    assert df["dim_0"].to_list() == [1, 2]
    assert df["value"].to_list() == [10.0, 20.0]
    df = read_gdx(
        LABELS, symbol="B", backend="gdxrs", key_mode="uel_id", key_filter={0: ["aa"]}
    )
    assert df["dim_0"].to_list() == [1]
    assert df["value"].to_list() == [10.0]


def test_resolve_labels_is_frozen():
    """Frozen: the resolver called with [2,1,2] returns rows
    (2, b'\\xffb'), (1, b'aa'), (2, b'\\xffb') in that order, with schema
    {uel_id: UInt32, label: Binary}; ID 0 or 3 rejects the whole request
    with ValueError."""
    with GdxrsReader(LABELS) as reader:
        table = reader.resolve_labels([2, 1, 2])
        assert table.schema.names == ["uel_id", "label"]
        assert [str(t) for t in table.schema.types] == ["uint32", "binary"]
        assert table.column(0).to_pylist() == [2, 1, 2]
        assert table.column(1).to_pylist() == [b"\xffb", b"aa", b"\xffb"]
        for bad in ([0], [3], [1, 0], [2, 3]):
            with pytest.raises(ValueError, match="zero or outside"):
                reader.resolve_labels(bad)


def test_resolve_labels_conversion_budget_rejects_before_partial_output():
    """Frozen: a request whose conversion allocation exceeds admission
    fails before returning a partial table."""
    with GdxrsReader(LABELS, dictionary_budget_bytes=64) as reader:
        with pytest.raises(RuntimeError, match="dictionary budget"):
            reader.resolve_labels([1, 2])
    # A zero-budget reader is invalid configuration.
    with pytest.raises(ValueError, match="positive"):
        GdxrsReader(LABELS, dictionary_budget_bytes=0)


def test_resolve_labels_admission_counts_all_simultaneous_buffers():
    """Finding 1: the resolver's admission counts every simultaneously
    live allocation — validation storage, the collected label buffer, the
    Arrow offsets/values/validity, borrowed slices and conversion
    temporaries, plus fixed cursor costs — so a request that needs more
    than the budget is rejected even when the raw label payload alone
    would fit. Fixed costs are checked before anything is allocated."""
    # Fixed and request-sized storage alone exceed a 64 KiB-minus-one
    # budget, before any label is even fetched.
    with GdxrsReader(LABELS, dictionary_budget_bytes=64 * 1024 - 1) as reader:
        with pytest.raises(RuntimeError, match="request-sized storage"):
            reader.resolve_labels([1, 2])
    # The preflight precedes validation: a huge request against a tiny
    # budget is rejected for its request-sized storage, not per-ID
    # validation (the ID-0 ValueError would prove the validation vector
    # was built first).
    with GdxrsReader(LABELS, dictionary_budget_bytes=128) as reader:
        with pytest.raises(RuntimeError, match="request-sized storage"):
            reader.resolve_labels([0] * 100_000)
    # Resolving all 20,000 labels of the growing fixture needs ~2.2 MB
    # of simultaneous buffers for a ~120 KB raw payload: a 1 MiB budget
    # must reject it, while the default budget resolves the full table.
    ids = list(range(1, 20_001))
    with GdxrsReader(GROW, dictionary_budget_bytes=1024 * 1024) as reader:
        with pytest.raises(RuntimeError, match="dictionary budget"):
            reader.resolve_labels(ids)
        assert reader.scan_statistics()["decoded_rows"] == 0
    with GdxrsReader(GROW) as reader:
        table = reader.resolve_labels(ids)
        assert table.num_rows == 20_000
        assert table.column(0).to_pylist()[:3] == [1, 2, 3]
        assert table.column(1).to_pylist()[:3] == [b"u00000", b"u00001", b"u00002"]


def test_label_ids_bounded_lookup():
    """The bounded label→ID lookup preserves request order and reports
    unknown labels as None, without materializing the dictionary: on the
    growing fixture a 512 KiB budget rejects the full-dictionary
    admission (~1.1 MB) while the lookup still resolves labels."""
    with GdxrsReader(GROW, dictionary_budget_bytes=512 * 1024) as reader:
        with pytest.raises(RuntimeError, match="dictionary budget"):
            reader.uel_table()
        assert reader.label_ids(["u00000", "nope", "u00000"]) == [1, None, 1]
    with GdxrsReader(DICT) as reader:
        assert reader.label_ids(["dd", "aa", "bb", "nope"]) == [4, 1, 2, None]


# ---------------------------------------------------------------------------
# Empty/scalar symbols in both modes
# ---------------------------------------------------------------------------


def test_empty_and_scalar_symbols_keep_planned_schema_in_both_modes():
    """Frozen: empty one-dimensional symbols keep the planned
    enum-or-UInt32/Float64 schema; scalars produce exactly one row with no
    key columns; a zero limit preserves the schema."""
    M7 = "tests/data/m7/m7-semantics.gdx"
    for mode in ("enum", "uel_id"):
        df = read_gdx(M7, symbol="D", backend="gdxrs", key_mode=mode)
        assert df.height == 0
        assert df.columns == ["J", "value"]
        if mode == "enum":
            assert isinstance(df.schema["J"], pl.Enum)
        else:
            assert df.schema["J"] == pl.UInt32
        assert df.schema["value"] == pl.Float64
        df = read_gdx(M7, symbol="S", backend="gdxrs", key_mode=mode)
        assert df.columns == ["value"]
        assert df.to_dicts() == [{"value": 1.25}]
        df = read_gdx(M7, symbol="Z", backend="gdxrs", key_mode=mode)
        assert df.to_dicts() == [{"value": 0.0}]
        assert (
            scan_gdx(M7, symbol="S", backend="gdxrs", key_mode=mode)
            .head(0)
            .collect()
            .height
            == 0
        )
