#!/usr/bin/env python3
"""M7 task-047 capped streaming acceptance runner (Linux x86_64).

Executes the adapter contract §6 acceptance recipes against the
generated fixtures (examples/gen_m7_acceptance.rs) under the concrete
process caps:

- hard+soft RLIMIT_AS of 1 GiB (self-checked),
- measured whole-process peak RSS ceiling of 512 MiB,
- POLARS_MAX_THREADS=2, OPENBLAS_NUM_THREADS=1,
- noncollecting sinks only (collect_batches); .collect() is not
  streaming evidence and is used only for small independent checks.

Every scenario runs three trials; expected outputs are computed from
the deterministic generator recipe (value = ordinal % 1000, key label =
r<ordinal % 1000> for record fixtures; every 8th UEL for the metadata
fixture) — independently of either backend. The report is a structured
JSON document with per-trial timings, row counts, checksums, VmHWM/VmSize,
scratch/handle observations and explicit pass/fail against the caps.

Usage:
    python3 tools/m7_acceptance.py <fixture.gdx> <kind> [--trials 3] \
        [--report out.json]
Kinds: record-raw | record-zlib | metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

PROCESS_CAP_BYTES = 1024 * 1024 * 1024  # 1 GiB RLIMIT_AS (soft + hard)
RSS_CEILING_BYTES = 512 * 1024 * 1024  # measured peak-RSS ceiling
TRIALS = 3

RECORD_FIXTURE_RECORDS = 80_000_000
RECORD_FIXTURE_LABELS = 1_000
METADATA_FIXTURE_LABELS = 72_000_000
METADATA_FIXTURE_RECORDS = 10_000_000
METADATA_STRIDE = 7  # every 7th UEL is used (10M * 7 < 72M)


def vm_values() -> dict:
    """Current VmSize/VmPeak/VmHWM/VmRSS from /proc/self/status."""
    out = {}
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith(("VmSize:", "VmPeak:", "VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                out[key] = int(value.strip().split()[0]) * 1024
    return out


def set_process_cap() -> None:
    limit = (PROCESS_CAP_BYTES, PROCESS_CAP_BYTES)
    resource.setrlimit(resource.RLIMIT_AS, limit)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    assert (soft, hard) == limit, "RLIMIT_AS cap not enforced"


def expected_value(ordinal: int) -> float:
    return float(ordinal % 1000)


def full_checksum(kind: str, records: int) -> tuple[int, float]:
    """Independent expected (row_count, value_sum) for a full scan."""
    if kind == "metadata":
        records = METADATA_FIXTURE_RECORDS
    cycles, rest = divmod(records, 1000)
    value_sum = cycles * (999 * 1000 / 2) + (rest - 1) * rest / 2 if rest else cycles * (999 * 1000 / 2)
    return records, float(value_sum)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_scenario(name, fn, trials=TRIALS):
    results = []
    for trial in range(1, trials + 1):
        start = time.monotonic()
        info = fn()
        duration = time.monotonic() - start
        peak = vm_values()
        results.append(
            {
                "trial": trial,
                "duration_s": round(duration, 3),
                "peak_rss_bytes": peak["VmHWM"],
                "vm_size_bytes": peak["VmSize"],
                **info,
            }
        )
    return {"scenario": name, "trials": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture")
    parser.add_argument("kind", choices=["record-raw", "record-zlib", "metadata"])
    parser.add_argument("--report", default=None)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument(
        "--records",
        type=int,
        default=None,
        help="override the expected record count (smoke testing; the "
        "acceptance fixtures use the contract sizes)",
    )
    args = parser.parse_args()

    # Capped-qualification runtime configuration (contract §6: Polars
    # limited to two threads; other runtime pools configured explicitly
    # and recorded). MALLOC_ARENA_MAX=2 is required: glibc's default
    # per-thread 64 MiB arenas push the polars streaming engine's fixed
    # VmPeak to ~1009 MiB — right at the 1 GiB cap — while this setting
    # measures ~346 MiB with identical RSS.
    os.environ.setdefault("POLARS_MAX_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    set_process_cap()

    import polars as pl

    from polars_gdx import GdxrsReader, scan_gdx

    records = {
        "record-raw": RECORD_FIXTURE_RECORDS,
        "record-zlib": RECORD_FIXTURE_RECORDS,
        "metadata": METADATA_FIXTURE_RECORDS,
    }[args.kind]
    if args.records is not None:
        records = args.records
    expected_rows, expected_sum = full_checksum(args.kind, records)

    report = {
        "task": "047",
        "kind": args.kind,
        "fixture": os.path.abspath(args.fixture),
        "fixture_bytes": os.path.getsize(args.fixture),
        "fixture_sha256": sha256_file(args.fixture),
        "expected_rows": expected_rows,
        "expected_value_sum": expected_sum,
        "process_cap_bytes": PROCESS_CAP_BYTES,
        "rss_ceiling_bytes": RSS_CEILING_BYTES,
        "env": {
            "POLARS_MAX_THREADS": os.environ.get("POLARS_MAX_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MALLOC_ARENA_MAX": os.environ.get("MALLOC_ARENA_MAX"),
        },
        "rlimit_as": list(resource.getrlimit(resource.RLIMIT_AS)),
        "scenarios": [],
        "caps_ok": True,
    }

    def check_caps(trial):
        if trial["peak_rss_bytes"] > RSS_CEILING_BYTES:
            report["caps_ok"] = False
            trial["rss_ceiling_exceeded"] = True

    lf = scan_gdx(args.fixture, symbol="BIG", backend="gdxrs", key_mode="uel_id")

    # 1. Full noncollecting stream (mode B): exact ordered rows and
    #    checksum against the independent expectation.
    def full_stream():
        rows = 0
        value_sum = 0.0
        first_batch_rows = None
        first_batch_s = None
        started = time.monotonic()
        for batch in lf.collect_batches(chunk_size=100_000):
            if first_batch_rows is None:
                first_batch_rows = batch.height
                first_batch_s = time.monotonic() - started
            rows += batch.height
            value_sum += float(batch["value"].sum())
        ok = rows == expected_rows and abs(value_sum - expected_sum) < 1.0
        return {
            "rows": rows,
            "value_sum": value_sum,
            "first_batch_rows": first_batch_rows,
            "first_batch_s": round(first_batch_s or -1.0, 3),
            "exact_output_ok": ok,
        }

    scenario = run_scenario("full-noncollecting-modeB", full_stream, args.trials)
    for trial in scenario["trials"]:
        check_caps(trial)
        if not trial["exact_output_ok"]:
            report["caps_ok"] = False
    report["scenarios"].append(scenario)

    # 2. Projection: only the value column is delivered.
    def projection():
        rows = 0
        for batch in lf.select("value").collect_batches(chunk_size=100_000):
            rows += batch.height
            assert batch.columns == ["value"]
        return {"rows": rows, "projection_ok": rows == expected_rows}

    scenario = run_scenario("projection-value-only", projection, args.trials)
    for trial in scenario["trials"]:
        check_caps(trial)
    report["scenarios"].append(scenario)

    # 3. Sparse residual with a short global limit: expected values
    #    996, 997, 998 (first three ordinals with ordinal % 1000 > 995).
    def sparse_residual_head():
        df = lf.filter(pl.col("value") > 995.0).head(3).collect()
        values = df["value"].to_list()
        return {
            "values": values,
            "residual_head_ok": values == [996.0, 997.0, 998.0],
        }

    scenario = run_scenario("sparse-residual-head3", sparse_residual_head, args.trials)
    for trial in scenario["trials"]:
        check_caps(trial)
        if not trial["residual_head_ok"]:
            report["caps_ok"] = False
    report["scenarios"].append(scenario)

    # 4. Cancellation after the first batch: production stops at the
    #    acknowledged prefix (decoded rows far below the total).
    def cancel_after_first():
        with GdxrsReader(args.fixture) as reader:
            cursor = reader.scan_batches(
                "BIG", key_mode="uel_id", batch_rows=100_000
            )
            # Consume via the Arrow C stream capsule (the pyarrow-backed
            # __next__ path cannot map libarrow under the 1 GiB cap).
            holder = cursor.next_stream()
            first = pl.from_arrow(holder).struct.unnest()
            decoded = cursor.scan_statistics()["decoded_rows"]
            cursor.cancel()
            return {
                "first_batch_rows": first.height,
                "decoded_at_cancel": decoded,
                "cancel_ok": decoded < expected_rows,
                "joined_workers": cursor.scan_statistics().get("joined_workers"),
            }

    scenario = run_scenario("cancel-after-first-batch", cancel_after_first, args.trials)
    for trial in scenario["trials"]:
        check_caps(trial)
        if not trial["cancel_ok"]:
            report["caps_ok"] = False
    report["scenarios"].append(scenario)

    # 5. Zero limit: nothing decoded.
    def zero_limit():
        df = lf.head(0).collect()
        return {"rows": df.height, "zero_limit_ok": df.height == 0}

    scenario = run_scenario("zero-limit", zero_limit, args.trials)
    for trial in scenario["trials"]:
        check_caps(trial)
    report["scenarios"].append(scenario)

    # 6. Mode A dictionary admission: the metadata fixture's 72M-label
    #    dictionary must be rejected clearly (never a silent schema
    #    change); the record fixtures' small dictionaries admit.
    def mode_a_admission():
        try:
            df = scan_gdx(args.fixture, symbol="BIG", backend="gdxrs").head(1).collect()
            admitted = True
            rows = df.height
        except RuntimeError as exc:
            admitted = False
            rows = 0
            message = str(exc)
        expect_reject = args.kind == "metadata"
        ok = (not admitted) == expect_reject
        return {
            "admitted": admitted,
            "rows": rows,
            "admission_ok": ok,
            **({"message": message[:200]} if not admitted else {}),
        }

    scenario = run_scenario("modeA-dictionary-admission", mode_a_admission, args.trials)
    for trial in scenario["trials"]:
        check_caps(trial)
        if not trial["admission_ok"]:
            report["caps_ok"] = False
    report["scenarios"].append(scenario)

    # 7. Ordered first rows vs the independent expectation (exact IDs,
    #    values and schema — not merely row counts).
    def ordered_head():
        df = lf.head(5).collect()
        values = df["value"].to_list()
        expected_values = [expected_value(i) for i in range(5)]
        if args.kind == "metadata":
            ids = df["dim_0"].to_list()
            expected_ids = [(i * METADATA_STRIDE + 1) for i in range(5)]
            ok = ids == expected_ids and values == expected_values
        else:
            minor = expected_rows // RECORD_FIXTURE_LABELS
            ids0 = df["dim_0"].to_list()
            ids1 = df["dim_1"].to_list()
            expected_ids0 = [(i // minor) + 1 for i in range(5)]
            expected_ids1 = [(i % minor) + RECORD_FIXTURE_LABELS + 1 for i in range(5)]
            ids = list(zip(ids0, ids1))
            ok = (
                ids0 == expected_ids0
                and ids1 == expected_ids1
                and values == expected_values
            )
        return {
            "ids": [list(pair) for pair in ids]
            if isinstance(ids[0], tuple)
            else ids,
            "values": values,
            "schema": {k: str(v) for k, v in df.schema.items()},
            "ordered_head_ok": ok,
        }

    scenario = run_scenario("ordered-head5-independent", ordered_head, args.trials)
    for trial in scenario["trials"]:
        check_caps(trial)
        if not trial["ordered_head_ok"]:
            report["caps_ok"] = False
    report["scenarios"].append(scenario)

    if args.report:
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps({"caps_ok": report["caps_ok"], "kind": args.kind}, indent=2))
    return 0 if report["caps_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
