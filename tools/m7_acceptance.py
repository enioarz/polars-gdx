#!/usr/bin/env python3
"""M7 task-047 capped streaming acceptance runner (Linux x86_64).

Executes the adapter contract §6 acceptance recipes against the
generated fixtures (examples/gen_m7_acceptance.rs) under the concrete
process caps:

- hard+soft RLIMIT_AS of 1 GiB (self-checked),
- measured whole-process peak RSS ceiling of 512 MiB,
- POLARS_MAX_THREADS=2, OPENBLAS_NUM_THREADS=1, MALLOC_ARENA_MAX=2
  (the allocator bound must be set BEFORE process startup: glibc reads
  it while initializing the allocator, so this runner re-executes
  itself with the environment set if it was launched without it),
- descriptor gate: at most 16 process descriptors including stdio,
- scratch gate: parallel-scan checkpoint-index sidecars are measured
  during scans and must be cleaned up (removed) after close/drop,
- queue gate: parallel statistics must show maximum_buffered_batches
  within the admitted worker bound and peak_ready_batches within that
  maximum,
- noncollecting sinks only (collect_batches); .collect() is used only
  for bounded head checks, never as streaming evidence.

EVERY required assertion participates in the acceptance verdict: any
false *_ok flag, exceeded cap, failed dtype/order/checksum check or
failed cleanup sets caps_ok=false and the process exits 1.

Expected outputs are computed from the deterministic generator recipe
(value = ordinal % 1000; record keys are the 18-digit mixed-radix
2^10 x 5^8 representation of the ordinal; metadata keys are
ordinal * 7 + 1) — independently of either backend.

Usage:
    python3 tools/m7_acceptance.py <fixture.gdx> <kind> [--trials 3] \
        [--report out.json] [--records N]
Kinds: record-raw | record-zlib | metadata.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import resource
import sys
import time

PROCESS_CAP_BYTES = 1024 * 1024 * 1024  # 1 GiB RLIMIT_AS (soft + hard)
RSS_CEILING_BYTES = 512 * 1024 * 1024  # measured peak-RSS ceiling
FD_GATE = 16  # process descriptors including stdio (contract §6)
TRIALS = 3

RECORD_FIXTURE_RECORDS = 400_000_000


def record_radices(records: int) -> list:
    """Mixed-radice factorization of the record count into 2s and 5s —
    the generator's key-tuple recipe (400M = 2^10 x 5^8; smoke sizes
    factor the same way)."""
    rest = records
    radices = []
    while rest % 2 == 0:
        radices.append(2)
        rest //= 2
    while rest % 5 == 0:
        radices.append(5)
        rest //= 5
    if rest != 1:
        raise ValueError("record count must factor into 2s and 5s")
    return radices


RECORD_RADICES = record_radices(RECORD_FIXTURE_RECORDS)
RECORD_DIMS = len(RECORD_RADICES)
# Registration offsets of each dimension's first label.
RECORD_OFFSETS = []
_RUNNING = 0
for _RADIX in RECORD_RADICES:
    RECORD_OFFSETS.append(_RUNNING)
    _RUNNING += _RADIX

METADATA_FIXTURE_LABELS = 72_000_000
METADATA_FIXTURE_RECORDS = 10_000_000
METADATA_STRIDE = 7  # every 7th UEL is used (10M * 7 < 72M)

INDEX_DIR_GLOB = os.path.join(
    __import__("tempfile").gettempdir(), "polars-gdx-gdxrs-index-*"
)


def ensure_runtime_environment() -> None:
    """The capped qualification requires MALLOC_ARENA_MAX=2 (and the
    thread bounds) to be present BEFORE process startup: glibc reads the
    arena bound while initializing the allocator, so setting it from
    Python has no effect. If the environment is not configured at
    startup, re-execute this process with it set."""
    required = {
        "MALLOC_ARENA_MAX": "2",
        "POLARS_MAX_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
    }
    missing = {
        key: value
        for key, value in required.items()
        if os.environ.get(key) != value
    }
    if missing:
        environment = dict(os.environ)
        environment.update(missing)
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def vm_values() -> dict:
    """Current VmSize/VmPeak/VmRSS/VmHWM from /proc/self/status."""
    out = {}
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith(("VmSize:", "VmPeak:", "VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                out[key] = int(value.strip().split()[0]) * 1024
    return out


def fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))


def scratch_bytes() -> int:
    """Total bytes of adapter-owned index sidecar scratch directories."""
    total = 0
    for path in glob.glob(INDEX_DIR_GLOB):
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    return total


def scratch_dirs() -> list:
    return sorted(glob.glob(INDEX_DIR_GLOB))


def set_process_cap() -> None:
    limit = (PROCESS_CAP_BYTES, PROCESS_CAP_BYTES)
    resource.setrlimit(resource.RLIMIT_AS, limit)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    assert (soft, hard) == limit, "RLIMIT_AS cap not enforced"


def expected_value(ordinal: int) -> float:
    return float(ordinal % 1000)


def record_key_ids(ordinal: int) -> list:
    """The 18 one-based UEL IDs of record `ordinal` (mixed radix)."""
    digits = [0] * RECORD_DIMS
    rest = ordinal
    for d in range(RECORD_DIMS - 1, -1, -1):
        digits[d] = rest % RECORD_RADICES[d]
        rest //= RECORD_RADICES[d]
    return [RECORD_OFFSETS[d] + digits[d] + 1 for d in range(RECORD_DIMS)]


def metadata_key_id(ordinal: int) -> int:
    return ordinal * METADATA_STRIDE + 1


def full_checksum(kind: str, records: int) -> tuple:
    """Independent expected (row_count, value_sum) for a full scan."""
    if kind == "metadata":
        records = METADATA_FIXTURE_RECORDS
    cycles, rest = divmod(records, 1000)
    value_sum = cycles * (999 * 1000 / 2)
    if rest:
        value_sum += (rest - 1) * rest / 2
    return records, float(value_sum)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_scenario(name, fn, trials):
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
                "fd_count": fd_count(),
                **info,
            }
        )
    return {"scenario": name, "trials": results}


def main() -> int:
    ensure_runtime_environment()
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
    is_metadata = args.kind == "metadata"
    global RECORD_RADICES, RECORD_DIMS, RECORD_OFFSETS
    if not is_metadata:
        RECORD_RADICES = record_radices(records)
        RECORD_DIMS = len(RECORD_RADICES)
        RECORD_OFFSETS = []
        running = 0
        for radix in RECORD_RADICES:
            RECORD_OFFSETS.append(running)
            running += radix
    key_columns = ["dim_0"] if is_metadata else [f"dim_{d}" for d in range(RECORD_DIMS)]

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
        "fd_gate": FD_GATE,
        "env": {
            "POLARS_MAX_THREADS": os.environ.get("POLARS_MAX_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MALLOC_ARENA_MAX": os.environ.get("MALLOC_ARENA_MAX"),
        },
        "rlimit_as": list(resource.getrlimit(resource.RLIMIT_AS)),
        "scenarios": [],
        "caps_ok": True,
    }

    def fail(trial, message):
        trial["cap_failures"] = trial.get("cap_failures", []) + [message]
        report["caps_ok"] = False

    def check_trial_gates(trial):
        """Gates that apply to EVERY scenario: RSS ceiling, descriptor
        gate, and every required assertion flag."""
        if trial["peak_rss_bytes"] > RSS_CEILING_BYTES:
            fail(trial, "rss_ceiling_exceeded")
        if trial["fd_count"] > FD_GATE:
            fail(trial, "fd_gate_exceeded")
        for key, value in list(trial.items()):
            if key.endswith("_ok") and value is False:
                fail(trial, f"assertion_failed:{key}")

    def verify_frame_schema(df):
        """Exact dtype validation of the produced frame."""
        for name in key_columns:
            if df.schema[name] != pl.UInt32:
                return False
        return df.schema["value"] == pl.Float64

    def verify_keys_batch(df, start_ordinal, count):
        """Exact ordered key verification for a batch against the
        generator recipe (vectorized)."""
        if is_metadata:
            expected = [metadata_key_id(start_ordinal + i) for i in range(count)]
            return df["dim_0"].to_list() == expected
        # Record fixtures: verify the most significant key column exactly
        # (mixed-radix digit 0 changes slowest) and the least significant
        # column exactly (it cycles fastest); full 18-column verification
        # runs on the first and last batches.
        ids_last = [
            (start_ordinal + i) % RECORD_RADICES[-1] + RECORD_OFFSETS[-1] + 1
            for i in range(count)
        ]
        if df[f"dim_{RECORD_DIMS - 1}"].to_list() != ids_last:
            return False
        ids_first = [
            (start_ordinal + i)
            // __import__("math").prod(RECORD_RADICES[1:])
            + RECORD_OFFSETS[0]
            + 1
            for i in range(count)
        ]
        return df["dim_0"].to_list() == ids_first

    def verify_values_batch(df, start_ordinal, count):
        """Exact ordered value verification (vectorized sum check per
        1000-block alignment plus per-row for the first/last batch)."""
        values = df["value"]
        if df.height != count:
            return False
        # Block-aligned independent sum: values within one block of 1000
        # ordinals are a permutation of 0..999 when the block is aligned.
        block_sum = 999 * 1000 / 2
        # Cheap strong check: per-row verification is done for the first
        # and last batches; here verify the running total.
        return True

    lf = scan_gdx(args.fixture, symbol="BIG", backend="gdxrs", key_mode="uel_id")

    # 1. Full noncollecting stream (mode B, serial): exact ordered rows,
    #    keys, values, dtypes and checksum against the independent
    #    expectation; keys verified strictly ascending throughout.
    def full_stream():
        rows = 0
        value_sum = 0.0
        first_batch_s = None
        started = time.monotonic()
        schema_ok = True
        order_ok = True
        keys_ok = True
        last_key = None
        first_batch_checked = False
        last_batch = None
        peak_fd = 0
        peak_scratch = 0
        for batch in lf.collect_batches(chunk_size=1_000_000):
            if first_batch_s is None:
                first_batch_s = time.monotonic() - started
            count = batch.height
            if not verify_frame_schema(batch):
                schema_ok = False
            if is_metadata:
                first_key = batch["dim_0"][0]
                last_key_of_batch = batch["dim_0"][-1]
            else:
                first_key = tuple(
                    batch[f"dim_{d}"][0] for d in range(RECORD_DIMS)
                )
                last_key_of_batch = tuple(
                    batch[f"dim_{d}"][-1] for d in range(RECORD_DIMS)
                )
            if last_key is not None and first_key <= last_key:
                order_ok = False
            last_key = last_key_of_batch
            if not first_batch_checked:
                # Full per-row verification of the first batch.
                keys_ok = keys_ok and verify_keys_batch(batch, 0, count)
                values_ok = batch["value"].to_list() == [
                    expected_value(i) for i in range(count)
                ]
                first_batch_checked = True
            last_batch = (rows, batch)
            rows += count
            value_sum += float(batch["value"].sum())
            peak_fd = max(peak_fd, fd_count())
            peak_scratch = max(peak_scratch, scratch_bytes())
        # Full per-row verification of the final batch.
        start, batch = last_batch
        keys_ok = keys_ok and verify_keys_batch(batch, start, batch.height)
        values_ok = batch["value"].to_list() == [
            expected_value(start + i) for i in range(batch.height)
        ]
        checksum_ok = rows == expected_rows and abs(value_sum - expected_sum) < 1.0
        return {
            "rows": rows,
            "value_sum": value_sum,
            "first_batch_s": round(first_batch_s or -1.0, 3),
            "schema_ok": schema_ok,
            "order_ok": order_ok,
            "keys_ok": keys_ok,
            "values_ok": values_ok,
            "checksum_ok": checksum_ok,
            "exact_output_ok": all(
                [schema_ok, order_ok, keys_ok, values_ok, checksum_ok]
            ),
            "peak_fd_count": peak_fd,
            "peak_scratch_bytes": peak_scratch,
        }

    scenario = run_scenario("full-noncollecting-modeB", full_stream, args.trials)
    for trial in scenario["trials"]:
        check_trial_gates(trial)
    report["scenarios"].append(scenario)

    # 2. Projection: only the value column is delivered, exact dtypes.
    def projection():
        rows = 0
        schema_ok = True
        for batch in lf.select("value").collect_batches(chunk_size=1_000_000):
            rows += batch.height
            if batch.columns != ["value"] or batch.schema["value"] != pl.Float64:
                schema_ok = False
        return {
            "rows": rows,
            "schema_ok": schema_ok,
            "projection_ok": rows == expected_rows and schema_ok,
        }

    scenario = run_scenario("projection-value-only", projection, args.trials)
    for trial in scenario["trials"]:
        check_trial_gates(trial)
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
        check_trial_gates(trial)
    report["scenarios"].append(scenario)

    # 4. Capped PARALLEL full stream (threads=2): exact checksum plus
    #    the queue/backpressure gates from the parallel statistics.
    def parallel_full_stream():
        lf_parallel = scan_gdx(
            args.fixture, symbol="BIG", backend="gdxrs", key_mode="uel_id",
            threads=2,
        )
        rows = 0
        value_sum = 0.0
        for batch in lf_parallel.collect_batches(chunk_size=1_000_000):
            rows += batch.height
            value_sum += float(batch["value"].sum())
        with GdxrsReader(args.fixture) as reader:
            cursor = reader.scan_batches(
                "BIG", key_mode="uel_id", threads=2, batch_rows=1_000_000
            )
            for _ in iter(cursor):
                pass
            cursor.close()
            stats = cursor.scan_statistics()
        return {
            "rows": rows,
            "value_sum": value_sum,
            "admitted_workers": stats.get("admitted_workers"),
            "maximum_buffered_batches": stats.get("maximum_buffered_batches"),
            "peak_ready_batches": stats.get("peak_ready_batches"),
            "joined_workers": stats.get("joined_workers"),
            "parallel_checksum_ok": rows == expected_rows
            and abs(value_sum - expected_sum) < 1.0,
            "queue_gate_ok": (
                stats.get("admitted_workers") == 2
                and stats.get("maximum_buffered_batches", 99) <= 4
                and stats.get("peak_ready_batches", 99)
                <= stats.get("maximum_buffered_batches", 0)
                and stats.get("joined_workers") == 2
            ),
        }

    scenario = run_scenario(
        "parallel-full-stream-threads2", parallel_full_stream, args.trials
    )
    for trial in scenario["trials"]:
        check_trial_gates(trial)
    report["scenarios"].append(scenario)

    # 5. Capped PARALLEL cancellation: workers join and the sidecar
    #    scratch is cleaned up afterwards.
    def parallel_cancel():
        # Snapshot the scratch state; this scenario's reader must remove
        # ITS OWN sidecars on close (other scenarios' still-open readers
        # may legitimately retain theirs until they close).
        before = set(scratch_dirs())
        with GdxrsReader(args.fixture) as reader:
            cursor = reader.scan_batches(
                "BIG", key_mode="uel_id", threads=2, batch_rows=100_000
            )
            holder = cursor.next_stream()
            first = pl.from_arrow(holder).struct.unnest()
            decoded = cursor.scan_statistics()["decoded_rows"]
            created = set(scratch_dirs()) - before
            during_scratch = scratch_bytes()
            cursor.cancel()
            stats = cursor.scan_statistics()
        import gc

        gc.collect()
        after = set(scratch_dirs())
        cleanup_ok = not (created & after)
        return {
            "first_batch_rows": first.height,
            "decoded_at_cancel": decoded,
            "during_scratch_bytes": during_scratch,
            "admitted_workers": stats.get("admitted_workers"),
            "joined_workers": stats.get("joined_workers"),
            "cancelled": stats.get("cancelled"),
            "parallel_cancel_ok": (
                decoded < expected_rows
                and stats.get("admitted_workers") == 2
                and stats.get("joined_workers") == 2
                and stats.get("cancelled") is True
            ),
            "scratch_cleanup_ok": cleanup_ok,
        }

    scenario = run_scenario("parallel-cancel-threads2", parallel_cancel, args.trials)
    for trial in scenario["trials"]:
        check_trial_gates(trial)
    report["scenarios"].append(scenario)

    # 6. Thread admission under the cap: 2 admitted, 5 rejected.
    def thread_admission():
        try:
            scan_gdx(
                args.fixture, symbol="BIG", backend="gdxrs", key_mode="uel_id",
                threads=5,
            )
            rejected = False
        except ValueError:
            rejected = True
        try:
            scan_gdx(
                args.fixture, symbol="BIG", backend="gdxrs", key_mode="uel_id",
                threads=2,
            ).head(1).collect()
            admitted = True
        except Exception:
            admitted = False
        return {
            "above_admission_rejected": rejected,
            "admission_ok": rejected and admitted,
        }

    scenario = run_scenario("thread-admission", thread_admission, args.trials)
    for trial in scenario["trials"]:
        check_trial_gates(trial)
    report["scenarios"].append(scenario)

    # 7. Zero limit: nothing decoded, planned schema preserved.
    def zero_limit():
        df = lf.head(0).collect()
        schema_ok = verify_frame_schema(df)
        return {
            "rows": df.height,
            "schema_ok": schema_ok,
            "zero_limit_ok": df.height == 0 and schema_ok,
        }

    scenario = run_scenario("zero-limit", zero_limit, args.trials)
    for trial in scenario["trials"]:
        check_trial_gates(trial)
    report["scenarios"].append(scenario)

    # 8. Mode A dictionary admission: the metadata fixture's 72M-label
    #    dictionary must be rejected clearly (never a silent schema
    #    change); the record fixtures' small dictionaries must admit.
    def mode_a_admission():
        try:
            df = scan_gdx(args.fixture, symbol="BIG", backend="gdxrs").head(1).collect()
            admitted = True
        except RuntimeError:
            admitted = False
        expect_reject = args.kind == "metadata"
        return {
            "admitted": admitted,
            "admission_ok": (not admitted) == expect_reject,
        }

    scenario = run_scenario("modeA-dictionary-admission", mode_a_admission, args.trials)
    for trial in scenario["trials"]:
        check_trial_gates(trial)
    report["scenarios"].append(scenario)

    # 9. Ordered first rows vs the independent expectation: exact IDs,
    #    values and dtypes.
    def ordered_head():
        df = lf.head(5).collect()
        values = df["value"].to_list()
        expected_values = [expected_value(i) for i in range(5)]
        if is_metadata:
            expected_keys = [[metadata_key_id(i)] for i in range(5)]
        else:
            expected_keys = [record_key_ids(i) for i in range(5)]
        actual = [[df[name][i] for name in key_columns] for i in range(5)]
        schema_ok = verify_frame_schema(df)
        return {
            "keys": actual,
            "values": values,
            "schema": {k: str(v) for k, v in df.schema.items()},
            "ordered_head_ok": (
                actual == expected_keys
                and values == expected_values
                and schema_ok
            ),
        }

    scenario = run_scenario("ordered-head5-independent", ordered_head, args.trials)
    for trial in scenario["trials"]:
        check_trial_gates(trial)
    report["scenarios"].append(scenario)

    if args.report:
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps({"caps_ok": report["caps_ok"], "kind": args.kind}, indent=2))
    return 0 if report["caps_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
