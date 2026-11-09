#!/usr/bin/env python3
"""M7 task-047 capped streaming acceptance runner (Linux x86_64).

Executes the adapter contract §6 acceptance recipes against the
generated fixtures (examples/gen_m7_acceptance.rs) under the concrete
process caps:

- hard+soft RLIMIT_AS of 1 GiB (self-checked),
- measured whole-process peak RSS ceiling of 512 MiB,
- POLARS_MAX_THREADS=2, OPENBLAS_NUM_THREADS=1, MALLOC_ARENA_MAX=2
  (set BEFORE process startup via re-exec: glibc reads the arena bound
  while initializing the allocator),
- descriptor gate: at most 16 process descriptors including stdio,
  enforced on the PEAK descriptor count observed during execution,
- scratch gate: parallel-scan checkpoint-index sidecars measured
  during scans, gated at the contract's 8 GiB aggregate quota, and
  cleaned up after close; the run uses a PRIVATE temp directory so
  unrelated processes' scratch is never attributed to this run,
- combined 256 MiB library envelope gate: the core reader's root
  budget accounting (live/peak/limit) is recorded and its peak must
  stay within the envelope,
- queue gate: parallel statistics (from the SAME execution whose
  output is verified) must show admitted workers within the request,
  maximum_buffered_batches within the contract bound and
  peak_ready_batches within that maximum,
- noncollecting sinks only (collect_batches); .collect() is used only
  for bounded head checks, never as streaming evidence.

EVERY required assertion participates in the verdict: any false *_ok
flag, exceeded cap or failed cleanup sets caps_ok=false and the
process exits 1. Whole-stream validation covers every batch: key
columns are verified against the generator's mixed-radix recipe and
values against ordinal % 1000, both computed vectorized, plus
strictly ascending order between batches and within batches.

Expected outputs are computed from the deterministic generator recipe —
independently of either backend. A run is a QUALIFICATION only at the
contract sizes with >= 3 trials; anything else is reported as a smoke
run. Nonpositive --trials is rejected.

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
import shutil
import sys
import tempfile
import time

PROCESS_CAP_BYTES = 1024 * 1024 * 1024  # 1 GiB RLIMIT_AS (soft + hard)
RSS_CEILING_BYTES = 512 * 1024 * 1024  # measured peak-RSS ceiling
FD_GATE = 16  # process descriptors including stdio (contract §6)
SCRATCH_QUOTA_BYTES = 8 * 1024 * 1024 * 1024  # aggregate scratch (§6)
ENVELOPE_BYTES = 256 * 1024 * 1024  # combined library envelope (§6)
QUEUE_GATE = 4  # at most 4 queued batches (§6)
TRIALS = 3

RECORD_FIXTURE_RECORDS = 400_000_000
METADATA_FIXTURE_RECORDS = 10_000_000
METADATA_STRIDE = 7  # every 7th UEL is used (10M * 7 < 72M)
BATCH_ROWS = 1_000_000


def record_radices(records: int) -> list:
    """Mixed-radix factorization of the record count into 2s and 5s —
    the generator's key-tuple recipe (400M = 2^10 x 5^8)."""
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


def ensure_runtime_environment() -> None:
    """The capped qualification requires MALLOC_ARENA_MAX=2 (and the
    thread bounds) BEFORE process startup, plus a PRIVATE temp
    directory so scratch accounting never sees unrelated processes. If
    the environment is not configured, re-execute with it set."""
    private_tmp = os.environ.get("_M7_ACCEPTANCE_TMP")
    required = {
        "MALLOC_ARENA_MAX": "2",
        "POLARS_MAX_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1",
    }
    if private_tmp is None:
        private_tmp = tempfile.mkdtemp(prefix="m7-acceptance-tmp-")
    missing = {
        key: value
        for key, value in required.items()
        if os.environ.get(key) != value
    }
    if os.environ.get("_M7_ACCEPTANCE_READY") != "1":
        environment = dict(os.environ)
        environment.update(missing)
        environment["TMPDIR"] = private_tmp
        environment["TMP"] = private_tmp
        environment["_M7_ACCEPTANCE_TMP"] = private_tmp
        environment["_M7_ACCEPTANCE_READY"] = "1"
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def vm_values() -> dict:
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
    """Total bytes of adapter-owned index sidecar scratch under this
    run's PRIVATE temp directory (never unrelated processes')."""
    total = 0
    for path in glob.glob(
        os.path.join(tempfile.gettempdir(), "polars-gdx-gdxrs-index-*")
    ):
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass
    return total


def scratch_dirs() -> set:
    return set(
        glob.glob(os.path.join(tempfile.gettempdir(), "polars-gdx-gdxrs-index-*"))
    )


def set_process_cap() -> None:
    limit = (PROCESS_CAP_BYTES, PROCESS_CAP_BYTES)
    resource.setrlimit(resource.RLIMIT_AS, limit)
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    assert (soft, hard) == limit, "RLIMIT_AS cap not enforced"


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
        help="override the expected record count (SMOKE testing; the "
        "acceptance fixtures use the contract sizes)",
    )
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be >= 1 (zero trials qualify nothing)")

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
    radices = None if is_metadata else record_radices(records)
    dims = 1 if is_metadata else len(radices)
    offsets = [] if is_metadata else []
    if not is_metadata:
        running = 0
        for radix in radices:
            offsets.append(running)
            running += radix
    key_columns = ["dim_0"] if is_metadata else [f"dim_{d}" for d in range(dims)]
    is_smoke = is_metadata and records != METADATA_FIXTURE_RECORDS
    is_smoke = is_smoke or (not is_metadata and records != RECORD_FIXTURE_RECORDS)
    is_smoke = is_smoke or args.trials < 3

    # Vectorized expected-key/value builders: the ordinal range of a
    # batch determines every key column and the value column exactly.
    def expected_keys_frame(start: int, count: int) -> pl.DataFrame:
        ordinals = pl.int_range(start, start + count, dtype=pl.Int64)
        if is_metadata:
            return pl.select((ordinals * METADATA_STRIDE + 1).alias("dim_0"))
        columns = []
        for d in range(dims):
            column = ordinals
            for later in range(d + 1, dims):
                column = column // radices[later]
            column = column % radices[d]
            columns.append((column + offsets[d] + 1).alias(f"dim_{d}"))
        return pl.select(columns)

    def expected_values_frame(start: int, count: int) -> pl.DataFrame:
        ordinals = pl.int_range(start, start + count, dtype=pl.Int64)
        return pl.select((ordinals % 1000).cast(pl.Float64).alias("value"))

    report = {
        "task": "047",
        "kind": args.kind,
        "smoke": is_smoke,
        "qualification": not is_smoke,
        "fixture": os.path.abspath(args.fixture),
        "fixture_bytes": os.path.getsize(args.fixture),
        "fixture_sha256": sha256_file(args.fixture),
        "expected_rows": expected_rows,
        "expected_value_sum": expected_sum,
        "process_cap_bytes": PROCESS_CAP_BYTES,
        "rss_ceiling_bytes": RSS_CEILING_BYTES,
        "fd_gate": FD_GATE,
        "scratch_quota_bytes": SCRATCH_QUOTA_BYTES,
        "envelope_bytes": ENVELOPE_BYTES,
        "queue_gate": QUEUE_GATE,
        "env": {
            "POLARS_MAX_THREADS": os.environ.get("POLARS_MAX_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MALLOC_ARENA_MAX": os.environ.get("MALLOC_ARENA_MAX"),
            "TMPDIR": tempfile.gettempdir(),
        },
        "rlimit_as": list(resource.getrlimit(resource.RLIMIT_AS)),
        "scenarios": [],
        "caps_ok": True,
    }

    def fail(trial, message):
        trial["cap_failures"] = trial.get("cap_failures", []) + [message]
        report["caps_ok"] = False

    def check_trial_gates(trial):
        """Gates that apply to EVERY scenario: RSS ceiling, PEAK
        descriptor count, scratch quota, the combined envelope, and
        every required assertion flag."""
        if trial["peak_rss_bytes"] > RSS_CEILING_BYTES:
            fail(trial, "rss_ceiling_exceeded")
        peak_fd = trial.get("peak_fd_count", trial["fd_count"])
        if peak_fd > FD_GATE:
            fail(trial, "fd_gate_exceeded")
        if trial.get("peak_scratch_bytes", 0) > SCRATCH_QUOTA_BYTES:
            fail(trial, "scratch_quota_exceeded")
        if trial.get("envelope_peak_bytes", 0) > ENVELOPE_BYTES:
            fail(trial, "envelope_exceeded")
        for key, value in list(trial.items()):
            if key.endswith("_ok") and value is False:
                fail(trial, f"assertion_failed:{key}")

    def verify_batch(df, start: int) -> bool:
        """Whole-batch exact verification: every key column and the
        value column against the generator recipe, vectorized."""
        count = df.height
        if count == 0:
            return False
        if not expected_keys_frame(start, count).equals(
            df.select(key_columns)
        ):
            return False
        return expected_values_frame(start, count).equals(df.select("value"))

    def frame_schema_ok(df) -> bool:
        for name in key_columns:
            if df.schema[name] != pl.UInt32:
                return False
        return df.schema["value"] == pl.Float64

    lf = scan_gdx(args.fixture, symbol="BIG", backend="gdxrs", key_mode="uel_id")

    # 1. Full noncollecting stream (mode B, serial): EVERY batch is
    #    verified exactly (keys, values, dtypes) and key order is
    #    checked within and between batches.
    def full_stream():
        rows = 0
        first_batch_s = None
        started = time.monotonic()
        schema_ok = True
        batch_ok = True
        order_ok = True
        last_key = None
        peak_fd = fd_count()
        peak_scratch = 0
        envelope_peak = 0
        with GdxrsReader(args.fixture) as reader:
            envelope_peak = max(
                envelope_peak, reader.budget_statistics()["peak_bytes"]
            )
        for batch in lf.collect_batches(chunk_size=BATCH_ROWS):
            if first_batch_s is None:
                first_batch_s = time.monotonic() - started
            count = batch.height
            if not frame_schema_ok(batch):
                schema_ok = False
            if not verify_batch(batch, rows):
                batch_ok = False
            # Within-batch ascending order: the batch equals its own
            # sorted self (vectorized lexicographic sort).
            keys = batch.select(key_columns)
            if keys.height > 1 and not keys.equals(keys.sort(key_columns)):
                order_ok = False
            first_key = keys.head(1)
            if last_key is not None and not _frame_lt(last_key, first_key):
                order_ok = False
            last_key = keys.tail(1)
            rows += count
            peak_fd = max(peak_fd, fd_count())
            peak_scratch = max(peak_scratch, scratch_bytes())
        checksum_ok = rows == expected_rows
        return {
            "rows": rows,
            "first_batch_s": round(first_batch_s or -1.0, 3),
            "schema_ok": schema_ok,
            "batch_ok": batch_ok,
            "order_ok": order_ok,
            "checksum_ok": checksum_ok,
            "exact_output_ok": all(
                [schema_ok, batch_ok, order_ok, checksum_ok]
            ),
            "peak_fd_count": peak_fd,
            "peak_scratch_bytes": peak_scratch,
            "envelope_peak_bytes": envelope_peak,
        }

    scenario = run_scenario("full-noncollecting-modeB", full_stream, args.trials)
    for trial in scenario["trials"]:
        check_trial_gates(trial)
    report["scenarios"].append(scenario)

    # 2. Projection: only the value column is delivered, exact dtypes.
    def projection():
        rows = 0
        schema_ok = True
        for batch in lf.select("value").collect_batches(chunk_size=BATCH_ROWS):
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
    #    996, 997, 998.
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

    # 4. Capped PARALLEL full stream (threads=2): ONE execution whose
    #    output is verified exactly AND whose worker/queue statistics
    #    are validated from that same cursor.
    def parallel_full_stream():
        rows = 0
        first_batch_s = None
        started = time.monotonic()
        batch_ok = True
        order_ok = True
        last_key = None
        peak_fd = fd_count()
        peak_scratch = 0
        with GdxrsReader(args.fixture) as reader:
            cursor = reader.scan_batches(
                "BIG", key_mode="uel_id", threads=2, batch_rows=BATCH_ROWS
            )
            while True:
                holder = cursor.next_stream()
                if holder is None:
                    break
                batch = pl.from_arrow(holder).struct.unnest()
                if first_batch_s is None:
                    first_batch_s = time.monotonic() - started
                if not verify_batch(batch, rows):
                    batch_ok = False
                keys = batch.select(key_columns)
                if last_key is not None and keys.head(1).equals(last_key):
                    order_ok = False
                last_key = keys.tail(1)
                rows += batch.height
                peak_fd = max(peak_fd, fd_count())
                peak_scratch = max(peak_scratch, scratch_bytes())
            stats = cursor.scan_statistics()
            cursor.close()
            envelope = reader.budget_statistics()
        return {
            "rows": rows,
            "first_batch_s": round(first_batch_s or -1.0, 3),
            "batch_ok": batch_ok,
            "order_ok": order_ok,
            "admitted_workers": stats.get("admitted_workers"),
            "requested_workers": stats.get("requested_workers"),
            "maximum_buffered_batches": stats.get("maximum_buffered_batches"),
            "peak_ready_batches": stats.get("peak_ready_batches"),
            "joined_workers": stats.get("joined_workers"),
            "complete": stats.get("complete"),
            "parallel_output_ok": (
                rows == expected_rows and batch_ok and order_ok
            ),
            "queue_gate_ok": (
                stats.get("requested_workers") == 2
                and stats.get("admitted_workers") == 2
                and stats.get("maximum_buffered_batches", 99) <= QUEUE_GATE
                and stats.get("peak_ready_batches", 99)
                <= stats.get("maximum_buffered_batches", 0)
                and stats.get("joined_workers") == 2
                and stats.get("complete") is True
            ),
            "peak_fd_count": peak_fd,
            "peak_scratch_bytes": peak_scratch,
            "envelope_peak_bytes": envelope["peak_bytes"],
        }

    scenario = run_scenario(
        "parallel-full-stream-threads2", parallel_full_stream, args.trials
    )
    for trial in scenario["trials"]:
        check_trial_gates(trial)
    report["scenarios"].append(scenario)

    # 5. Capped PARALLEL cancellation: workers join and this reader's
    #    own sidecar scratch (private temp dir) is cleaned up.
    def parallel_cancel():
        with GdxrsReader(args.fixture) as reader:
            cursor = reader.scan_batches(
                "BIG", key_mode="uel_id", threads=2, batch_rows=100_000
            )
            holder = cursor.next_stream()
            first = pl.from_arrow(holder).struct.unnest()
            decoded = cursor.scan_statistics()["decoded_rows"]
            during_scratch = scratch_bytes()
            created = scratch_dirs()
            cursor.cancel()
            stats = cursor.scan_statistics()
        after = scratch_dirs()
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
        schema_ok = frame_schema_ok(df)
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
    #    dictionary must be rejected clearly; the record fixtures'
    #    small dictionaries must admit.
    def mode_a_admission():
        try:
            scan_gdx(args.fixture, symbol="BIG", backend="gdxrs").head(1).collect()
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

    # 9. Ordered first rows vs the independent expectation.
    def ordered_head():
        df = lf.head(5).collect()
        if not frame_schema_ok(df):
            return {"ordered_head_ok": False}
        ok = verify_batch(df, 0) and df.height == 5
        return {
            "keys": [[df[name][i] for name in key_columns] for i in range(5)],
            "values": df["value"].to_list(),
            "schema": {k: str(v) for k, v in df.schema.items()},
            "ordered_head_ok": ok,
        }

    scenario = run_scenario("ordered-head5-independent", ordered_head, args.trials)
    for trial in scenario["trials"]:
        check_trial_gates(trial)
    report["scenarios"].append(scenario)

    if args.report:
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2)
    print(
        json.dumps(
            {
                "caps_ok": report["caps_ok"],
                "kind": args.kind,
                "smoke": is_smoke,
            },
            indent=2,
        )
    )
    return 0 if report["caps_ok"] else 1


def _frame_lt(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    """Lexicographic tuple comparison of two single-row key frames."""
    for name in left.columns:
        a = left[name][0]
        b = right[name][0]
        if a < b:
            return True
        if a > b:
            return False
    return False


if __name__ == "__main__":
    sys.exit(main())
