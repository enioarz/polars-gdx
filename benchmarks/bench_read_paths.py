"""Measure full, limited, selective and guaranteed-empty reads.

Generate the fixture with:
  cargo run -p gdx --release --example make_bench -- /tmp/bench.gdx 2000000 par-only
Run against each release build with:
  python benchmarks/bench_read_paths.py /tmp/bench.gdx

Each measurement opens a fresh Reader through scan_gdx; the filesystem and
process-wide restart cache are warmed. No gamsapi/GAMS installation is needed.
"""

import argparse
import json
import platform
import statistics
import time

import polars as pl
import pyarrow as pa

from polars_gdx import scan_gdx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--runs", type=int, default=9)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    def scan(**kwargs):
        return scan_gdx(args.path, symbol="bigpar", **kwargs)

    scenarios = {
        "full": lambda: scan().collect(),
        "head10": lambda: scan().head(10).collect(),
        "filter_dim0": lambda: scan(key_filter={0: ["i00042"]}).collect(),
        "absent_dim1": lambda: scan(key_filter={1: ["absent"]}).collect(),
        "empty_dim1": lambda: scan(key_filter={1: []}).collect(),
        "absent_dim1_threads4": lambda: scan(
            key_filter={1: ["absent"]}, threads=4
        ).collect(),
    }
    results = {}
    for name, read in scenarios.items():
        read()
        durations = []
        for _ in range(args.runs):
            start = time.perf_counter()
            frame = read()
            durations.append((time.perf_counter() - start) * 1000)
        results[name] = {
            "median_ms": statistics.median(durations),
            "min_ms": min(durations),
            "rows": frame.height,
        }
    print(json.dumps({
        "python": platform.python_version(),
        "polars": pl.__version__,
        "pyarrow": pa.__version__,
        "runs": args.runs,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
