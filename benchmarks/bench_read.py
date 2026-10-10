"""Measure GDX loading without a GAMS installation.

Generate the six-million-row compressed fixture:
  COMPRESS=1 OUT=/tmp/big_compressed.gdx cargo run --release -p gdx --example gen_fixture
Run each thread count in a new process so the first read includes index creation:
  python benchmarks/bench_read.py /tmp/big_compressed.gdx --threads 1 --expect-rows 6000000
  python benchmarks/bench_read.py /tmp/big_compressed.gdx --threads 4 --expect-rows 6000000

Reports first-read and repeated-read timings separately. These are process-cold,
not disk-cold measurements: this script does not flush the OS page cache.
"""

import argparse
import gc
import json
import statistics
import time

import pyarrow  # noqa: F401 -- initialize Arrow before measuring the reader

from polars_gdx import read_gdx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--symbol", default="big")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--expect-rows", type=int)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    samples = []
    for _ in range(args.runs + 1):
        gc.collect()
        start = time.perf_counter()
        frame = read_gdx(args.path, symbol=args.symbol, threads=args.threads)
        samples.append(time.perf_counter() - start)
        rows = frame.height
        if args.expect_rows is not None and rows != args.expect_rows:
            raise AssertionError(f"Expected {args.expect_rows} rows, got {rows}")
        del frame

    print(json.dumps({
        "path": args.path,
        "threads": args.threads,
        "rows": rows,
        "first_seconds": samples[0],
        "repeated_median_seconds": statistics.median(samples[1:]),
        "repeated_seconds": samples[1:],
    }, indent=2))


if __name__ == "__main__":
    main()
