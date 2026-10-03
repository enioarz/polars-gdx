"""Benchmark: polars-gdx vs gamsapi (gams.transfer, pandas-backed).

Run:  python benchmarks/bench_vs_gamsapi.py [n_rows]

Requires: pip install polars pandas gamsapi gamspy
Set SYSDIR below (gamspy_base ships all required native libraries).
"""
import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from polars_gdx import scan_gdx  # noqa: E402

import gams.transfer as gt  # noqa: E402

SYSDIR = next(Path(__file__).resolve().parent.parent.glob(
    ".venv/lib/python*/site-packages/gamspy_base"))

GDX = sys.argv[1] if len(sys.argv) > 1 else "/tmp/bench_big.gdx"
FILTER_LABEL = "i00042"   # one of 2000 first-dim labels
N_WARMUP = 1
N_RUNS = 5


def timeit(fn, *args):
    best = None
    for _ in range(N_WARMUP):
        fn(*args)
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        out = fn(*args)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, out


def main():
    print(f"file: {GDX}")
    print(f"sysdir: {SYSDIR}")

    # ---- polars-gdx: lazy full read ----
    def pgx_full():
        return scan_gdx(GDX, symbol="bigpar").collect()

    # ---- polars-gdx: lazy read with native prefilter ----
    def pgx_prefilter():
        return scan_gdx(GDX, symbol="bigpar", key_filter={0: [FILTER_LABEL]}).collect()

    # ---- polars-gdx: lazy read + polars predicate (pushed into prefilter) ----
    def pgx_predicate():
        return scan_gdx(GDX, symbol="bigpar").filter(
            pl.col("dim_0") == FILTER_LABEL).collect()

    # ---- gamsapi: gams.transfer full read into pandas ----
    def gamsapi_full():
        c = gt.Container(system_directory=str(SYSDIR))
        c.read(GDX, symbols=["bigpar"])
        return c["bigpar"].records

    # ---- gamsapi: full read then pandas filter (the only option it has) ----
    def gamsapi_filter():
        c = gt.Container(system_directory=str(SYSDIR))
        c.read(GDX, symbols=["bigpar"])
        return c["bigpar"].records[c["bigpar"].records["uni_0"] == FILTER_LABEL]

    results = []
    for name, fn in [
        ("polars-gdx  full read", pgx_full),
        ("gamsapi     full read", gamsapi_full),
        ("polars-gdx  prefiltered (native)", pgx_prefilter),
        ("polars-gdx  predicate (native prefilter)", pgx_predicate),
        ("gamsapi     full read + pandas filter", gamsapi_filter),
    ]:
        dt, out = timeit(fn)
        n = len(out) if hasattr(out, "__len__") else out.height
        results.append((name, dt, n))
        print(f"{name:44s} {dt:8.3f} s   rows={n}")

    print()
    full_pgx = next(dt for n, dt, _ in results if "full" in n and "polars-gdx" in n)
    full_gdx = next(dt for n, dt, _ in results if "full" in n and "gamsapi" in n)
    pre = next(dt for n, dt, _ in results if "prefiltered" in n)
    print(f"speedup full read:    {full_gdx / full_pgx:.1f}x")
    print(f"speedup prefiltered:  {full_gdx / pre:.1f}x vs gamsapi full+filter")


if __name__ == "__main__":
    main()
