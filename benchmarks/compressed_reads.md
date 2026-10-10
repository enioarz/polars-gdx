# Compressed GDX read benchmark

The native decoder now reads byte tags and scalar values directly from its
current buffer. Values spanning buffers use the existing refill path, including
zlib decompression. The scalar accessors are inline and the concrete stream is
final, allowing the compiler to remove virtual calls from the record loop.
The file format, zlib implementation, and Python API are unchanged.

## Measurement

Measured against `c795069a34d363d388ae7eaa5cf7c57eea224e7a` on Linux x86-64
with a four-CPU quota, Python 3.12, Rust 1.99, Polars 2.0, and PyArrow 26.0.
The `gen_fixture` example produces six million five-dimensional records:
about 3.3 MiB compressed or 60 MiB uncompressed.

Each result is the median of 14 repeated full reads across two fresh Python
processes. Each process performs one first read before seven timed repetitions;
the first read is excluded from the table. Before/after order is reversed in the
second round. The OS file cache is warm; these results do not measure disk-cold
I/O. Parallel repetitions reuse the restart index built by the first read.

| File | Threads | Before | After | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| Compressed | 1 | 580 ms | 529 ms | 8.8% |
| Compressed | 4 | 505 ms | 481 ms | 4.7% |
| Uncompressed | 1 | 539 ms | 493 ms | 8.5% |
| Uncompressed | 4 | 489 ms | 488 ms | Within measurement noise |

These are synthetic-fixture results; dimensions, value distribution, storage,
and CPU affect the gain. The first parallel read still has to build its restart
index. No performance assertion is added to tests.

## Reproduce

With the development virtual environment and Rust toolchain active:

```sh
maturin develop --release --locked
COMPRESS=1 OUT=/tmp/big_compressed.gdx cargo run --locked --release -p gdx --example gen_fixture
OUT=/tmp/big.gdx cargo run --locked --release -p gdx --example gen_fixture
python benchmarks/bench_read.py /tmp/big_compressed.gdx --threads 1 --expect-rows 6000000
python benchmarks/bench_read.py /tmp/big_compressed.gdx --threads 4 --expect-rows 6000000
python benchmarks/bench_read.py /tmp/big.gdx --threads 1 --expect-rows 6000000
python benchmarks/bench_read.py /tmp/big.gdx --threads 4 --expect-rows 6000000
```

Run the same commands on the baseline revision with the same dependencies to
compare. The script reports first-read latency separately from repeated reads.

## Correctness coverage

- Native stream tests cover every scalar alignment across 32 KiB blocks,
  compressed and uncompressed input, opposite byte order, and compressed
  checkpoint resumes. This suite also runs in CI.
- The Python compressed fixture is checked against every expected key/value
  tuple, followed by serial/parallel parity checks.
- The five large positional/span tests pass with both six-million-record
  fixtures present, including value equality as well as key equality.
- The 52 Python tests, regular Rust suite, formatting, Clippy, and existing
  two-million-record GAMS comparison benchmark pass. Four optional Rust tests
  requiring an external GAMS fixture remain ignored by the regular suite.
