# Contributing to polars-gdx

Thanks for your interest in contributing! This project welcomes contributions
of all kinds: bug reports, fixes, performance improvements, documentation, and
new features.

## Getting started

1. Fork the repository and clone your fork.
2. Set up a development environment:

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   pip install maturin patchelf polars pyarrow pytest
   maturin develop --release
   ```

   Building requires a Rust toolchain (≥ 1.80), a C++17 compiler, and
   CMake ≥ 3.5. The vendored GDX C library is built automatically.

3. Run the test suite:

   ```sh
   cargo test --workspace
   pytest tests/ -q
   ```

## Making changes

* Keep changes focused: one logical change per PR.
* Follow the existing code style. Rust code must pass `cargo fmt` and
  `cargo clippy --workspace --all-targets -- -D warnings` (CI enforces both).
* Add or update tests for any behavior change.
* For performance-sensitive changes, include a benchmark comparison when
  practical (`benchmarks/bench_vs_gamsapi.py`).

## Reporting bugs

Open a GitHub issue with:

* A minimal reproducer (Python snippet plus, if possible, the GDX file or the
  code that generated it).
* The polars-gdx, polars, and platform versions.
* The full traceback or error output.

## Security issues

Please do not open a public issue for security problems — see
[SECURITY.md](SECURITY.md).

## Licensing

By contributing, you agree that your contributions will be licensed under the
same MIT license that covers this project (see [LICENSE](LICENSE)).
