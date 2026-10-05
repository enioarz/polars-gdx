# Agent context: finding small performance wins with a real extra-large GDX file

You are an agent working in a local clone of `enioarz/polars-gdx`. Your task:
find **small, low-risk performance wins** in the GDX read path, validated against a
**real extra-large file** (100M+ records) that exists on the local machine.

Scope: small wins. Do not redesign the architecture, do not rewrite the decode loop
wholesale, do not add new dependencies. A "win" is typically 1-5% on a hot path or a
constant-factor saving on a per-read cost, verified by a repeatable measurement.

## What this project is

A Polars IO plugin reading GAMS GDX files via a hand-written FFI layer to a vendored
MIT C++ library (`crates/gdx-sys/third_party/gdx`, extracted from lolow/gdxcomp,
building on GAMS-dev/gdx). Reads are index-based (raw UEL numbers), converted to Arrow
dictionary arrays only at the end. Key selling point vs the official `gamsapi`: lazy,
prefiltered reads that never materialize unwanted records.

## Layout

- `crates/gdx-sys/` — FFI bindings (`src/lib.rs`) and the vendored C++ library.
  - The C++ sources under `third_party/gdx/src/` carry `(polars-gdx extension)`
    markers for our additions; the rest is upstream GAMS-dev code — keep changes
    surgical and marked.
  - Generated wrappers: `third_party/gdx/generated/gdxcclib.cpp` (C API surface) and
    `gdxcwrap.h` (inline C++ shims). If you add a library method, you must add all
    three: the method in `gxfile.cpp`/`gdx.hpp`, the C wrapper in `gdxcclib.cpp`
    (including the lowercase `#define C__Foo c__foo` alias — the Rust FFI links
    against the lowercase symbol), the inline shim in `gdxcwrap.h`, and the Rust
    binding in `gdx-sys/src/lib.rs`.
  - `build.rs` builds with CMake (C++17) and emits link/rpath directives. It
    re-runs when the watched vendored files change; if you edit other files there,
    `touch crates/gdx-sys/build.rs` or `cargo clean -p gdx-sys`.
- `crates/gdx/` — safe Rust wrapper. `src/reader.rs` is the hot path:
  `read_symbol_raw*` (serial), `read_symbol_raw_parallel` (dim-0 UEL-range split,
  with range-skip), `read_symbol_raw_parallel_pos` (byte-position split for
  non-leading-dimension filters, using a cached restart-position index).
- `crates/polars-gdx/` — PyO3 layer; `src/lib.rs` builds the Arrow record batch and
  dispatches serial vs UEL-range vs positional parallel.
- `python/polars_gdx/` — Python API; `lazy.py` implements `scan_gdx`/`read_gdx`,
  predicate pushdown, and `threads="auto"` resolution.
- `tests/` — pytest suite (45 tests). `tests/test_lazy.py` covers parity between
  serial and parallel reads — keep it green.

## The read pipeline (where time goes)

1. Python `scan_gdx(...).collect()` → `_read` in `lazy.py`.
2. `Reader::read_arrow` in `polars-gdx/src/lib.rs`: resolve filters to UEL index
   bitmaps (one bit per UEL; membership check is a single word load), build the
   predicate closure, dispatch on threads/filter shape.
3. C bulk callback (`gdxDataReadRawFastEx` or the range variants): the C++ loop
   decodes records (delta-encoded keys + values) and calls back into Rust
   (`store_record_ex` in `gdx/src/reader.rs`), which applies the bitmap predicate
   and pushes accepted records into flat `Vec<u32>`/`Vec<f64>` buffers.
4. Buffers are wrapped into Arrow: keys as `DictionaryArray` over a shared UEL
   string-values array, values as `Float64Array`, handed to Polars zero-copy.

Parallel paths: dim-0 filter → UEL-range split (whole ranges skipped without I/O);
no dim-0 filter → positional split at exact restart-record boundaries (byte offsets
from a per-(file, symbol) cached index; workers seek and decode only their range;
record-count contract falls back to serial on mismatch).

Known hard constraint: **block-compressed symbol data** cannot use the positional or
range paths (logical vs physical positions diverge); those reads fall back to serial
silently. Check whether your extra-large file is compressed before interpreting
timings (see "Diagnosing the file" below).

## Building and testing

- Rust: `cargo build --release -p gdx-sys -p gdx -p polars-gdx`
- Rust tests: `cargo test --release -p gdx` (includes `tests/pos_parallel.rs`,
  which needs `/tmp/big.gdx`; it skips when absent — generate with
  `cargo run --release -p gdx --example gen_fixture`)
- Python extension: `maturin develop --release` (venv with maturin/uv), then
  `python -m pytest tests -q`
- Lint gates (CI runs these): `cargo fmt --check`, `cargo clippy --all-targets`
  (zero warnings expected), pytest.
- Wheel/CI matrix: 5 platforms (linux x86_64/aarch64, windows-msvc, macos
  x86_64/aarch64). **Anything touching C++ must stay portable**: no
  `std::filesystem` (unavailable on macOS targets < 10.15; MSVC narrows), careful
  with `long` vs `long long` (macOS `INT64` is `long`, `int64_t` is `long long` —
  marshal through locals), no POSIX-only APIs in code paths built on Windows.
  These exact bugs shipped before; don't re-ship them.

## Working with the extra-large file

The user's real file has a parameter with 5 index dimensions, ~300M records, one
dimension with ~8000 elements, others ~20s. Expect it on local disk; ask the user
for the path if not found. Workable proxies you can generate yourself:

- `cargo run --release -p gdx --example gen_fixture` → `/tmp/big.gdx`
  (6M records, 5 dims; env `OUT=` for another path, `COMPRESS=1` for a
  block-compressed copy via the `GDXCOMPRESS` env var).
- `cargo run --release -p gdx --example make_bench -- /tmp/bench_big.gdx 20000000`
  (2-dim, N records — the CI benchmark fixture generator).

Always benchmark against a file at least 10M records; smaller files hide constant
costs and measure noise.

### Diagnosing the file

- `list_symbols(path)` — record counts, dims, domains per symbol.
- Compression: if the file is far smaller than `records * (dim + 8)` bytes, it is
  block-compressed; confirm by timing `threads=8` vs `threads=1` on a
  non-leading-dim filter (no difference → compressed → serial fallback).
- Sizing the split cache: `read_symbol_raw_parallel_pos` runs one sequential
  planning pass per (file, symbol), cached process-wide. On a 300M-record symbol
  that pass is measurable the first time; consider whether the restart index
  could be derived more cheaply (see candidates).

### Measurement discipline

- Warm up once (page cache + restart-index cache), then take min of 3+ runs.
- Compare like with like: same filter selectivity, same row count returned.
- Validate correctness with parity asserts: serial vs threads=2..8 must produce
  identical frames (`assert a.equals(b)`), and a filtered read must equal the
  serial read filtered post-hoc.
- Watch out: `threads="auto"` resolves differently based on machine core count;
  pin explicit thread counts in benchmarks.

## Candidate small wins (leads, not conclusions)

Ordered roughly by expected effort-to-payoff. Verify each with data before
committing to it; discard anything that doesn't reproduce.

1. **Reduce restart-index cost.** The planning pass decodes every record just to
   note restart positions. The decode is mostly the C++ `DoRead` per record
   (~40M records/s/core). A "positions-only" mode that skips the value bytes
   (they can be skipped arithmetically from the value-type tag without full
   decode) could cut planning time meaningfully on 300M-record symbols. The skip
   must handle special-value bytes exactly as `DoRead` does.
2. **Bit-slice filter pushdown into C.** The Rust bitmap predicate runs per
   record via FFI callback. The `gdxDataReadRawFastFilt` C entry point already
   exists (filtered read with a UEL filter string) but is unused. Moving the
   membership test into C (passing the bitmap as a UEL filter or a raw bitmask
   pointer) saves one indirect call per record on filtered reads.
3. **Per-worker preallocation.** `RawSymbolData` buffers grow by doubling from
   empty. For full (unfiltered) reads, the symbol's record count is known up
   front — `Vec::with_capacity(records)` per worker range avoids reallocation
   copies. Small, contained, measurable on 100M+ record reads.
4. **Arrow construction cost.** Keys are copied into per-column buffers in
   `store_record_ex`, then wrapped. Check if the dictionary key arrays can be
   built once over concatenated worker buffers without per-record branching in
   the hot loop (e.g., unfiltered reads could bypass the predicate entirely).
5. **`read_domains` on huge symbols** — the user originally complained this was
   slow. It uses `gdxGetDomainElements` in one bulk scan; profile whether the
   per-element callback or the UEL mapping dominates. There may be a cheap
   win in the callback (marking seen flags) or in how the UEL strings are
   materialized afterwards.
6. **Decode-loop micro-optimizations in `DoRead`** (upstream code): e.g., the
   special-value branch reads a tag byte per value field per record; a common
   case check (all-normal records) could branch-predict better. Mark any change
   with `(polars-gdx extension)` and keep it optional/local — this is shared
   upstream code.
7. **Python-side overhead.** `lazy.py` resolves UEL strings per call for filter
   labels; the UEL table is cached per Reader, but each `collect()` re-opens and
   re-parses the symbol table. For repeated collects over one file, a Reader
   reuse path might be a visible constant win for small-result filtered reads.

## What NOT to do

- No new dependencies (repo rule).
- No API breaks; `scan_gdx`/`read_gdx` signatures are public.
- No changes to the vendored upstream code beyond marked, surgical additions —
  upstream sync gets painful otherwise.
- No `unsafe` added to the Python or gdx crates without an existing pattern to
  follow.
- Don't chase wins that only show on the 6M fixture — the user's file is 300M
  records; constant factors and per-record costs dominate there.

## Definition of done for a win

1. A repeatable before/after number (command + timings) on a 10M+ record file.
2. Parity asserts green: serial vs parallel, filtered vs post-filtered.
3. `cargo fmt --check` + clippy clean + pytest 45 passing.
4. A commit per win with the measurement in the message, on a branch, PR to main.
