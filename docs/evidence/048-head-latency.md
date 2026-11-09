# Task 048 — Head latency: finite-limit early-stop mode resolution (evidence)

Task: head-latency investigation follow-up on `develop/m7` (post-#46 merge
`eec3513`). This is the focused adapter fix from the investigation's
implementation sequence, step 1 only. Validated-session reuse (step 2),
remaining metadata/conversion profiling (step 3) and the acceptance-gate
extension (step 4, coordinated with task 047) are separately scoped
follow-ups; none of them is implemented or claimed here.

## Problem

A `head()` query on a large compressed symbol with `threads="auto"` paid
whole-input parallel setup before producing a single row:

- `scan_gdx` resolved `threads="auto"` to an integer worker count from the
  symbol's **total record count**, before execution provided the
  accepted-row limit.
- `GdxrsReader::scan_batches` forwarded the limit to `ScanOptions` but still
  called `parallel_index` whenever workers were requested: one checkpoint
  sidecar build (full-file hash plus symbol decode replay) and one strict
  reopen (another full-file hash plus replay) on the first query, and
  another strict reopen on every repeated query.
- The legacy backend switches to its serial early-stop path when a row
  limit exists; the gdxrs backend did not.

Diagnostic measurements (local Linux x86_64, Python 3.12.14, polars 2.1.0,
pyarrow 26.0.1, release extension, `POLARS_MAX_THREADS=2`,
`MALLOC_ARENA_MAX=2`, page-cache-warm fixtures, diagnostic only — not task
047 qualification):

- 400M-record 18-dimensional zlib fixture (raw 4,710,404,715 bytes, zlib
  564,838,937 bytes), Enum keys, reused LazyFrame, `head(5)`:
  legacy auto 0.027 s (first) / 0.0013 s (repeated); gdxrs serial
  0.0015 s / 0.0010 s; gdxrs auto **78.4 s / 39.3 s** (build + strict
  open, then strict reopen). An isolated head-aware facade that routed
  the finite pushed limit to the serial cursor measured 0.0038 s /
  0.0011 s, proving the routing fix.
- 1M-record raw fixture, Enum `head(5)`: gdxrs serial 0.0035 s, gdxrs
  four workers 2.13 s (first) / 0.98 s (repeated).

## Fix (adapter only; no core change, no dependency bump)

- **Rust** (`crates/polars-gdx/src/gdxrs_backend.rs`, `scan_batches`): after
  the unchanged explicit worker-admission validation, a finite `n_rows`
  resolves the execution mode to the serial early-stop cursor before any
  sidecar lookup, creation, strict reopen or worker admission — for auto
  and explicit worker requests alike. Unlimited scans keep explicit
  parallel execution over the checkpoint index.
- **Python** (`python/polars_gdx/lazy.py`): `threads="auto"` stays a
  requested policy instead of being resolved to a worker count from the
  symbol's total records at `scan_gdx` time. The IO-source callback
  resolves the effective mode per execution: a finite accepted-row limit
  pushed to the source resolves to serial, and an unprepared source under
  `auto` resolves to serial without constructing or strictly reopening an
  index. Explicit requests above the 2..=4 admission still raise
  `ValueError` at `scan_gdx` time. A residual predicate keeps its limit
  out of the source (`native_limit_safe` is False), so its output limit is
  never pushed ahead of filtering. The legacy backend's `auto` behavior
  (all cores for symbols ≥ 5M records) is unchanged.
- Documented workaround for the reported workload: `threads=None` or
  `threads=1`. Mandatory full-dictionary Enum admission is unchanged.

## Deterministic regressions

`tests/test_m7_head_latency.py` asserts, on raw and zlib fixtures, Enum and
UEL-ID key modes, direct-reader and LazyFrame execution:

- `head` 0/1/5/1024 produce the independently expected prefix and match the
  serial read, for `threads=None`, `threads="auto"` and explicit
  `threads=2..=4`.
- Finite early-stop setup admits zero workers and neither builds nor
  strictly reopens a checkpoint index: the direct-reader cursor's
  `scan_statistics` carries no parallel lifecycle counters, and no index
  sidecar directory appears for the process.
- Projections, translatable key predicates and residual value predicates
  compose with the early-stop path; residual `filter(...).head(n)` keeps
  exact semantics.
- Disposed batch consumers (`collect_batches` + `close()`) join cleanly on
  the early-stop path.
- `auto` resolves to serial even for a symbol above the legacy 5M-record
  threshold (threshold monkeypatched below the fixture's record count),
  and an env-gated probe covers a genuinely ≥5M-record fixture when
  `POLARS_GDX_HEAD_LATENCY_BIG` names one.
- Explicit unlimited parallel scans still run the parallel path (parallel
  lifecycle counters present) and stay byte-identical to serial.

## Not done here (separately scoped)

- Validated-session reuse via `ParallelScanCursor::into_index()` with
  leased, bounded, integrity-checked session caching (investigation step 2).
- Metadata/Enum-conversion profiling and a selected-symbol metadata API
  (step 3).
- Timing/work-count acceptance gates and the ≥300M-record compressed
  before/after comparison, coordinated with task 047 (step 4). The
  measurements above are diagnostic; full timing qualification remains
  implementation work.
