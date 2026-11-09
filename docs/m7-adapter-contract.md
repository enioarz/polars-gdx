# M7 adapter contract — gdxrs backend for polars-gdx

Status: implementation contract (task 041); implementation and acceptance remain pending. This document defines the contract that tasks
042–047 implement against. It authorizes no implementation by itself.

Scope boundary: everything in this contract lives in `enioarz/polars-gdx`. No
Polars, Arrow or PyO3 dependency may enter `enioarz/gdxrs`; any missing core
primitive requires its own scoped proposal in gdxrs rather than adapter-side
workarounds.

## 1. Pinned revisions and supported versions

| Component | Pinned / supported |
| --- | --- |
| gdxrs core revision | `9a7411764f176adb613185fa1e45d64b06123683` (gdxrs `main`, post-M6) — adapter depends on it as a git dependency with this rev pinned; bump only via a reviewed change that reruns the affected gates |
| polars-gdx adapter revision | `23beb2429d4a362c3c078c239d9136ae2f463e27` (`main`, legacy backend only) |
| Rust | gdxrs backend requires 1.99.1 (the pinned core's `rust-version`); use 1.99.1 for M7 CI/builds. The legacy workspace declares 1.80, which does not apply to gdxrs-enabled builds and was not requalified here |
| Python | >= 3.10 (abi3-py310 wheel); qualified on 3.12.15; CI classifiers 3.10–3.14 |
| PyO3 | 0.25 (`extension-module`, `abi3-py310`), matching arrow-pyarrow 56 |
| Arrow (Rust) | 56 (`arrow`, `arrow-pyarrow`, `default-features = false`, features `pyarrow`, `ipc`) |
| polars (Python) | >= 2.0; qualified on 2.1.0 |
| pyarrow (Python) | >= 11; qualified on 26.0.1 |
| Platforms | Linux x86_64, macOS ARM64, macOS Intel, Windows x64 |

## 2. Behavior inventory (current legacy adapter, to be preserved or explicitly changed)

Public Python surface of the legacy backend (all must keep working under either
backend unless an exclusion below says otherwise):

- `list_symbols(path)` → DataFrame `name, type, dim, records, domains, text`.
- `read_domains(path, symbol=...)` → per-dimension unique labels actually used,
  in file order, padded with nulls.
- `scan_gdx(path, symbol=, value_field=, key_filter=, threads=)` → `pl.LazyFrame`;
  `read_gdx(...)` is the eager wrapper (`.collect()`).
- `Reader` context manager: `symbols()`, `uel_table()`, `domain_elements(symbol, dim)`,
  `resolve_uel_indices_used(...)`, `read_arrow(...)`, `close()`.
- Scan parameters honored today: `value_field` ∈ {level, marginal, lower, upper,
  scale} (default level; Sets/Parameters/Aliases expose a single `value` column),
  `key_filter={dim_index: [labels]}` exact case-sensitive native prefilter,
  `threads` (None/0/1 serial; explicit N workers, admitted 2..=4 on gdxrs;
  `"auto"` = all cores for symbols ≥ 5M records on legacy — on gdxrs it is a
  per-execution policy: a finite accepted-row limit pushed to the source and
  an unprepared source both resolve to the serial early-stop cursor, so a
  bounded prefix query never pays the whole-file checkpoint-index build plus
  strict reopen; see the head-latency evidence), Polars IO-source callback
  parameters `with_columns` (projection),
  `predicate`, `n_rows` (global head), `batch_size` (currently ignored, `del batch_size`).
- Schema: key columns are `pl.Enum(uel_table)` over the full file UEL table in
  file order (Arrow dictionary arrays, zero-copy reinterpretation); value column
  is `pl.Float64`; key column names come from domain names with positional
  de-duplication and never collide with the value column name.
- Predicate pushdown: conjunctions of `col(key) == "label"` and
  `col(key).is_in([...])` fold into the native prefilter and are re-applied
  exactly on the produced frame; all other predicates are residual Polars-side
  filters. A native `n_rows` limit is only pushed when the predicate is fully
  translatable; otherwise rows are read unlimited and `head(n)` applies after.
- Parallel reads produce byte-identical ordered results to the serial read, with
  verified fallback to serial on any boundary/count mismatch.
- Repeated scans of the same LazyFrame are deterministic (`is_pure=True`).

Existing fixtures: `tests/data/trnsport.gdx`, `tests/data/compressed.gdx`,
`tests/data/edgecase.gdx`; independent native expectations are authored in
`tests/` (see the evidence record), not derived from either backend.

## 3. Compatibility requirements, exclusions, special-value/acronym policy

Compatibility requirements:

- Case-sensitive, byte-for-byte label matching for `key_filter` and predicates;
  unknown labels match nothing; an empty resolved filter means zero rows and
  must not be dropped.
- Record order of scans must equal the serial ordered core scan.
- Empty symbols, scalar (0-dimensional) symbols, alias/universe domains and all
  value-field semantics must be preserved.

Intentional exclusions (adapter does not provide):

- Writing/appending GDX files (reader-only adapter).
- GAMS-dependent functionality: no GAMS system directory, no `gamsapi`
  (`gams.transfer`) path, no C++ or native GDX library in the gdxrs build/runtime
  path. Legacy backend and native-oracle comparisons remain explicit development
  or optional paths.
- Universal zero-copy: ownership-permitting reuse only.

Special-value / acronym conversion policy (value column is `pl.Float64`):

- Core `Value::NumberBits` converts with `FloatConversion::Exact`; lossy bits are
  rejected, never silently coerced.
- `SpecialValue::PositiveInfinity`/`NegativeInfinity` map to `±inf`.
- `SpecialValue::Undefined`, `NotAvailable` map to `null`.
- `SpecialValue::Epsilon` explicitly maps to positive zero, bits
  `0x0000000000000000`, preserving the legacy Float64 projection. This loses
  EPS identity by design; count EPS projections in scan statistics. Match on
  the typed core value and construct the adapter result directly. Do not call
  `FloatConversion::Exact` for EPS: it correctly rejects loss of typed identity.
  Core values and core conversion behavior remain unchanged.
- `Value::Acronym` has no Float64 representation: with
  `AcronymReadPolicy::Preserve` the adapter surfaces the value as `null` and
  counts occurrences in a scan statistic; with `ToNotAvailable` it maps to
  `null` as NotAvailable. Any different mapping is a contract change requiring
  independent expected results. The authored expectations are fixed now in
  [m7-fixture-expectations.md](m7-fixture-expectations.md); tasks 042/043 add the
  generators and executable assertions. Undefined/NA-to-null intentionally differs
  from the legacy backend's NaN projection; test that difference explicitly.

## 4. Backend selection

- `scan_gdx`/`read_gdx` gain an explicit `backend=` parameter: `"legacy"`
  (vendored C library; default in 0.6.x) or `"gdxrs"` (pinned core revision).
- No silent fallback: a gdxrs source/decode/resource failure is an error, never
  a retry through the legacy backend. The legacy backend stays available for
  differential tests and migration.

## 5. Schema and dictionary modes

- Mode A (default, budget-admitted): `pl.Enum` over the full UEL table, as today.
- Mode B: `key_mode="uel_id"` uses original one-based UEL IDs as UInt32 key
  columns. `key_mode="enum"` selects mode A and remains the default. No automatic
  mode switching; bounded string output is deferred beyond M7. Batch-local
  category codes must never be reinterpreted as global IDs.
- Mode A strictly decodes UEL labels as UTF-8 and rejects invalid bytes before
  output; no replacement characters. Mode B does not decode labels. Its explicit
  bounded resolver, `Reader.resolve_labels(ids)`, returns columns `uel_id: UInt32`
  and `label: Binary`, preserving request order, duplicate requests and exact
  original bytes. Zero/out-of-range IDs raise ValueError before any result;
  requests exceeding the configured conversion budget raise a resource error.
  String filters encode as UTF-8 and retain exact byte matching in either mode.
- Full-dictionary Enum is admitted only under the budget in §6; insufficient
  admission is rejected clearly, never by silently changing the planned schema.
- Schema is chosen before execution and preserved across empty batches,
  partitions and repeated scans.

## 6. End-to-end resource envelope and concrete caps

One envelope, accounted separately per layer:

| Layer | Cap / accounting |
| --- | --- |
| gdxrs core (`OpenOptions`) | `memory_budget_bytes` library root cap (default 64 MiB, min 1 MiB) covering metadata cache (`metadata_cache_bytes`, default 128 KiB), per-batch output (`maximum_batch_bytes` / `ScanOptions.batch_bytes`, default 64 KiB), decoder scratch and cursor state; M6-qualified containment: root ≤ ~29 MiB observed at 512 MiB process cap, ≤ 4 descriptors incl. stdio |
| Adapter conversions | Arrow/Polars conversion buffers, dictionary unification and temporary copies bounded by the same envelope; ownership documented; no eager full-symbol or mandatory full-dictionary collection |
| Queues / workers | bounded producer queue; admitted workers × (stack + guard) reserved; Polars and core worker pools never nest unbudgeted |
| Streaming sink | noncollecting sink with bounded producer buffering and backpressure; caller `.collect()` is materializing, not streaming evidence |
| Python/Polars runtime overhead | accounted separately from the envelope; reported as whole-process RSS/VM, not charged to the core budget |
| Caller-retained output | outside the envelope; reported separately |

Concrete task-047 caps (targets to qualify, not existing adapter measurements):

- Core plus adapter-owned buffers: **256 MiB combined**, with shares reserved
  before allocation. This includes live core state, dictionaries, conversion
  copies, worker stacks/guards, queued output and sink staging. No independent
  256 MiB allowance per layer or worker.
- Linux process: enforced self-checked soft/hard **1 GiB RLIMIT_AS**, plus a
  **512 MiB measured peak-RSS ceiling**. RSS is measured, not claimed as an
  enforced RLIMIT_RSS. Python/Polars overhead and caller-retained output remain
  outside the library accounting but inside the process measurements.
- Scratch: **8 GiB aggregate per active scan**, including temporary indexes,
  label-resolution spill, merge runs and unpublished sink output. Reserve before
  writes; exhaustion fails explicitly. Input files and already published caller
  outputs are excluded. Verify cleanup on success, failure and cancellation,
  especially on Windows; do not remove caller-owned sources or sidecars.
- Handles: **16 process descriptors including stdio**, of which at most **12**
  are adapter-owned source/index/spill/sink handles. Admission is aggregate,
  never per worker. M6's observed four descriptors are historical native-worker
  evidence, not a viable adapter/sink cap.
- At most **4 core workers**, with at most **4 queued batches** and at most
  **1 MiB per batch**. Conversion/sink staging is charged simultaneously with
  queued output; fixed slots and explicit backpressure prevent extra hidden
  batches. Polars execution is limited to two threads for capped qualification;
  configure other runtime pools explicitly and record their settings. Reject
  explicit requests above configured worker admission; auto never exceeds it.

The process-cap change is deliberate: in the setup environment, a fresh Python
3.12 process importing Polars 2.1.0 and PyArrow 26.0.1 with
`POLARS_MAX_THREADS=2 OPENBLAS_NUM_THREADS=1` reported VmPeak 645,908 KiB and
VmSize 584,196 KiB before any GDX scan. The proposed 512 MiB address-space cap
would reject that baseline. The new 1 GiB cap still requires task-047 validation;
it is not a measured streaming pass. Functionality targets all four native
platforms; initial enforced OS-cap measurements target Linux only. macOS/Windows
resource measurements remain unqualified until separately executed and reported.

Acceptance recipes: raw and zlib numeric symbols with **40M/80M records** at
120 logical bytes per row; metadata-heavy universes with **72M/144M UELs**,
63-byte labels plus terminator (64 logical bytes each). Both sizes exceed four
times the 1 GiB process cap logically. Require physical raw/metadata sources
also to exceed four times the cap; report actual bytes/hashes and fail the size
gate if they do not. Zlib qualification is logical; compressed physical bytes
may fit RAM. Required generator count limits must be extended/checked in 047.
Use noncollecting sinks, at least three trials per case, controlled thread pools,
and published hardware/storage/cache settings. Report RSS/VM, simultaneous
per-layer buffers, scratch/handles, first batch, full output and cancellation.
Compare backends using equivalent decoded work and the declared value projection;
report losses/regressions without universal speedup claims.

## 7. Lifecycle: ownership, cancellation, errors, repeated scans

Current consumer API: gdxrs scan frames and their derived frames provide an owned
`collect_batches()` iterator with explicit `close()` and context-manager support.
For operations that return a plain LazyFrame (for example, `pl.concat` or group-by
aggregation), call `polars_gdx.collect_batches(plan)` to retain this ownership.
The adapter does not patch Polars' global methods. Plain Polars batch consumers
keep the installed Polars version's own disposal behavior.

Owned batch iterators implement `__arrow_c_stream__(requested_schema=None)`,
delegating schema negotiation to Polars and forwarding Arrow schemas, metadata
and arrays unchanged. Export transfers execution ownership: closing or dropping
the original iterator cannot cancel the exported stream. Stream release closes
the execution and joins its workers, even if the original iterator remains alive.
Unreleased streams are closed before interpreter teardown. Arrow release has no
error return; cleanup failures there use Python's unraisable-error hook.

Owned execution binds the existing live sources through an in-process Polars
plan clone, preserving the original reader/source identity. Custom Python UDF
captures must be serializable. Cloning copies plan metadata and any in-memory
inputs in the plan; it does not decode or accumulate GDX records. Task 047 must
include this metadata staging in resource qualification. Portable serialization
of live GDX readers is not provided.

- Each yielded Arrow/Polars batch owns valid data after the lending core cursor
  advances or closes; no borrowing across batches.
- Fallible close/cancellation: dropping the consumer, sink exceptions and Python
  cancellation wake blocked producers/consumers and join all owned workers; core
  errors propagate without replaying acknowledged rows; blocked synchronous OS
  reads cannot be forcibly interrupted (documented, not hidden).
- GIL released only under valid ownership rules; no legacy global-lock or unsafe
  Send assumptions are imported.
- Repeated scans are independent and deterministic; source identity is preserved.

## 8. Acceptance (task 041 deliverable)

- This contract (reviewed behavior/resource matrix above).
- Evidence record: [evidence/041.md](evidence/041.md) with named existing
  regression tests, reproducible baseline commands and results, independently
  authored expected results for gaps, the supported-version/platform matrix, and
  an explicit list of commands/fixtures that remain to be implemented.

## 9. Development branch and merge gate

`develop/m7` is the integration target for tasks 041–047. All M7 task PRs target
that branch, including PR #39; tasks may span multiple reviewed PRs. Branch/task
merges mean progress only. They do not authorize merging `develop/m7` into `main`.

The final `develop/m7` → `main` PR stays draft until every task 041–047 is complete,
review findings are resolved, and evidence links identify the exact adapter/core
revisions qualified. Require the full legacy and gdxrs backend suites, all four
native functionality/packaging platforms, the bounded streaming/schema/predicate/
lifecycle gates, gdxrs-only installation without GAMS, and task-047 capped growing
workloads with three trials. Record failures, skips and unrun cases explicitly;
unresolved required checks keep the main merge blocked. Keep legacy as default
and available until a separately reviewed migration change passes its gates.
No task PR or agent may bypass this gate or merge the integration branch to main
merely because one PR passed. Final merge requires explicit maintainer approval.

Track each task and its evidence in [m7-development.md](m7-development.md).
