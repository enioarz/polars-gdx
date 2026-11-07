# M7 adapter contract — gdxrs backend for polars-gdx

Status: draft for review (task 041). This document defines the contract that tasks
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
| Rust | gdxrs toolchain 1.99.1; adapter MSRV 1.80 (workspace `rust-version`), built with 1.99.1 in this qualification |
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
  `threads` (None/0/1 serial; N workers; `"auto"` = all cores for symbols ≥ 5M
  records), Polars IO-source callback parameters `with_columns` (projection),
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

Independent fixtures: `tests/data/trnsport.gdx`, `tests/data/compressed.gdx`,
`tests/data/edgecase.gdx`; independent native expectations are authored in
`tests/` (see §7), not derived from either backend.

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
- `SpecialValue::Epsilon` maps to its exact stored double bits via the exact
  conversion (no rounding to 0).
- `Value::Acronym` has no Float64 representation: with
  `AcronymReadPolicy::Preserve` the adapter surfaces the value as `null` and
  counts occurrences in a scan statistic; with `ToNotAvailable` it maps to
  `null` as NotAvailable. Any different mapping is a contract change requiring
  independent expected results (task 043 fixtures).

## 4. Backend selection

- `scan_gdx`/`read_gdx` gain an explicit `backend=` parameter: `"legacy"`
  (vendored C library; default in 0.6.x) or `"gdxrs"` (pinned core revision).
- No silent fallback: a gdxrs source/decode/resource failure is an error, never
  a retry through the legacy backend. The legacy backend stays available for
  differential tests and migration.

## 5. Schema and dictionary modes

- Mode A (default, budget-admitted): `pl.Enum` over the full UEL table, as today.
- Mode B (bounded large dictionaries): key columns carry original UEL IDs
  (UInt32 physical) with explicit label resolution, and/or bounded string
  batches; batch-local category codes are never reinterpreted as global UEL IDs.
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

Concrete caps for acceptance (task 047 executes them): 256 MiB library root cap,
enforced self-checked soft/hard 512 MiB OS process cap (RLIMIT_AS on Linux; the
OS-cap measurement platform set is Linux, macOS, Windows x64 — functionality is
qualified on all four platforms), ≤ 4 open descriptors incl. stdio, per-worker
stack reservations admitted explicitly, scratch files under a bounded scratch cap
with release verified especially on Windows.

Acceptance recipes (defined here, executed in 047): growing, independently
oversized record-heavy (raw and zlib-compressed) and metadata-heavy inputs at ≥ 4×
the OS cap, streamed through noncollecting sinks, ≥ 3 trials with documented
hardware/storage/cache state, measuring whole-process RSS/VM plus per-layer
buffers; compare legacy vs gdxrs backends on equivalent decoded work; no
universal speedup claims.

## 7. Lifecycle: ownership, cancellation, errors, repeated scans

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
