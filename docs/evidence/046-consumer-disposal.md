# Task 046 consumer-disposal fix

Review base: PR #46 at `207d56fad19fb1ffec8915b00e6b0bcc1f05ff0a`.
Core remains pinned at `9a7411764f176adb613185fa1e45d64b06123683`.

Polars 2.1 can retain a Python source generator after its batch iterator and
LazyFrame are deleted. The generator's `finally` and the process-exit registry
therefore do not provide prompt consumer cancellation. Reproduction before the
fix left one registered cursor with four workers admitted and none joined.

The adapter now returns an owned batch iterator for gdxrs scan frames. It binds
each execution to its own source callbacks through a local Polars plan clone;
the binding retains the existing reader rather than reopening the file. Only
live source references are serialized, using the standard library. No dependency
was added. Normal `collect()` retains its existing path.

The consumer's finalizer closes its execution's cursors directly, independently
of source-generator finalization. Explicit `close()` and context exit report
cleanup failures. Setup stays registered through cursor publication, so a close
that races with setup waits and closes late-published cursors. Concurrent consumers
of the same LazyFrame receive separate bindings and cannot cancel each other.
Already yielded frames retain their buffers.

Use `polars_gdx.collect_batches(plan)` for composed plans such as `pl.concat`
that return a plain Polars LazyFrame. This is an explicit adapter ownership API;
Polars' global methods are unchanged. Plan metadata and in-memory inputs are
copied during binding, and nonserializable Python UDF captures are rejected before
execution. GDX records remain incremental. This metadata staging remains part of
task-047 qualification.

Focused Linux reproduction after the fix: consuming one batch, deleting the
iterator and frame, and running GC left zero registered cursors, joined all four
workers, and removed four OS threads. The retained output frame remained valid.

Regression coverage includes disposal with and without retaining the LazyFrame,
independent concurrent consumers, sink exceptions, lazy close before startup,
composed plans, close during cursor publication, source changes before execution,
and execution with `cloudpickle` deliberately unavailable. Scratch assertions now
track only directories created by the tested reader, allowing unrelated readers
to be reclaimed by GC without creating false cleanup failures.

Linux validation (Python 3.12.14, Polars 2.1.0, PyArrow 26.0.1):

| Check | Result |
| --- | --- |
| Default-backend Python suite | 226 passed |
| Gdxrs-default Python suite | 192 passed, 34 skipped |
| Installed gdxrs-only wheel in an isolated environment | 174 passed, 40 skipped |
| `cargo fmt --all --check` | pass |
| `cargo clippy --workspace --all-targets -- -D warnings` | pass |
| `cargo test --workspace` | 16 passed, 4 ignored |

Manual checks also verified mixed legacy/gdxrs concatenation, group-by output,
and projected label predicates in both key modes against ordinary collection.
Four-platform CI and task-047 OS-cap qualification are not claimed by these
Linux results.
