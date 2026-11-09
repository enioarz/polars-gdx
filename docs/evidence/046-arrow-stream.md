# Task 046 Arrow stream ownership

Review base: PR #46 at `b88164ae9fa2394fddf85d3db589a763bd50860d`.
Core remains pinned at `9a7411764f176adb613185fa1e45d64b06123683`.

The owned batch wrapper now implements `__arrow_c_stream__`, restoring
`pyarrow.RecordBatchReader.from_stream(lf.collect_batches())`. Polars handles
requested-schema negotiation before ownership moves. Export exhausts the original
iterator and transfers execution ownership to a lease held by the C stream;
closing or disposing of that iterator cannot cancel the reader. Stream release
closes the execution even while the original iterator is still alive. Retained
output buffers remain valid after cancellation.

A small Rust C ABI wrapper forwards schema, next-array and error callbacks to
the original producer without importing or reconstructing its output. Importing
and re-exporting through Arrow 56 would lose the dictionary-ordered flag;
opaque forwarding preserves that flag, extension metadata and producer errors.
The unsafe boundary consists of the documented Arrow capsule ownership move,
boxed stream-state pointer and C callback delegation/release. There are no new
unsafe Send/Sync implementations or dependencies, and no changes to the core.

The upstream stream is released before the execution lease. Release may run on
a foreign thread, so the lease acquires the GIL for Python cleanup. Execution
close disarms its callback guard; live leases also have exit finalizers. A final
shutdown gate stops admission and drains already admitted release callbacks with
the GIL released before interpreter teardown. Release has no error return;
cleanup failures use Python's unraisable-error hook. Ordinary explicit batch
close retains its existing fallible behavior.

`tests/test_m7_arrow_stream.py` adds 23 cases covering:

- Exact table/schema metadata parity with Polars in enum/UEL-ID, serial/parallel
  and eager/lazy modes, including ordered dictionary flags.
- Stream survival after the original consumer closes or is dropped; cancellation
  on reader close, GC or another thread, including when the original remains alive.
- Joining all four workers, removing owned cursor registrations, and retaining
  already yielded output after release.
- Unconsumed capsule release, requested-schema negotiation and source-change errors.
- Unreleased-stream process exit with both import orders, and shutdown waiting for
  an admitted release callback to complete.

Linux validation (Python 3.12.14, Polars 2.1.0, PyArrow 26.0.1):

| Check | Result |
| --- | --- |
| Default-backend Python suite | 249 passed |
| Gdxrs-default Python suite | 215 passed, 34 skipped |
| Installed gdxrs-only wheel in an isolated environment | 197 passed, 40 skipped |
| `cargo fmt --all --check` | pass |
| `cargo clippy --workspace --all-targets -- -D warnings` | pass |
| Legacy-only clippy with all targets and warnings denied | pass |
| `cargo test --workspace` | 16 passed, 4 ignored |

Four-platform CI and task-047 OS-cap qualification remain separate gates;
these are local Linux results, with no new resource/performance qualification claim.
