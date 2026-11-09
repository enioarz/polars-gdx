# Task 046 code hygiene follow-up

Base: PR #46 at `0e02a6b5211d121d89efc882e7349c6c1d62412e`.
Core dependency remains pinned at `9a7411764f176adb613185fa1e45d64b06123683`.

Changes:

- Share the serial/parallel batch dispatch and cursor close/cancel paths.
- Reserve conversion vectors fallibly and allocate value storage only when
  the projection includes values.
- Saturate the row-to-byte multiplication before applying the batch byte cap.
  A regression test requests a representable row count whose byte size would
  otherwise wrap to zero, and verifies that the first batch uses the byte cap.
- Restrict scratch-directory assertions to the current process, so unrelated
  scans cannot cause false cleanup failures. Read fixture bytes with an owned
  file lifetime and bound shutdown-test subprocesses with a 60-second timeout.
- Verify background Python progress during core production, before any thread
  join can release the GIL and produce a false positive.
- Remove unused imports and correct stale streaming, label-residual and worker
  lifetime comments. Update the integration tracker with confirmed merge commits.

The Python shutdown gate, callback accounting, cursor publication and
close-before-unregister ordering are unchanged.

Validation on Linux x86_64, Rust 1.99.1, Python 3.12.14, Polars 2.1.0,
PyArrow 26.0.1:

| Check | Result |
| --- | --- |
| `cargo fmt --all --check` | pass |
| `cargo clippy --workspace --all-targets -- -D warnings` | pass |
| gdxrs-only clippy, all adapter targets | pass |
| `cargo test --workspace` | 16 passed, 4 ignored |
| `maturin develop --release --locked` | pass |
| `python -m pytest tests/ -q` | 218 passed |
| `POLARS_GDX_BACKEND=gdxrs python -m pytest tests/ -q` | 184 passed, 34 skipped |

The default suite ran while a separate Python process repeatedly created,
scanned and closed parallel readers. The scratch checks passed and the external
producer exited successfully. No dependencies were added. Native platform and
OS-cap qualification remain the task-047 gates.
