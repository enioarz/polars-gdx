# Improvement plans — polars-gdx

Self-contained action plans from the code review (2026-10). Each file is
independent enough to hand to a separate agent. Read `AGENTS.md` first for
repo facts and the build/test commands; every plan assumes that context.

| Plan | Title | Area | Type |
| --- | --- | --- | --- |
| [01](01-uel-caching.md) | Cache UEL table/index on the Reader | crates/gdx, crates/polars-gdx, lazy.py | perf |
| [02](02-drop-probe-read.md) | Drop the redundant probe read | crates/gdx | perf |
| [03](03-n-rows-early-exit.md) | True `n_rows` pushdown | all layers | perf / feature |
| [04](04-filter-hot-path.md) | Filter hot path (single-label fast case, TLS sink, map_special) | crates/gdx, crates/polars-gdx | perf |
| [05](05-label-case-normalization.md) | Case-insensitive label matching | lazy.py, crates/polars-gdx | correctness / GAMS parity |
| [06](06-reader-lifecycle-and-names.md) | Reader lifecycle + key/value name collision | crates/polars-gdx, lazy.py | robustness |
| [07](07-narrow-dictionary-indices.md) | Narrow dictionary indices / Categorical keys | crates/gdx, crates/polars-gdx, lazy.py | perf / dtype (user-visible) |

## Suggested order & merge conflicts

- **No dependencies between plans**; they can run in parallel.
- Expected overlap (rebase order suggestion, smallest-conflict last):
  - 01 and 07 both touch `to_record_batch` in crates/polars-gdx/src/lib.rs
    (07 builds on 01's cached UEL table — if serialized, do 01 → 07).
  - 03 and 04 both touch `read_symbol_raw`/`read_symbol_raw_locked` in
    crates/gdx/src/reader.rs (04 also reworks the `SINK` thread-local).
  - 05 and 06 both touch `scan_gdx`/`_key_names` in lazy.py.
- Each plan's changes are small enough that rebasing over the others should
  stay mechanical.

## Definition of done (applies to every plan)

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace`
- `maturin develop --release` (with VIRTUAL_ENV set, per AGENTS.md) and
  `pytest tests/`
- No regression in the README benchmark numbers for the affected scenario;
  improvements recorded in the plan's PR description.
