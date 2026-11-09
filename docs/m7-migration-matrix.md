# M7 compatibility / migration matrix

Status: task 047 (in progress). This matrix records the supported
combinations, the default-backend migration decision and the gates that
must pass before any default change. It does not itself authorize a
release or a default switch.

## Supported version matrix (contract §1)

| Component | Pinned / supported |
| --- | --- |
| gdxrs core | `9a7411764f176adb613185fa1e45d64b06123683` (git dependency, `index` feature enabled; bump only via reviewed change rerunning affected gates) |
| Rust | 1.99.1 for gdxrs-enabled builds (legacy-only builds keep the 1.80 declaration) |
| Python | >= 3.10 (abi3-py310 wheel); qualified on 3.12.15 |
| PyO3 | 0.25 (`extension-module`, `abi3-py310`) |
| Arrow (Rust) | 56 (`arrow`, `arrow-pyarrow`, `ipc`) |
| polars (Python) | >= 2.0; qualified on 2.1.0 |
| pyarrow (Python) | >= 11; qualified on 26.0.1 |
| Platforms | Linux x86_64, macOS ARM64, macOS Intel, Windows x64 |

## Backend matrix

| Capability | legacy | gdxrs |
| --- | --- | --- |
| `scan_gdx`/`read_gdx`/`list_symbols`/`read_domains` | yes (default in 0.6.x) | yes (`backend="gdxrs"`) |
| Enum mode A (full-dictionary, budget-admitted, strict UTF-8) | yes (dictionary not budget-admitted) | yes |
| Bounded ID mode B (`key_mode="uel_id"`, `resolve_labels`, `label_ids`) | no (clear error) | yes |
| Streaming batches (`batch_size`, `collect_batches`), owned batches | single batch (batch_size ignored) | yes |
| Projection pushdown (key/value masks) | no | yes |
| Global accepted-row limits (pushed and residual) | yes | yes |
| Threads | 2..N (native parallel paths) | 2..=4 admitted (core parallel scan over an explicit checkpoint index); >4 rejected |
| GIL release during production | no | yes |
| Cancellation/join lifecycle | n/a | `close()`/`cancel()`, lock-free signal, shutdown protocol |
| Special values | NaN for UNDEF/NA, native acronym numbers | null for UNDEF/NA/acronyms, EPS -> +0.0 (counted) — documented difference |
| Build without GAMS/C++/native GDX library | no | yes (`--no-default-features --features gdxrs`) |

## Migration decision

Legacy remains the default in 0.6.x. Changing the default backend or
removing `gdx-sys` requires a **separately reviewable migration change**
after:

1. All task-041..047 gates pass at the pinned adapter/core revisions,
   including the four-platform functional matrix
   (`.github/workflows/platform-matrix.yml`) and the Linux capped
   streaming acceptance (`tools/m7_acceptance.py`) with three trials.
2. The capped-process findings are dispositioned (see
   [evidence/047.md](evidence/047.md)): the 1 GiB RLIMIT_AS requires
   `MALLOC_ARENA_MAX=2` (glibc per-thread arenas push the polars
   streaming engine's fixed VmPeak to ~1009 MiB without it), and the
   pyarrow-backed `GdxrsBatchCursor.__next__` cannot map `libarrow`
   under the cap — the capped path uses the Arrow C stream capsules.
3. A maintainer-approved migration PR flips the default and reruns the
   affected gates.

## Known limitations

- Generated-row-index-only projections (#45): sources cannot express
  "N rows with no columns"; the gdxrs backend raises clearly, the
  legacy backend silently returns zero rows.
- Blocked synchronous OS reads inside an in-flight worker step cannot
  be forcibly interrupted; cancellation joins the current step.
