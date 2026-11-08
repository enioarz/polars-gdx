# Task 043 — general code quality follow-up

Code revision `4aaaebb`; reviewed base `65eaa55`. This follow-up stays within the
current adapter APIs and task-043 scope. No core revision, dependency or default
backend changes are included. [Structured results](043-quality-results.json)
record source/wheel hashes and identities of the full local logs.

## Changes

- Admission arithmetic now shares a checked layout calculator: multiplication
  and addition overflow return errors. A regression exercises a huge sized
  sequence without accessing or materializing it, and a Rust unit test covers
  product/sum overflow. Admitted vectors use fallible reservation.
- Label resolution extracts only the admitted sequence length, supports tuples
  and preserves exact bytes/order/duplicates. Label lookup borrows Python UTF-8
  views instead of cloning each string into two Rust buffers. Conservative
  admission estimates remain in place.
- Arrow dictionary/binary builders receive the checked payload size up front,
  avoiding a temporary vector of borrowed slices and repeated buffer growth.
  Payloads outside the signed 32-bit Arrow offset range reject explicitly.
- Mixed label/value predicates convert label comparisons into integer ID
  expressions. Residual expressions keep their original types. The adapter no
  longer builds Python lists or temporary String columns for every output row.
  One comparison parser is shared by full pushdown and residual handling.
  Existing restrictions on disabled pushdown and same-column label/numeric
  mixing are retained and now documented in the public docstring.
- Column-name collisions/duplicates reject before dictionary admission or record
  decoding. Used-label requests validate symbols/dimensions even when no labels
  resolve. Used-label intersection retains only requested IDs, rather than a full
  used-domain vector and hash set; underlying scan costs are not claimed to vanish.
- Domain cursors close explicitly on success/failure, and secondary cursor or
  scratch-cleanup diagnostics are retained. Scratch directories use exclusive
  creation and never adopt an existing directory. Cleanup-denial injection was
  not executed in this follow-up.
- Closing a reader releases its dictionary cache owners. Previously returned
  Arrow batches retain valid data through their own ownership, covered by a new
  regression. Poisoned reader locks return errors instead of panicking.

## Verification at the code revision

| Check | Result |
| --- | --- |
| Locked release extension build | Pass |
| Default Python suite | 145 passed |
| gdxrs-default Python suite | 111 passed, 34 legacy-specific skips |
| Isolated gdxrs-only wheel, fresh test venv | 93 passed, 40 legacy-specific skips |
| Locked workspace Rust tests | 16 passed including one compile doctest; four default-ignored tests |
| Workspace all-target Clippy, warnings denied | Pass |
| gdxrs-only all-target Clippy, warnings denied | Pass |
| Formatting and diff whitespace | Pass |

The seven new Python regression cases cover overflow before sequence access,
UTF-8 lookup order, early name validation, empty-filter argument validation,
retained Arrow ownership after close and ID predicates without Python row copies.
Existing independent expectations and assertions remain enabled. No threshold or
skip was relaxed; generator/fixture bytes are unchanged.

Fresh [CI](https://github.com/enioarz/polars-gdx/actions/runs/42474079052) passes
test, benchmark and gdxrs-only jobs. At documentation time the
[wheel workflow](https://github.com/enioarz/polars-gdx/actions/runs/42474079050)
passed ARM Linux and both macOS builds; x86 Linux and Windows were still running.
These are new results; the prior wheel failures remain historical and their Linux
Docker exit-code-2 cause was not established. The documentation push refreshes CI.

This is implementation self-verification. Full cursor-backed batches, broader
predicate/lifecycle integration and OS-capped M7 acceptance remain tasks 044–047.
The `develop/m7` → `main` gate remains unchanged; this PR is not a milestone release.
