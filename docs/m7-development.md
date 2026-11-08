# M7 development branch

**Integration branch: `develop/m7`. All M7 task PRs target this branch. Do not
merge it into `main` until every task 041–047 and its required gates are done.**

Development may span multiple PRs per task. A task branch is created from the
current integration head and merged back after review. Keep changes bounded by
the task prerequisites; merging a task PR is not a milestone release. PR #39
establishes the documentation and expectations for task 041 on this branch.

Task definitions are in
[gdxrs PR #34](https://github.com/enioarz/gdxrs/pull/34); they are proposed task
specifications, not evidence that the later tasks are implemented. Core revision:
`9a7411764f176adb613185fa1e45d64b06123683`. All adapter work stays in this repo.

| Task | Scope | Implementation status / evidence |
| --- | --- | --- |
| 041 | Contract, resource envelope and independent expectations | Documentation complete; [evidence](evidence/041.md), PR #39 |
| 042 | Explicit gdxrs backend and legacy differential path | Merged in PR #41 at `b008b57`; [evidence](evidence/042.md) |
| 043 | Stable schemas, admitted enums and bounded ID mode | Implementation complete; [evidence](evidence/043.md), [quality follow-up](evidence/043-quality.md); PR #42 pending review/merge |
| 044 | Cursor-backed owned batch source with backpressure | Implementation complete; [evidence](evidence/044.md); task PR pending review/merge |
| 045 | Exact predicate/projection/global-limit behavior | Implementation complete; [evidence](evidence/045.md); task PR pending review/merge |
| 046 | Scan ownership, repeated scans, threads and cancellation | Implementation complete; [evidence](evidence/046.md); task PR pending review/merge |
| 047 | Four-platform functionality, capped streaming and migration review | Pending |

For every implementation PR, state its task(s), base `develop/m7`, checks actually
run and outstanding gates. Add evidence links and update this table when a task
passes its acceptance checks; do not mark a task done merely because code exists.

## Final main merge gate

The aggregate `develop/m7` → `main` PR must remain draft until:

1. Every task 041–047 is complete with reviewed evidence at exact adapter/core
   revisions; no unresolved blocking review findings remain.
2. Existing and new tests pass across both backends, including schemas, value
   projections, predicates, limits, repeated scans and failure/cancellation paths.
3. Linux x86_64, macOS ARM64/Intel and Windows x64 native functionality and
   packaging pass. A gdxrs-only installation works without native GDX/GAMS.
4. Task 047's growing record-heavy and metadata-heavy noncollecting sinks pass
   the contract's combined memory, process, scratch, handle and queue caps, with
   three trials, exact output checks and explicit measured-platform scope.
5. Performance regressions, supported versions, skipped/unrun cases and migration
   decisions are documented. Required unrun/failed checks block merging.
6. A maintainer explicitly approves merging the completed milestone into main.

Legacy remains the default until a separately reviewed migration change. No
automatic release or general performance guarantee follows from this branch.
The historical main-fast-forward merging note in AGENTS.md must never be used to
bypass this milestone gate.
