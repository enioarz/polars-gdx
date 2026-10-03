# Plan 05 — Case-insensitive label matching (or clearly documented case sensitivity)

## Problem

Label matching against UELs is strictly case-sensitive:

- Rust: `label_to_index.get(l.as_str())` in
  `Reader::resolve_uel_indices` (crates/polars-gdx/src/lib.rs:129).
- Python: `l in uels` in `_predicate_key_filter`
  (python/polars_gdx/lazy.py:168-173).

GAMS itself matches set elements case-insensitively (labels are typically
stored uppercased in GDX files, but not guaranteed). Consequence:

```python
scan_gdx(f, symbol="x").filter(pl.col("dim_0") == "SEATTLE").collect()
# silently returns 0 rows even though "seattle" exists
```

`gamsapi`/`gams.transfer` would match here. This is a silent-wrong-answer
gotcha, not just an inconvenience.

## Goal

Make `filter(pl.col(key) == label)` and `key_filter={d: [labels]}` match the
way GAMS does (case-insensitive), **without** changing what is stored in the
output frame (original label casing must be preserved in the produced
columns).

## Decision to make first

Two acceptable outcomes — pick one, note it in the PR:

A. **Case-insensitive matching** (recommended; matches GAMS/gamsapi).
B. Keep case-sensitive matching but document it loudly (docstrings +
   README) and raise nothing silently. (Lower effort; acceptable fallback
   if A turns out to be invasive.)

## Changes (for option A)

### UEL reverse map

- Build `uel_index` with an ASCII-uppercase-folded key: store the map as
  `HashMap<String, i32>` where the key is the label **as stored**, plus a
  second map (or a single folded map with a fallback probe) for folded
  lookup. Simplest robust design: one `HashMap<String, i32>` keyed on
  `label.to_uppercase()` (or `to_ascii_uppercase`) for **filter resolution**,
  kept separate from the display table so display casing is untouched.
  - Watch for collision: two UELs differing only by case. GDX files can
    technically contain both. Decide deterministically (first wins, matching
    today's `HashMap::insert`-first-wins? verify: today the loop inserts in
    UEL order, so later insert overwrites — check reader.rs:158-166 and
    pick first-wins for stability, noting any behavior change).
- Lookup side folds the user label the same way before `get()`.

### Python side

- `_predicate_key_filter` membership test (`l in uels`) must fold the same
  way: build a folded set once (or rely on Plan 01's cached structures and
  fold there).
- The folded label that is appended to the native filter list must be the
  **stored** (file-cased) label, because downstream comparison happens on
  raw UEL indices resolved from stored labels. Easiest: resolve label →
  UEL number in Python? No — keep current flow, but make the Python
  membership check pass only labels that exist under folding, and pass the
  **user-supplied** label through: Rust's `resolve_uel_indices` re-resolves
  with folding, so consistency is preserved as long as both sides fold
  identically.

### Predicate re-application

- The predicate is always re-applied by Polars after the native prefilter
  (lazy.py:146-149). Polars' `==` on strings is case-sensitive. If the native
  prefilter (case-insensitive) admits a record whose stored label differs in
  case from the user label, the Polars-side re-filter would then **drop** it,
  reintroducing the silent-miss. Fix options:
  1. When folding changed a label (user label != stored label), rewrite the
     re-applied predicate: since the plan is serialized JSON and rebuilt,
     add a translation that replaces `col == "USER"` with an equivalent —
     but Polars-side case-insensitive equality is `str.to_uppercase() ==
     ...`, not expressible in the recognized Eq-of-literal form. Simplest
     correct approach: when any folded label differs from a stored label,
     **skip the native translation** for that predicate (fall back to
     Polars-side filtering entirely) — correctness first, perf only when
     casing matches exactly.
  2. Or: translate `==` to membership in the set of stored-case variants
     and keep it in the filter list (labels list can carry multiple stored
     labels that fold to the user label). The re-applied Polars `==` still
     fails on non-exact casing, so this doesn't work alone — must combine
     with a case-insensitive rewrite of the re-applied expression, which is
     out of scope of the current plan-walker. Prefer option 1.

Given the interaction above, **option 1** is the concrete recommendation:
native prefilter only when the user label matches a UEL exactly
(case-sensitive); otherwise fall back to Polars-side filtering and (small
extra) emit a `warnings.warn` once per scan when a label's case-folding
matched — so users learn the data's real casing. Keep `key_filter` (the
explicit API) fully case-insensitive, since there is no re-applied predicate
for it — this is where GAMS parity matters most.

## Verification

- New pytest: fixture `trnsport.gdx` — `key_filter={0: ["SEATTLE"]}` returns
  the seattle rows (after the change).
- New pytest: `filter(pl.col("dim_0") == "SEATTLE")` returns the same rows as
  the lowercase form (may go through the fallback path — assert the *result*,
  not the path).
- Existing tests must stay green (they use exact-case labels, so behavior
  there is unchanged).
- Document the final behavior in `scan_gdx`/`read_gdx` docstrings and the
  README ("Label matching" note), whichever option is chosen.

## Gotchas

- GDX label casing is not guaranteed uppercase; GAMS *usually* uppercases —
  never assume.
- `to_uppercase()` (Unicode) vs `to_ascii_uppercase()`: GAMS labels are
  typically ASCII; use `to_ascii_uppercase` for determinism and speed, and
  note it in docs.
