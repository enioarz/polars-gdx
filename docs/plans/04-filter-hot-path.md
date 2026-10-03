# Plan 04 — Filter hot path: single-label fast case + cheaper membership + TLS sink access

## Problem

The filtered read path is the library's headline feature, and its innermost
loop has avoidable overhead (crates/polars-gdx/src/lib.rs:107-114):

```rust
Some(&|idx: &[i32]| {
    index_filters.iter().all(|&(d, ref allowed)|
        allowed.contains(&idx[d]))   // HashSet<i32> hash per record per dim
})
})
```

- `HashSet<i32>` hashing per record per dimension — for the overwhelmingly
  common single-label filter (`dim_0 == "seattle"`) a single integer
  compare would do.
- The bulk callback (`store_record`, crates/gdx/src/reader.rs:552-568) takes
  and re-sets the `SINK` thread-local **per record** (two TLS accesses per
  record).
- `map_special` (crates/gdx/src/reader.rs:442-455) runs up to 4 double
  comparisons per record on the hot path.

## Goal

Cut per-record constant overhead in both the filtered and bulk loops.

## Changes

### 1. Filter representation (crates/polars-gdx/src/lib.rs)

Replace `Vec<(usize, HashSet<i32>)>` with an enum, built once per read:

```rust
enum IndexFilter {
    Single(usize, i32),                    // most common: one dim, one label
    Set(usize, Vec<i32>),                  // sorted; binary search when long
    Conjunction(Vec<IndexFilter>),         // multiple dims — or keep Vec<..>
}
```

Simpler alternative that keeps most of the win: keep `Vec<(usize, Vec<i32>)>`
with **sorted** index vectors, plus a `Single` special case when the whole
filter is exactly one `(d, [idx])` pair:

```rust
let pred: gdx::IndexPred<'_> = match index_filters.as_slice() {
    [] => None,
    [(d, [idx])] => Some(&|idx_slice: &[i32]| idx_slice[*d] == *idx),
    many => Some(&|idx_slice: &[i32]| many.iter().all(|(d, allowed)|
        allowed.binary_search(&idx_slice[*d]).is_ok())),
};
```

(The closure must still be `&dyn Fn`; the current `Some(&|...|)` pattern
already does this — keep it.)

- `resolve_uel_indices` produces sorted `Vec<i32>` instead of `HashSet<i32>`.
- Keep the empty-set semantics: a filter whose label resolved to nothing
  must still be present with an empty allowed list (matches no records).
  With the `Single` shape an empty set degrades to the `many` arm — handle it
  (an empty `allowed` in `many` correctly matches nothing via `binary_search`
  on an empty slice).

### 2. TLS sink access (crates/gdx/src/reader.rs)

`store_record` currently does `SINK.with(|s| s.take())` then `s.set(..)` per
record. Rework to read the sink without taking:

- Keep the set/take bracketing **around** the `c__gdxdatareadrawfast` call
  (required for soundness on panic paths), but inside the callback replace
  take/set with a plain read:

  ```rust
  thread_local! {
      static SINK: Cell<Option<RecordSink>> = const { Cell::new(None) };
  }

  extern "C" fn store_record(indx: *const i32, vals: *const f64) {
      let Some(sink) = SINK.with(|s| s.get().unwrap_ptr()) else { return; };
      // ... store; no set() back
  }
  ```

  `Cell<Option<RecordSink>>` can't hand out `&RecordSink` — options:
  - change to `RefCell<Option<RecordSink>>` and `borrow()` (one TLS access +
    cheap flag check), or
  - store `Cell<*const RecordSink>` where the `RecordSink` lives on the
    caller's stack (as today: `let sink = RecordSink {..}; SINK.set(&sink)`),
    then `SINK.with(|s| s.get())` is a raw pointer load. The sink already
    lives on the stack for the duration of the call — this is sound under the
    same reasoning documented at reader.rs:546-550.
- Whatever variant: keep the "bracketed by set/take around the FFI call"
  invariant and the existing soundness comment's claims true.

### 3. map_special (crates/gdx/src/reader.rs)

Reorder comparisons to the cheapest common case and short-circuit:

```rust
fn map_special(v: f64, special: &[f64; ffi::GMS_SVIDX_MAX]) -> f64 {
    if v == special[ffi::GMS_SVIDX_PINF] { return f64::INFINITY; }
    if v == special[ffi::GMS_SVIDX_MINF] { return f64::NEG_INFINITY; }
    if v == special[ffi::GMS_SVIDX_UNDEF] || v == special[ffi::GMS_SVIDX_NA] {
        return f64::NAN;
    }
    if v == special[ffi::GMS_SVIDX_EPS] { return 0.0; }
    v
}
```

(Identical semantics; ordinary values now pay 1-2 compares instead of
potentially 4.) Optionally hoist `special[...]` loads out of the loop in the
filtered path by copying the 4 sentinels into locals before the `while`.

## Verification

- `cargo test --workspace`, clippy `-D warnings`, fmt.
- pytest suite — especially `test_scan_prefilter`,
  `test_predicate_uses_native_prefilter`, `test_predicate_unknown_label_yields_no_rows`,
  `test_predicate_conjunction_prefilter`.
- Benchmark the filtered scenario from README (1000 of 2M rows) before/after;
  record the delta in the plan's PR.
- New unit test: single-label filter with a label absent from the file still
  yields zero rows (guards the empty-set degradation).

## Gotchas

- Predicates must remain `&dyn Fn(&[i32]) -> bool` (`gdx::IndexPred`) — do
  not change the gdx crate's public type.
- Do not remove the "empty resolved set means zero rows" behavior — it is a
  documented design fact (AGENTS.md).
