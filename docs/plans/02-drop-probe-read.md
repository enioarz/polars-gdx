# Plan 02 — Remove the redundant probe read (use SymbolInfo.records)

## Problem

`GdxFile::read_symbol_raw_locked` (crates/gdx/src/reader.rs:210-245) performs
a **probe read** to learn the record count:

```rust
ffi::c__gdxdatareadrawstart(self.obj, info.number as i32, &mut nrecs) == 0  // probe
ffi::c__gdxdatareaddone(self.obj);                                        // reset
```

and then calls `gdxDataReadRawStart` **again** for the actual read (a third
start on the filtered path). For compressed GDX files the start may touch or
decompress the symbol's data block, so the probe can double that work per
read.

Meanwhile `SymbolInfo.records` (crates/gdx/src/types.rs:100) is already
populated at file-open time from `gdxSymbolInfoX`
(crates/gdx/src/reader.rs:338-346) and holds exactly this count.

## Goal

Drop the probe. Use `info.records` for `Vec::with_capacity` and start the
actual read exactly once per call.

## Changes

### crates/gdx/src/reader.rs — `read_symbol_raw_locked`

- Delete the probe `gdxDataReadRawStart`/`gdxDataReadDone` pair.
- `let mut data = RawSymbolData::with_capacity(info.dim, info.records);`
- The unfiltered (bulk callback) path needs no count beyond capacity.
- The filtered path starts the read once and loops as today.
- Keep the `gdxDataReadRawStart` error mapping (`gdxDataReadRawStart`
  operation name) for the one remaining call.

### Notes / gotchas

- `SymbolInfo.records` is `usize` built from the `c_int` `reccnt` — for very
  large symbols `c_int` (i32) is what the library reports; capacity is only
  an optimization, so a truncated or zero value is harmless (Vec grows).
  If you want belt-and-braces: `nrecs.max(0) as usize` is already the
  pattern; using `info.records` directly is fine.
- Do **not** remove the `gdxDataReadDone` call that terminates the actual
  read loop — only the probe's Done.
- A stale/zero `records` value in a malformed file must not break reading:
  the read loop terminates on `gdxDataReadRaw` returning 0, not on a count.

## Verification

- `cargo test --workspace` (the `parity` tests in reader.rs cover raw reads)
- Full read equivalence: pytest `tests/test_lazy.py` unchanged and green
  (`read_gdx`, prefilter, predicate tests all exercise both paths).
- Optional micro-benchmark: `cargo run -p gdx --release --example make_bench
  -- /tmp/bench_big.gdx 2000000` then time full reads before/after; expect a
  small but measurable improvement (~a few ms of the ~90 ms C floor).

## Scope guard

This is a change inside `read_symbol_raw_locked` only. Do not touch
`read_records_raw` (the `Record`-based path keeps its own start), the bulk
callback sink, or the Python layer.
