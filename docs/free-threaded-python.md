# Free-threaded Python readiness

The current library is **not ready for GIL-disabled execution**. This is a
source and build-configuration audit; no free-threaded interpreter was available
in the development environment for runtime validation.

## Current blockers

- `crates/polars-gdx/src/lib.rs` uses `#[pymodule]` with PyO3 0.25. Its default
  is `gil_used = true`. A source build imported into free-threaded CPython
  normally causes the interpreter to enable the GIL (unless overridden).
- The same file declares `unsafe impl Send/Sync for SendGdxFile`. The contained
  `GdxFile` has `RefCell` caches in `crates/gdx/src/reader.rs`. In particular,
  `uel_index()` and the initial cache check in `uel_table()` access them outside
  the global GDX lock. Concurrent calls sharing a Reader cannot safely rely on
  that lock alone. The outer cache mutexes do not cover all these accesses.
- Python-facing native reads retain the GIL; there is no `allow_threads` around
  the work. Native `threads=` parallelism uses independent worker handles and
  does not establish thread safety for concurrent Python calls on one Reader.
- `crates/polars-gdx/Cargo.toml` enables `abi3-py310`. The build workflow publishes
  conventional abi3 wheels and tests regular Python 3.10/3.13/3.14. It has no
  free-threaded interpreter, version-specific free-threaded wheels, or tests
  asserting that the GIL remains disabled after importing dependencies.

PyO3 0.25 can build against free-threaded CPython, ignoring the abi3 setting
for that build. This does **not** make the existing abi3 wheels or the current
unsafe wrapper free-threading safe. See the
[PyO3 0.25 free-threading guide](https://pyo3.rs/v0.25.1/free-threading.html).

## Work needed before declaring support

1. Synchronize handle/cache access, or enforce exclusive ownership per call;
   audit every unsafe Send/Sync implementation and concurrent close/read path.
2. Detach from Python during blocking native operations and lock acquisition
   where appropriate. Audit lock ordering and interaction with interpreter
   synchronization, including Arrow conversion and native worker joins.
3. Validate the PyO3/Arrow/PyArrow/Polars dependency combination on supported
   free-threaded interpreters and produce their version-specific wheels.
4. Add CI coverage that checks `sys._is_gil_enabled()` after imports and exercises
   shared/separate Readers, cache initialization, simultaneous reads and closes,
   compressed/parallel paths, and error handling.
5. Only after the audit and tests pass, opt in with `#[pymodule(gil_used = false)]`.

Forcing `PYTHON_GIL=0` with the current wrapper is not a supported workaround.
