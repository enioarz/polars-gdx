"""Shared pytest fixtures and backend-selection helpers (M7 task 042).

The default backend is ``legacy`` unless ``POLARS_GDX_BACKEND`` names an
available backend (docs/m7-adapter-contract.md §4). The existing suite runs
against both backends by running it twice, once per default; tests that
assert legacy-backend-specific mechanics (native-prefilter spies, parallel
threads, a monkeypatched legacy ``Reader``) are marked
``legacy_backend_only`` and skip when the default is ``gdxrs``.
"""

import os

import pytest

import polars_gdx

DEFAULT_BACKEND = os.environ.get("POLARS_GDX_BACKEND", "legacy")

legacy_backend_only = pytest.mark.skipif(
    DEFAULT_BACKEND == "gdxrs" or polars_gdx.Reader is None,
    reason="asserts legacy-backend-specific mechanics (native prefilter "
    "spy, parallel threads or the legacy Reader class) and the default "
    "backend is not legacy, or the legacy backend is not compiled",
)
