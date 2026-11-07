"""Polars IO plugin for reading GAMS GDX files lazily with prefiltering."""
from polars_gdx import _core
from polars_gdx.lazy import (  # noqa: F401
    scan_gdx,
    read_gdx,
    read_domains,
    list_symbols,
    DEFAULT_BACKEND,
    AVAILABLE_BACKENDS,
)

#: Legacy backend class (vendored native GDX C library); ``None`` when the
#: extension was built without the ``legacy`` cargo feature.
Reader = getattr(_core, "Reader", None)
#: gdxrs backend class (pinned pure-Rust core, M7 task 042); ``None`` when
#: the extension was built without the ``gdxrs`` cargo feature.
GdxrsReader = getattr(_core, "GdxrsReader", None)

__version__ = "0.6.0"
__all__ = [
    "Reader",
    "GdxrsReader",
    "scan_gdx",
    "read_gdx",
    "read_domains",
    "list_symbols",
    "DEFAULT_BACKEND",
    "AVAILABLE_BACKENDS",
]
