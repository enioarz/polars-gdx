"""Polars IO plugin for reading GAMS GDX files lazily with prefiltering."""
from polars_gdx._core import Reader  # noqa: F401
from polars_gdx.lazy import (  # noqa: F401
    scan_gdx,
    read_gdx,
    read_domains,
    list_symbols,
)

__version__ = "0.3.1"
__all__ = ["Reader", "scan_gdx", "read_gdx", "read_domains", "list_symbols"]
