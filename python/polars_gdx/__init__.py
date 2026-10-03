"""Polars IO plugin for reading GAMS GDX files lazily with prefiltering."""

from polars_gdx._core import Reader  # noqa: F401
from polars_gdx.lazy import scan_gdx, list_symbols  # noqa: F401

__version__ = "0.1.0"
__all__ = ["Reader", "scan_gdx", "list_symbols"]
