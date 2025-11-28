"""Catalog access module.

Unified interface to multiple photometric reference catalogs including
ATLAS, PanSTARRS, GAIA, SDSS, USNO-B, and Makak.

This module may be modularized into separate catalog units in the future.
"""

from pyrt.catalog.catalog import Catalog, CatalogFilter

__all__ = ["Catalog", "CatalogFilter"]
