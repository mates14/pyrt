"""
PYRT - Photometric Reduction Tool

A powerful photometric calibration and astrometric refinement package
for astronomical image processing.
"""

__version__ = "0.1.0"
__author__ = "Martin Jelínek (ASÚ AV ČR Ondřejov)"

# Main API exports - users can import core functionality
from pyrt.core.fotfit import fotfit
from pyrt.core.termfit import termfit
from pyrt.core.zpnfit import zpnfit
from pyrt.core.data_handling import PhotometryData
from pyrt.catalog.catalog import Catalog

__all__ = [
    "fotfit",
    "termfit",
    "zpnfit",
    "PhotometryData",
    "Catalog",
    "__version__",
]
