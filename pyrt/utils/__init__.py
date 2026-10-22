"""General purpose utility functions.

Includes plotting, file handling, filter matching, airmass calculations,
and other utilities that may have broader use beyond dophot3.
"""

from pyrt.utils.dophot_config import load_config
from pyrt.utils.file_utils import *
from pyrt.utils.filter_matching import *
from pyrt.utils.airmass import calculate_airmass_array

__all__ = [
    "load_config",
    "create_residual_plots",
    "create_correction_volume_plots",
    "calculate_airmass_array",
]

def __getattr__(name):
    """Plotting (and matplotlib with it, ~0.3 s to import) only on first use"""
    if name in ("create_residual_plots", "create_correction_volume_plots"):
        from pyrt.utils import plotting
        return getattr(plotting, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
