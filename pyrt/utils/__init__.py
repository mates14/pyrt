"""General purpose utility functions.

Includes plotting, file handling, filter matching, airmass calculations,
and other utilities that may have broader use beyond dophot3.
"""

from pyrt.utils.dophot_config import load_config
from pyrt.utils.file_utils import *
from pyrt.utils.filter_matching import *
from pyrt.utils.plotting import create_residual_plots, create_correction_volume_plots
from pyrt.utils.airmass import calculate_airmass_array

__all__ = [
    "load_config",
    "create_residual_plots",
    "create_correction_volume_plots",
    "calculate_airmass_array",
]
