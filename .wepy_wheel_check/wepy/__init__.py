"""Electrochemical data-processing utilities.

The package currently exposes the established ``basics`` and ``iv_curve``
APIs. Older analysis scripts commonly import these modules explicitly.
"""

__version__ = "0.1.2"

from . import basics, iv_curve, plots
from .basics import *
from .iv_curve import *

__all__ = [
    "basics",
    "iv_curve",
    "plots",
    "get_sample_name",
    "read_mpr",
    "read_mpt_dataframe",
]
