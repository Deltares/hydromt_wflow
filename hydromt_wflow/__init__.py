"""hydroMT plugin for wflow models."""

__version__ = "1.0.0.dev0"

from .naming import *
from .utils import *
from .wflow import *
from .wflow_sediment import *

# PCRaster functions
try:
    import pcraster as pcr

    HAS_PCRASTER = True
    from .pcrm import *
except ImportError:
    HAS_PCRASTER = False
