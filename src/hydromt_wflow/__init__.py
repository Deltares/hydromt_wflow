"""hydroMT plugin for wflow models."""

from .naming import *
from .utils import *
from .version import __version__
from .wflow import *
from .wflow_sediment import *

# PCRaster functions
try:
    import pcraster as pcr

    HAS_PCRASTER = True
    from .pcrm import *
except ImportError:
    HAS_PCRASTER = False
