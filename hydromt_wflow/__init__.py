"""hydroMT plugin for wflow models."""

from os.path import abspath, dirname, join

__version__ = "0.4.1"

try:
    import pcraster as pcr

    HAS_PCRASTER = True
except ImportError:
    HAS_PCRASTER = False

DATADIR = join(dirname(abspath(__file__)), "data")

from .utils import *
from .wflow import *
from .wflow_sediment import *
