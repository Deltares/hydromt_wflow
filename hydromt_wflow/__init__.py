"""HydroMT plugin for wflow models."""

from pathlib import Path

from hydromt_wflow.version import __version__
from hydromt_wflow.wflow_base import WflowBaseModel
from hydromt_wflow.wflow_sbm import WflowSbmModel
from hydromt_wflow.wflow_sediment import WflowSedimentModel

DATA_DIR = Path(__file__).parent / "data"

__all__ = ["WflowBaseModel", "WflowSbmModel", "WflowSedimentModel", "DATA_DIR"]
