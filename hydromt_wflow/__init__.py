"""HydroMT plugin for wflow models."""

import warnings

try:
    # Rasterio 1.5 installs a broken sys.excepthook that recurses infinitely
    # on interpreter shutdown. Reset it to a safe version that prints the exception
    # and then calls the original hook wrapped in a try-except.
    import sys as _sys
    import traceback as _traceback

    import rasterio as _rasterio

    _original_excepthook = _sys.excepthook

    def safe_excepthook(exc_type, exc_value, exc_tb):
        try:
            _original_excepthook(exc_type, exc_value, exc_tb)
        except Exception:
            _traceback.print_exception(exc_type, exc_value, exc_tb, file=_sys.stderr)

    _sys.excepthook = safe_excepthook

except Exception:
    warnings.warn(
        "Failed to patch the broken sys.excepthook installed by rasterio 1.5. "
        "This may cause a harmless, but annoying recursive error on interpreter "
        "shutdown. To fix this, upgrade rasterio to a version that resolves the issue "
        "or downgrade to rasterio <1.5. You can safely ignore this warning if you do "
        "not experience shutdown errors.",
        category=RuntimeWarning,
    )

from pathlib import Path

from hydromt_wflow.version import __version__
from hydromt_wflow.wflow_base import WflowBaseModel
from hydromt_wflow.wflow_sbm import WflowSbmModel
from hydromt_wflow.wflow_sediment import WflowSedimentModel

DATA_DIR = Path(__file__).parent / "data"

__all__ = ["WflowBaseModel", "WflowSbmModel", "WflowSedimentModel", "DATA_DIR"]
