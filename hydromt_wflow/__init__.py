"""HydroMT plugin for wflow models."""

try:
    # Rasterio 1.5 installs a broken sys.excepthook that recurses infinitely
    # on interpreter shutdown. Reset it to a safe version that prints the exception
    # and then calls the original hook wrapped in a try-except.
    import sys as _sys
    import traceback as _traceback

    import rasterio as _rasterio

    _original_excepthook = _sys.excepthook

    def safe_excepthook(exc_type, exc_value, exc_tb):
        print("REAL EXCEPTION:", file=_sys.stderr)
        _traceback.print_exception(exc_type, exc_value, exc_tb, file=_sys.stderr)
        try:
            _original_excepthook(exc_type, exc_value, exc_tb)
        except Exception:
            pass  # suppress the broken hook

    _sys.excepthook = safe_excepthook

except Exception:
    print(
        "Failed to reset broken exception hook installed by rasterio 1.5. "
        "You may encounter issues with exception handling.",
        file=_sys.stderr,
    )

from pathlib import Path

from hydromt_wflow.version import __version__
from hydromt_wflow.wflow_base import WflowBaseModel
from hydromt_wflow.wflow_sbm import WflowSbmModel
from hydromt_wflow.wflow_sediment import WflowSedimentModel

DATA_DIR = Path(__file__).parent / "data"

__all__ = ["WflowBaseModel", "WflowSbmModel", "WflowSedimentModel", "DATA_DIR"]
