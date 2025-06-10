"""Utility of the wflow components."""

from os.path import relpath
from pathlib import Path
from typing import Any

import xarray as xr
from tomlkit import TOMLDocument
from tomlkit.items import Table


## Config/ pathing related
def _relpath(
    value: Any,
    root: Path,
):
    """Generate a relative path."""
    if isinstance(value, str) and Path(value).is_absolute():
        value = Path(value)
    if isinstance(value, Path):
        try:
            value = Path(relpath(value, root))
        except ValueError:
            pass  # `value` path is not relative to root
        finally:
            return value.as_posix()
    return value


def make_config_paths_relative(
    data: TOMLDocument,
    root: Path,
):
    """Make the configurations path relative to the root.

    This only concerns itself with paths that are absolute and on
    the same mount.

    Parameters
    ----------
    data : TOMLDocument
        The configurations in a TOMLDocument format.
    root : Path
        The root to which the paths are made relative.
        Most of the time, this will be the parent directory of the
        configurations file.
    """
    for key, val in data.items():
        if isinstance(val, Table):
            data.update({key: make_config_paths_relative(val, root)})
        else:
            data.update({key: _relpath(val, root)})
    return data


## Grid related
def get_mask_layer(mask: str | xr.DataArray | None, *args) -> xr.DataArray | None:
    """Get the proper mask layer based on itself or a layer in a Dataset.

    Parameters
    ----------
    mask : str | xr.DataArray | None
        The mask itself or the name of the mask layer in another dataset.
    *args : list
        These have to be xarray Datasets in which the mask (as a string)
        can be present
    """
    if mask is None:
        return mask
    if isinstance(mask, xr.DataArray):
        return mask != mask.raster.nodata
    if not isinstance(mask, str):
        raise ValueError(f"Unknown type for determining mask: {type(mask).__name__}")
    for ds in args:
        if mask in ds:
            return ds[mask] != ds[mask].raster.nodata
    return None  # Nothin found
