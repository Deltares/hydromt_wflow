"""Utility of the wflow components."""

import re
from os.path import relpath
from pathlib import Path
from typing import Any

import xarray as xr

MOUNT_PATTERN = re.compile(r"(^\/(\w+)\/|^(\w+):\/).*$")


## Config/ pathing related
def _mount(
    value: str,
) -> str | None:
    """Get the mount of a path.

    As this not properly solved currently by pathlib or os.
    """
    m = MOUNT_PATTERN.match(value)
    if m is None:
        return None
    return m.group(1)


def _relpath(
    value: Any,
    root: Path,
) -> str | Any:
    """Generate a relative path.

    Being able to go either up or down in directories.
    Also return the original path if the mount differs.
    Otherwise it will error..
    """
    if not isinstance(value, (Path, str)) or not Path(value).is_absolute():
        return value
    value = Path(value)
    try:
        if _mount(value.as_posix()) == _mount(root.as_posix()):
            value = Path(relpath(value, root))
    except ValueError:
        pass  # `value` path is not relative to root
    finally:
        return value.as_posix()


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
        return None
    if isinstance(mask, xr.DataArray):
        return mask != mask.raster.nodata
    if not isinstance(mask, str):
        raise ValueError(f"Unknown type for determining mask: {type(mask).__name__}")
    for ds in args:
        if mask in ds:
            return ds[mask] != ds[mask].raster.nodata
    return None  # Nothin found


def make_config_paths_relative(
    data: dict,
    root: Path,
):
    """Make the configurations path relative to the root.

    This only concerns itself with paths that are absolute and on
    the same mount.

    Parameters
    ----------
    data : dict
        The configurations in a dictionary format.
    root : Path
        The root to which the paths are made relative.
        Most of the time, this will be the parent directory of the
        configurations file.
    """
    for key, val in data.items():
        if isinstance(val, dict):
            data.update({key: make_config_paths_relative(val, root)})
        else:
            data.update({key: _relpath(val, root)})
    return data
