"""Utility of the wflow components."""

import re
from os.path import relpath
from pathlib import Path
from typing import Any

import numpy as np
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
    return value.as_posix()


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


def align_grid_with_region(
    data: xr.DataArray,
    region_grid: xr.DataArray,
    region_component_name: str,
) -> xr.DataArray:
    """Align the data grid with the region grid.

    Aligns the name of the x and y dimensions and flips the y dimension if needed.

    Parameters
    ----------
    data : xr.DataArray
        The data array to align.
    region_grid : xr.DataArray
        The region grid to align with.
    region_component_name : str
        The name of the region component.

    Returns
    -------
    xr.DataArray
        The aligned data array.

    Raises
    ------
    ValueError
        If the data grid is not identical to the region grid.
    """
    if not data.raster.identical_grid(region_grid):
        y_dim = region_grid.raster.y_dim
        x_dim = region_grid.raster.x_dim
        # First try to rename dimensions
        data = data.rename(
            {
                data.raster.x_dim: x_dim,
                data.raster.y_dim: y_dim,
            }
        )
        # Flip latitude if needed
        if (
            np.diff(data[y_dim].values)[0] > 0
            and np.diff(region_grid[y_dim].values)[0] < 0
        ):
            data = data.reindex({y_dim: data[y_dim][::-1]})

    if not data.raster.identical_grid(region_grid):
        raise ValueError(
            f"Data grid must be identical to {region_component_name} component"
        )

    return data


def test_equal_grid_data(
    grid: xr.Dataset, other_grid: xr.Dataset
) -> tuple[bool, dict[str, str]]:
    """
    Test if two grid datasets are equal.

    Checks the CRS, dimensions, and data variables for equality.

    Parameters
    ----------
    grid : xr.Dataset
        The first grid dataset to compare.
    other_grid : xr.Dataset
        The second grid dataset to compare.

    Returns
    -------
    tuple[bool, dict[str, str]]
        True if the grids are equal, and a dict with the associated errors per property
        checked.
    """
    errors: dict[str, str] = {}
    invalid_maps: dict[str, str] = {}
    invalid_coords: dict[str, str] = {}

    # Check if grid is empty
    if len(grid) == 0:
        if len(other_grid) == 0:
            return True, {}
        else:
            errors["grid"] = "first grid is empty, second is not"
            return False, errors

    # Check CRS and dims
    maps = grid.raster.vars

    if not np.all(grid.raster.crs == other_grid.raster.crs):
        errors["crs"] = "the two grids have different crs"

    # check on dims names and values
    for dim in other_grid.dims:
        try:
            xr.testing.assert_identical(other_grid[dim], grid[dim])
        except AssertionError:
            errors["dims"] = f"dim {dim} not identical"
        except KeyError:
            errors["dims"] = f"dim {dim} not in grid"

    # Check if new maps in other grid
    new_maps = []
    for name in other_grid.raster.vars:
        if name not in maps:
            new_maps.append(name)
    if len(new_maps) > 0:
        errors["Other grid has additional maps"] = f"{', '.join(new_maps)}"

    # Check per map (dtype, value, nodata)
    missing_maps = []
    for name in maps:
        map0 = grid[name].fillna(0)
        if name not in other_grid.data_vars:
            missing_maps.append(name)
            continue
        map1 = other_grid[name].fillna(0)

        # hilariously np.nan == np.nan returns False, hence the additional check
        equal_nodata = map0.raster.nodata == map1.raster.nodata
        if not equal_nodata and (
            np.isnan(map0.raster.nodata) and np.isnan(map1.raster.nodata)
        ):
            equal_nodata = True

        if (
            not np.allclose(map0, map1, atol=1e-3, rtol=1e-3)
            or map0.dtype != map1.dtype
            or not equal_nodata
        ):
            if len(map0.dims) > 2:  # 3 dim map
                map0 = map0[0, :, :]
                map1 = map1[0, :, :]
            # Check on dtypes
            err = (
                ""
                if map0.dtype == map1.dtype
                else f"{map1.dtype} instead of {map0.dtype}"
            )
            # Check on nodata
            err = (
                err
                if equal_nodata
                else f"nodata {map1.raster.nodata} instead of \
{map0.raster.nodata}; {err}"
            )
            not_close = ~np.equal(map0, map1)
            n_cells = int(np.sum(not_close))
            if n_cells > 0:
                diff = (map0.values - map1.values)[not_close].mean()
                err = f"mean diff ({n_cells:d} cells): {diff:.4f}; {err}"
            invalid_maps[name] = err
    for coord in grid.coords:
        if coord in other_grid.coords:
            try:
                xr.testing.assert_identical(other_grid[coord], grid[coord])
            except AssertionError:
                invalid_coords[coord] = "not identical"
            except KeyError:
                invalid_coords[coord] = "not in grid"

    if len(invalid_coords) > 0:
        errors[f"{len(invalid_coords)} invalid coords"] = invalid_coords

    if len(missing_maps) > 0:
        errors["Other grid is missing maps"] = f"{', '.join(missing_maps)}"

    if len(invalid_maps) > 0:
        errors[f"{len(invalid_maps)} invalid maps"] = invalid_maps

    return len(errors) == 0, errors
