"""Some utilities from the Wflow plugin."""

from os.path import abspath, dirname, join
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xarray as xr
from hydromt.io import open_timeseries_from_table
from hydromt.vector import GeoDataArray
from hydromt.workflows.grid import grid_from_constant

DATADIR = join(dirname(abspath(__file__)), "data")

__all__ = ["read_csv_results", "get_config", "get_grid_from_config"]


def read_csv_results(
    fn: Path | str, config: Dict, maps: xr.Dataset
) -> Dict[str, GeoDataArray]:
    """Read wflow results csv timeseries and parse to dictionary.

    Parses the wflow csv results file into different ``hydromt.GeoDataArrays``, one per
    column (csv section and csv.column sections of the TOML). The xy coordinates are the
    coordinates of the station or of the representative point of the subcatch/area. The
    variable name in the ``GeoDataArray`` corresponds to the csv header attribute or
    header_map when available.

    Parameters
    ----------
    fn: str
        Path to the wflow csv results file.
    config: dict
        wflow.toml configuration.
    maps: xr.Dataset
        wflow staticmaps.nc dataset

    Returns
    -------
    csv_dict: dict
        Dictionary of hydromt.GeoDataArrays for the different csv.column section \
of the config.
    """
    # Count items by csv.column
    count = 1
    csv_dict = dict()
    # Loop over csv.column
    for col in config["csv"].get("column"):
        header = col["header"]
        # Column based on map
        if "map" in col.keys():
            # Read the corresponding map and derive the different locations
            # The centroid of the geometry is used as coordinates for the timeseries
            map_name = config["input"].get(f"{col['map']}")
            da = maps[map_name]
            gdf = da.raster.vectorize()
            gdf.geometry = gdf.geometry.representative_point()
            gdf.index = gdf.value.astype(da.dtype)
            gdf.index.name = "index"
            # Read the timeseries
            usecols = [0]
            usecols = np.append(usecols, np.arange(count, count + len(gdf.index)))
            count += len(gdf.index)
            da_ts = open_timeseries_from_table(
                fn, name=f"{header}_{col['map']}", usecols=usecols
            )
            da = GeoDataArray.from_gdf(gdf, da_ts, index_dim="index")
        # Column based on xy coordinates or reducer for the full model domain domain
        else:
            # Read the timeseries
            usecols = [0]
            usecols = np.append(usecols, np.arange(count, count + 1))
            count += 1
            try:
                da_ts = open_timeseries_from_table(fn, name=header, usecols=usecols)
            except Exception:
                colnames = ["time", "0"]
                da_ts = open_timeseries_from_table(
                    fn,
                    name=header,
                    usecols=usecols,
                    header=0,
                    names=colnames,
                )
            # Add point coordinates
            # Column based on xy coordinates
            if "coordinate" in col.keys():
                scoords = {
                    "x": xr.IndexVariable("index", [col["coordinate"]["x"]]),
                    "y": xr.IndexVariable("index", [col["coordinate"]["y"]]),
                }
            # Column based on index
            elif "index" in col.keys():
                # x and y index, works on the full 2D grid
                if isinstance(col["index"], dict):
                    # index in julia starts at 1
                    # coordinates are always ascending
                    xi = maps.raster.xcoords.values[col["index"]["x"] - 1]
                    yi = np.sort(maps.raster.ycoords.values)[col["index"]["y"] - 1]
                    scoords = {
                        "x": xr.IndexVariable("index", [xi]),
                        "y": xr.IndexVariable("index", [yi]),
                    }
                # index of the full array
                else:
                    # Create grid with full 2D Julia indices
                    # Dimensions are ascending and ordered as (x,y,layer,time)
                    # Indices are created before ordering for compatibility with
                    # raster.idx_to_xy
                    full_index = maps[f"{config['input'].get('subcatchment')}"].copy()
                    res_x, res_y = full_index.raster.res
                    if res_y < 0:
                        full_index = full_index.reindex(
                            {
                                full_index.raster.y_dim: full_index[
                                    full_index.raster.y_dim
                                ][::-1]
                            }
                        )
                    data = np.arange(0, np.size(full_index)).reshape(
                        np.size(full_index, 0), np.size(full_index, 1)
                    )
                    full_index[:, :] = data
                    full_index = full_index.transpose(
                        full_index.raster.x_dim, full_index.raster.y_dim
                    )
                    # Index depends on the struct
                    # For land uses the active subcatch IDs
                    if (
                        "vertical" in col["parameter"]
                        or "lateral.land" in col["parameter"]
                    ):
                        mask = maps[f"{config['input'].get('subcatchment')}"].copy()
                    elif "reservoir" in col["parameter"]:
                        mask = maps[
                            f"{config['input']['lateral']['river']['reservoir'].get('locs')}"
                        ].copy()
                    elif "lake" in col["parameter"]:
                        mask = maps[
                            f"{config['input']['lateral']['river']['lake'].get('locs')}"
                        ].copy()
                    # Else lateral.river
                    else:
                        mask = maps[f"{config['input'].get('river_location')}"].copy()
                    # Rearrange the mask
                    res_x, res_y = mask.raster.res
                    if res_y < 0:
                        mask = mask.reindex(
                            {mask.raster.y_dim: mask[mask.raster.y_dim][::-1]}
                        )
                    mask = mask.transpose(mask.raster.x_dim, mask.raster.y_dim)
                    # Filter and reduce full_index based on mask
                    full_index = full_index.where(mask != mask.raster.nodata, 0)
                    full_index.attrs.update(_FillValue=0)
                    mask_index = full_index.values.flatten()
                    mask_index = mask_index[mask_index != 0]
                    # idx corresponding to the wflow index
                    idx = mask_index[col["index"] - 1]
                    # Reorder full_index as (y,x) to use raster.idx_to_xy method
                    xi, yi = full_index.transpose(
                        full_index.raster.y_dim, full_index.raster.x_dim
                    ).raster.idx_to_xy(idx)
                    scoords = {
                        "x": xr.IndexVariable("index", xi),
                        "y": xr.IndexVariable("index", yi),
                    }
            # Based on model bbox center for column based on reducer for
            # the full model domain
            else:
                xmin, ymin, xmax, ymax = maps.raster.bounds
                scoords = {
                    "x": xr.IndexVariable("index", [(xmax + xmin) / 2]),
                    "y": xr.IndexVariable("index", [(ymax + ymin) / 2]),
                }
            da = da_ts.assign_coords(scoords)

        csv_dict[f"{da.name}"] = da

    return csv_dict


def get_config(
    *args,
    config: Dict = {},
    fallback=None,
    root: Path = None,
    abs_path: Optional[bool] = False,
):
    """
    Get a config value at key(s).

    Copy of hydromt.Model.get_config method to parse config outside of Model functions.

    See Also
    --------
    hydromt.Model.get_config

    Parameters
    ----------
    args : tuple or string
        keys can given by multiple args: ('key1', 'key2')
        or a string with '.' indicating a new level: ('key1.key2')
    config : dict, optional
        config dict to get the values from.
    fallback: any, optional
        fallback value if key(s) not found in config, by default None.
    abs_path: bool, optional
        If True return the absolute path relative to the model root,
        by default False.

    Returns
    -------
    value : any type
        dictionary value
    """
    args = list(args)
    if len(args) == 1 and "." in args[0]:
        args = args[0].split(".") + args[1:]
    branch = config.copy()  # reads config at first call
    for key in args[:-1]:
        branch = branch.get(key, {})
        if not isinstance(branch, dict):
            branch = dict()
            break
    value = branch.get(args[-1], fallback)
    if abs_path and isinstance(value, str):
        value = Path(value)
        if not value.is_absolute():
            if root is None:
                raise ValueError(
                    "root path is required to get absolute path from relative path"
                )
            value = Path(abspath(join(root, value)))
    return value


def get_grid_from_config(
    *args,
    config: Dict = {},
    grid: xr.Dataset | None = None,
    fallback=None,
    root: Path | None = None,
    abs_path: Optional[bool] = False,
    nodata: int | float = -9999,
    mask_name: str | None = None,
) -> xr.DataArray:
    """
    Get actual grid values from config including scale and offset.

    Calls get_config and applies value, scale and offset if available.

    Parameters
    ----------
    args : tuple or string
        keys can given by multiple args: ('key1', 'key2')
        or a string with '.' indicating a new level: ('key1.key2')
    config : dict
        config dict to get the values from.
    grid : xr.Dataset
        grid dataset to get the values from.
    fallback: any, optional
        fallback value if key(s) not found in config, by default None.
    abs_path: bool, optional
        If True return the absolute path relative to the model root,
        by default False.
    nodata: int or float, optional
        nodata value to use for the DataArray, by default -9999. Used only if the
        variable in config is described using value only.
    mask_name: str, optional
        Name of the mask variable in grid. Used only if the variable in config is
        described using value only.

    Returns
    -------
    da : xr.DataArray
        DataArray with actual grid values

    See Also
    --------
    get_config
    hydromt.workflows.grid.grid_from_constant
    """
    # get config value
    var = get_config(
        *args,
        config=config,
        fallback=fallback,
        root=root,
        abs_path=abs_path,
    )

    # direct map in grid
    if isinstance(var, str):
        if var in grid:
            da = grid[var]
        else:
            raise ValueError(f"grid variable {var} not found in staticmaps.")

    else:  # dict type
        # constant value in config
        if "value" in var:
            da = grid_from_constant(
                grid,
                constant=var["value"],
                name=args[-1],
                nodata=nodata,
                mask_name=mask_name,
            )

        # else scale and offset
        else:
            var_name = get_config("netcdf.variable.name", config=var)
            scale = var.get("scale", 1.0)
            offset = var.get("offset", 0.0)
            # apply scale and offset
            if var_name not in grid:
                raise ValueError(f"grid variable {var_name} not found in staticmaps.")
            da = grid[var_name] * scale + offset

    return da
