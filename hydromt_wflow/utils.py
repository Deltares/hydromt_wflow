"""Some utilities from the Wflow plugin."""

import logging
from functools import reduce
from os.path import abspath, join
from pathlib import Path
from typing import Any, Callable, Union

import geopandas as gpd
import numpy as np
import xarray as xr
from hydromt.gis import GeoDataArray
from hydromt.io import open_timeseries_from_table
from hydromt.model.processes.grid import grid_from_constant

logger = logging.getLogger(f"hydromt.{__name__}")

DATADIR = Path(__file__).parent / "data"

__all__ = [
    "get_config",
    "set_config",
    "get_grid_from_config",
    "read_csv_output",
]


def get_config(
    config: dict,
    key: str,
    root: Path | None = None,
    fallback: Any | None = None,
    abs_path: bool = False,
):
    """
    Get a config value at key.

    Parameters
    ----------
    config : dict
        The config settings.
    key : str
        keys are string with '.' indicating a new level: ('key1.key2')
    root: Path, optional
        The model root.
    fallback: Any, optional
        fallback value if key(s) not found in config, by default None.
    abs_path: bool, optional
        If True return the absolute path relative to the model root,
        by default False.
        NOTE: this assumes the config is located in model root!

    Returns
    -------
    value : Any
        dictionary value

    Examples
    --------
    >> config = {'a': 1, 'b': {'c': {'d': 2}}

    >> get_config(config, 'a')
    >> 1

    >> get_config(config, 'b.c.d')
    >> 2

    """
    parts = key.split(".")
    num_parts = len(parts)
    current = config
    value = fallback
    for i, part in enumerate(parts):
        if i < num_parts - 1:
            current = current.get(part, {})
        else:
            value = current.get(part, fallback)

    if abs_path and isinstance(value, (str, Path)):
        value = Path(value)
        if not value.is_absolute():
            if root is None:
                raise ValueError(
                    "root path is required to get absolute path from relative path"
                )
            value = Path(abspath(join(root, value)))

    return value


def set_config(config: dict, key: str, value: Any):
    """
    Update the config toml at key(s) with values.

    Parameters.
    ----------
    config : dict
        The config settings.
    key : str
        key is a string,  with '.' indicating a new level: ('key1.key2').

    Examples
    --------
    .. code-block:: ipython
        >> config
        >> {'a': 1, 'b': {'c': {'d': 2}}}
        >> set_config(config, 'a', 99)
        >> {'a': 99, 'b': {'c': {'d': 2}}}
        >> set_config(config, 'b.d.e', 99)
        >> {'a': 1, 'b': {'c': {'d': 99}}}
    """
    if not isinstance(key, str):
        raise TypeError("key must be string")
    keys = key.split(".")
    reduce(lambda d, k: d.setdefault(k, {}), keys[:-1], config)[keys[-1]] = value


def read_csv_output(
    fn: Path | str, config: dict, maps: xr.Dataset
) -> dict[str, GeoDataArray]:
    """Read wflow output csv timeseries and parse to dictionary.

    Parses the wflow csv output file into different ``hydromt.GeoDataArrays``, one per
    column (csv section and csv.column sections of the TOML). The xy coordinates are the
    coordinates of the station or of the representative point of the subcatch/area. The
    variable name in the ``GeoDataArray`` corresponds to the csv header attribute or
    header_map when available.

    Parameters
    ----------
    fn: str
        Path to the wflow csv output file.
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
    csv_dict = {}
    # Loop over csv.column
    for col in config["output"]["csv"].get("column"):
        header = col["header"]
        logger.debug(f"Reading csv column '{header}'")
        # Column based on map
        if "map" in col.keys():
            # Read the corresponding map and derive the different locations
            # The centroid of the geometry is used as coordinates for the timeseries
            map_name = config["input"].get(f"{col['map']}")
            if map_name not in maps:
                logger.warning(
                    f"Map '{map_name}' not found in staticmaps. Skip reading."
                )
                return {}
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
                    full_index = maps[
                        f"{config['input'].get('subbasin_location__count')}"
                    ].copy()
                    _, res_y = full_index.raster.res
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
                    if "reservoir" in col["parameter"]:
                        mask = maps[
                            f"{config['input'].get('reservoir_location__count')}"
                        ].copy()
                    elif "river" in col["parameter"]:
                        mask = maps[
                            f"{config['input'].get('river_location__mask')}"
                        ].copy()
                    # Else all the rest should be for the whole subcatchment
                    else:
                        mask = maps[
                            f"{config['input'].get('subbasin_location__count')}"
                        ].copy()
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


def get_grid_from_config(
    var_name: str,
    config: dict = {},
    grid: xr.Dataset | None = None,
    root: Path | None = None,
    abs_path: bool = False,
    nodata: int | float = -9999,
    mask_name: str | None = None,
) -> xr.DataArray:
    """
    Get actual grid values from config including scale and offset.

    Calls get_config and applies value, scale and offset if available.

    Parameters
    ----------
    var_name : string
        Wflow variable name to get from config. Input, static and cyclic will be
        deduced. Eg 'land_surface__slope'.
    config : dict
        config dict to get the values from.
    grid : xr.Dataset
        grid dataset to get the values from.
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
    # try with input only
    var_name = get_wflow_var_fullname(var_name, config)
    var = get_config(
        key=var_name,
        config=config,
        fallback=None,
        root=root,
        abs_path=abs_path,
    )
    if var is None:
        # try with input.static.var.value
        var = config["input"]["static"].get(f"{var_name}.value", None)
    if var is None:
        raise ValueError(f"variable {var_name} not found in config.")

    # direct map in grid
    if isinstance(var, str):
        if var in grid:
            da = grid[var]
        else:
            raise ValueError(f"grid variable {var} not found in staticmaps.")

    else:  # dict type
        # constant value in config
        if isinstance(var, (int, float)) or "value" in var:
            value = var if isinstance(var, (int, float)) else var["value"]
            da = grid_from_constant(
                grid,
                constant=value,
                name=var_name,
                nodata=nodata,
                mask_name=mask_name,
            )

        # else scale and offset
        else:
            var_name = get_config(key="netcdf_variable_name", config=var)
            scale = var.get("scale", 1.0)
            offset = var.get("offset", 0.0)
            # apply scale and offset
            if var_name not in grid:
                raise ValueError(f"grid variable {var_name} not found in staticmaps.")
            da = grid[var_name] * scale + offset

    return da


def get_wflow_var_fullname(input_var: str, config: dict) -> str:
    """Get the full variable name for a Wflow variable."""
    # Check if the variable is in the input section
    if get_config(config, f"input.{input_var}") is not None:
        return f"input.{input_var}"
    # Check if the variable is in the static section
    if get_config(config, f"input.static.{input_var}") is not None:
        return f"input.static.{input_var}"
    # Check if the variable is in the cyclic section
    if get_config(config, f"input.cyclic.{input_var}") is not None:
        return f"input.cyclic.{input_var}"
    # Check if the variable is in the forcing section
    if get_config(config, f"input.forcing.{input_var}") is not None:
        return f"input.forcing.{input_var}"
    # If not found, return the original variable name
    return input_var


def _mask_data_array(data_array: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """Mask the data array based on the mask."""
    # If the data is boolean, we set it to False where the mask is False
    if data_array.dtype == np.bool:
        return data_array.where(mask, False)
    # Otherwise we set it to nodata where the mask is False
    else:
        return data_array.where(mask, data_array.raster.nodata)


def mask_raster_from_layer(
    data: xr.Dataset | xr.DataArray, mask: xr.DataArray
) -> xr.Dataset | xr.DataArray:
    """Mask the data in the supplied grid based on the value in one of the layers.

        This for example can be used to mask a grid based on subcatchment data.
        All data in the rest of the data variables in the dataset will be set to
        the `raster.nodata` value except for boolean variables which will be set to
        False. If the supplied grid is not an xarray dataset, or does not contain
        the correct layer, it will be returned unaltered.

        The layer supplied can be either boolean or numeric. Array elements where the
        layer is larger than 0 (after type conversion) will be masked.

    Parameters
    ----------
        data (xr.Dataset, xr.DataArray):
            The grid data containing the data that should be masked
        mask (xr.DataArray):
            mask that the data will be masked to. Values can be boolean or numeric.
            Places where this layer is different than the raster nodata will be
            masked in the other data.

    Returns
    -------
        xr.Dataset, xr.DataArray: The grid with all of the data variables masked.
    """
    # Skip masking if different grid
    if data.sizes != mask.sizes:
        logger.warning("Skipping masking due to different grid sizes.")
        return data

    mask = mask != mask.raster.nodata

    if isinstance(data, xr.DataArray):
        data = _mask_data_array(data, mask)
    else:
        for var in data.data_vars:
            data[var] = _mask_data_array(data[var], mask)

    return data


def planar_operation_in_utm(
    gdf: gpd.GeoDataFrame,
    operation: Callable[[gpd.GeoSeries], Union[gpd.GeoSeries, gpd.GeoDataFrame]],
) -> Union[gpd.GeoSeries, gpd.GeoDataFrame]:
    """
    Apply a planar geometric operation on a GeoDataFrame's geometry.

    To ensure the operation is performed after converting to an appropriate UTM CRS,
    then reprojected the result back to the original CRS.

    Parameters
    ----------
        gdf (GeoDataFrame): Input GeoDataFrame with a defined CRS.
        operation (Callable): A function that operates on gdf.geometry, such as:
                              lambda geom: geom.centroid, geom.buffer(...), etc.

    Returns
    -------
        GeoSeries or GeoDataFrame: Result of the operation, reprojected to the original
        CRS.

    Examples
    --------
        >>> import geopandas as gpd
        >>> from shapely.geometry import (
        ...     Point,
        ... )
        >>> gdf = gpd.GeoDataFrame(
        ...     {
        ...         "geometry": [
        ...             Point(
        ...                 1, 2
        ...             )
        ...         ]
        ...     },
        ...     crs="EPSG:4326",
        ... )
        >>> centroid = planar_operation_in_utm(
        ...     gdf,
        ...     lambda geom: geom.centroid,
        ... )

    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame must have a defined CRS.")

    original_crs = gdf.crs
    utm_crs = gdf.estimate_utm_crs()

    gdf_projected = gdf.to_crs(utm_crs)
    result = operation(gdf_projected.geometry)

    if isinstance(result, gpd.GeoSeries):
        return result.set_crs(utm_crs).to_crs(original_crs)
    elif isinstance(result, gpd.GeoDataFrame):
        return result.to_crs(original_crs)
    else:
        raise TypeError("Operation must return a GeoSeries or GeoDataFrame.")
