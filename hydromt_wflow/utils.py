"""Some utilities from the Wflow plugin."""

import logging
from os.path import abspath, dirname, join
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tomlkit
import xarray as xr
from hydromt.io import open_timeseries_from_table
from hydromt.vector import GeoDataArray
from hydromt.workflows.grid import grid_from_constant
from tomlkit.items import Key

from .naming import (
    WFLOW_NAMES,
    WFLOW_SEDIMENT_NAMES,
    WFLOW_SEDIMENT_STATES_NAMES,
    WFLOW_STATES_NAMES,
)

logger = logging.getLogger(__name__)

DATADIR = Path(join(dirname(abspath(__file__)), "data"))

__all__ = [
    "convert_to_wflow_v1_sbm",
    "convert_to_wflow_v1_sediment",
    "get_config",
    "get_grid_from_config",
    "read_csv_results",
    "set_config",
]


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
    for col in config["output"]["csv"].get("column"):
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
                    full_index = maps[
                        f"{config['input'].get('subbasin_location__count')}"
                    ].copy()
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
                    if "reservoir" in col["parameter"]:
                        mask = maps[
                            f"{config['input'].get('reservoir_location__count')}"
                        ].copy()
                    elif "lake" in col["parameter"]:
                        mask = maps[
                            f"{config['input'].get('lake_location__count')}"
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


def get_config(
    config: tomlkit.TOMLDocument,
    *args,
    root: Path | None = None,
    fallback: Any | None = None,
    abs_path: bool = False,
):
    """
    Get a config value at key.

    Parameters
    ----------
    args : tuple, str
        keys can given by multiple args: ('key1', 'key2')
        or a string with '.' indicating a new level: ('key1.key2')
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

    >> get_config(config, 'b', 'c', 'd') # identical to get_config(config, 'b.c.d')
    >> 2

    >> get_config(config, 'b.c') # # identical to get_config(config, 'b','c')
    >> {'d': 2}
    """
    args = list(args)
    if len(args) == 1 and "." in args[0]:
        args = args[0].split(".") + args[1:]
    branch = config  # reads config at first call
    for key in args[:-1]:
        branch = branch.get(key, {})
        if not isinstance(branch, dict):
            branch = dict()
            break
    value = branch.get(args[-1], fallback)
    if abs_path and isinstance(value, str):
        value = Path(value)
        if not value.is_absolute():
            value = Path(abspath(join(root, value)))

    if isinstance(value, tomlkit.items.Item):
        return value.unwrap()
    elif value is None:
        return fallback
    else:
        return value


def set_config(config: tomlkit.TOMLDocument, *args):
    """
    Update the config toml at key(s) with values.

    This function is made to maintain the structure of your toml file.
    When adding keys it will look for the most specific header present in
    the toml file and add it under that.

    meaning that if you have a config toml that is empty and you run
    ``set_config("input.forcing.scale", 1)``

    it will result in the following file:

    .. code-block:: toml

        input.forcing.scale = 1


    however if your toml file looks like this before:

    .. code-block:: toml

        [input.forcing]

    (i.e. you have a header in there that has no keys)

    then after the insertion it will look like this:

    .. code-block:: toml

        [input.forcing]
        scale = 1


    .. warning::

        Due to limitations of the underlying library it is currently not possible to
        create new headers (i.e. groups like ``input.forcing`` in the example above)
        programmatically, and they will need to be added to the default config
        toml document


    .. warning::

        Even though the underlying config object behaves like a dictionary, it is
        not, it is a ``tomlkit.TOMLDocument``. Due to implementation limitations,
        error scan easily be introduced if this structure is modified by hand.
        Therefore we strongly discourage users from manually modying it, and
        instead ask them to use this ``set_config`` function to avoid problems.

    Parameters
    ----------
    config : tomlkit.TOMLDocument
        The config settings in TOMLDocument object.
    args : str, tuple, list
        if tuple or list, minimal length of two
        keys can given by multiple args: ('key1', 'key2', 'value')
        or a string with '.' indicating a new level: ('key1.key2', 'value')

    Examples
    --------
    .. code-block:: ipython

        >> config
        >> {'a': 1, 'b': {'c': {'d': 2}}}

        >> set_config(config, 'a', 99)
        >> {'a': 99, 'b': {'c': {'d': 2}}}

        >> set_config(config, 'b', 'c', 'd', 99) # identical to \
set_config(config, 'b.d.e', 99)
        >> {'a': 1, 'b': {'c': {'d': 99}}}
    """
    if len(args) < 2:
        raise TypeError("set_config() requires a least one key and one value.")
    if not all([isinstance(part, str) for part in args[:-1]]):
        raise TypeError("All but last argument for set_config must be str")

    args = list(args)
    value = args.pop(-1)
    keys = [part for arg in args for part in arg.split(".")]

    # if we try to set dictionaries as values directly tomlkit will mess up the
    # key bookkeeping, resulting in invalid toml, so instead
    # if we see a mapping, we go over it recursively
    # and manually add all of its keys, because of cloning issues.
    if isinstance(value, (dict, tomlkit.items.AbstractTable)):
        for key, inner_value in value.items():
            set_config(config, *keys, key, inner_value)

    # if the first key is not present
    # we can just set the entire thing straight
    if keys[0] not in config:
        config.append(_tomlkit_key(keys), value)
        return

    # If there is only one key we also just set that directly as
    # a string key instead of the dotted variant
    if len(keys) == 1:
        config.update({keys[0]: value})
        return

    current = config
    for idx in range(len(keys)):
        if idx != len(keys) - 1:
            remaining_key = _tomlkit_key(keys[idx:])
        else:
            remaining_key = keys[idx]

        if keys[idx] not in current or not isinstance(current[keys[idx]], dict):
            break

        current = current[keys[idx]]

    # tomlkit's update function doesn't work properly
    # so instead of updating we take the key out if it is in there
    # and readd it afterwards
    if remaining_key in current:
        _ = current.pop(remaining_key)

    current[remaining_key] = value


def _tomlkit_key(keys: list) -> Key:
    return tomlkit.key(keys[0] if len(keys) == 1 else keys)


def get_grid_from_config(
    var_name: str,
    config: Dict = {},
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
    var = get_config(
        config,
        f"input.{var_name}",
        fallback=None,
        root=root,
        abs_path=abs_path,
    )
    if var is None:
        # try with input.static
        var = get_config(
            config,
            f"input.static.{var_name}",
            fallback=None,
            root=root,
            abs_path=abs_path,
        )
    if var is None:
        # try with input.static.var.value
        var = config["input"]["static"].get(f"{var_name}.value", None)
    if var is None:
        # try with input.cyclic
        var = get_config(
            config,
            f"input.cyclic.{var_name}",
            fallback=None,
            root=root,
            abs_path=abs_path,
        )
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
            var_name = get_config(var, "netcdf.variable.name")
            scale = var.get("scale", 1.0)
            offset = var.get("offset", 0.0)
            # apply scale and offset
            if var_name not in grid:
                raise ValueError(f"grid variable {var_name} not found in staticmaps.")
            da = grid[var_name] * scale + offset

    return da


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
        layer_name (string):
            mask that the data will be masked to. Values can be boolean or
            numeric. Places where this layer is different than the rater nodata will be
            masked in the other data variables

    Returns
    -------
        xr.Dataset, xr.DataArray: The grid with all of the data variables masked.
    """
    mask = mask != mask.raster.nodata
    # Need to duplicate or else data should have a name ie we duplicate functionality
    # of GridModel.set_grid
    if isinstance(data, xr.DataArray):
        # nodata is required for all but boolean fields
        if data.dtype != "bool":
            data = data.where(mask, data.raster.nodata)
        else:
            data = data.where(mask, False)
    else:
        for var in data.data_vars:
            # nodata is required for all but boolean fields
            if data[var].dtype != "bool":
                data[var] = data[var].where(mask, data[var].raster.nodata)
            else:
                data[var] = data[var].where(mask, False)

    return data


def _solve_var_name(var: str | dict, path: str, add: list):
    """Solve the config file into individual entries.

    Every entry is the entire path ("river.lateral.< something >") plus its value.

    Parameters
    ----------
    var : str | dict,
        Either the direct settings entry or a dictionary containing nested settings.
    path : str,
        Prepend the entries with the value (e.g. "lateral" or "lateral.river")
    add : list
        Usually an empty list in which the temporary headers are stored.
    """
    if not isinstance(var, dict):
        sep = "." if path else ""
        add_str = ".".join(add) if add else ""
        yield (var, path + sep + add_str)
        return
    for key, item in var.items():
        yield from _solve_var_name(item, path, add + [key])


def _convert_to_wflow_v1(
    config: tomlkit.TOMLDocument,
    wflow_vars: Dict,
    states_vars: Dict,
    model_options: Dict = {},
    cross_options: Dict = {},  # TODO we shouldnt pass mutables as defaults
    input_options: Dict = {},
    input_variables: list = [],
    additional_variables: Dict = {},
    logger: logging.Logger = logger,
) -> Dict:
    """Convert the config to Wflow v1 format.

    Parameters
    ----------
    config: dict
        The config to convert.
    wflow_vars: dict
        The Wflow variables dict to use for the conversion between versions.
        Either WFLOW_NAMES or WFLOW_SEDIMENT_NAMES.
    states_vars: dict
        The Wflow states variables dict to use for the conversion between versions.
        Either WFLOW_STATES_NAMES or WFLOW_SEDIMENT_STATES_NAMES.
    model_options: dict, optional
        Options in the [model] section of the TOML that were updated in Wflow v1.
    input_options: dict, optional
        Options in the [input] section of the TOML that were updated in Wflow v1.
    input_variables: list, optional
        Variables that were moved to input rather than input.static.
    logger: logging.Logger, optional
        The logger to use, by default logger.

    Returns
    -------
    config_out: dict
        The converted config.
    """
    WFLOW_CONVERSION = {v["wflow_v0"]: v["wflow_v1"] for v in wflow_vars.values()}
    for k, v in states_vars.items():
        WFLOW_CONVERSION[v["wflow_v0"]] = v["wflow_v1"]
    # Add a few extra output variables that are supported by the conversion
    WFLOW_CONVERSION.update(additional_variables)

    # Logging function for the other non supported variables
    def _warn_str(wflow_var, output_type):
        logger.warning(
            f"Output variable {wflow_var} not supported for the conversion. "
            f"Skipping from {output_type} output."
        )

    # Update function for the output.netcdf_grid
    def _update_output_netcdf_grid(wflow_var, var_name):
        if wflow_var in WFLOW_CONVERSION.keys():
            config_out["output"]["netcdf_grid"]["variables"][
                WFLOW_CONVERSION[wflow_var]
            ] = var_name
        else:
            _warn_str(var_name, "netcdf_grid")

    # Initialize the output config
    logger.info("Converting config to Wflow v1 format")
    logger.info("Converting config general, time and model sections")
    config_out = tomlkit.TOMLDocument()

    # Start with the general section - split into general, time and logging in v1
    input_section = {
        "general": ["dir_input", "dir_output", "fews_run"],
        "time": [
            "calendar",
            "starttime",
            "endtime",
            "time_units",
            "timestepsecs",
        ],
        "logging": ["loglevel", "path_log", "silent"],
    }
    for section, options in input_section.items():
        for key in options:
            value = get_config(config, key, fallback=None)
            if value is not None:
                if section == "general":
                    # only fews_run was renamed in general options
                    if key == "fews_run":
                        key = "fews_run__flag"
                    config_out[key] = value
                else:
                    config_out[section] = config_out.get(section, {})
                    config_out[section][key] = value

    # Model section
    config_out["model"] = {}
    for value, config_var in _solve_var_name(config["model"], "", []):
        if config_var not in model_options:
            continue
        new_config_var = model_options[config_var]
        if isinstance(new_config_var, (list, tuple)):
            for elem in new_config_var:
                set_config(config_out, f"model.{elem}", value)
            continue
        set_config(config_out, f"model.{new_config_var}", value)

    # Cross options
    for opt_old, opt_new in cross_options.items():
        value = get_config(config, opt_old)
        if value is None:
            continue
        set_config(config_out, opt_new, value)

    # State
    logger.info("Converting config state section")
    config_out["state"] = {
        "path_input": get_config(
            config, "state.path_input", fallback="instate/instates.nc"
        ),
        "path_output": get_config(
            config, "state.path_output", fallback="outstate/outstates.nc"
        ),
    }
    # Go through the states variables
    config_out["state"]["variables"] = {}
    for key, variables in states_vars.items():
        name = get_config(config, f"state.{variables['wflow_v0']}", fallback=None)
        if name is not None and variables["wflow_v1"] is not None:
            config_out["state"]["variables"][variables["wflow_v1"]] = name

    # Input section
    logger.info("Converting config input section")
    cyclic_variables = []
    forcing_variables = []

    config_out["input"] = {}
    for key, name in config["input"].items():
        if key in input_options.keys():
            config_out["input"][input_options[key]] = name
        elif key == "forcing":
            forcing_variables = name
        elif key == "cyclic":
            cyclic_variables = name
        elif key not in ["vertical", "lateral"]:  # variables are done separately
            config_out["input"][key] = name

    # Go through the input variables
    config_out["input"]["forcing"] = {}
    config_out["input"]["cyclic"] = {}
    config_out["input"]["static"] = {}
    for key, variables in wflow_vars.items():
        print(f"key: {key}, variables: {variables}")
        name = get_config(config, f"input.{variables['wflow_v0']}", fallback=None)
        if variables["wflow_v0"] == "vertical.g_ttm" and name is None:
            # this change is probably too recent for most models
            name = get_config(config, "input.vertical.g_tt", fallback=None)
        if name is not None and variables["wflow_v1"] is not None:
            if variables["wflow_v0"] in input_options.keys():
                continue
            elif variables["wflow_v0"] in input_variables:
                config_out["input"][variables["wflow_v1"]] = name
            elif variables["wflow_v0"] in forcing_variables:
                config_out["input"]["forcing"][variables["wflow_v1"]] = name
            elif variables["wflow_v0"] in cyclic_variables:
                config_out["input"]["cyclic"][variables["wflow_v1"]] = name
            else:
                config_out["input"]["static"][variables["wflow_v1"]] = name

    # Output netcdf_grid section
    logger.info("Converting config output sections")
    if get_config(config, "output", fallback=None) is not None:
        config_out["output"] = {}
        config_out["output"]["netcdf_grid"] = {
            "path": get_config(config, "output.path", fallback="output.nc"),
            "compressionlevel": get_config(
                config, "output.compressionlevel", fallback=1
            ),
        }
        config_out["output"]["netcdf_grid"]["variables"] = {}
        for key, value in config["output"].items():
            if key in ["path", "compressionlevel"]:
                continue

            for var_name, wflow_var in _solve_var_name(value, key, []):
                _update_output_netcdf_grid(wflow_var, var_name)

    # Output netcdf_scalar section
    if get_config(config, "netcdf", fallback=None) is not None:
        if "output" not in config_out:
            config_out["output"] = {}
        config_out["output"]["netcdf_scalar"] = {
            "path": get_config(config, "netcdf.path", fallback="output_scalar.nc"),
        }
        config_out["output"]["netcdf_scalar"]["variable"] = []
        nc_scalar_vars = get_config(config, "netcdf.variable", fallback=[])
        for nc_scalar in nc_scalar_vars:
            if nc_scalar["parameter"] in WFLOW_CONVERSION.keys():
                nc_scalar["parameter"] = WFLOW_CONVERSION[nc_scalar["parameter"]]
                if "map" in nc_scalar and nc_scalar["map"] in input_options.keys():
                    nc_scalar["map"] = input_options[nc_scalar["map"]]
                config_out["output"]["netcdf_scalar"]["variable"].append(nc_scalar)
            else:
                _warn_str(nc_scalar["parameter"], "netcdf_scalar")

    # Output csv section
    if get_config(config, "csv", fallback=None) is not None:
        if "output" not in config_out:
            config_out["output"] = {}
        config_out["output"]["csv"] = {}

        config_out["output"]["csv"]["path"] = get_config(
            config, "csv.path", fallback="output.csv"
        )
        config_out["output"]["csv"]["column"] = []
        csv_vars = get_config(config, "csv.column", fallback=[])
        for csv_var in csv_vars:
            if csv_var["parameter"] in WFLOW_CONVERSION.keys():
                csv_var["parameter"] = WFLOW_CONVERSION[csv_var["parameter"]]
                if csv_var.get("map", None) in input_options.keys():
                    csv_var["map"] = input_options[csv_var["map"]]
                config_out["output"]["csv"]["column"].append(csv_var)
            else:
                _warn_str(csv_var["parameter"], "csv")

    return config_out


def convert_to_wflow_v1_sbm(
    config: Dict,
    logger: logging.Logger = logger,
) -> Dict:
    """Convert the config to Wflow v1 format for SBM.

    Parameters
    ----------
    config: dict
        The config to convert.
    logger: logging.Logger, optional
        The logger to use, by default logger.

    Returns
    -------
    config_out: dict
        The converted config.
    """
    additional_variables = {
        "vertical.interception": "vegetation_canopy_water__interception_volume_flux",
        "vertical.actevap": "land_surface__evapotranspiration_volume_flux",
        "vertical.actinfilt": "soil_water__infiltration_volume_flux",
        "vertical.excesswatersoil": "soil~compacted_surface_water__excess_volume_flux",
        "vertical.excesswaterpath": "soil~non-compacted_surface_water__excess_volume_flux",  # noqa : E501
        "vertical.exfiltustore": "soil_surface_water_unsat-zone__exfiltration_volume_flux",  # noqa : E501
        "vertical.exfiltsatwater": "land.soil.variables.exfiltsatwater",
        "vertical.recharge": "soil_water_sat-zone_top__net_recharge_volume_flux",
        "vertical.vwc_percroot": "soil_water_root-zone__volume_percentage",
        "lateral.land.q_av": "land_surface_water__volume_flow_rate",
        "lateral.land.h_av": "land_surface_water__depth",
        "lateral.land.to_river": "land_surface_water~to-river__volume_flow_rate",
        "lateral.subsurface.to_river": "subsurface_water~to-river__volume_flow_rate",
        "lateral.subsurface.drain.flux": "land_drain_water~to-subsurface__volume_flow_rate",  # noqa : E501
        "lateral.subsurface.flow.aquifer.head": "subsurface_water__hydraulic_head",
        "lateral.subsurface.river.flux": "river_water~to-subsurface__volume_flow_rate",
        "lateral.subsurface.recharge.rate": "subsurface_water_sat-zone_top__net_recharge_volume_flow_rate",  # noqa : E501
        "lateral.river.q_av": "river_water__volume_flow_rate",
        "lateral.river.h_av": "river_water__depth",
        "lateral.river.volume": "river_water__instantaneous_volume",
        "lateral.river.inwater": "river_water_inflow~lateral__volume_flow_rate",
        "lateral.river.floodplain.volume": "floodplain_water__instantaneous_volume",
        "lateral.river.reservoir.volume": "reservoir_water__instantaneous_volume",
        "lateral.river.reservoir.totaloutflow": "reservoir_water~outgoing__volume_flow_rate",  # noqa : E501
        "lateral.river.lake.storage": "lake_water__instantaneous_volume",
        "lateral.river.lake.totaloutflow": "lake_water~outgoing__volume_flow_rate",
    }

    # Options in model section that were renamed
    model_options = {
        "reinit": "cold_start__flag",
        "sizeinmetres": "cell_length_in_meter__flag",
        "reservoirs": "reservoir__flag",
        "lakes": "lake__flag",
        "snow": "snow__flag",
        "glacier": "glacier__flag",
        "pits": "pit__flag",
        "masswasting": "snow_gravitional_transport__flag",
        "thicknesslayers": "soil_layer__thickness",
        "min_streamorder_land": "land_streamorder__min_count",
        "min_streamorder_river": "river_streamorder__min_count",
        "drains": "drain__flag",
        "kin_wave_iteration": "kinematic_wave__adaptive_time_step_flag",
        "kw_land_tstep": "land_kinematic_wave__time_step",
        "kw_river_tstep": "river_kinematic_wave__time_step",
        "inertial_flow_alpha": [
            "river_local_inertial_flow__alpha_coefficient",
            "land_local_inertial_flow__alpha_coefficient",
        ],
        "h_thresh": [
            "river_water_flow_threshold__depth",
            "land_surface_water_flow_threshold__depth",
        ],
        "froude_limit": [
            "river_water_flow__froude_limit_flag",
            "land_surface_water_flow__froude_limit_flag",
        ],
        "floodplain_1d": "floodplain_1d__flag",
        "inertial_flow_theta": "land_local_inertial_flow__theta_coefficient",
        "soilinfreduction": "soil_infiltration_reduction__flag",
        "transfermethod": "topog_sbm_transfer__flag",
        "water_demand.domestic": "water_demand.domestic__flag",
        "water_demand.industry": "water_demand.industry__flag",
        "water_demand.livestock": "water_demand.livestock__flag",
        "water_demand.paddy": "water_demand.paddy__flag",
        "water_demand.nonpaddy": "water_demand.nonpaddy__flag",
        "constanthead": "constanthead__flag",
    }

    # Options in input section that were renamed
    input_options = {
        "ldd": "basin__local_drain_direction",
        "river_location": "river_location__mask",
        "subcatchment": "subbasin_location__count",
    }

    # variables that were moved to input rather than input.static
    input_variables = [
        "lateral.river.lake.areas",
        "lateral.river.lake.locs",
        "lateral.river.lake.linkedlakelocs",
        "lateral.river.reservoir.areas",
        "lateral.river.reservoir.locs",
    ]

    # Wflow entries that cross main headers (i.e. [input, state, model, output])
    cross_options = {
        "input.lateral.subsurface.conductivity_profile": "model.conductivity_profile",
    }

    config_out = _convert_to_wflow_v1(
        config=config,
        wflow_vars=WFLOW_NAMES,
        states_vars=WFLOW_STATES_NAMES,
        model_options=model_options,
        cross_options=cross_options,
        input_options=input_options,
        input_variables=input_variables,
        additional_variables=additional_variables,
        logger=logger,
    )

    return config_out


def convert_to_wflow_v1_sediment(
    config: Dict,
    logger: logging.Logger = logger,
) -> Dict:
    """Convert the config to Wflow v1 format for sediment.

    Parameters
    ----------
    config: dict
        The config to convert.
    logger: logging.Logger, optional
        The logger to use, by default logger.

    Returns
    -------
    config_out: dict
        The converted config.
    """
    additional_variables = {
        "vertical.soilloss": "soil_erosion__mass_flow_rate",
        "lateral.river.SSconc": "river_water_sediment~suspended__mass_concentration",
        "lateral.river.Bedconc": "river_water_sediment~bedload__mass_concentration",
        "lateral.river.outsed": "land_surface_water_sediment__mass_flow_rate",
        "lateral.land.inlandclay": "land_surface_water_clay~to-river__mass_flow_rate",
        "lateral.land.inlandsilt": "land_surface_water_silt~to-river__mass_flow_rate",
        "lateral.land.inlandsand": "land_surface_water_sand~to-river__mass_flow_rate",
        "lateral.land.inlandsagg": "land_surface_water_aggregates~small~to-river__mass_flow_rate",  # noqa: E501
        "lateral.land.inlandlagg": "land_surface_water_aggregates~large~to-river__mass_flow_rate",  # noqa: E501
    }

    # Options in model section that were renamed
    model_options = {
        "reinit": "cold_start__flag",
        "sizeinmetres": "cell_length_in_meter__flag",
        "runrivermodel": "run_river_model__flag",
        "doreservoir": "reservoir__flag",
        "dolake": "lake__flag",
        "rainerosmethod": "rainfall_erosion",
        "landtransportmethod": "land_transport",
        "rivtransportmethod": "river_transport",
    }

    # Options in input section that were renamed
    input_options = {
        "ldd": "basin__local_drain_direction",
        "river_location": "river_location__mask",
        "subcatchment": "subbasin_location__count",
    }

    # variables that were moved to input rather than input.static
    input_variables = [
        "vertical.lakeareas",
        "lateral.river.lakelocs",
        "vertical.resareas",
        "lateral.river.reslocs",
    ]

    config_out = _convert_to_wflow_v1(
        config=config,
        wflow_vars=WFLOW_SEDIMENT_NAMES,
        states_vars=WFLOW_SEDIMENT_STATES_NAMES,
        model_options=model_options,
        input_options=input_options,
        input_variables=input_variables,
        additional_variables=additional_variables,
        logger=logger,
    )

    return config_out
