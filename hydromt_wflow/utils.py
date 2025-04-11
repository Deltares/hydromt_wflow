"""Some utilities from the Wflow plugin."""

import logging
from os.path import abspath, dirname, join
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import xarray as xr
from hydromt.io import open_timeseries_from_table
from hydromt.vector import GeoDataArray
from hydromt.workflows.grid import grid_from_constant

from .naming import WFLOW_NAMES, WFLOW_STATES_NAMES

logger = logging.getLogger(__name__)

DATADIR = join(dirname(abspath(__file__)), "data")

__all__ = [
    "read_csv_results",
    "get_config",
    "get_grid_from_config",
    "convert_to_wflow_v1",
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
                        f"{config['input'].get('subcatchment_location__count')}"
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
                            f"{config['input'].get('subcatchment_location__count')}"
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
    var_name: str,
    config: Dict = {},
    grid: xr.Dataset | None = None,
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
        f"input.{var_name}",
        config=config,
        fallback=None,
        root=root,
        abs_path=abs_path,
    )
    if var is None:
        # try with input.static
        var = get_config(
            f"input.static.{var_name}",
            config=config,
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
            f"input.cyclic.{var_name}",
            config=config,
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
            var_name = get_config("netcdf.variable.name", config=var)
            scale = var.get("scale", 1.0)
            offset = var.get("offset", 0.0)
            # apply scale and offset
            if var_name not in grid:
                raise ValueError(f"grid variable {var_name} not found in staticmaps.")
            da = grid[var_name] * scale + offset

    return da


def mask_raster_from_layer(
    data: Union[xr.Dataset, xr.DataArray], mask: xr.DataArray
) -> Union[xr.Dataset, xr.DataArray]:
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


def convert_to_wflow_v1(
    config: Dict,
    logger: Optional[logging.Logger] = logger,
) -> Dict:
    """Convert the config to Wflow v1 format.

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
    WFLOW_CONVERSION = {v["wflow_v0"]: v["wflow_v1"] for v in WFLOW_NAMES.values()}
    for k, v in WFLOW_STATES_NAMES.items():
        WFLOW_CONVERSION[v["wflow_v0"]] = v["wflow_v1"]
    # Add a few extra output variables that are supported by the conversion
    additional_variables = {
        "vertical.interception": "vegetation_canopy_water__interception_volume_flux",
        "vertical.actevap": "land_surface__evapotranspiration_volume_flux",
        "vertical.actinfilt": "soil_water__infiltration_volume_flux",
        "vertical.excesswatersoil": "land.soil.variables.excesswatersoil",
        "vertical.excesswaterpath": "land.soil.variables.excesswaterpath",
        "vertical.exfiltustore": "land.soil.variables.exfiltustore",
        "vertical.exfiltsatwater": "land.soil.variables.exfiltsatwater",
        "vertical.recharge": "soil_water_sat-zone_top__net_recharge_volume_flux",
        "vertical.vwc_percroot": "soil_water_root-zone__volume_percentage",
        "lateral.land.q_av": "land_surface_water__volume_flow_rate",
        "lateral.land.h_av": "land_surface_water__depth",
        "lateral.land.to_river": "routing.overland_flow.variables.to_river",
        "lateral.subsurface.to_river": "subsurface_water~to-river__volume_flow_rate",
        "lateral.river.q_av": "river_water__volume_flow_rate",
        "lateral.river.h_av": "river_water__depth",
        "lateral.river.volume": "routing.river_flow.variables.storage",
        "lateral.river.inwater": "river_water_inflow~lateral__volume_flow_rate",
        "lateral.river.floodplain.volume": "routing.river_flow.floodplain.variables.storage",  # noqa: E501
        "lateral.river.reservoir.volume": "reservoir_water__instantaneous_volume",
        "lateral.river.reservoir.totaloutflow": "reservoir_water~outgoing__volume_flow_rate",  # noqa: E501
        "lateral.river.lake.storage": "routing.river_flow.boundary_conditions.lake.variables.storage",  # noqa: E501
        "lateral.river.lake.totaloutflow": "lake_water~outgoing__volume_flow_rate",
    }
    WFLOW_CONVERSION.update(additional_variables)

    # Logging function for the other non supported variables
    def warn_str(wflow_var, output_type):
        logger.warning(
            f"Output variable {wflow_var} not supported for the conversion. "
            f"Skipping from {output_type} output."
        )

    # Initialize the output config
    logger.info("Converting config to Wflow v1 format")
    logger.info("Converting config general, time and model sections")
    config_out = dict()

    # Start with the general section
    for key in ["dir_input", "dir_output"]:
        value = get_config(key, config=config, fallback=None)
        if value is not None:
            config_out[key] = value

    # Time section
    config_out["time"] = {
        "calendar": get_config(
            "calendar", config=config, fallback="proleptic_gregorian"
        ),
        "starttime": get_config(
            "starttime", config=config, fallback="2010-02-01T00:00:00"
        ),
        "endtime": get_config("endtime", config=config, fallback="2010-02-10T00:00:00"),
        "time_units": get_config(
            "time_units", config=config, fallback="days since 1900-01-01 00:00:00"
        ),
        "timestepsecs": get_config("timestepsecs", config=config, fallback=86400),
    }

    # Logging
    config_out["logging"] = {
        "loglevel": get_config("loglevel", config=config, fallback="info")
    }

    # Model section
    config_out["model"] = config["model"]
    if "masswasting" in config_out["model"]:
        config_out["model"]["gravitational_snow_transport"] = config_out["model"].pop(
            "masswasting"
        )

    # State
    logger.info("Converting config state section")
    config_out["state"] = {
        "path_input": get_config(
            "state.path_input", config=config, fallback="instate/instates.nc"
        ),
        "path_output": get_config(
            "state.path_output", config=config, fallback="outstate/outstates.nc"
        ),
    }
    # Go through the states variables
    config_out["state.variables"] = {}
    for key, variables in WFLOW_STATES_NAMES.items():
        name = get_config(
            f"state.{variables['wflow_v0']}", config=config, fallback=None
        )
        if name is not None and variables["wflow_v1"] is not None:
            config_out["state.variables"][variables["wflow_v1"]] = name

    # Input section
    logger.info("Converting config input section")
    input_name_updated = {
        "ldd": "local_drain_direction",
        "river_location": "river_location__mask",
        "subcatchment": "subcatchment_location__count",
    }
    # variables that were moved to input rather than input.static
    input_variables = [
        "lateral.river.lake.areas",
        "lateral.river.lake.locs",
        "lateral.river.reservoir.areas",
        "lateral.river.reservoir.locs",
        "lateral.subsurface.conductivity_profile",
    ]
    cyclic_variables = []
    forcing_variables = []

    config_out["input"] = {}
    for key, name in config["input"].items():
        if key in input_name_updated.keys():
            config_out["input"][input_name_updated[key]] = name
        elif key == "forcing":
            forcing_variables = name
        elif key == "cyclic":
            cyclic_variables = name
        elif key not in ["vertical", "lateral"]:  # variables are done separately
            config_out["input"][key] = name

    # Go through the input variables
    config_out["input.forcing"] = {}
    config_out["input.cyclic"] = {}
    config_out["input.static"] = {}
    for key, variables in WFLOW_NAMES.items():
        name = get_config(
            f"input.{variables['wflow_v0']}", config=config, fallback=None
        )
        if variables["wflow_v0"] == "vertical.g_ttm" and name is None:
            # this change is probably too recent for most models
            name = get_config("input.vertical.g_tt", config=config, fallback=None)
        if name is not None and variables["wflow_v1"] is not None:
            if variables["wflow_v0"] in input_name_updated.keys():
                continue
            elif variables["wflow_v0"] in input_variables:
                config_out["input"][variables["wflow_v1"]] = name
            elif variables["wflow_v0"] in forcing_variables:
                config_out["input.forcing"][variables["wflow_v1"]] = name
            elif variables["wflow_v0"] in cyclic_variables:
                config_out["input.cyclic"][variables["wflow_v1"]] = name
            else:
                config_out["input.static"][variables["wflow_v1"]] = name

    # Output netcdf_grid section
    logger.info("Converting config output sections")
    if get_config("output", config=config, fallback=None) is not None:
        config_out["output.netcdf_grid"] = {
            "path": get_config("output.path", config=config, fallback="output.nc"),
            "compressionlevel": get_config(
                "output.compressionlevel", config=config, fallback=1
            ),
        }
        config_out["output.netcdf_grid.variables"] = {}
        for key, value in config["output"].items():
            if key in ["path", "compressionlevel"]:
                continue
            else:
                # vertical
                if key == "vertical":
                    for var, var_name in value.items():
                        wflow_var = f"vertical.{var}"
                        if wflow_var in WFLOW_CONVERSION.keys():
                            config_out["output.netcdf_grid.variables"][
                                WFLOW_CONVERSION[wflow_var]
                            ] = var_name
                        else:
                            warn_str(wflow_var, "netcdf_grid")
                # lateral
                elif key == "lateral":
                    # land
                    if "land" in value.keys():
                        for var, var_name in value["land"].items():
                            wflow_var = f"lateral.land.{var}"
                            if wflow_var in WFLOW_CONVERSION.keys():
                                config_out["output.netcdf_grid.variables"][
                                    WFLOW_CONVERSION[wflow_var]
                                ] = var_name
                            else:
                                warn_str(wflow_var, "netcdf_grid")
                    # subsurface
                    if "subsurface" in value.keys():
                        for var, var_name in value["subsurface"].items():
                            wflow_var = f"lateral.subsurface.{var}"
                            if wflow_var in WFLOW_CONVERSION.keys():
                                config_out["output.netcdf_grid.variables"][
                                    WFLOW_CONVERSION[wflow_var]
                                ] = var_name
                            else:
                                warn_str(wflow_var, "netcdf_grid")
                    # river (do not support reservoir and lake outputs)
                    if "river" in value.keys():
                        for var, var_name in value["river"].items():
                            if var in ["reservoir", "floodplain", "lake"]:
                                for var2, var_name2 in value["river"][var].items():
                                    wflow_var = f"lateral.river.{var}.{var2}"
                                    if wflow_var in WFLOW_CONVERSION.keys():
                                        config_out["output.netcdf_grid.variables"][
                                            WFLOW_CONVERSION[wflow_var]
                                        ] = var_name2
                                    else:
                                        warn_str(wflow_var, "netcdf_grid")
                            else:
                                wflow_var = f"lateral.river.{var}"
                                if wflow_var in WFLOW_CONVERSION.keys():
                                    config_out["output.netcdf_grid.variables"][
                                        WFLOW_CONVERSION[wflow_var]
                                    ] = var_name
                                else:
                                    warn_str(wflow_var, "netcdf_grid")

    # Output netcdf_scalar section
    if get_config("netcdf", config=config, fallback=None) is not None:
        config_out["output.netcdf_scalar"] = {
            "path": get_config(
                "netcdf.path", config=config, fallback="output_scalar.nc"
            ),
        }
        config_out["output.netcdf_scalar.variable"] = []
        nc_scalar_vars = get_config("netcdf.variable", config=config, fallback=[])
        for nc_scalar in nc_scalar_vars:
            if nc_scalar["parameter"] in WFLOW_CONVERSION.keys():
                nc_scalar["parameter"] = WFLOW_CONVERSION[nc_scalar["parameter"]]
                if "map" in nc_scalar and nc_scalar["map"] in input_name_updated.keys():
                    nc_scalar["map"] = input_name_updated[nc_scalar["map"]]
                config_out["output.netcdf_scalar.variable"].append(nc_scalar)
            else:
                warn_str(nc_scalar["parameter"], "netcdf_scalar")

    # Output csv section
    if get_config("csv", config=config, fallback=None) is not None:
        config_out["output.csv"] = {
            "path": get_config("csv.path", config=config, fallback="output.csv"),
        }
        config_out["output.csv.column"] = []
        csv_vars = get_config("csv.column", config=config, fallback=[])
        for csv_var in csv_vars:
            if csv_var["parameter"] in WFLOW_CONVERSION.keys():
                csv_var["parameter"] = WFLOW_CONVERSION[csv_var["parameter"]]
                if "map" in csv_var and csv_var["map"] in input_name_updated.keys():
                    csv_var["map"] = input_name_updated[csv_var["map"]]
                config_out["output.csv.column"].append(csv_var)
            else:
                warn_str(csv_var["parameter"], "csv")

    return config_out
