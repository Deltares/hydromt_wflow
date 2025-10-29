"""Workflow for wflow model states."""

import numpy as np
import pandas as pd
import xarray as xr
from hydromt.gis import full_like
from hydromt.model.processes.grid import grid_from_constant

from hydromt_wflow.utils import get_grid_from_config

__all__ = ["prepare_cold_states"]


def prepare_cold_states(
    ds_like: xr.Dataset,
    config: dict,
    timestamp: str = None,
    mask_name_land: str = "subcatchment",
    mask_name_river: str = "river_mask",
) -> tuple[xr.Dataset, dict[str, str]]:
    """
    Prepare cold states for Wflow.

    Compute cold states variables:

    * **soil_saturated_depth**: saturated store [mm]
    * **snow_leq_depth**: snow storage [mm]
    * **soil_temp**: top soil temperature [°C]
    * **soil_unsaturated_depth**: amount of water in the unsaturated store, per layer
      [mm]
    * **snow_water_depth**: liquid water content in the snow pack [mm]
    * **vegetation_water_depth**: canopy storage [mm]
    * **river_instantaneous_q**: river discharge [m3/s]
    * **river_h**: river water level [m]
    * **subsurface_q**: subsurface flow [m3/d]
    * **land_h**: land water level [m]
    * **land_instantaneous_q** or **land_instantaneous_qx**+**land_instantaneous_qy**:
      overland flow for kinwave [m3/s] or overland flow in x/y directions for
      local_inertial [m3/s]

    If reservoirs, also adds:

    * **reservoir_water_level**: reservoir water level [m]

    If glaciers, also adds:

    * **glacier_leq_depth**: water within the glacier [mm]

    If paddy, also adds:

    * **demand_paddy_h**: water on the paddy fields [mm]

    Parameters
    ----------
    ds_like : xr.Dataset
        Dataset containing the staticmaps grid and variables to prepare some of the
        states.

        * Required variables: `mask_name_land`, `mask_name_river`

        * Other required variables (exact name will be read from the wflow config):
            soil_brooks_corey_c, soilthickness,
            theta_s, theta_r, ksat_vertical, f, slope, subsurface_ksat_horizontal_ratio

        * Optional variables (exact name from the wflow config):
            glacierstore, reservoir_water_surface__initial_elevation
    config : dict
        Wflow configuration dictionary.
    timestamp : str, optional
        Timestamp of the cold states. By default uses the starttime
        from the config.
    mask_name_land : str, optional
        Name of the land mask variable in the ds_like dataset. By default
        subcatchment.
    mask_name_river : str, optional
        Name of the river mask variable in the ds_like dataset. By default
        river_mask.

    Returns
    -------
    xr.Dataset
        Dataset containing the cold states.
    dict
        Config dictionary with the cold states variable names.
    """
    # Defaults
    nodata = -9999.0
    dtype = "float32"

    # Times
    if timestamp is None:
        # states time = starttime for Wflow.jl > 0.7.0)
        if "time" in config:
            starttime = pd.to_datetime(config["time"].get("starttime", None))
            timestamp = starttime  # - pd.Timedelta(seconds=timestepsecs)
        if timestamp is None:
            raise ValueError("No timestamp provided and no starttime in config.")
    else:
        timestamp = pd.to_datetime(timestamp)
    timestepsecs = config.get("time.timestepsecs", 86400)

    # Create empty dataset
    ds_out = xr.Dataset()

    # Base output config dict for states
    states_config = {
        "state.variables.soil_water_saturated_zone__depth": "soil_saturated_depth",
        "state.variables.snowpack_dry_snow__leq_depth": "snow_leq_depth",
        "state.variables.soil_surface__temperature": "soil_temp",
        "state.variables.soil_layer_water_unsaturated_zone__depth": "soil_unsaturated_depth",  # noqa: E501
        "state.variables.snowpack_liquid_water__depth": "snow_water_depth",
        "state.variables.vegetation_canopy_water__depth": "vegetation_water_depth",
        "state.variables.subsurface_water__volume_flow_rate": "subsurface_q",
        "state.variables.land_surface_water__depth": "land_h",
        "state.variables.river_water__instantaneous_volume_flow_rate": "river_instantaneous_q",  # noqa: E501
        "state.variables.river_water__depth": "river_h",
    }

    # Map with constant values or zeros for basin
    zeromap = [
        "soil_temp",
        "snow_leq_depth",
        "snow_water_depth",
        "vegetation_water_depth",
        "land_h",
    ]
    land_routing = config["model"].get("land_routing", "kinematic_wave")
    if land_routing == "local_inertial":
        zeromap.extend(["land_instantaneous_qx", "land_instantaneous_qy"])
        states_config[
            "state.variables.land_surface_water__x_component_of_instantaneous_volume_flow_rate"
        ] = "land_instantaneous_qx"
        states_config[
            "state.variables.land_surface_water__y_component_of_instantaneous_volume_flow_rate"
        ] = "land_instantaneous_qy"
    else:
        zeromap.extend(["land_instantaneous_q"])
        states_config[
            "state.variables.land_surface_water__instantaneous_volume_flow_rate"
        ] = "land_instantaneous_q"

    for var in zeromap:
        if var == "soil_temp":
            value = 10.0
        else:
            value = 0.0
        da_param = grid_from_constant(
            ds_like,
            constant=value,
            name=var,
            dtype=dtype,
            nodata=nodata,
            mask_name=mask_name_land,
        )
        ds_out[var] = da_param

    # soil_unsaturated_depth (zero per layer)
    # layers are based on brooks_corey_c parameter
    c = get_grid_from_config(
        "soil_layer_water__brooks_corey_exponent",
        config=config,
        grid=ds_like,
    )
    usld = full_like(c, nodata=nodata)
    for sl in usld["layer"]:
        usld.loc[dict(layer=sl)] = xr.where(ds_like[mask_name_land], 0.0, nodata)
    ds_out["soil_unsaturated_depth"] = usld

    # Compute other soil states
    # Get required variables from config
    st = get_grid_from_config(
        "soil__thickness",
        config=config,
        grid=ds_like,
    )
    ts = get_grid_from_config(
        "soil_water__saturated_volume_fraction",
        config=config,
        grid=ds_like,
    )
    tr = get_grid_from_config(
        "soil_water__residual_volume_fraction",
        config=config,
        grid=ds_like,
    )
    ksh = get_grid_from_config(
        "subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio",
        config=config,
        grid=ds_like,
    )
    ksv = get_grid_from_config(
        "soil_surface_water__vertical_saturated_hydraulic_conductivity",
        config=config,
        grid=ds_like,
    )
    f = get_grid_from_config(
        "soil_water__vertical_saturated_hydraulic_conductivity_scale_parameter",
        config=config,
        grid=ds_like,
    )
    sl = get_grid_from_config(
        "land_surface__slope",
        config=config,
        grid=ds_like,
    )

    # soil_saturated_depth
    swd = 0.85 * st * (ts - tr)
    swd = swd.where(ds_like[mask_name_land] > 0, nodata)
    swd.raster.set_nodata(nodata)
    ds_out["soil_saturated_depth"] = swd

    # subsurface_q
    zi = np.maximum(0.0, st - swd / (ts - tr))
    kh0 = ksh * ksv * 0.001 * (86400 / timestepsecs)
    ssf = (kh0 * np.maximum(0.00001, sl) / (f * 1000)) * (
        np.exp(-f * 1000 * zi * 0.001)
    ) - (np.exp(-f * 1000 * st) * np.sqrt(ds_like.raster.area_grid()))
    ssf = ssf.where(ds_like[mask_name_land] > 0, nodata)
    ssf.raster.set_nodata(nodata)
    ds_out["subsurface_q"] = ssf

    # River
    zeromap_riv = ["river_instantaneous_q", "river_h"]
    # 1D floodplain
    if config["model"].get("floodplain_1d__flag", False):
        zeromap_riv.extend(["floodplain_instantaneous_q", "floodplain_h"])
        states_config[
            "state.variables.floodplain_water__instantaneous_volume_flow_rate"
        ] = "floodplain_instantaneous_q"
        states_config["state.variables.floodplain_water__depth"] = "floodplain_h"
    for var in zeromap_riv:
        value = 0.0
        da_param = grid_from_constant(
            ds_like,
            constant=value,
            name=var,
            dtype=dtype,
            nodata=nodata,
            mask_name=mask_name_river,
        )
        da_param = da_param.rename(var)
        ds_out[var] = da_param

    # reservoir
    if config["model"].get("reservoir__flag", False):
        rl = get_grid_from_config(
            "reservoir_water_surface__initial_elevation",
            config=config,
            grid=ds_like,
        )
        rl = rl.where(rl != rl.raster.nodata, nodata)
        rl.raster.set_nodata(nodata)
        ds_out["reservoir_water_level"] = rl

        states_config["state.variables.reservoir_water_surface__elevation"] = (
            "reservoir_water_level"
        )

    # glacier
    if config["model"].get("glacier__flag", False):
        gs_vn = get_grid_from_config(
            "glacier_ice__initial_leq_depth",
            config=config,
            grid=ds_like,
        )
        if gs_vn.name in ds_like:
            ds_out["glacier_leq_depth"] = ds_like[gs_vn.name]
        else:
            glacstore = grid_from_constant(
                ds_like,
                value=5500.0,
                name="glacier_leq_depth",
                nodata=nodata,
                dtype=dtype,
                mask=mask_name_land,
            )
            ds_out["glacier_leq_depth"] = glacstore

        states_config["state.variables.glacier_ice__leq_depth"] = "glacier_leq_depth"

    # paddy
    if config["model"].get("water_demand.paddy__flag", False):
        h_paddy = grid_from_constant(
            ds_like,
            value=0.0,
            name="demand_paddy_h",
            nodata=nodata,
            dtype=dtype,
            mask=mask_name_land,
        )
        ds_out["demand_paddy_h"] = h_paddy

        states_config["state.variables.paddy_surface_water__depth"] = "demand_paddy_h"

    # Add time dimension
    ds_out = ds_out.expand_dims(dim=dict(time=[timestamp]))

    return ds_out, states_config
