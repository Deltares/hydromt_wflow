"""Workflow for wflow model states."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from hydromt.raster import full_like
from hydromt.workflows.grid import grid_from_constant

from ..utils import get_grid_from_config

__all__ = ["prepare_cold_states"]


def prepare_cold_states(
    ds_like: xr.Dataset,
    config: dict,
    timestamp: str = None,
    mask_name_land: str = "wflow_subcatch",
    mask_name_river: str = "wflow_river",
) -> Tuple[xr.Dataset, Dict[str, str]]:
    """
    Prepare cold states for Wflow.

    Compute cold states variables:

    * **satwaterdepth**: saturated store [mm]
    * **snow**: snow storage [mm]
    * **tsoil**: top soil temperature [°C]
    * **ustorelayerdepth**: amount of water in the unsaturated store, per layer [mm]
    * **snowwater**: liquid water content in the snow pack [mm]
    * **canopystorage**: canopy storage [mm]
    * **q_river**: river discharge [m3/s]
    * **h_river**: river water level [m]
    * **h_av_river**: river average water level [m]
    * **ssf**: subsurface flow [m3/d]
    * **h_land**: land water level [m]
    * **h_av_land**: land average water level[m]
    * **q_land** or **qx_land**+**qy_land**: overland flow for kinwave [m3/s] or
      overland flow in x/y directions for local-inertial [m3/s]

    If lakes, also adds:

    * **waterlevel_lake**: lake water level [m]

    If reservoirs, also adds:

    * **volume_reservoir**: reservoir volume [m3]

    If glaciers, also adds:

    * **glacierstore**: water within the glacier [mm]

    If paddy, also adds:

    * **h_paddy**: water on the paddy fields [mm]

    Parameters
    ----------
    ds_like : xr.Dataset
        Dataset containing the staticmaps grid and variables to prepare some of the
        states.

        * Required variables: wflow_subcatch, wflow_river

        * Other required variables (exact name from the wflow config): c, soilthickness,
            theta_s, theta_r, kv_0, f, slope, ksathorfrac

        * Optional variables (exact name from the wflow config): reservoir.locs,
            glacierstore, reservoir.maxvolume, reservoir.targetfullfrac,
            lake.waterlevel
    config : dict
        Wflow configuration dictionary.
    timestamp : str, optional
        Timestamp of the cold states. By default uses the starttime
        from the config.
    mask_name_land : str, optional
        Name of the land mask variable in the ds_like dataset. By default
        wflow_subcatch.
    mask_name_river : str, optional
        Name of the river mask variable in the ds_like dataset. By default
        wflow_river.

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
        "state.variables.soil_water_sat-zone__depth": "satwaterdepth",
        "state.variables.snowpack~dry__leq-depth": "snow",
        "state.variables.soil_surface__temperature": "tsoil",
        "state.variables.soil_layer_water_unsat-zone__depth": "ustorelayerdepth",
        "state.variables.snowpack~liquid__depth": "snowwater",
        "state.variables.vegetation_canopy_water__depth": "canopystorage",
        "state.variables.subsurface_water__volume_flow_rate": "ssf",
        "state.variables.land_surface_water__instantaneous_depth": "h_land",
        "state.variables.river_water__instantaneous_volume_flow_rate": "q_river",
        "state.variables.river_water__instantaneous_depth": "h_river",
    }

    # Map with constant values or zeros for basin
    zeromap = ["tsoil", "snow", "snowwater", "canopystorage", "h_land", "h_av_land"]
    land_routing = config["model"].get("land_routing", "kinematic-wave")
    if land_routing == "local-inertial":
        zeromap.extend(["qx_land", "qy_land"])
        states_config[
            "state.variables.land_surface_water__x_component_of_instantaneous_volume_flow_rate"
        ] = "qx_land"
        states_config[
            "state.variables.land_surface_water__y_component_of_instantaneous_volume_flow_rate"
        ] = "qy_land"
    else:
        zeromap.extend(["q_land"])
        states_config[
            "state.variables.land_surface_water__instantaneous_volume_flow_rate"
        ] = "q_land"

    for var in zeromap:
        if var == "tsoil":
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

    # ustorelayerdepth (zero per layer)
    # layers are based on c parameter
    c = get_grid_from_config(
        "soil_layer_water__brooks-corey_exponent",
        config=config,
        grid=ds_like,
    )
    usld = full_like(c, nodata=nodata)
    for sl in usld["layer"]:
        usld.loc[dict(layer=sl)] = xr.where(ds_like[mask_name_land], 0.0, nodata)
    ds_out["ustorelayerdepth"] = usld

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
        "subsurface_water__horizontal-to-vertical_saturated_hydraulic_conductivity_ratio",
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

    # satwaterdepth
    swd = 0.85 * st * (ts - tr)
    swd = swd.where(ds_like[mask_name_land] > 0, nodata)
    swd.raster.set_nodata(nodata)
    ds_out["satwaterdepth"] = swd

    # ssf
    zi = np.maximum(0.0, st - swd / (ts - tr))
    kh0 = ksh * ksv * 0.001 * (86400 / timestepsecs)
    ssf = (kh0 * np.maximum(0.00001, sl) / (f * 1000)) * (
        np.exp(-f * 1000 * zi * 0.001)
    ) - (np.exp(-f * 1000 * st) * np.sqrt(ds_like.raster.area_grid()))
    ssf = ssf.where(ds_like[mask_name_land] > 0, nodata)
    ssf.raster.set_nodata(nodata)
    ds_out["ssf"] = ssf

    # River
    zeromap_riv = ["q_river", "h_river", "h_av_river"]
    # 1D floodplain
    if config["model"].get("floodplain_1d__flag", False):
        zeromap_riv.extend(["q_floodplain", "h_floodplain"])
        states_config[
            "state.variables.floodplain_water__instantaneous_volume_flow_rate"
        ] = "q_floodplain"
        states_config["state.variables.floodplain_water__instantaneous_depth"] = (
            "h_floodplain"
        )
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
        tff = get_grid_from_config(
            "reservoir_water~full-target__volume_fraction",
            config=config,
            grid=ds_like,
        )
        mv = get_grid_from_config(
            "reservoir_water__max_volume",
            config=config,
            grid=ds_like,
        )
        locs = get_grid_from_config(
            "reservoir_location__count",
            config=config,
            grid=ds_like,
        )
        resvol = tff * mv
        resvol = xr.where(locs > 0, resvol, nodata)
        resvol.raster.set_nodata(nodata)
        ds_out["volume_reservoir"] = resvol

        states_config["state.variables.volume_reservoir"] = "volume_reservoir"

    # lake
    if config["model"].get("lake__flag", False):
        ll = get_grid_from_config(
            "lake_water_surface__initial_elevation",
            config=config,
            grid=ds_like,
        )
        ll = ll.where(ll != ll.raster.nodata, nodata)
        ll.raster.set_nodata(nodata)
        ds_out["waterlevel_lake"] = ll

        states_config["state.variables.lake_water_surface__instantaneous_elevation"] = (
            "waterlevel_lake"
        )

    # glacier
    if config["model"].get("glacier__flag", False):
        gs_vn = get_grid_from_config(
            "glacier_ice__initial_leq-depth",
            config=config,
            grid=ds_like,
        )
        if gs_vn.name in ds_like:
            ds_out["glacierstore"] = ds_like[gs_vn.name]
        else:
            glacstore = grid_from_constant(
                ds_like,
                value=5500.0,
                name="glacierstore",
                nodata=nodata,
                dtype=dtype,
                mask=mask_name_land,
            )
            ds_out["glacierstore"] = glacstore

        states_config["state.variables.glacier_ice__leq-depth"] = "glacierstore"

    # paddy
    if config["model"].get("water_demand.paddy__flag", False):
        h_paddy = grid_from_constant(
            ds_like,
            value=0.0,
            name="h_paddy",
            nodata=nodata,
            dtype=dtype,
            mask=mask_name_land,
        )
        ds_out["h_paddy"] = h_paddy

        states_config["state.variables.land_surface_water~paddy__depth"] = "h_paddy"

    # Add time dimension
    ds_out = ds_out.expand_dims(dim=dict(time=[timestamp]))

    return ds_out, states_config
