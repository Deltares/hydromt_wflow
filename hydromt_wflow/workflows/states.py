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
        starttime = pd.to_datetime(config.get("starttime", None))
        timestamp = starttime  # - pd.Timedelta(seconds=timestepsecs)
        if timestamp is None:
            raise ValueError("No timestamp provided and no starttime in config.")
    else:
        timestamp = pd.to_datetime(timestamp)
    timestepsecs = config.get("timestepsecs", 86400)

    # Create empty dataset
    ds_out = xr.Dataset()

    # Base output config dict for states
    states_config = {
        "state.vertical.satwaterdepth": "satwaterdepth",
        "state.vertical.snow": "snow",
        "state.vertical.tsoil": "tsoil",
        "state.vertical.ustorelayerdepth": "ustorelayerdepth",
        "state.vertical.snowwater": "snowwater",
        "state.vertical.canopystorage": "canopystorage",
        "state.lateral.subsurface.ssf": "ssf",
        "state.lateral.land.h": "h_land",
        "state.lateral.land.h_av": "h_av_land",
        "state.lateral.river.q": "q_river",
        "state.lateral.river.h": "h_river",
        "state.lateral.river.h_av": "h_av_river",
    }

    # Map with constant values or zeros for basin
    zeromap = ["tsoil", "snow", "snowwater", "canopystorage", "h_land", "h_av_land"]
    land_routing = config["model"].get("land_routing", "kinematic-wave")
    if land_routing == "local-inertial":
        zeromap.extend(["qx_land", "qy_land"])
        states_config["state.lateral.land.qx"] = "qx_land"
        states_config["state.lateral.land.qy"] = "qy_land"
    else:
        zeromap.extend(["q_land"])
        states_config["state.lateral.land.q"] = "q_land"

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
            mask_name="wflow_subcatch",
        )
        ds_out[var] = da_param

    # ustorelayerdepth (zero per layer)
    # layers are based on c parameter
    c = get_grid_from_config(
        "input.vertical.c", config=config, grid=ds_like, fallback="c"
    )
    usld = full_like(c, nodata=nodata)
    for sl in usld["layer"]:
        usld.loc[dict(layer=sl)] = xr.where(ds_like["wflow_subcatch"], 0.0, nodata)
    ds_out["ustorelayerdepth"] = usld

    # Compute other soil states
    # Get required variables from config
    st = get_grid_from_config(
        "input.vertical.soilthickness",
        config=config,
        grid=ds_like,
        fallback="SoilThickness",
    )
    ts = get_grid_from_config(
        "input.vertical.theta_s", config=config, grid=ds_like, fallback="thetaS"
    )
    tr = get_grid_from_config(
        "input.vertical.theta_r", config=config, grid=ds_like, fallback="thetaR"
    )
    ksh = get_grid_from_config(
        "input.lateral.subsurface.ksathorfrac",
        config=config,
        grid=ds_like,
        fallback="KsatHorFrac",
    )
    ksv = get_grid_from_config(
        "input.vertical.kv_0", config=config, grid=ds_like, fallback="KsatVer"
    )
    f = get_grid_from_config(
        "input.vertical.f", config=config, grid=ds_like, fallback="f"
    )
    sl = get_grid_from_config(
        "input.vertical.slope", config=config, grid=ds_like, fallback="Slope"
    )

    # satwaterdepth
    swd = 0.85 * st * (ts - tr)
    swd = swd.where(ds_like["wflow_subcatch"] > 0, nodata)
    swd.raster.set_nodata(nodata)
    ds_out["satwaterdepth"] = swd

    # ssf
    zi = np.maximum(0.0, st - swd / (ts - tr))
    kh0 = ksh * ksv * 0.001 * (86400 / timestepsecs)
    ssf = (kh0 * np.maximum(0.00001, sl) / (f * 1000)) * (
        np.exp(-f * 1000 * zi * 0.001)
    ) - (np.exp(-f * 1000 * st) * np.sqrt(ds_like.raster.area_grid()))
    ssf = ssf.where(ds_like["wflow_subcatch"] > 0, nodata)
    ssf.raster.set_nodata(nodata)
    ds_out["ssf"] = ssf

    # River
    zeromap_riv = ["q_river", "h_river", "h_av_river"]
    # 1D floodplain
    if config["model"].get("floodplain_1d", False):
        zeromap_riv.extend(["q_floodplain", "h_floodplain"])
        states_config["state.lateral.floodplain.q"] = "q_floodplain"
        states_config["state.lateral.floodplain.h"] = "h_floodplain"
    for var in zeromap_riv:
        value = 0.0
        da_param = grid_from_constant(
            ds_like,
            constant=value,
            name=var,
            dtype=dtype,
            nodata=nodata,
            mask_name="wflow_river",
        )
        da_param = da_param.rename(var)
        ds_out[var] = da_param

    # reservoir
    if config["model"].get("reservoirs", False):
        tff = get_grid_from_config(
            "input.lateral.river.reservoir.targetfullfrac",
            config=config,
            grid=ds_like,
            fallback="ResTargetFullFrac",
        )
        mv = get_grid_from_config(
            "input.lateral.river.reservoir.maxvolume",
            config=config,
            grid=ds_like,
            fallback="ResMaxVolume",
        )
        locs = get_grid_from_config(
            "input.lateral.river.reservoir.locs",
            config=config,
            grid=ds_like,
            fallback="wflow_reservoirlocs",
        )
        resvol = tff * mv
        resvol = xr.where(locs > 0, resvol, nodata)
        resvol.raster.set_nodata(nodata)
        ds_out["volume_reservoir"] = resvol

        states_config["state.lateral.river.reservoir.volume"] = "volume_reservoir"

    # lake
    if config["model"].get("lakes", False):
        ll = get_grid_from_config(
            "input.lateral.river.lake.waterlevel",
            config=config,
            grid=ds_like,
            fallback="LakeAvgLevel",
        )
        ll = ll.where(ll != ll.raster.nodata, nodata)
        ll.raster.set_nodata(nodata)
        ds_out["waterlevel_lake"] = ll

        states_config["state.lateral.river.lake.waterlevel"] = "waterlevel_lake"

    # glacier
    if config["model"].get("glacier", False):
        gs_vn = config["input"]["vertical"].get("glacierstore", "wflow_glacierstore")
        if gs_vn in ds_like:
            ds_out["glacierstore"] = ds_like[gs_vn]
        else:
            glacstore = grid_from_constant(
                ds_like,
                value=5500.0,
                name="glacierstore",
                nodata=nodata,
                dtype=dtype,
                mask="wflow_subcatch",
            )
            ds_out["glacierstore"] = glacstore

        states_config["state.vertical.glacierstore"] = "glacierstore"

    # Add time dimension
    ds_out = ds_out.expand_dims(dim=dict(time=[timestamp]))

    return ds_out, states_config
