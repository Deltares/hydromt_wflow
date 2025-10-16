"""Reservoir workflows for Wflow plugin."""

import json
import logging
from os.path import join
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr

from hydromt_wflow import utils

logger = logging.getLogger(f"hydromt.{__name__}")


__all__ = [
    "reservoir_id_maps",
    "reservoir_simple_control_parameters",
    "reservoir_parameters",
    "merge_reservoirs",
    "merge_reservoirs_sediment",
    "create_reservoirs_geoms",
    "create_reservoirs_geoms_sediment",
]

RESERVOIR_COMMON_PARAMETERS = [
    "reservoir_area",
    "reservoir_initial_depth",
    "reservoir_rating_curve",
    "reservoir_storage_curve",
]

RESERVOIR_CONTROL_PARAMETERS = [
    "reservoir_max_volume",
    "reservoir_target_min_fraction",
    "reservoir_target_full_fraction",
    "reservoir_demand",
    "reservoir_max_release",
]

RESERVOIR_UNCONTROL_PARAMETERS = [
    "reservoir_b",
    "reservoir_e",
    "reservoir_outflow_threshold",
    "reservoir_lower_id",
]

RESERVOIR_LAYERS = (
    RESERVOIR_COMMON_PARAMETERS
    + RESERVOIR_CONTROL_PARAMETERS
    + RESERVOIR_UNCONTROL_PARAMETERS
    + [
        "reservoir_area_id",
        "reservoir_outlet_id",
    ]
)

RESERVOIR_LAYERS_SEDIMENT = [
    "reservoir_area_id",
    "reservoir_outlet_id",
    "reservoir_area",
    "reservoir_trapping_efficiency",
]


def reservoir_id_maps(
    gdf: gpd.GeoDataFrame,
    ds_like: xr.Dataset,
    min_area: float = 0.0,
    uparea_name: str = "uparea",
) -> tuple[xr.Dataset | None, gpd.GeoDataFrame | None]:
    """Return reservoir location maps (see list below).

    At model resolution based on gridded upstream area data input or outlet coordinates.

    The following reservoir maps are calculated:

    - reservoir_area_id : reservoir areas mask [ID]
    - reservoir_outlet_id : reservoir outlets [ID]

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing reservoirs/lakes geometries and attributes.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    min_area : float, optional
        Minimum reservoir area threshold [km2], by default 0.0 km2.
    uparea_name : str, optional
        Name of uparea variable in ds_like. If None then database coordinates will be
        used to setup outlets

    Returns
    -------
    ds_out : xarray.DataArray
        Dataset containing gridded reservoir data
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing (updated) reservoir outlet coordinates.
    """
    # Check if uparea_name in ds_like
    if uparea_name not in ds_like.data_vars:
        logger.warning(
            "Upstream area map for reservoir outlet setup not found. "
            "Database coordinates used instead"
        )
        uparea_name = None

    # skip small size reservoirs
    if "Area_avg" in gdf.columns and gdf.geometry.size > 0:
        min_area_m2 = min_area * 1e6
        gdf = gdf[gdf.Area_avg >= min_area_m2]
    else:
        logger.warning(
            "Reservoir's database has no area attribute. "
            "All reservoirs will be considered."
        )

    # check if any reservoirs are left after filtering
    nb_wb = gdf.geometry.size
    logger.info(f"{nb_wb} reservoir(s) of sufficient size found within region.")
    if nb_wb == 0:
        return None, None

    ### Compute reservoir maps
    # Rasterize the GeoDataFrame to get the areas mask of reservoirs
    res_id = gdf["waterbody_id"].values
    da_wbmask = ds_like.raster.rasterize(
        gdf,
        col_name="waterbody_id",
        nodata=-999,
        all_touched=True,
        dtype=None,
        sindex=False,
    )
    da_wbmask = da_wbmask.rename("reservoir_area_id")
    da_wbmask.attrs.update(_FillValue=-999)
    ds_out = da_wbmask.to_dataset()
    if not np.all(np.isin(res_id, ds_out["reservoir_area_id"])):
        gdf = gdf.loc[np.isin(res_id, ds_out["reservoir_area_id"])]
        nskipped = res_id.size - gdf.index.size
        res_id = gdf["waterbody_id"].values
        logger.warning(
            f"{nskipped} reservoirs are not successfully rasterized and skipped!!"
            " Consider increasing the lakes min_area threshold."
        )

    # Initialize the reservoir outlet map
    ds_out["reservoir_outlet_id"] = xr.full_like(ds_out["reservoir_area_id"], -999)
    # If an upstream area map is present in the model, gets outlets coordinates using/
    # the maximum uparea in each reservoir mask to match model river network.
    if uparea_name is not None and uparea_name in ds_like.data_vars:
        logger.debug("Setting reservoir outlet map based maximum upstream area.")
        # create dataframe with x and y coord to be filled in either from uparea or from
        # xout and yout in hydrolakes data
        outdf = gdf[["waterbody_id"]].assign(xout=np.nan, yout=np.nan)
        ydim = ds_like.raster.y_dim
        xdim = ds_like.raster.x_dim
        for i in res_id:
            res_acc = (
                ds_like[uparea_name].where(ds_out["reservoir_area_id"] == i).load()
            )
            max_res_acc = res_acc.where(res_acc == res_acc.max(), drop=True).squeeze()
            yacc = max_res_acc[ydim].values
            xacc = max_res_acc[xdim].values
            ds_out["reservoir_outlet_id"].loc[{f"{ydim}": yacc, f"{xdim}": xacc}] = i
            outdf.loc[outdf.waterbody_id == i, "xout"] = xacc
            outdf.loc[outdf.waterbody_id == i, "yout"] = yacc
        outgdf = gpd.GeoDataFrame(
            outdf, geometry=gpd.points_from_xy(outdf.xout, outdf.yout)
        )

    # ELse use coordinates from the reservoir database
    elif "xout" in gdf.columns and "yout" in gdf.columns:
        logger.debug("Setting reservoir outlet map based on coordinates.")
        outdf = gdf[["waterbody_id", "xout", "yout"]]
        outgdf = gpd.GeoDataFrame(
            outdf, geometry=gpd.points_from_xy(outdf.xout, outdf.yout)
        )
        ds_out["reservoir_outlet_id"] = ds_like.raster.rasterize(
            outgdf, col_name="waterbody_id", nodata=-999
        )
    # Else outlet map is equal to areas mask map
    else:
        ds_out["reservoir_outlet_id"] = ds_out["reservoir_area_id"]
        logger.warning(
            "Neither upstream area map nor reservoir's outlet coordinates found. "
            "Setting reservoir outlet map equal to the area map."
        )
        # dummy outgdf
        outgdf = gdf[["waterbody_id"]]
    ds_out["reservoir_outlet_id"].attrs.update(_FillValue=-999)
    # fix dtypes
    ds_out["reservoir_outlet_id"] = ds_out["reservoir_outlet_id"].astype("int32")
    ds_out["reservoir_area_id"] = ds_out["reservoir_area_id"].astype("float32")

    # update/replace xout and yout in gdf_org from outgdf:
    gdf.loc[:, "xout"] = outgdf["xout"].values
    gdf.loc[:, "yout"] = outgdf["yout"].values

    return ds_out, gdf


def reservoir_simple_control_parameters(
    gdf: gpd.GeoDataFrame,
    ds_reservoirs: xr.Dataset,
    timeseries_fn: str = None,
    output_folder: str | Path | None = None,
) -> tuple[xr.Dataset, gpd.GeoDataFrame]:
    """Return reservoir attributes (see list below) needed for modelling.

    When specified, some of the reservoir attributes can be derived from \
earth observation data.
    Two options are currently available: 1. Global Water Watch data (Deltares, 2022) \
using gwwapi and 2. JRC (Peker, 2016) using hydroengine.

    The following reservoir attributes are calculated:

    - reservoir_max_volume : reservoir maximum volume [m3]
    - reservoir_area : reservoir area [m2]
    - reservoir_initial_depth : reservoir initial water level [m]
    - reservoir_rating_curve : option to compute rating curve [-]
    - reservoir_storage_curve : option to compute storage curve [-]
    - reservoir_demand : reservoir demand flow [m3/s]
    - reservoir_max_release : reservoir maximum release flow [m3/s]
    - reservoir_target_full_fraction : reservoir targeted full volume fraction [m3/m3]
    - reservoir_target_min_fraction : reservoir targeted minimum volume fraction [m3/m3]

    Two additional tables will be prepared and saved if output_folder is specified:
    "reservoir_timeseries_{timeseries_fn}.csv" contains the timeseries downloaded from
    ``timeseries_fn``; "reservoir_accuracy.csv" contains debugging values for reservoir
    building.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing reservoirs geometries and attributes.
    ds_reservoirs : xarray.Dataset
        Dataset containing reservoir location and outlet id at model resolution.
    timeseries_fn : str, optional
        Name of database from which time series of reservoir surface water area
        will be retrieved.
        Currently available: ['jrc', 'gww']
        Defaults to Deltares' Global Water Watch database.
    output_folder: str or Path, optional
        Folder to save the reservoir time series data and parameter accuracy as .csv
        files. If None, no file will be saved.

    Returns
    -------
    ds_reservoirs : xarray.Dataset
        Dataset containing reservoir locations and parameters at model resolution.
    gdf_out : geopandas.GeoDataFrame
        GeoDataFrame containing reservoir parameters.
    """
    layers = (
        ["waterbody_id"] + RESERVOIR_COMMON_PARAMETERS + RESERVOIR_CONTROL_PARAMETERS
    )
    # if present use directly
    if np.all(np.isin(layers, gdf.columns)):
        df_reservoirs = gdf[layers]
    # else compute
    else:
        df_reservoirs = compute_reservoir_simple_control_parameters(
            gdf=gdf,
            timeseries_fn=timeseries_fn,
            output_folder=output_folder,
        )

    # create a geodf with id of reservoir and geometry at outflow location
    gdf_points = gpd.GeoDataFrame(
        gdf["waterbody_id"],
        geometry=gpd.points_from_xy(gdf.xout, gdf.yout),
    )
    gdf_points = gdf_points.merge(df_reservoirs, on="waterbody_id")  # merge
    # add parameter attributes to polygon gdf:
    gdf = gdf.merge(df_reservoirs, on="waterbody_id")

    # rasterize parameters to model resolution and add to ds_reservoirs
    for name in gdf_points.columns[2:]:
        gdf_points[name] = gdf_points[name].astype("float32")
        ds_reservoirs[name] = ds_reservoirs.raster.rasterize(
            gdf_points, col_name=name, dtype="float32", nodata=-999
        )
    ds_reservoirs = set_rating_curve_layer_data_type(ds_reservoirs)
    return ds_reservoirs, gdf


def compute_reservoir_simple_control_parameters(
    gdf: gpd.GeoDataFrame,
    timeseries_fn: str = None,
    perc_norm: int = 50,
    perc_min: int = 20,
    output_folder: str | Path | None = None,
) -> pd.DataFrame:
    """Return reservoir attributes (see list below) needed for modelling.

    When specified, some of the reservoir attributes can be derived from \
earth observation data.
    Two options are currently available: 1. Global Water Watch data (Deltares, 2022) \
using gwwapi and 2. JRC (Peker, 2016) using hydroengine.

    The following reservoir attributes are calculated:

    - reservoir_max_volume : reservoir maximum volume [m3]
    - reservoir_area : reservoir area [m2]
    - reservoir_initial_depth : reservoir initial water level [m]
    - reservoir_rating_curve : option to compute rating curve [-]
    - reservoir_storage_curve : option to compute storage curve [-]
    - reservoir_demand : reservoir demand flow [m3/s]
    - reservoir_max_release : reservoir maximum release flow [m3/s]
    - reservoir_target_full_fraction : reservoir targeted full volume fraction [m3/m3]
    - reservoir_target_min_fraction : reservoir targeted minimum volume fraction [m3/m3]

    Two additional tables will be prepared and saved if output_folder is specified:
    "reservoir_timeseries_{timeseries_fn}.csv" contains the timeseries downloaded from
    ``timeseries_fn``; "reservoir_accuracy.csv" contains debugging values for reservoir
    building.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing reservoirs geometries and attributes.
    timeseries_fn : str, optional
        Name of database from which time series of reservoir surface water area
        will be retrieved.
        Currently available: ['jrc', 'gww']
        Defaults to Deltares' Global Water Watch database.
    perc_norm : int, optional
        Percentile for normal (operational) surface area
    perc_min: int, optional
        Percentile for minimal (operational) surface area
    output_folder: str or Path, optional
        Folder to save the reservoir time series data and parameter accuracy as .csv
        files. If None, no file will be saved.

    Returns
    -------
    df_out : pandas.DataFrame
        DataFrame containing reservoir parameters.
    """
    if timeseries_fn == "jrc":
        try:
            import hydroengine as he

            logger.info("Using reservoir timeseries from JRC via hydroengine.")
        except Exception:
            raise ImportError(
                "hydroengine package not found, cannot download jrc \
reservoir timeseries."
            )

    elif timeseries_fn == "gww":
        try:
            from gwwapi import client as cli
            from gwwapi import utils

            logger.info("Using reservoir timeseries from GWW via gwwapi.")
        except Exception:
            raise ImportError(
                "gwwapi package not found, cannot download gww reservoir timeseries."
            )
    elif timeseries_fn is not None:
        raise ValueError(
            f"timeseries_fn argument {timeseries_fn} not understood, \
please use one of [gww, jrc] or None."
        )
    else:
        logger.debug(
            "Using reservoir attributes from reservoir database to compute parameters."
        )

    # Initialize output DataFrame with empty values and reservoir ID
    df_out = pd.DataFrame(
        index=range(len(gdf["waterbody_id"])),
        columns=(
            [
                "resid",
                "resmaxvolume",
                "resarea",
                "resinitialdepth",
                "resratingcurve",
                "resstoragecurve",
                "resdemand",
                "resmaxrelease",
                "resfullfrac",
                "resminfrac",
            ]
        ),
        dtype=np.float32,
    )
    df_out["resid"] = gdf["waterbody_id"].values

    # Create similar dataframe for EO time series
    df_eo = pd.DataFrame(
        index=range(len(gdf["waterbody_id"])),
        columns=(["resid", "maxarea", "normarea", "minarea", "capmin", "capmax"]),
    )
    df_eo["resid"] = gdf["waterbody_id"].values

    # Create dtaframe for accuracy plots
    df_plot = pd.DataFrame(
        index=range(len(gdf["waterbody_id"])),
        columns=(["resid", "factor", "accuracy_min", "accuracy_norm"]),
    )
    df_plot["resid"] = gdf["waterbody_id"].values

    # Create empty dataframe for timeseries exports
    df_ts = pd.DataFrame()

    # Get EO data for each reservoir
    if timeseries_fn == "jrc":
        for i in range(len(gdf["waterbody_id"])):
            ids = str(gdf["waterbody_id"].iloc[i])
            try:
                logger.debug(f"Downloading HydroEngine time series for reservoir {ids}")
                time_series = he.get_lake_time_series(
                    int(gdf["Hylak_id"].iloc[i]), "water_area"
                )

                # Append series to df_ts which will contain all time series
                # of all downloaded reservoirs, with an outer join on the datetime index
                ts_index = pd.to_datetime([k * 1000000 for k in time_series["time"]])
                ts_values = time_series["water_area"]
                ts_series = pd.Series(
                    data=ts_values, index=ts_index, name=int(gdf["Hylak_id"].iloc[i])
                )
                ts_series = ts_series[ts_series > 0].dropna()
                df_ts = pd.concat([df_ts, ts_series], join="outer", axis=1)

                # Save area stats
                area_series = np.array(time_series["water_area"])  # [m2]
                area_series_nozeros = area_series[area_series > 0]
                df_eo.loc[i, "maxarea"] = area_series_nozeros.max()
                df_eo.loc[i, "normarea"] = np.percentile(
                    area_series_nozeros, perc_norm, axis=0
                )
                df_eo.loc[i, "minarea"] = np.percentile(
                    area_series_nozeros, perc_min, axis=0
                )
            except Exception:
                logger.warning(
                    f"No HydroEngine time series available for reservoir {ids}!"
                )

    if timeseries_fn == "gww":
        # get bounds from gdf input as JSON object that can be used in
        # post request with the gww api
        gdf_bounds = json.dumps(
            shapely.geometry.box(*gdf.total_bounds, ccw=True).__geo_interface__
        )
        # get reservoirs within these bounds
        gww_reservoirs = cli.get_reservoirs_by_geom(gdf_bounds)
        # from the response, create a dictionary, linking the gww_id to the hylak_id
        # (used in the default reservoir database)
        idlink = {
            k["properties"]["source_id"]: k["id"] for k in gww_reservoirs["features"]
        }

        for i in range(len(gdf["waterbody_id"])):
            ids = str(gdf["waterbody_id"].iloc[i])
            try:
                logger.debug(
                    f"Downloading Global Water Watch time series for reservoir {ids}"
                )
                time_series = cli.get_reservoir_ts_monthly(
                    idlink[int(gdf["Hylak_id"].iloc[i])]
                )

                # Append series to df_ts which will contain all time series
                # of all downloaded reservoirs, with an outer join on the datetime index
                ts_series = utils.to_timeseries(
                    time_series, name=f"{int(gdf['Hylak_id'].iloc[i])}"
                ).drop_duplicates()
                df_ts = pd.concat([df_ts, ts_series], join="outer", axis=1)

                # Compute stats
                area_series = utils.to_timeseries(time_series)["area"].to_numpy()
                area_series_nozeros = area_series[area_series > 0]
                df_eo.loc[i, "maxarea"] = area_series_nozeros.max()
                df_eo.loc[i, "normarea"] = np.percentile(
                    area_series_nozeros, perc_norm, axis=0
                )
                df_eo.loc[i, "minarea"] = np.percentile(
                    area_series_nozeros, perc_min, axis=0
                )
            except Exception:
                logger.warning(f"No GWW time series available for reservoir {ids}!")

    # Sort timeseries dataframe (will be saved to root as .csv later)
    df_ts = df_ts.sort_index()

    # Compute resdemand and resmaxrelease from average discharge
    if "Dis_avg" in gdf.columns:
        df_out["resdemand"] = gdf["Dis_avg"].values * 0.5
        df_out["resmaxrelease"] = gdf["Dis_avg"].values * 4.0

    # Get resarea either from EO or database depending
    if "Area_avg" in gdf.columns:
        df_out["resarea"] = gdf["Area_avg"].values.astype(np.float32)
        if timeseries_fn is not None:
            df_out.loc[pd.notna(df_eo["maxarea"]), "resarea"] = df_eo["maxarea"][
                pd.notna(df_eo["maxarea"])
            ].values.astype(np.float32)
        else:
            df_out.loc[pd.isna(df_out["resarea"]), "resarea"] = df_eo["maxarea"][
                pd.isna(df_out["resarea"])
            ].values.astype(np.float32)
    else:
        df_out["resarea"] = df_eo["maxarea"].values.astype(np.float32)

    # Get resmaxvolume from database
    if "Vol_avg" in gdf.columns:
        df_out["resmaxvolume"] = gdf["Vol_avg"].values.astype(np.float32)

    # Compute target min and max fractions
    # First look if data is available from the database
    if "Capacity_min" in gdf.columns:
        df_out["resminfrac"] = gdf["Capacity_min"].values / df_out["resmaxvolume"]
        df_plot["accuracy_min"] = np.repeat(1.0, len(df_plot["accuracy_min"]))
    if "Capacity_norm" in gdf.columns:
        df_out["resfullfrac"] = gdf["Capacity_norm"].values / df_out["resmaxvolume"]
        df_plot["accuracy_norm"] = np.repeat(1.0, len(df_plot["accuracy_norm"]))

    # Add storage and rating curve values (1 and 4)
    df_out["resratingcurve"] = np.repeat(4, len(df_out["resratingcurve"]))
    df_out["resstoragecurve"] = np.repeat(1, len(df_out["resstoragecurve"]))

    # Then compute from EO data and fill or replace the previous values
    # (if a valid source is provided)
    gdf = gdf.where(pd.notna(gdf), np.nan).infer_objects(copy=False)
    for i in range(len(gdf["waterbody_id"])):
        # Initialise values
        dam_height = np.nanmax([gdf["Dam_height"].iloc[i], 0.0])
        max_level = np.nanmax([gdf["Depth_avg"].iloc[i], 0.0])
        max_area = np.nanmax([df_out["resarea"].iloc[i], 0.0])
        max_cap = np.nanmax([df_out["resmaxvolume"].iloc[i], 0.0])
        norm_area = np.nanmax([df_eo["normarea"].iloc[i], 0.0])
        if "Capacity_norm" in gdf.columns:
            norm_cap = np.nanmax([gdf["Capacity_norm"].iloc[i], 0.0])
        else:
            norm_cap = 0.0
        min_area = np.nanmax([df_eo["minarea"].iloc[i], 0.0])
        if "Capacity_min" in gdf.columns:
            min_cap = np.nanmax([gdf["Capacity_min"].iloc[i], 0.0])
        else:
            min_cap = 0.0
        mv = 0.0

        # Maximum level
        # a validation has shown that GRanD dam height is a more reliable value than
        # HydroLAKES depth average (when it is a larger value)
        if dam_height > max_level:
            max_level_f = dam_height
            factor_shape = dam_height / max_level
            logger.info(
                "GRanD dam height used as max. level instead of HydroLAKES depth "
                f"average. Difference factor: {round(factor_shape, 2):.4f}"
            )
        else:
            max_level_f = max_level
            factor_shape = 1.0

        # coefficient for linear relationship
        lin_coeff = max_area / max_level_f  # [m2/m]
        factor_used = 1.0

        # Operational (norm) level
        if norm_cap != mv and norm_area != mv:
            norm_level = factor_used * (norm_cap / norm_area)  # [m]
            norm_area_f = norm_area
            norm_cap_f = norm_cap
            accuracy_norm = 1
        elif norm_cap != mv:
            norm_level = (norm_cap / lin_coeff) ** (1 / 2)  # [m]
            norm_area_f = norm_cap / norm_level  # [m]
            norm_cap_f = norm_cap
            accuracy_norm = 1
        elif norm_area != mv:
            norm_level = norm_area / lin_coeff  # [m]
            norm_area_f = norm_area
            norm_cap_f = norm_area * norm_level  # [m3]
            accuracy_norm = 2
        else:
            # the calculation is based on the max area (and not max level or max
            # capacity) as it is the most reliable value
            norm_area_f = max_area * 0.666  # [m]
            norm_level = norm_area_f / lin_coeff  # [m]
            norm_cap_f = norm_area_f * norm_level  # [m3]
            accuracy_norm = 3

        # CHECK norm level (1)
        if accuracy_norm == 1 and norm_level > max_level_f:
            # it is assumed that the norm capacity value is not reliable, so this value
            # is delated and the linear relationship assumption is introduced
            norm_level = norm_area_f / lin_coeff  # [m]
            norm_cap_f = norm_area_f * norm_level  # [m3]
            accuracy_norm = 21
        elif accuracy_norm == 2 and norm_level > max_level_f:
            norm_area_f = max_area * 0.666  # [m]
            norm_level = norm_area_f / lin_coeff  # [m]
            norm_cap_f = norm_area_f * norm_level  # [m3]
            accuracy_norm = 31

        # Minimum level
        if min_area != mv and min_cap != mv:
            min_level = min_cap / min_area  # [m]
            min_area_f = min_area
            min_cap_f = min_cap
            accuracy_min = 1
        elif min_cap != mv:
            min_level = (min_cap / lin_coeff) ** (1 / 2)  # [m]
            min_area_f = min_cap / min_level  # [m]
            min_cap_f = min_cap  # [m3]
            accuracy_min = 1
        elif min_area != mv:
            min_level = min_area / lin_coeff  # [m]
            min_area_f = min_area  # [m]
            min_cap_f = min_area * min_level  # [m3]
            accuracy_min = 2
        else:
            # the calculation is based on the max area (and not max level or max
            # capacity) as it is the most reliable value
            min_area_f = max_area * 0.333  # [m]
            min_level = min_area_f / lin_coeff  # [m]
            min_cap_f = min_area_f * min_level  # [m3]
            accuracy_min = 3

        # CHECK minimum level (1)
        if accuracy_min == 1 and min_level > norm_level:
            accuracy_min = 21
            min_level = min_area_f / lin_coeff  # [m]
            min_cap_f = min_area_f * min_level  # [m3]
        elif accuracy_min == 2 and min_level > norm_level:
            accuracy_min = 31
            min_area_f = max_area * 0.333  # [m]
            min_level = min_area_f / lin_coeff  # [m]
            min_cap_f = (min_area_f * min_level) / 100  # [m3]
        elif min_level > norm_level:
            min_area_f = norm_area_f * 0.45  # [m]
            min_level = min_area_f / lin_coeff  # [m]
            min_cap_f = min_area_f * min_level  # [m3]
            accuracy_min = 4

        # CHECK norm level (2)
        if norm_cap_f > max_cap:
            logger.warning("norm_cap > max_cap! setting norm_cap equal to max_cap.")
            norm_cap_f = max_cap
            accuracy_norm = 5

        # CHECK minimum level (2)
        if min_cap_f > norm_cap_f:
            logger.warning("min_cap > norm_cap! setting min_cap equal to norm_cap.")
            min_cap_f = norm_cap_f
            accuracy_min = 5

        # Resume results
        df_eo.loc[i, "capmin"] = min_cap_f
        df_eo.loc[i, "capmax"] = norm_cap_f
        df_plot.loc[i, "factor"] = factor_shape
        df_plot.loc[i, "accuracy_min"] = accuracy_min
        df_plot.loc[i, "accuracy_norm"] = accuracy_norm

    # Depending on priority EO update fullfrac and min frac
    if timeseries_fn is not None:
        df_out.resminfrac = df_eo["capmin"].values / df_out["resmaxvolume"].values
        df_out.resfullfrac = df_eo["capmax"].values / df_out["resmaxvolume"].values
    else:
        mask = pd.isna(df_out["resminfrac"])
        df_out.loc[mask, "resminfrac"] = df_eo.loc[mask, "capmin"].values.astype(
            np.float64
        ) / (df_out.loc[mask, "resmaxvolume"].values).astype(np.float32)

        mask = pd.isna(df_out["resfullfrac"])
        df_out.loc[mask, "resfullfrac"] = df_eo.loc[mask, "capmax"].values.astype(
            np.float64
        ) / (df_out.loc[mask, "resmaxvolume"].values).astype(np.float32)

    # Add initial depth
    df_out["resinitialdepth"] = (
        df_out["resfullfrac"] * df_out["resmaxvolume"] / df_out["resarea"]
    ).astype(np.float32)

    # rename to wflow naming convention
    tbls = {
        "resarea": "reservoir_area",
        "resstoragecurve": "reservoir_storage_curve",
        "resratingcurve": "reservoir_rating_curve",
        "resinitialdepth": "reservoir_initial_depth",
        "resdemand": "reservoir_demand",
        "resfullfrac": "reservoir_target_full_fraction",
        "resminfrac": "reservoir_target_min_fraction",
        "resmaxrelease": "reservoir_max_release",
        "resmaxvolume": "reservoir_max_volume",
        "resid": "waterbody_id",
    }
    df_out = df_out.rename(columns=tbls)

    # Save accuracy information on reservoir parameters
    if output_folder is not None:
        df_plot.to_csv(join(output_folder, "reservoir_accuracy.csv"))
        df_ts.to_csv(join(output_folder, f"reservoir_timeseries_{timeseries_fn}.csv"))

    return df_out


def reservoir_parameters(
    ds: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    rating_dict: dict = {},
) -> tuple[xr.Dataset, gpd.GeoDataFrame, dict]:
    """
    Return (uncontrolled) reservoir attributes (see list below) needed for modelling.

    If rating_dict is not empty, prepares also rating tables for wflow.

    The following reservoir attributes are calculated:

    - waterbody_id : waterbody id
    - reservoir_area : reservoir area [m2]
    - reservoir_initial_depth: reservoir average level or initial depth (cold state) [m]
    - meta_reservoir_mean_outflow: reservoir average outflow [m3/s]
    - reservoir_b: reservoir rating curve coefficient [-]
    - reservoir_e: reservoir rating curve exponent [-]
    - reservoir_storage_curve: option to compute storage curve [-]
    - reservoir_rating_curve: option to compute rating curve [-]
    - reservoir_outflow_threshold: minimum threshold for reservoir outflow [m]
    - reservoir_lower_id: id of linked reservoir location if any

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the reservoir locations and area
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing the reservoir locations and area
    rating_dict : dict, optional
        Dictionary containing the rating curve parameters, by default dict()

    Returns
    -------
    ds : xr.Dataset
        Dataset containing the reservoir locations with the attributes
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing the reservoir locations with the attributes
    rating_curves : dict
        Dictionary containing the rating curves in wflow format
    """
    # rename to param values
    gdf = gdf.rename(
        columns={
            "Area_avg": "reservoir_area",
            "Depth_avg": "reservoir_initial_depth",
            "Dis_avg": "meta_reservoir_mean_outflow",
        }
    )

    # Minimum value for mean_outflow
    mean_outflow = gdf["meta_reservoir_mean_outflow"].copy()
    gdf["meta_reservoir_mean_outflow"] = np.maximum(
        gdf["meta_reservoir_mean_outflow"], 0.01
    )
    if "reservoir_b" not in gdf.columns:
        gdf["reservoir_b"] = gdf["meta_reservoir_mean_outflow"].values / (
            gdf["reservoir_initial_depth"].values
        ) ** (2)
    if "reservoir_e" not in gdf.columns:
        gdf["reservoir_e"] = 2
    if "reservoir_outflow_threshold" not in gdf.columns:
        gdf["reservoir_outflow_threshold"] = 0.0
    if "reservoir_lower_id" not in gdf.columns:
        gdf["reservoir_lower_id"] = 0
    if "reservoir_storage_curve" not in gdf.columns:
        gdf["reservoir_storage_curve"] = 1
    if "reservoir_rating_curve" not in gdf.columns:
        gdf["reservoir_rating_curve"] = 3

    # Check if some mean_outflow values have been replaced
    if not np.all(mean_outflow == gdf["meta_reservoir_mean_outflow"]):
        logger.warning(
            "Some values of meta_reservoir_mean_outflow have been replaced by "
            "a minimum value of 0.01m3/s"
        )

    # Check if rating curve is provided
    rating_curves = {}
    if len(rating_dict) != 0:
        # Assume one rating curve per reservoir index
        for wid in gdf["waterbody_id"].values:
            wid = int(wid)
            if wid in rating_dict.keys():
                df_rate = rating_dict[wid]
                # Prepare the right tables for wflow
                # Update storage and rating curves
                # Storage
                if "volume" in df_rate.columns:
                    gdf.loc[gdf["waterbody_id"] == wid, "reservoir_storage_curve"] = 2
                    df_stor = df_rate[["elevtn", "volume"]].dropna(
                        subset=["elevtn", "volume"]
                    )
                    df_stor.rename(columns={"elevtn": "H", "volume": "S"}, inplace=True)
                    # add to rating_curves
                    rating_curves[f"reservoir_sh_{wid}"] = df_stor
                else:
                    logger.warning(
                        f"Storage data not available for reservoir {wid}. "
                        "Using default S=AH"
                    )
                # Rating
                if "discharge" in df_rate.columns:
                    gdf.loc[gdf["waterbody_id"] == wid, "reservoir_rating_curve"] = 1
                    df_rate = df_rate[["elevtn", "discharge"]].dropna(
                        subset=["elevtn", "discharge"]
                    )
                    df_rate.rename(
                        columns={"elevtn": "H", "discharge": "Q"},
                        inplace=True,
                    )
                    # Repeat Q for the 365 JDOY
                    df_q = pd.concat([df_rate.Q] * (366), axis=1, ignore_index=True)
                    df_q[0] = df_rate["H"]
                    df_q.rename(columns={0: "H"}, inplace=True)
                    # add to rating_curves
                    rating_curves[f"reservoir_hq_{wid}"] = df_q
                else:
                    logger.warning(
                        f"Rating data not available for reservoir {wid}. "
                        "Using default Modified Puls Approach"
                    )

    # Create raster of reservoir params
    reservoir_params = (
        ["waterbody_id", "meta_reservoir_mean_outflow"]
        + RESERVOIR_COMMON_PARAMETERS
        + RESERVOIR_UNCONTROL_PARAMETERS
    )

    gdf_org_points = gpd.GeoDataFrame(
        gdf[reservoir_params],
        geometry=gpd.points_from_xy(gdf.xout, gdf.yout),
    )

    for name in reservoir_params[1:]:
        da_reservoir = ds.raster.rasterize(
            gdf_org_points, col_name=name, dtype="float32", nodata=-999
        )
        ds[name] = da_reservoir

    ds = set_rating_curve_layer_data_type(ds)
    return ds, gdf, rating_curves


def _check_duplicated_ids_in_merge(
    ds: xr.Dataset,
    duplicate_id: str = "error",
) -> xr.Dataset | None:
    """
    Check if reservoir IDs in ds are not duplicated in ds_like.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the merged reservoir layers.
    duplicate_id: str, optional {"error", "skip"}
        Action to take if duplicate reservoir IDs are found when merging with
        existing reservoirs. Options are "error" to raise an error (default); "skip"
        to skip adding new reservoirs.

    Returns
    -------
    xr.Dataset | None
        Returns None if there are duplicated reservoir IDs, otherwise returns the
        dataset.
    """
    # Check if there is no duplicated reservoir_outlet_id after merging
    ids = ds["reservoir_outlet_id"].raster.mask_nodata().values
    # Remove NaN values from ids
    ids = ids[~np.isnan(ids)]
    # Get unique ids and their counts
    ids_unique, counts = np.unique(ids, return_counts=True)
    # Check if any id is duplicated
    if np.any(counts > 1):
        if duplicate_id == "error":
            raise ValueError(
                f"Reservoir ID(s) {ids_unique[counts > 1]} are duplicated in the merged"
                " dataset. This may lead to incorrect results. Either update the IDs "
                "in the source dataset or use a different duplicate_id action."
            )
        elif duplicate_id == "skip":
            logger.warning(
                f"Reservoir ID(s) {ids_unique[counts > 1]} are duplicated in the merged"
                " dataset. This may lead to incorrect results. Skip merging reservoirs."
            )
            return None
        else:
            raise ValueError(
                f"Unknown duplicate_id action: {duplicate_id}. "
                "Choose from {'error', 'skip'}"
            )

    return ds


def merge_reservoirs(
    ds: xr.Dataset,
    ds_like: xr.Dataset,
    duplicate_id: str = "error",
    id_layer: str = "reservoir_outlet_id",
) -> xr.Dataset | None:
    """
    Merge reservoir layers in ds to layers in ds_like.

    It will first check if the IDs in ds are not duplicated in ds_like.
    If they are, the function will raise a warning and return None.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the reservoir layers to be merged.
    ds_like : xr.Dataset
        Dataset containing the reservoir layers to merge into.
    duplicate_id: str, optional {"error", "skip"}
        Action to take if duplicate reservoir IDs are found when merging with
        existing reservoirs. Options are "error" to raise an error (default); "skip"
        to skip adding new reservoirs.
    id_layer : str, optional
        Name of the layer containing the reservoir IDs, by default
        "reservoir_outlet_id".

    Returns
    -------
    xr.Dataset | None
        Merged dataset of reservoir parameters.
    """
    # Loop over layers to merge
    ds_out = ds.copy()
    for layer in RESERVOIR_LAYERS:
        if "area_id" in layer:
            # Set the mask in case the area_id map is selected (spatial coverage of the
            # waterbody)
            mask = ds[layer] > 0
        else:
            # Set the mask to the outlet_id map (point location of the waterbody)
            mask = ds[id_layer] > 0

        # if layer is not in ds, skip it
        # NaN can be ok: e.g. natural lake does not have reservoir_demand
        if layer not in ds and layer in ds_like:
            ds_out[layer] = ds_like[layer]

        # if layer is in ds_like, merge it
        if layer in ds and layer in ds_like:
            # merge the layer
            ds_out[layer] = ds[layer].where(mask, ds_like[layer])
            # ensure the nodata value is set correctly
            ds_out[layer].raster.set_nodata(ds_like[layer].raster.nodata)
        # else we just keep ds[layer] as it is
    ds_out = set_rating_curve_layer_data_type(ds_out)
    return _check_duplicated_ids_in_merge(ds_out, duplicate_id=duplicate_id)


def merge_reservoirs_sediment(
    ds: xr.Dataset,
    ds_like: xr.Dataset,
    duplicate_id: str = "error",
) -> xr.Dataset | None:
    """
    Merge reservoir layers in ds to layers in ds_like for wflow sediment.

    It will check if the IDs in ds are not duplicated in ds_like.
    If they are, the function will raise a warning and return None.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the reservoir layers to be merged.
    ds_like : xr.Dataset
        Dataset containing the reservoir layers to merge into.
    duplicate_id: str, optional {"error", "skip"}
        Action to take if duplicate reservoir IDs are found when merging with
        existing reservoirs. Options are "error" to raise an error (default); "skip"
        to skip adding new reservoirs.

    Returns
    -------
    xr.Dataset | None
        Merged dataset of reservoir parameters.
    """
    # Loop over layers to merge
    ds_out = ds.copy()
    for layer in RESERVOIR_LAYERS_SEDIMENT:
        # if layer is in ds_like, merge it
        if layer in ds and layer in ds_like:
            # merge the layer
            ds_out[layer] = ds[layer].where(
                ds[layer] != ds[layer].raster.nodata, ds_like[layer]
            )
        else:
            # all parameters for sediment should be in both
            logger.warning(
                f"Reservoir layer {layer} is not present in either the new dataset or"
                "the wflow model. Skipping adding new reservoirs. Consider overwritting"
                "to solve this issue."
            )
            return None

    return _check_duplicated_ids_in_merge(ds_out, duplicate_id=duplicate_id)


def create_reservoirs_geoms_sediment(
    ds_res: xr.Dataset,
) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame of reservoir geometries for the sediment model.

    Parameters
    ----------
    ds_res : xr.Dataset
        Dataset containing the reservoir data.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the reservoir geometries and properties.
    """
    return create_reservoirs_geoms(ds_res, layers=RESERVOIR_LAYERS_SEDIMENT)


def create_reservoirs_geoms(
    ds_res: xr.Dataset,
    layers: list[str] = RESERVOIR_LAYERS,
) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame of reservoir geometries for the sediment model.

    Parameters
    ----------
    ds_res : xr.Dataset
        Dataset containing the reservoir data.
    layers : list[str], optional
        List of layer names to include in the GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing the reservoir geometries and properties.
    """
    # Vectorize the outlets in order to sample parameters values
    gdf_outlets = ds_res["reservoir_outlet_id"].raster.vectorize()
    # Convert to points
    centroid = utils.planar_operation_in_utm(
        gdf_outlets["geometry"], lambda geom: geom.centroid
    )
    gdf_outlets["geometry"] = centroid

    # Sample reservoir properties at the outlet locations
    res_layers = [layer for layer in layers if layer in ds_res]
    params = ds_res[res_layers].raster.sample(gdf_outlets)
    # Convert to dataframe
    df_params = params.to_dataframe()

    # Now we can vectorize the reservoir shapes and add the parameters by ID
    gdf_reservoirs = ds_res["reservoir_area_id"].raster.vectorize()
    # Merge by index with df_params
    gdf_reservoirs = gdf_reservoirs.merge(
        df_params, left_on="value", right_on="reservoir_area_id", how="left"
    )
    # Only keep geometry, layers and y, x columns
    gdf_reservoirs = gdf_reservoirs[
        res_layers + [ds_res.raster.y_dim, ds_res.raster.x_dim, "geometry"]
    ]
    # Rename to xout and yout
    gdf_reservoirs = gdf_reservoirs.rename(
        columns={ds_res.raster.x_dim: "xout", ds_res.raster.y_dim: "yout"}
    )

    return gdf_reservoirs


def set_rating_curve_layer_data_type(ds_res: xr.Dataset) -> xr:
    """Set reservoir rating curve layers to int data type.

    Parameters
    ----------
    ds_res : xr.Dataset
        Dataset containing the reservoir layers.

    Returns
    -------
    xr
        returns the dataset with the rating curve layers set to int data type.
    """
    convert_to_int = [
        "reservoir_rating_curve",
        "reservoir_storage_curve",
        "reservoir_lower_id",
    ]

    for var in convert_to_int:
        if var in ds_res:
            fill_value = ds_res[var].raster.nodata
            fill_value_new = int(fill_value) if not np.isnan(fill_value) else -999
            # replace NaN with fill_value_new
            ds_res[var] = ds_res[var].fillna(fill_value_new)
            ds_res[var] = ds_res[var].where(ds_res[var] != fill_value, fill_value_new)
            ds_res[var] = ds_res[var].astype(np.int32)
            ds_res[var].raster.set_nodata(fill_value_new)
    return ds_res
