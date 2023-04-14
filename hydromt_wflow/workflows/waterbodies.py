# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gp
import logging
import shapely
import json

logger = logging.getLogger(__name__)


__all__ = ["waterbodymaps", "reservoirattrs"]


def waterbodymaps(
    gdf,
    ds_like,
    wb_type="reservoir",
    uparea_name="uparea",
    logger=logger,
):
    """Returns waterbody (reservoir/lake) maps (see list below) at model resolution based on gridded 
    upstream area data input or outlet coordinates. 

    The following waterbody maps are calculated:\
    - resareas/lakeareas : waterbody areas mask [ID]\
    - reslocs/lakelocs : waterbody outlets [ID]
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing reservoirs/lakes geometries and attributes.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    uparea_name : str, optional
        Name of uparea variable in ds_like. If None then database coordinates will be 
        used to setup outlets
    wb_type : str, optional either "reservoir" or "lake"
        Option to change the name of the maps depending on reservoir or lake

    Returns
    -------
    ds_out : xarray.DataArray
        Dataset containing gridded waterbody data
    """
    # Rasterize the GeoDataFrame to get the areas mask of waterbodies
    res_id = gdf["waterbody_id"].values
    da_wbmask = ds_like.raster.rasterize(
        gdf,
        col_name="waterbody_id",
        nodata=-999,
        all_touched=True,
        dtype=None,
        sindex=False,
    )
    da_wbmask = da_wbmask.rename("resareas")
    da_wbmask.attrs.update(_FillValue=-999)
    ds_out = da_wbmask.to_dataset()
    if not np.all(np.isin(res_id, ds_out["resareas"])):
        gdf = gdf.loc[np.isin(res_id, ds_out["resareas"])]
        nskipped = res_id.size - gdf.index.size
        res_id = gdf["waterbody_id"].values
        logger.warning(
            f"{nskipped} reservoirs are not succesfully rasterized and skipped!!"
            " Consider increasing the lakes min_area threshold."
        )

    # Initialize the waterbody outlet map
    ds_out["reslocs"] = xr.full_like(ds_out["resareas"], -999)
    # If an upstream area map is present in the model, gets outlets coordinates using/
    # the maximum uparea in each waterbody mask to match model river network.
    if uparea_name is not None and uparea_name in ds_like.data_vars:
        logger.debug(f"Setting {wb_type} outlet map based maximum upstream area.")
        # create dataframe with x and y coord to be filled in either from uparea or from xout and yout in hydrolakes data
        outdf = gdf[["waterbody_id"]].assign(xout=np.nan, yout=np.nan)
        ydim = ds_like.raster.y_dim
        xdim = ds_like.raster.x_dim
        for i in res_id:
            res_acc = ds_like[uparea_name].where(ds_out["resareas"] == i)
            # max_res_acc = np.amax(res_acc.values())
            max_res_acc = res_acc.where(res_acc == res_acc.max(), drop=True).squeeze()
            yacc = max_res_acc[ydim].values
            xacc = max_res_acc[xdim].values
            ds_out["reslocs"].loc[{f"{ydim}": yacc, f"{xdim}": xacc}] = i
            outdf.loc[outdf.waterbody_id == i, "xout"] = xacc
            outdf.loc[outdf.waterbody_id == i, "yout"] = yacc
        outgdf = gp.GeoDataFrame(
            outdf, geometry=gp.points_from_xy(outdf.xout, outdf.yout)
        )

    # ELse use coordinates from the waterbody database
    elif "xout" in gdf.columns and "yout" in gdf.columns:
        logger.debug(f"Setting {wb_type} outlet map based on coordinates.")
        outdf = gdf[["waterbody_id", "xout", "yout"]]
        outgdf = gp.GeoDataFrame(
            outdf, geometry=gp.points_from_xy(outdf.xout, outdf.yout)
        )
        ds_out["reslocs"] = ds_like.raster.rasterize(
            outgdf, col_name="waterbody_id", nodata=-999
        )
    # Else outlet map is equal to areas mask map
    else:
        ds_out["reslocs"] = ds_out["resareas"]
        logger.warning(
            f"Neither upstream area map nor {wb_type}'s outlet coordinates found. "
            f"Setting {wb_type} outlet map equal to the area map."
        )
        # dummy outgdf
        outgdf = gdf[["waterbody_id"]]
    ds_out["reslocs"].attrs.update(_FillValue=-999)
    # fix dtypes
    ds_out["reslocs"] = ds_out["reslocs"].astype("int32")
    ds_out["resareas"] = ds_out["resareas"].astype("float32")

    if wb_type == "lake":
        ds_out = ds_out.rename({"resareas": "lakeareas", "reslocs": "lakelocs"})

    return ds_out, outgdf


def reservoirattrs(gdf, timeseries_fn=None, perc_norm=50, perc_min=20, logger=logger):
    """Returns reservoir attributes (see list below) needed for modelling. 
    When specified, some of the reservoir attributes can be derived from earth observation data. 
    Two options are currently available: 1. Global Water Watch data (Deltares, 2022) using gwwapi and 2. JRC (Peker, 2016) using hydroengine.

    The following reservoir attributes are calculated:\
    - resmaxvolume : reservoir maximum volume [m3]\
    - resarea : reservoir area [m2]\
    - resdemand : reservoir demand flow [m3/s]\
    - resmaxrelease : reservoir maximum release flow [m3/s]\
    - resfullfrac : reservoir targeted full volume fraction [m3/m3]\
    - resminfrac : reservoir targeted minimum volume fraction [m3/m3]\
    
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing reservoirs geometries and attributes.
    timeseries_fn : str, optional 
        Name of database from which time series of reservoir surface water area will be retrieved.
        Currently available: ['jrc', 'gww']
        Defaults to Deltares' Global Water Watch database. 
    perc_norm : int, optional
        Percentile for normal (operational) surface area
    perc_min: int, optional 
        Percentile for minimal (operational) surface area

    Returns
    -------
    df_out : pandas.DataFrame
        DataFrame containing reservoir attributes.
    df_plot : pandas.DataFrame
        DataFrame containing debugging values for reservoir building.
    df_ts : pandas.DataFrame
        DataFrame containing all downloaded reservoir time series. 
    """

    if timeseries_fn == "jrc":
        try:
            import hydroengine as he

            logger.info("Using reservoir timeseries from JRC via hydroengine.")
        except:
            raise ImportError(
                "hydroengine package not found, cannot download jrc reservoir timeseries."
            )

    elif timeseries_fn == "gww":
        try:
            from gwwapi import client as cli
            from gwwapi import utils

            logger.info("Using reservoir timeseries from GWW via gwwapi.")
        except:
            raise ImportError(
                "gwwapi package not found, cannot download gww reservoir timeseries."
            )
    elif timeseries_fn is not None:
        raise ValueError(
            f"timeseries_fn argument {timeseries_fn} not understood, please use one of [gww, jrc] or None."
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
                "resdemand",
                "resmaxrelease",
                "resfullfrac",
                "resminfrac",
            ]
        ),
    )
    df_out["resid"] = gdf["waterbody_id"].values

    # Create similar dataframe for EO time series
    df_EO = pd.DataFrame(
        index=range(len(gdf["waterbody_id"])),
        columns=(["resid", "maxarea", "normarea", "minarea", "capmin", "capmax"]),
    )
    df_EO["resid"] = gdf["waterbody_id"].values

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
                # of all dowloaded reservoirs, with an outer join on the datetime index
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
                df_EO.loc[i, "maxarea"] = area_series_nozeros.max()
                df_EO.loc[i, "normarea"] = np.percentile(
                    area_series_nozeros, perc_norm, axis=0
                )
                df_EO.loc[i, "minarea"] = np.percentile(
                    area_series_nozeros, perc_min, axis=0
                )
            except:
                logger.warning(
                    f"No HydroEngine time series available for reservoir {ids}!"
                )

    if timeseries_fn == "gww":
        # get bounds from gdf input as JSON object that can be used in post request with the gww api
        gdf_bounds = json.dumps(
            shapely.geometry.box(*gdf.total_bounds, ccw=True).__geo_interface__
        )
        # get reservoirs wihtin these bounds
        gww_reservoirs = cli.get_reservoirs_by_geom(gdf_bounds)
        # from the response, create a dictonary, linking the gww_id to the hylak_id (used in the default reservoir database)
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
                # of all dowloaded reservoirs, with an outer join on the datetime index
                ts_series = utils.to_timeseries(
                    time_series, name=f'{int(gdf["Hylak_id"].iloc[i])}'
                ).drop_duplicates()
                df_ts = pd.concat([df_ts, ts_series], join="outer", axis=1)

                # Compute stats
                area_series = utils.to_timeseries(time_series)["area"].to_numpy()
                area_series_nozeros = area_series[area_series > 0]
                df_EO.loc[i, "maxarea"] = area_series_nozeros.max()
                df_EO.loc[i, "normarea"] = np.percentile(
                    area_series_nozeros, perc_norm, axis=0
                )
                df_EO.loc[i, "minarea"] = np.percentile(
                    area_series_nozeros, perc_min, axis=0
                )
            except:
                logger.warning(f"No GWW time series available for reservoir {ids}!")

    # Sort timeseries dataframe (will be saved to root as .csv later)
    df_ts = df_ts.sort_index()

    # Compute resdemand and resmaxrelease either from average discharge
    if "Dis_avg" in gdf.columns:
        df_out["resdemand"] = gdf["Dis_avg"].values * 0.5
        df_out["resmaxrelease"] = gdf["Dis_avg"].values * 4.0

    # Get resarea either from EO or database depending
    if "Area_avg" in gdf.columns:
        df_out["resarea"] = gdf["Area_avg"].values
        if timeseries_fn is not None:
            df_out.loc[pd.notna(df_EO["maxarea"]), "resarea"] = df_EO["maxarea"][
                pd.notna(df_EO["maxarea"])
            ].values
        else:
            df_out.loc[pd.isna(df_out["resarea"]), "resarea"] = df_EO["maxarea"][
                pd.isna(df_out["resarea"])
            ].values
    else:
        df_out["resarea"] = df_EO["maxarea"].values

    # Get resmaxvolume from database
    if "Vol_avg" in gdf.columns:
        df_out["resmaxvolume"] = gdf["Vol_avg"].values

    # Compute target min and max fractions
    # First look if data is available from the database
    if "Capacity_min" in gdf.columns:
        df_out["resminfrac"] = gdf["Capacity_min"].values / df_out["resmaxvolume"]
        df_plot["accuracy_min"] = np.repeat(1.0, len(df_plot["accuracy_min"]))
    if "Capacity_norm" in gdf.columns:
        df_out["resfullfrac"] = gdf["Capacity_norm"].values / df_out["resmaxvolume"]
        df_plot["accuracy_norm"] = np.repeat(1.0, len(df_plot["accuracy_norm"]))

    # Then compute from EO data and fill or replace the previous values (if a valid source is provided)
    # TODO for now assumes that the reservoir-db is used (combination of GRanD and HydroLAKES)
    gdf = gdf.fillna(value=np.nan)
    for i in range(len(gdf["waterbody_id"])):
        # Initialise values
        # import pdb; pdb.set_trace()
        dam_height = np.nanmax([gdf["Dam_height"].iloc[i], 0.0])
        max_level = np.nanmax([gdf["Depth_avg"].iloc[i], 0.0])
        max_area = np.nanmax([df_out["resarea"].iloc[i], 0.0])
        max_cap = np.nanmax([df_out["resmaxvolume"].iloc[i], 0.0])
        norm_area = np.nanmax([df_EO["normarea"].iloc[i], 0.0])
        norm_cap = np.nanmax([gdf["Capacity_norm"].iloc[i], 0.0])
        min_area = np.nanmax([df_EO["minarea"].iloc[i], 0.0])
        min_cap = np.nanmax([gdf["Capacity_min"].iloc[i], 0.0])
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

        #        # adjust factor based on chosen method
        #        if method == 0:
        #            factor_used = factor_shape
        #        elif method == 1:
        #            factor_used = 1.0
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

        # CHECK minumum level (1)
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

        # CHECK minumum level (2)
        if min_cap_f > norm_cap_f:
            logger.warning("min_cap > norm_cap! setting min_cap equal to norm_cap.")
            min_cap_f = norm_cap_f
            accuracy_min = 5

        # Resume results
        df_EO.loc[i, "capmin"] = min_cap_f
        df_EO.loc[i, "capmax"] = norm_cap_f
        df_plot.loc[i, "factor"] = factor_shape
        df_plot.loc[i, "accuracy_min"] = accuracy_min
        df_plot.loc[i, "accuracy_norm"] = accuracy_norm

    # Depending on priority EO update fullfrac and min frac
    if timeseries_fn is not None:
        df_out.resminfrac = df_EO["capmin"].values / df_out["resmaxvolume"].values
        df_out.resfullfrac = df_EO["capmax"].values / df_out["resmaxvolume"].values
    else:
        df_out.loc[pd.isna(df_out["resminfrac"]), "resminfrac"] = (
            df_EO.loc[pd.isna(df_out["resminfrac"]), "capmin"].values
            / df_out.loc[pd.isna(df_out["resminfrac"]), "resmaxvolume"].values
        )
        df_out.loc[pd.isna(df_out["resfullfrac"]), "resfullfrac"] = (
            df_EO.loc[pd.isna(df_out["resfullfrac"]), "capmax"].values
            / df_out.loc[pd.isna(df_out["resfullfrac"]), "resmaxvolume"].values
        )

    return df_out, df_plot, df_ts
