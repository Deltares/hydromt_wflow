# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gp
import logging

logger = logging.getLogger(__name__)


__all__ = ["waterbodymaps", "reservoirattrs", "lakeattrs"]


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


def reservoirattrs(
    gdf,
    priorityJRC=False,
    perc_norm=50,
    perc_min=20,
    usehe=True,
    logger=logger,
):
    """Returns reservoir attributes (see list below) needed for modelling. 
    For some attributes, download data from the JRC database (Peker, 2016) using hydroengine.

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
    priorityJRC : boolean, optional
        Specify if attributes are more reliable from the gdf attributes or from the JRC database.
    perc_norm : int, optional
        Percentile for normal (operational) surface area
    perc_min: int, optional 
        Percentile for minimal (operational) surface area
    usehe : bool, optional
        If True use hydroengine to get reservoir timeseries

    Returns
    -------
    df_out : pandas.DataFrame
        DataFrame containing reservoir attributes.
    df_plot : pandas.DataFrame
        DataFrame containing debugging values for reservoir building.
    """
    if usehe:
        try:
            import hydroengine as he
        except ImportError:
            usehe = False
            logger.debug(
                "HydroEngine package not found, using default reservoir attribute values."
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

    # Create similar dataframe for JRC timeseries
    df_JRC = pd.DataFrame(
        index=range(len(gdf["waterbody_id"])),
        columns=(["resid", "maxarea", "normarea", "minarea", "capmin", "capmax"]),
    )
    df_JRC["resid"] = gdf["waterbody_id"].values

    # Create dtaframe for accuracy plots
    df_plot = pd.DataFrame(
        index=range(len(gdf["waterbody_id"])),
        columns=(["resid", "factor", "accuracy_min", "accuracy_norm"]),
    )
    df_plot["resid"] = gdf["waterbody_id"].values

    # Get JRC data for each reservoir from hydroengine
    df_JRC.loc[:, "maxarea"] = np.nan
    df_JRC.loc[:, "normarea"] = np.nan
    df_JRC.loc[:, "minarea"] = np.nan

    if usehe:
        for i in range(len(gdf["waterbody_id"])):
            ids = str(gdf["waterbody_id"].iloc[i])
            try:
                logger.debug(f"Downloading HydroEngine timeseries for reservoir {ids}")
                time_series = he.get_lake_time_series(
                    int(gdf["Hylak_id"].iloc[i]), "water_area"
                )
                area_series = np.array(time_series["water_area"])  # [m2]
                area_series_nozeros = area_series[area_series > 0]
                df_JRC.loc[i, "maxarea"] = area_series_nozeros.max()
                df_JRC.loc[i, "normarea"] = np.percentile(
                    area_series_nozeros, perc_norm, axis=0
                )
                df_JRC.loc[i, "minarea"] = np.percentile(
                    area_series_nozeros, perc_min, axis=0
                )
            except:
                logger.warning(
                    f"No HydroEngine time series available for reservoir {ids}!"
                )

    # Compute resdemand and resmaxrelease either from average discharge
    if "Dis_avg" in gdf.columns:
        df_out["resdemand"] = gdf["Dis_avg"].values * 0.5
        df_out["resmaxrelease"] = gdf["Dis_avg"].values * 4.0

    # Get resarea either from JRC or database depending on priorityJRC
    if "Area_avg" in gdf.columns:
        df_out["resarea"] = gdf["Area_avg"].values
        if priorityJRC:
            df_out.loc[pd.notna(df_JRC["maxarea"]), "resarea"] = df_JRC["maxarea"][
                pd.notna(df_JRC["maxarea"])
            ].values
        else:
            df_out.loc[pd.isna(df_out["resarea"]), "resarea"] = df_JRC["maxarea"][
                pd.isna(df_out["resarea"])
            ].values
    else:
        df_out["resarea"] = df_JRC["maxarea"].values

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

    # Then compute from JRC data and fill or replace the previous values depending on priorityJRC
    # TODO for now assumes that the reservoir-db is used (combination of GRanD and HydroLAKES)
    for i in range(len(gdf["waterbody_id"])):
        # Initialise values
        # import pdb; pdb.set_trace()
        dam_height = np.nanmax([gdf["Dam_height"].iloc[i], 0.0])
        max_level = np.nanmax([gdf["Depth_avg"].iloc[i], 0.0])
        max_area = np.nanmax([df_out["resarea"].iloc[i], 0.0])
        max_cap = np.nanmax([df_out["resmaxvolume"].iloc[i], 0.0])
        norm_area = np.nanmax([df_JRC["normarea"].iloc[i], 0.0])
        norm_cap = np.nanmax([gdf["Capacity_norm"].iloc[i], 0.0])
        min_area = np.nanmax([df_JRC["minarea"].iloc[i], 0.0])
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
        df_JRC.loc[i, "capmin"] = min_cap_f
        df_JRC.loc[i, "capmax"] = norm_cap_f
        df_plot.loc[i, "factor"] = factor_shape
        df_plot.loc[i, "accuracy_min"] = accuracy_min
        df_plot.loc[i, "accuracy_norm"] = accuracy_norm

    # Depending on priority JRC update fullfrac and min frac
    if priorityJRC:
        df_out.resminfrac = df_JRC["capmin"].values / df_out["resmaxvolume"].values
        df_out.resfullfrac = df_JRC["capmax"].values / df_out["resmaxvolume"].values
    else:
        df_out.loc[pd.isna(df_out["resminfrac"]), "resminfrac"] = (
            df_JRC.loc[pd.isna(df_out["resminfrac"]), "capmin"].values
            / df_out.loc[pd.isna(df_out["resminfrac"]), "resmaxvolume"].values
        )
        df_out.loc[pd.isna(df_out["resfullfrac"]), "resfullfrac"] = (
            df_JRC.loc[pd.isna(df_out["resfullfrac"]), "capmax"].values
            / df_out.loc[pd.isna(df_out["resfullfrac"]), "resmaxvolume"].values
        )

    return df_out, df_plot


def lakeattrs(
    ds: xr.Dataset,
    gdf: gp.GeoDataFrame,
    rating_dict: dict = dict(),
    logger=logger,
):
    """
    Returns lake attributes (see list below) needed for modelling. 
    If rating_dict is not empty, prepares also rating tables for wflow

    The following reservoir attributes are calculated:\
    - waterbody_id : waterbody id\
    - LakeArea : lake area [m2]\
    - LakeAvgLevel: lake average level [m]\
    - LakeAvgOut: lake average outflow [m3/s]\
    - Lake_b: lake rating curve coefficient [-]\
    - Lake_e: lake rating curve exponent [-]\
    - LakeStorFunc: option to compute storage curve [-]\
    - LakeOutflowFunc: option to compute rating curve [-]\
    - LakeThreshold: minimium threshold for lake outflow [m]\
    - LinkedLakeLocs: id of linked lake location if any\
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the lake locations and area
    gdf : gp.GeoDataFrame
        GeoDataFrame containing the lake locations and area
    rating_dict : dict, optional
        Dictionary containing the rating curve parameters, by default dict()
    
    Returns
    -------
    ds : xr.Dataset
        Dataset containing the lake locations with the attributes
    gdf : gp.GeoDataFrame
        GeoDataFrame containing the lake locations with the attributes
    rating_curves : dict
        Dictionary containing the rating curves in wflow format
    """
    # rename to param values
    gdf = gdf.rename(
        columns={
            "Area_avg": "LakeArea",
            "Depth_avg": "LakeAvgLevel",
            "Dis_avg": "LakeAvgOut",
        }
    )
    # Minimum value for LakeAvgOut
    LakeAvgOut = gdf["LakeAvgOut"].copy()
    gdf["LakeAvgOut"] = np.maximum(gdf["LakeAvgOut"], 0.01)
    if "Lake_b" not in gdf.columns:
        gdf["Lake_b"] = gdf["LakeAvgOut"].values / (gdf["LakeAvgLevel"].values) ** (2)
    if "Lake_e" not in gdf.columns:
        gdf["Lake_e"] = 2
    if "LakeThreshold" not in gdf.columns:
        gdf["LakeThreshold"] = 0.0
    if "LinkedLakeLocs" not in gdf.columns:
        gdf["LinkedLakeLocs"] = 0
    if "LakeStorFunc" not in gdf.columns:
        gdf["LakeStorFunc"] = 1
    if "LakeOutflowFunc" not in gdf.columns:
        gdf["LakeOutflowFunc"] = 3

    # Check if some LakeAvgOut values have been replaced
    if not np.all(LakeAvgOut == gdf["LakeAvgOut"]):
        logger.warning(
            "Some values of LakeAvgOut have been replaced by a minimum value of 0.01m3/s"
        )

    # Check if rating curve is provided
    rating_curves = dict()
    if len(rating_dict) != 0:
        # Assume one rating curve per lake index
        for id in gdf["waterbody_id"].values:
            if id in rating_dict.keys():
                df_rate = rating_dict[id]
                # Prepare the right tables for wflow
                # Update LakeStor and LakeOutflowFunc
                # Storage
                if "volume" in df_rate.columns:
                    gdf.loc[gdf["waterbody_id"] == id, "LakeStorFunc"] = 2
                    df_stor = df_rate[["elevtn", "volume"]].dropna(
                        subset=["elevtn", "volume"]
                    )
                    df_stor.rename(columns={"elevtn": "H", "volume": "S"}, inplace=True)
                    # add to rating_curves
                    rating_curves[f"lake_sh_{id}"] = df_stor
                else:
                    logger.warning(
                        f"Storage data not available for lake {id}. Using default S=AH"
                    )
                # Rating
                if "discharge" in df_rate.columns:
                    gdf.loc[gdf["waterbody_id"] == id, "LakeOutflowFunc"] = 1
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
                    rating_curves[f"lake_hq_{id}"] = df_q
                else:
                    logger.warning(
                        f"Rating data not available for lake {id}. Using default Modified Puls Approach"
                    )

    # Create raster of lake params
    lake_params = [
        "waterbody_id",
        "LakeArea",
        "LakeAvgLevel",
        "LakeAvgOut",
        "Lake_b",
        "Lake_e",
        "LakeStorFunc",
        "LakeOutflowFunc",
        "LakeThreshold",
        "LinkedLakeLocs",
    ]

    gdf_org_points = gp.GeoDataFrame(
        gdf[lake_params],
        geometry=gp.points_from_xy(gdf.xout, gdf.yout),
    )

    for name in lake_params[1:]:
        da_lake = ds.raster.rasterize(
            gdf_org_points, col_name=name, dtype="float32", nodata=-999
        )
        ds[name] = da_lake

    return ds, gdf, rating_curves
