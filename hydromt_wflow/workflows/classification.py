# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gp
import logging

logger = logging.getLogger(__name__)


__all__ = ["classification_hru"]


def classification_hru(
    ds,
    ds_like,
    hand_th = 5.9,
    slope_th = 0.07,
    logger=logger,
):
    """Returns percentage hillslope, plateau and wetland at model resolution based on
    high resolution hand and slope data. 
    
    Parameters
    ----------
    ds : xarray.DataArray
        Dataset containing gridded high resolution hand and slope data.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    hand_th : float, optional
        hand threshold [m] to delineate hydrological response units (HRU)
    slope_th : float, optional
        slope threshold [-] to delineate hydrological response units (HRU)

    Returns
    -------
    ds_out : xarray.DataArray
        Dataset containing gridded percentages of each HRU
    """
    logger.info("Classification of the basin into hydrological response units (HRU)")

    hand = ds["hnd"]
    slope = ds["lndslp"]

    ds["hru"] = xr.where((hand>hand_th) & (slope>slope_th), 1, #hillslope
                           xr.where((hand>hand_th) & (slope<=slope_th), 2, #plateau
                           3, #wetland
                           ))
    
    ds["percentH"] = xr.where(ds["hru"] == 1, 1.0, 0)
    ds["percentP"] = xr.where(ds["hru"] == 2, 1.0, 0)
    ds["percentW"] = xr.where(ds["hru"] == 3, 1.0, 0)

    ds_out = ds[["percentH", "percentP", "percentW"]].raster.reproject_like(ds_like, method = "average")

    # for writing pcraster map files a scalar nodata value is required
    nodata = -9999.0
    for var in ds_out:
        ds_out[var] = ds_out[var].fillna(nodata)
        ds_out[var].raster.set_nodata(nodata)

    return ds_out

