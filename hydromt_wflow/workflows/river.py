# -*- coding: utf-8 -*-
"""
@author: haag (2019)

Derives river/stream widths to construct a riverwidth map (wflow_riverwidth.map) to be used in a wflow model.
"""

import os
from os.path import join
import json
import numpy as np
from scipy.optimize import curve_fit
import xarray as xr
import pandas as pd
import logging
import pyflwdir
from pyflwdir import FlwdirRaster

from hydromt import gis_utils, stats, flw
from hydromt_wflow import DATADIR  # global var

logger = logging.getLogger(__name__)

__all__ = [
    "river",
    "river_width",
]

RESAMPLING = {"climate": "nearest", "precip": "average"}
NODATA = {"discharge": -9999}


def power_law(x, a, b):
    return a * np.power(x, b)


def river(
    ds,
    ds_like=None,
    river_upa=30.0,
    slope_len=1e3,
    channel_dir="up",
    min_rivlen_ratio=0.1,
    logger=logger,
    **kwargs,
):
    """Returns river maps

    The output maps are:\
    - rivmsk : river mask based on upstream area threshold on upstream area\
    - rivlen : river length [m], minimum set to 1/4 cell res\
    - rivslp : smoothed river slope [m/m]\
    - rivwth_obs : river width at pixel outlet (if in ds)
    - rivbed : elevation of the river bed based on pixel outlet 

    Parameters
    ----------
    ds: xr.Dataset
        dataset containing "flwdir", "uparea", "elevtn" variables; and optional
        "rivwth" variable
    ds_like: xr.Dataset, optional
        dataset with output grid, must contain "uparea", for subgrid rivlen/slp
        must contain "x_out", "y_out". If None, takes ds grid for output
    river_upa: float
        minimum threshold to define river cell & pixels, by default 30 [km2]
    slope_len: float
        minimum length over which to calculate the river slope, by default 1000 [m]
    min_rivlen_ratio: float
        minimum global river length to avg. cell resolution ratio, by default 0.1
    channel_dir: {"up", "down"}
        flow direcition in which to calculate (subgrid) river length and width
    
    Returns:
    ds_out: xr.Dataset
        Dataset with output river attributes
    """

    # sort options
    dvars = ["flwdir", "uparea", "elevtn"]
    dvars_like = ["x_out", "y_out", "uparea"]
    subgrid = True
    for name in dvars:
        if name not in ds.data_vars:
            raise ValueError(f"Dataset variable {name} not in ds.")
    if ds_like is None or not np.all([v in ds_like for v in dvars_like]):
        subgrid = False
        logger.info("River length and slope are calculated at model resolution.")
        ds_like = ds

    logger.debug(f"Set river mask with upstream area threshold: {river_upa} km2.")
    dims = ds_like.raster.dims
    _mask = ds_like["uparea"] > river_upa  # initial mask
    _mask_org = ds["uparea"].values >= river_upa  # highres riv mask

    logger.debug("Derive river length.")
    flwdir = flw.flwdir_from_da(ds["flwdir"], **kwargs)
    if subgrid == False:
        # get cell index of river cells
        idxs_out = np.arange(ds_like.raster.size).reshape(ds_like.raster.shape)
        idxs_out = np.where(_mask, idxs_out, flwdir._mv)
    else:
        # get subgrid outlet pixel index
        idxs_out = ds.raster.xy_to_idx(
            xs=ds_like["x_out"].values,
            ys=ds_like["y_out"].values,
            mask=_mask.values,
            nodata=flwdir._mv,
        )

    # get river length based on on-between distance between two outlet pixels
    msk = _mask.values
    rivlen = flwdir.subgrid_rivlen(
        idxs_out=idxs_out,
        mask=_mask_org,
        direction=channel_dir,
        unit="m",
    )

    # minimum river length equal 10% of cellsize
    xres, yres = ds_like.raster.res
    if ds_like.raster.crs.is_geographic:  # convert degree to meters
        xres, yres = gis_utils.cellres(ds_like.raster.ycoords.values.mean(), xres, yres)
    min_len = np.mean(np.abs([xres, yres])) * min_rivlen_ratio
    rivlen = np.where(msk, np.maximum(rivlen, min_len), -9999)

    # bed level
    # readout elevation of bedlevel on outlet pixels
    bed_level = ds["elevtn"].values.flat[idxs_out]

    # make model resolution masked flwdir for rivers
    da_flw_model = ds_like["flwdir"].copy()
    da_flw_model = da_flw_model.assign_coords(mask=_mask)
    # indices van flwdir
    flwdir_model = flw.flwdir_from_da(da_flw_model, **kwargs)

    # hydrologically adjust
    bed_level_adjust = flwdir_model.dem_adjust(bed_level)
    #    (bed_level[msk] != bed_level_adjust[msk]).sum()

    # mask to keep river cells
    bed_level_adjust = np.where(msk, bed_level_adjust, -9999)

    # add diff and log
    diff = flwdir_model.downstream(bed_level_adjust) - bed_level_adjust
    if np.all(diff <= 0) == False:
        logger.warning(
            "Erroneous increase in riverbed level in downstream direction found."
        )

    # set mean length at pits when taking the downstream length
    if channel_dir == "down":
        rivlen.flat[flwdir.idxs_pit] = np.mean(rivlen[_mask])

    # get river slope as derivative of elevation around outlet pixels
    logger.debug("Derive river slope.")
    rivslp = flwdir.subgrid_rivslp(
        idxs_out=idxs_out,
        elevtn=ds["elevtn"].values,
        mask=_mask_org,
        length=slope_len,
    )
    rivslp = np.where(msk, rivslp, -9999)

    # create xarray dataset for river mask, length and width
    ds_out = xr.Dataset(coords=ds_like.raster.coords)
    ds_out["rivmsk"] = xr.Variable(dims, msk, attrs=dict(_FillValue=0))
    attrs = dict(_FillValue=-9999, unit="m")
    ds_out["rivlen"] = xr.Variable(dims, rivlen, attrs=attrs)
    attrs = dict(_FillValue=-9999, unit="m.m-1")
    ds_out["rivslp"] = xr.Variable(dims, rivslp, attrs=attrs)
    attrs = dict(_FillValue=-9999, unit="m")
    ds_out["rivbed"] = xr.Variable(dims, bed_level_adjust, attrs=attrs)

    # add river width at outlet pixels if in source
    if "rivwth" in ds:
        rivwth = np.full(msk.shape, -9999.0, dtype=np.float32)
        rivwth[msk] = ds["rivwth"].values.flat[idxs_out[msk]]
        attrs = dict(_FillValue=-9999, unit="m")
        ds_out["rivwth_obs"] = xr.Variable(dims, rivwth, attrs=attrs)

    return ds_out, flwdir


def river_width(
    ds_like,
    flwdir,
    data=dict(),
    fit=False,
    fill=True,
    fill_outliers=True,
    min_wth=1,
    mask_names=[],
    predictor="discharge",
    rivwth_name="rivwth",
    obs_postfix="_obs",
    rivmsk_name="rivmsk",
    a=None,
    b=None,
    logger=logger,
    **kwargs,
):
    nopars = a is None or b is None  # no manual a, b parameters
    fit = fit or (nopars and predictor not in ["discharge"])  # fit power-law on the fly
    nowth = (
        f"{rivwth_name}{obs_postfix}" not in ds_like
    )  # no obseved with in staticmaps
    fill = fill and nowth == False  # fill datagaps and masked areas (lakes/res) in obs

    if nowth and fit:
        raise ValueError(
            f'Observed rivwth "{rivwth_name}{obs_postfix}" required to fit riverwidths.'
        )
    elif nowth:
        logger.warning(f'Observed rivwth "{rivwth_name}{obs_postfix}" not found.')

    # get predictor
    logger.debug(f'Deriving predictor "{predictor}" values')
    if predictor == "discharge":
        values, pars = _discharge(ds_like, flwdir=flwdir, logger=logger, **data)
        if fit == False:
            a, b = pars
    # TODO: check units
    elif predictor == "precip":
        values = _precip(ds_like, flwdir=flwdir, logger=logger, **data)
    else:
        if predictor not in ds_like:
            raise ValueError(f"required {predictor} variable missing in staticmaps.")
        values = ds_like[predictor].values

    # read river width observations
    if nowth == False:
        rivwth_org = ds_like[f"{rivwth_name}{obs_postfix}"].values
        nodata = ds_like[f"{rivwth_name}{obs_postfix}"].raster.nodata
        rivmsk = rivwth_org != nodata
        rivwth_org = np.where(rivmsk, rivwth_org, -9999)

        # mask zero and negative riverwidth and waterbodies if present
        mask = rivwth_org > min_wth
        for name in mask_names:
            if name in ds_like:
                mask[ds_like[name].values != ds_like[name].raster.nodata] = False
                logger.debug(f"{name} masked out in rivwth data.")
            else:
                logger.warning(f'mask variable "{name}" not found in maps.')

        # fit predictor
        wth = rivwth_org[mask]
        val = values[mask]
        if fit:
            logger.info(f"Fitting power-law a,b parameters based on {predictor}")
            a, b = _width_fit(wth, val, mask, logger=logger)
        wsim = power_law(val, a, b)
        res = np.abs(wsim - wth)
        outliers = np.logical_and(res > 200, (res / wsim) > 1.0)
        pout = np.sum(outliers) / outliers.size * 100

        # validate
        wsim = power_law(val[~outliers], a, b)
        nse = stats._nse(wsim, wth[~outliers])
        logger.info(
            f'Using "width = a*{predictor}^b"; where a={a:.2f}, b={b:.2f} '
            f"(nse={nse:.3f}; outliers={pout:.2f}%)"
        )

    # compute new riverwidth
    rivmsk = ds_like[rivmsk_name].values
    rivwth_out = np.full(ds_like.raster.shape, -9999.0, dtype=np.float32)
    rivwth_out[rivmsk] = np.maximum(min_wth, power_law(values[rivmsk], a, b))

    # overwrite rivwth data with original data at valid points
    if fill:
        if fill_outliers:
            mask[mask] = ~outliers
            logger.info(f"masking {np.sum(outliers):.0f} outliers")
        rivwth_out[mask] = wth[~outliers]

    fills = "for missing data" if fill else "globally"
    logger.info(f"rivwth set {fills} based on width-{predictor} relationship.")

    # return xarray dataarray
    attrs = dict(_FillValue=-9999, unit="m")
    return xr.DataArray(rivwth_out, dims=ds_like.raster.dims, attrs=attrs)


def _width_fit(
    wth,
    val,
    mask,
    p0=[0.15, 0.65],
    logger=logger,  # rhine uparea based
):
    outliers = np.full(np.sum(mask), False, dtype=np.bool)
    a, b = None, None
    # check if sufficient data
    if np.sum(mask) > 10:
        n = 0
        # fit while removing outliers.
        while n < 20:
            if np.sum(mask[mask][~outliers]) > 10:
                pars = curve_fit(
                    f=power_law,
                    xdata=val[~outliers],
                    ydata=wth[~outliers],
                    p0=p0,
                    bounds=([0.01, 0.01], [100.0, 1.0]),
                )[0]
                # print(n, pars, np.sum(~outliers))
                if np.allclose(p0, pars):
                    break
                # outliers based on absolute and relative diff
                wsim = power_law(val, *pars)
                res = np.abs(wsim - wth)
                outliers = np.logical_and(res > 200, (res / wsim) > 1.0)
                if np.sum(outliers) > np.sum(mask) * 0.2:  # max 20 % outliers
                    pars = p0
                    outliers[:] = False
                    logger.warning("Fit not successful.")
                    break
                p0 = pars
                n += 1
        a, b = pars
    elif a is None or b is None:
        a, b = p0
        logger.warning("Insufficient data points, using global parameters.")

    return a, b


def _precip(ds_like, flwdir, da_precip, logger=logger):
    """"""
    precip = (
        da_precip.raster.reproject_like(ds_like, method=RESAMPLING["precip"]).values
        / 1e3
    )  # [m/yr]
    precip = np.maximum(precip, 0)
    lat, lon = ds_like.raster.ycoords.values, ds_like.raster.xcoords.values
    area = gis_utils.reggrid_area(lat, lon)
    # 10 x average flow
    accu_precip = flwdir.accuflux(precip * area / (86400 * 365) * 10)  # [m3/s]
    return accu_precip


def _discharge(ds_like, flwdir, da_precip, da_climate, logger=logger):
    """"""
    # read clim classes and regression parameters from data dir
    source_precip = da_precip.name
    source_climate = da_climate.name
    fn_regr = join(DATADIR, "rivwth", f"regr_{source_precip}.csv")
    fn_clim = join(DATADIR, "rivwth", f"{source_climate}.csv")
    clim_map = pd.read_csv(fn_clim, index_col="class")
    regr_map = pd.read_csv(fn_regr, index_col="source").loc[source_climate]
    regr_map = regr_map.set_index("base_class")

    # get overall catchment climate classification (based on mode)
    # TODO: convert to xarray method by using scipy.stats.mode in xr.apply_ufuncs
    # TODO: reproject climate and mask cells outside basin
    np_climate = da_climate.values
    basin_climate = np.bincount(np_climate.flatten()).argmax()
    # convert detailed climate classification to higher level base class
    base_class, class_name = clim_map.loc[basin_climate]
    logger.debug(f"Basin base climate classification: {base_class} ({class_name})")
    params = regr_map.loc[base_class]

    # precipitation (TO-DO: add checks that were present in old version?)
    precip = da_precip.raster.reproject_like(
        ds_like, method=RESAMPLING["precip"]
    ).values
    # apply scaling factor to make sure accumulated precipitation will stay
    # (approximately) the same
    scaling_factor_1 = np.round(da_precip.raster.res[0] / ds_like.raster.res[0], 4)
    scaling_factor_2 = np.round(da_precip.raster.res[1] / ds_like.raster.res[1], 4)
    scaling_factor = (scaling_factor_1 + scaling_factor_2) / 2
    precip = precip / scaling_factor ** 2

    # derive cell areas (m2)
    lat, lon = ds_like.raster.ycoords.values, ds_like.raster.xcoords.values
    areagrid = gis_utils.reggrid_area(lat, lon) / 1e6

    # calculate "local runoff" (note: set missings in precipitation to zero)
    runoff = (np.maximum(precip, 0) * params["precip"]) + (areagrid * params["area"])
    runoff = np.maximum(runoff, 0)  # make sure runoff is not negative

    # get discharges
    discharge = flwdir.accuflux(runoff, nodata=NODATA["discharge"], direction="up")

    return discharge, (params["discharge_a"], params["discharge_b"])
