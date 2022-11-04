# -*- coding: utf-8 -*-
"""Workflows to derive river for a wflow model."""

from os.path import join
import numpy as np
from scipy.optimize import curve_fit
import xarray as xr
import pandas as pd
import geopandas as gpd
import logging
import pyflwdir

from hydromt import gis_utils, stats, flw, workflows
from hydromt_wflow import DATADIR  # global var

logger = logging.getLogger(__name__)

__all__ = ["river", "river_bathymetry", "river_width"]

RESAMPLING = {"climate": "nearest", "precip": "average"}
NODATA = {"discharge": -9999}


def power_law(x, a, b):
    return a * np.power(x, b)


def river(
    ds,
    ds_model=None,
    river_upa=30.0,
    slope_len=2e3,
    min_rivlen_ratio=0.0,
    channel_dir="up",
    logger=logger,
):
    """Returns river maps

    The output maps are:\
    - rivmsk : river mask based on upstream area threshold on upstream area\
    - rivlen : river length [m]\
    - rivslp : smoothed river slope [m/m]\
    - rivzs : elevation of the river bankfull height based on pixel outlet 
    - rivwth : river width at pixel outlet (if in ds)
    - qbankfull : bankfull discharge at pixel outlet (if in ds)

    Parameters
    ----------
    ds: xr.Dataset
        hydrography dataset containing "flwdir", "uparea", "elevtn" variables; 
        and optional "rivwth" and "qbankfull" variable
    ds_model: xr.Dataset, optional
        model dataset with output grid, must contain "uparea", for subgrid rivlen/slp
        must contain "x_out", "y_out". If None, ds is assumed to be the model grid
    river_upa: float
        minimum threshold to define river cell & pixels, by default 30 [km2]
    slope_len: float
        minimum length over which to calculate the river slope, by default 1000 [m]
    min_rivlen_ratio: float
        minimum global river length to avg. Cell resolution ratio used as threshold 
        in window based smoothing of river length, by default 0.0. 
        The smoothing is skipped if min_riverlen_ratio = 0.
    channel_dir: {"up", "down"}
        flow direcition in which to calculate (subgrid) river length and width
    
    Returns:
    ds_out: xr.Dataset
        Dataset with output river attributes
    """
    # check data variables.
    dvars = ["flwdir", "uparea", "elevtn"]
    dvars_model = ["flwdir", "uparea"]
    if not np.all([v in ds for v in dvars]):
        raise ValueError(f"One or more variables missing from ds: {dvars}.")
    if ds_model is not None and not np.all([v in ds_model for v in dvars_model]):
        raise ValueError(f"One or more variables missing from ds_model: {dvars_model}")
    # sort sugrid
    subgrid = True
    if ds_model is None or not np.all([v in ds_model for v in ["x_out", "y_out"]]):
        subgrid = False
        ds_model = ds
        logger.info("River length and slope are calculated at model resolution.")

    ## river mask and flow directions at model grid
    logger.debug(f"Set river mask (min uparea: {river_upa} km2) and prepare flow dirs.")
    riv_mask = ds_model["uparea"] > river_upa  # initial mask
    mod_mask = ds_model["uparea"] != ds_model["uparea"].raster.nodata

    ## (high res) flwdir and outlet indices
    riv_mask_org = ds["uparea"].values >= river_upa  # highres riv mask
    flwdir = flw.flwdir_from_da(ds["flwdir"], mask=True)
    if subgrid == False:
        # get cell index of river cells
        idxs_out = np.arange(ds_model.raster.size).reshape(ds_model.raster.shape)
        idxs_out = np.where(mod_mask, idxs_out, flwdir._mv)
    else:
        # get subgrid outlet pixel index
        idxs_out = ds.raster.xy_to_idx(
            xs=ds_model["x_out"].values,
            ys=ds_model["y_out"].values,
            mask=mod_mask.values,
            nodata=flwdir._mv,
        )

    ## river length
    # get river length based on on-between distance between two outlet pixels
    logger.debug("Derive river length.")
    rivlen = flwdir.subgrid_rivlen(
        idxs_out=idxs_out,
        mask=riv_mask_org,
        direction=channel_dir,
        unit="m",
    )
    xres, yres = ds_model.raster.res
    if ds_model.raster.crs.is_geographic:  # convert degree to meters
        lat_avg = ds_model.raster.ycoords.values.mean()
        xres, yres = gis_utils.cellres(lat_avg, xres, yres)
    rivlen = np.where(riv_mask.values, rivlen, -9999)
    # set mean length at most downstream (if channel_dir=down) or upstream (if channel_dir=up) river lengths
    if np.any(rivlen == 0):
        rivlen[rivlen == 0] = np.mean(rivlen[rivlen > 0])
    # smooth river length based on minimum river length
    if min_rivlen_ratio > 0 and hasattr(flwdir, "smooth_rivlen"):
        res = np.mean(np.abs([xres, yres]))
        min_len = res * min_rivlen_ratio
        flwdir_model = flw.flwdir_from_da(ds_model["flwdir"], mask=riv_mask)
        rivlen2 = flwdir_model.smooth_rivlen(rivlen, min_len, nodata=-9999)
        min_len2 = rivlen2[riv_mask].min()
        pmod = (rivlen != rivlen2).sum() / riv_mask.sum() * 100
        logger.debug(
            f"River length smoothed (min length: {min_len:.0f} m; cells modified: {pmod:.1f})%."
        )
        rivlen = rivlen2
    elif min_rivlen_ratio > 0:
        logger.warning(
            "River length smoothing skipped as it requires newer version of pyflwdir."
        )

    ## river slope as derivative of elevation around outlet pixels
    logger.debug("Derive river slope.")
    rivslp = flwdir.subgrid_rivslp(
        idxs_out=idxs_out,
        elevtn=ds["elevtn"].values,
        mask=riv_mask_org,
        length=slope_len,
    )
    rivslp = np.where(riv_mask.values, rivslp, -9999)

    # create xarray dataset for all river variables
    ds_out = xr.Dataset(coords=ds_model.raster.coords)
    dims = ds_model.raster.dims
    # save as uint8 as bool is not supported in nc and tif files
    riv_mask = riv_mask.astype(np.uint8)
    riv_mask.raster.set_nodata(0)
    ds_out["rivmsk"] = riv_mask
    attrs = dict(_FillValue=-9999, unit="m")
    ds_out["rivlen"] = xr.Variable(dims, rivlen, attrs=attrs)
    attrs = dict(_FillValue=-9999, unit="m.m-1")
    ds_out["rivslp"] = xr.Variable(dims, rivslp, attrs=attrs)

    for name in ["rivwth", "qbankfull"]:
        if name in ds:
            logger.debug(f"Derive {name} from hydrography dataset.")
            data = np.full_like(riv_mask, -9999, dtype=np.float32)
            data0 = ds[name].values.flat[idxs_out[riv_mask.values]]
            data[riv_mask.values] = np.where(data0 > 0, data0, -9999)
            ds_out[name] = xr.Variable(dims, data, attrs=dict(_FillValue=-9999))

    return ds_out, flwdir


def river_bathymetry(
    ds_model: xr.Dataset,
    gdf_riv: gpd.GeoDataFrame,
    method: str = "powlaw",
    smooth_len: float = 5e3,
    min_rivdph: float = 1.0,
    min_rivwth: float = 30.0,
    logger=logger,
    **kwargs,
) -> xr.Dataset:
    """Get river width and bankfull discharge from `gdf_riv` to estimate river depth
    using :py:meth:`hydromt.workflows.river_depth`. Missing values in rivwth are first
    filled using downward filling and remaining  (upstream) missing values are set
    to min_rivwth (for rivwth) and 0 (for qbankfull).

    Parameters
    ----------
    ds_model : xr.Dataset
        Model dataset with 'flwdir', 'rivmsk', 'rivlen', 'x_out' and 'y_out' variables.
    gdf_riv : gpd.GeoDataFrame
        River geometry with 'rivwth' and 'qbankfull' columns.
    method : {'gvf', 'manning', 'powlaw'}
        see py:meth:`hydromt.workflows.river_depth` for details, by default "powlaw"
    smooth_len : float, optional
        Length [m] over which to smooth the output river width and depth, by default 5e3
    min_rivdph : float, optional
        Minimum river depth [m], by default 1.0
    min_rivwth : float, optional
        Minimum river width [m], by default 30.0

    Returns
    -------
    xr.Dataset
        Dataset with 'rivwth' and 'rivdph' variables
    """
    dims = ds_model.raster.dims
    # check data variables.
    dvars_model = ["flwdir", "rivmsk", "rivlen"]
    if method != "powlaw":
        dvars_model += ["rivslp", "rivzs"]
    if not np.all([v in ds_model for v in dvars_model]):
        raise ValueError(f"One or more variables missing from ds_model: {dvars_model}")

    # setup flow direction for river
    riv_mask = ds_model["rivmsk"].values == 1
    flwdir_river = flw.flwdir_from_da(ds_model["flwdir"], mask=riv_mask)
    rivlen_avg = ds_model["rivlen"].values[riv_mask].mean()

    ## river width and bunkfull discharge
    vars0 = ["rivwth", "qbankfull"]
    # find nearest values from river shape if provided
    # if None assume the data is in ds_model
    if gdf_riv is not None:
        vars = [c for c in vars0 if c in gdf_riv.columns]
        if len(vars) == 0:
            raise ValueError(f" columns {vars0} not found in gdf_riv")
        logger.debug(f"Derive {vars} from shapefile.")
        if "x_out" in ds_model and "y_out" in ds_model:
            # get subgrid outlet pixel index and coordinates
            xs_out = ds_model["x_out"].values[riv_mask]
            ys_out = ds_model["y_out"].values[riv_mask]
        else:
            # get river cell coordinates
            row, col = np.where(riv_mask)
            xs_out = ds_model.raster.xcoords.values[col]
            ys_out = ds_model.raster.ycoords.values[row]
        gdf_out = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(xs_out, ys_out), crs=ds_model.raster.crs
        )
        idx_nn, dst_nn = gis_utils.nearest(gdf_out, gdf_riv)
        # get valid river data within max half pixel distance
        xres, yres = ds_model.raster.res
        if ds_model.raster.crs.is_geographic:  # convert degree to meters
            lat_avg = ds_model.raster.ycoords.values.mean()
            xres, yres = gis_utils.cellres(lat_avg, xres, yres)
        max_dist = np.mean(np.abs([xres, yres])) / 2.0
        nriv, nsnap = xs_out.size, int(np.sum(dst_nn < max_dist))
        logger.debug(
            f"Valid for {nsnap}/{nriv} river cells (max dist: {max_dist:.0f} m)."
        )
        for name in vars:
            data = np.full_like(riv_mask, -9999, dtype=np.float32)
            data[riv_mask] = np.where(
                dst_nn < max_dist, gdf_riv.loc[idx_nn, name].fillna(-9999).values, -9999
            )
            ds_model[name] = xr.Variable(dims, data, attrs=dict(_FillValue=-9999))
    # TODO fallback option when qbankfull is missing.
    assert "qbankfull" in ds_model and "rivwth" in ds_model
    # fill gaps in data using downward filling along flow directions
    for name in vars0:
        data = ds_model[name].values
        nodata = ds_model[name].raster.nodata
        if np.all(data[riv_mask] != nodata):
            continue
        data = flwdir_river.fillnodata(data, nodata, direction="down", how="max")
        ds_model[name].values = np.maximum(0, data)
    # smooth by averaging along flow directions and set minimum
    if smooth_len > 0:
        nsmooth = min(1, int(round(smooth_len / rivlen_avg / 2)))
        kwgs = dict(n=nsmooth, restrict_strord=True)
        ds_model["rivwth"].values = flwdir_river.moving_average(
            ds_model["rivwth"].values, nodata=ds_model["rivwth"].raster.nodata, **kwgs
        )
    ds_model["rivwth"] = np.maximum(min_rivwth, ds_model["rivwth"]).where(
        riv_mask, ds_model["rivwth"].raster.nodata
    )

    ## river depth
    # distance to outlet; required for manning and gvf rivdph methods
    if method != "powlaw" and "rivdst" not in ds_model:
        rivlen = ds_model["rivlen"].values
        nodata = ds_model["rivlen"].raster.nodata
        rivdst = flwdir_river.accuflux(rivlen, nodata=nodata, direction="down")
        ds_model["rivdst"] = xr.Variable(dims, rivdst, attrs=dict(_FillValue=nodata))
    # add river distance to outlet -> required for manning/gvf method
    rivdph = workflows.river_depth(
        data=ds_model,
        flwdir=flwdir_river,
        method=method,
        min_rivdph=min_rivdph,
        **kwargs,
    )
    attrs = dict(_FillValue=-9999, unit="m")
    ds_model["rivdph"] = xr.Variable(dims, rivdph, attrs=attrs).fillna(-9999)
    # smooth by averaging along flow directions and set minimum
    if smooth_len > 0:
        ds_model["rivdph"].values = flwdir_river.moving_average(
            ds_model["rivdph"].values, nodata=-9999, **kwgs
        )
    ds_model["rivdph"] = np.maximum(min_rivdph, ds_model["rivdph"]).where(
        riv_mask, -9999
    )

    return ds_model[["rivwth", "rivdph"]]


# TODO: methods below are redundant after version v0.1.4 and will be removed in
# future versions


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
    rivmsk = ds_like[rivmsk_name].values != 0
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
    """ """
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
    """ """
    # read clim classes and regression parameters from data dir
    precip_fn = da_precip.name
    climate_fn = da_climate.name
    fn_regr = join(DATADIR, "rivwth", f"regr_{precip_fn}.csv")
    fn_clim = join(DATADIR, "rivwth", f"{climate_fn}.csv")
    clim_map = pd.read_csv(fn_clim, index_col="class")
    regr_map = pd.read_csv(fn_regr, index_col="source").loc[climate_fn]
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
    precip = precip / scaling_factor**2

    # derive cell areas (m2)
    lat, lon = ds_like.raster.ycoords.values, ds_like.raster.xcoords.values
    areagrid = gis_utils.reggrid_area(lat, lon) / 1e6

    # calculate "local runoff" (note: set missings in precipitation to zero)
    runoff = (np.maximum(precip, 0) * params["precip"]) + (areagrid * params["area"])
    runoff = np.maximum(runoff, 0)  # make sure runoff is not negative

    # get discharges
    discharge = flwdir.accuflux(runoff, nodata=NODATA["discharge"], direction="up")

    return discharge, (params["discharge_a"], params["discharge_b"])
