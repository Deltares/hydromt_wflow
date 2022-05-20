# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xarray as xr
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import logging

from . import ptf

logger = logging.getLogger(__name__)

__all__ = ["soilgrids", "soilgrids_sediment"]

# soilgrids_2017
soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
soildepth_mm = 10.0 * soildepth_cm
# for 2017 - midpoint and midpoint surface are equal to original
soildepth_cm_midpoint = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
soildepth_mm_midpoint = 10.0 * soildepth_cm
soildepth_mm_midpoint_surface = 10.0 * soildepth_cm
M_minmax = 10000.0
nodata = -9999.0

# default z [cm] of soil layers wflow_sbm: 10, 40, 120, > 120
# mapping of wflow_sbm parameter c to soil layer SoilGrids
c_sl_index = [2, 4, 6, 7]  # v2017 direct mapping
# c_sl_index = [2, 3, 5, 6] #v2021 if direct mapping - not used, averages are taken instead.


def average_soillayers_block(ds, soilthickness):
    """
    Determine the weighted average of soil property at different depths over soil thickness,
    assuming that the properties are computed at the mid-point of the interval and are considered
    constant over the whole depth interval (Sousa et al., 2020). https://doi.org/10.5194/soil-2020-65
    This function is used for soilgrids_2020.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing soil property over each soil depth profile [sl1 - sl6].
    soilthickness : xarray.DataArray
        Dataset containing soil thickness [cm].

    Returns
    -------
    da_av : xarray.DataArray
        Dataset containing weighted average of soil property.

    """
    da_sum = soilthickness * 0.0
    # set NaN values to 0.0 (to avoid RuntimeWarning in comparison soildepth)
    d = soilthickness.fillna(0.0)

    for i in ds.sl:

        da_sum = da_sum + (
            (soildepth_cm[i] - soildepth_cm[i - 1])
            * ds.sel(sl=i)
            * (d >= soildepth_cm[i])
            + (d - soildepth_cm[i - 1])
            * ds.sel(sl=i)
            * ((d < soildepth_cm[i]) & (d > soildepth_cm[i - 1]))
        )

    da_av = xr.apply_ufunc(
        lambda x, y: x / y,
        da_sum,
        soilthickness,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )

    da_av.raster.set_nodata(np.nan)

    return da_av


def average_soillayers(ds, soilthickness):

    """
    Determine weighted average of soil property at different depths over soil thickness,
    using the trapezoidal rule.
    See also: Hengl, T., Mendes de Jesus, J., Heuvelink, G. B. M., Ruiperez Gonzalez, M., Kilibarda,
    M., Blagotic, A., et al.: SoilGrids250m: Global gridded soil information based on machine learning,
    PLoS ONE, 12, https://doi.org/10.1371/journal.pone.0169748, 2017.
    This function is used for soilgrids (2017).

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing soil property at each soil depth [sl1 - sl7].
    soilthickness : xarray.DataArray
        Dataset containing soil thickness [cm].

    Returns
    -------
    da_av : xarray.DataArray
        Dataset containing weighted average of soil property.

    """

    da_sum = soilthickness * 0.0
    # set NaN values to 0.0 (to avoid RuntimeWarning in comparison soildepth)
    d = soilthickness.fillna(0.0)

    for i in range(1, len(ds.sl)):  # range(1, 7):

        da_sum = da_sum + (
            (soildepth_cm[i] - soildepth_cm[i - 1])
            * (ds.sel(sl=i) + ds.sel(sl=i + 1))
            * (d >= soildepth_cm[i])
            + (d - soildepth_cm[i - 1])
            * (ds.sel(sl=i) + ds.sel(sl=i + 1))
            * ((d < soildepth_cm[i]) & (d > soildepth_cm[i - 1]))
        )

    da_av = xr.apply_ufunc(
        lambda x, y: x * (1 / (y * 2)),
        da_sum,
        soilthickness,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )

    da_av.raster.set_nodata(np.nan)

    return da_av


def pore_size_distrution_index_layers(ds, thetas):

    """
    Determine pore size distribution index per soil layer depth based on PTF.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing soil properties at each soil depth [sl1 - sl7].
    thetas: xarray.Dataset
        Dataset containing thetaS at each soil layer depth.

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing pore size distribution index [-] for each soil layer
        depth.

    """

    ds_out = xr.apply_ufunc(
        ptf.pore_size_index_brakensiek,
        ds["sndppt"],
        thetas,
        ds["clyppt"],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )
    ds_out.name = "pore_size"
    ds_out.raster.set_nodata(np.nan)
    # ds_out = ds_out.raster.interpolate_na("nearest")
    return ds_out


def kv_layers(ds, thetas, ptf_name):

    """
    Determine vertical saturated hydraulic conductivity (KsatVer) per soil layer depth based on PTF.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing soil properties at each soil depth [sl1 - sl7].
    thetas: xarray.Dataset
        Dataset containing thetaS at each soil layer depth.
    ptf_name : str
        PTF to use for calculation KsatVer.

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing KsatVer [mm/day] for each soil layer depth.
    """
    if ptf_name == "brakensiek":
        ds_out = xr.apply_ufunc(
            ptf.kv_brakensiek,
            thetas,
            ds["clyppt"],
            ds["sndppt"],
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
        )
    elif ptf_name == "cosby":
        ds_out = xr.apply_ufunc(
            ptf.kv_cosby,
            ds["clyppt"],
            ds["sndppt"],
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
        )

    ds_out.name = "kv"
    ds_out.raster.set_nodata(np.nan)

    return ds_out


def func(x, b):
    return np.exp(-b * x)


def do_linalg(x, y):
    """
    Apply np.linalg.lstsq and return fitted parameter.

    Parameters
    ----------
    x : array_like (float)
        “Coefficient” matrix.
    y : array_like (float)
        dependent variable.

    Returns
    -------
    popt_0 : float
        Optimal value for the parameter fit.

    """
    idx = ((~np.isinf(np.log(y)))) & ((~np.isnan(y)))
    return np.linalg.lstsq(x[idx, np.newaxis], np.log(y[idx]), rcond=None)[0][0]


def do_curve_fit(x, y):
    """
    Apply scipy.optimize.curve_fit and return fitted parameter. If least-squares minimization
    fails with an inital guess p0 of 1e-3, and 1e-4, np.linalg.lstsq is used for curve fitting.

    Parameters
    ----------
    x : array_like of M length (float)
        independent variable.
    y : array_like of M length (float)
        dependent variable.

    Returns
    -------
    popt_0 : float
        Optimal value for the parameter fit.

    """
    idx = ((~np.isinf(np.log(y)))) & ((~np.isnan(y)))
    if len(y[idx]) == 0:
        popt_0 = np.nan
    else:
        try:
            # try curve fitting with certain p0
            popt_0 = curve_fit(func, x[idx], y[idx], p0=(1e-3))[0]
        except RuntimeError:
            try:
                # try curve fitting with lower p0
                popt_0 = curve_fit(func, x[idx], y[idx], p0=(1e-4))[0]
            except RuntimeError:
                # do linalg  regression instead
                popt_0 = np.linalg.lstsq(
                    x[idx, np.newaxis], np.log(y[idx]), rcond=None
                )[0][0]
    return popt_0


def constrain_M(M, popt_0, M_minmax):
    """Constrain M parameter with the value M_minmax"""
    M = xr.where((M > 0) & (popt_0 == 0), M_minmax, M)
    M = xr.where(M > M_minmax, M_minmax, M)
    M = xr.where(M < 0, M_minmax, M)
    return M


def soilgrids(ds, ds_like, ptfKsatVer, soil_fn, logger=logger):
    """
    Returns soil parameter maps at model resolution based on soil properties from SoilGrids datasets.
    Both soilgrids 2017 and 2020 are supported. Soilgrids 2017 provides soil properties at 7 specific depths, while soilgrids_2020 provides soil properties averaged over 6 depth intervals. 
    Ref: Hengl, T., Mendes de Jesus, J., Heuvelink, G. B. M., Ruiperez Gonzalez, M., Kilibarda, 
    M., Blagotic, A., et al.: SoilGrids250m: Global gridded soil information based on machine learning, 
    PLoS ONE, 12, https://doi.org/10.1371/journal.pone.0169748, 2017.
    Ref: de Sousa, L.M., Poggio, L., Batjes, N.H., Heuvelink, G., Kempen, B., Riberio, E. and Rossiter, D., 2020. 
    SoilGrids 2.0: producing quality-assessed soil information for the globe. SOIL Discussions, pp.1-37.
    https://doi.org/10.5194/soil-2020-65 

    The following soil parameter maps are calculated:\
    - `thetaS`            : average saturated soil water content [m3/m3]\
    - `thetaR`            : average residual water content [m3/m3]\
    - `KsatVer`           : vertical saturated hydraulic conductivity at soil surface [mm/day]\
    - `SoilThickness`     : soil thickness [mm]\
    - `SoilMinThickness`  : minimum soil thickness [mm] (equal to SoilThickness)\
    - `M`                 : model parameter [mm] that controls exponential decline of KsatVer with soil depth 
                            (fitted with curve_fit (scipy.optimize)), bounds of M are checked\
    - `M_`                : model parameter [mm] that controls exponential decline of KsatVer with soil depth 
                            (fitted with numpy linalg regression), bounds of `M_`are checked\
    - `M_original`        : `M` without checking bounds\
    - `M_original_`       : `M_`without checking bounds\
    - `f`                 : scaling parameter controlling the decline of KsatVer [mm-1] (fitted with curve_fit (scipy.optimize)), bounds are checked\
    - `f_`                : scaling parameter controlling the decline of KsatVer [mm-1] (fitted with numpy linalg regression), bounds are checked\
    - `c_0`               : Brooks Corey coefficient [-] based on pore size distribution index at depth
                            1st soil layer (100 mm) wflow_sbm\
    - `c_1`               : idem `c_0` at depth 2nd soil layer (400 mm) wflow_sbm\
    - `c_2`               : idem `c_0` at depth 3rd soil layer (1200 mm) wflow_sbm\
    - `c_3`               : idem `c_0` at depth 4th soil layer (> 1200 mm) wflow_sbm\
    - `KsatVer_[z]cm`     : KsatVer [mm/day] at soil depths [z] of SoilGrids data [0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0]\
    - `wflow_soil`        : USDA Soil texture based on percentage clay, silt, sand mapping: [1:Clay, 2:Silty Clay, 3:Silty Clay-Loam, 4:Sandy Clay, 5:Sandy Clay-Loam, 6:Clay-Loam, 7:Silt, 8:Silt-Loam, 9:Loam, 10:Sand, 11: Loamy Sand, 12:Sandy Loam]\
                
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing soil properties.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    ptfKsatVer : str
        PTF to use for calculcation KsatVer.
    soil_fn : str
        soilgrids version {'soilgrids', 'soilgrids_2020'}

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing gridded soil parameters.
    """

    if soil_fn == "soilgrids_2020":
        # use midpoints of depth intervals for soilgrids_2020.
        soildepth_cm_midpoint = np.array([2.5, 10.0, 22.5, 45.0, 80.0, 150.0])
        soildepth_cm_midpoint_surface = np.array([0, 10.0, 22.5, 45.0, 80.0, 150.0])
        soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm = 10.0 * soildepth_cm
        soildepth_mm_midpoint = 10.0 * soildepth_cm_midpoint
        soildepth_mm_midpoint_surface = 10.0 * soildepth_cm_midpoint_surface
    else:
        soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm = 10.0 * soildepth_cm
        soildepth_cm_midpoint = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm_midpoint = 10.0 * soildepth_cm
        soildepth_mm_midpoint_surface = 10.0 * soildepth_cm

    ds_out = xr.Dataset(coords=ds_like.raster.coords)

    # set nodata values in dataset to NaN (based on soil property SLTPPT at first soil layer)
    # ds = xr.where(ds["sltppt_sl1"] == ds["sltppt_sl1"].raster.nodata, np.nan, ds)
    ds = ds.raster.mask_nodata()

    # add new coordinate sl to merge layers
    ds = ds.assign_coords(sl=np.arange(1, len(soildepth_cm_midpoint) + 1))

    for var in ["bd", "oc", "ph", "clyppt", "sltppt", "sndppt"]:
        da_prop = []
        for i in np.arange(1, len(soildepth_cm_midpoint) + 1):
            da_prop.append(ds[f"{var}_sl{i}"])
            # remove layer from ds
            ds = ds.drop_vars(f"{var}_sl{i}")
        da = xr.concat(
            da_prop,
            pd.Index(
                np.arange(1, len(soildepth_cm_midpoint) + 1, dtype=int), name="sl"
            ),
        ).transpose("sl", ...)
        da.name = var
        # add concat maps to ds
        ds[f"{var}"] = da

    logger.info("calculate and resample thetaS")
    thetas_sl = xr.apply_ufunc(
        ptf.thetas_toth,
        ds["ph"],
        ds["bd"],
        ds["clyppt"],
        ds["sltppt"],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )

    if soil_fn == "soilgrids_2020":
        thetas = average_soillayers_block(thetas_sl, ds["soilthickness"])
    else:
        thetas = average_soillayers(thetas_sl, ds["soilthickness"])
    thetas = thetas.raster.reproject_like(ds_like, method="average")
    ds_out["thetaS"] = thetas.astype(np.float32)

    logger.info("calculate and resample thetaR")
    thetar_sl = xr.apply_ufunc(
        ptf.thetar_toth,
        ds["oc"],
        ds["clyppt"],
        ds["sltppt"],
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )

    if soil_fn == "soilgrids_2020":
        thetar = average_soillayers_block(thetar_sl, ds["soilthickness"])
    else:
        thetar = average_soillayers(thetar_sl, ds["soilthickness"])
    thetar = thetar.raster.reproject_like(ds_like, method="average")
    ds_out["thetaR"] = thetar.astype(np.float32)

    soilthickness_hr = ds["soilthickness"]
    soilthickness = soilthickness_hr.raster.reproject_like(ds_like, method="average")
    # wflow_sbm cannot handle (yet) zero soil thickness
    soilthickness = soilthickness.where(soilthickness > 0.0, np.nan)
    soilthickness.raster.set_nodata(np.nan)
    soilthickness = soilthickness.astype(np.float32)
    ds_out["SoilThickness"] = soilthickness * 10.0  # from [cm] to [mm]
    ds_out["SoilMinThickness"] = xr.DataArray.copy(ds_out["SoilThickness"], deep=False)

    logger.info("calculate and resample KsatVer")
    kv_sl_hr = kv_layers(ds, thetas_sl, ptfKsatVer)
    kv_sl = np.log(kv_sl_hr)
    kv_sl = kv_sl.raster.reproject_like(ds_like, method="average")
    kv_sl = np.exp(kv_sl)

    logger.info("calculate and resample pore size distribution index")
    lambda_sl_hr = pore_size_distrution_index_layers(ds, thetas_sl)
    lambda_sl = np.log(lambda_sl_hr)
    lambda_sl = lambda_sl.raster.reproject_like(ds_like, method="average")
    lambda_sl = np.exp(lambda_sl)

    if soil_fn == "soilgrids_2020":
        # calculate c for each layer of lambda
        c_sl = 3.0 + (2.0 / lambda_sl)
        c_sl.name = "c_sl"

        # make empty dataarray for c for the 4 sbm layers
        ds_c = xr.full_like(soilthickness, np.nan)
        ds_c.name = "c"
        ds_c = ds_c.expand_dims(dim=dict(layer=[0, 1, 2, 3])).copy()
        # calc weighted average values of c over the sbm soil layers
        ds_c.loc[dict(layer=0)] = (
            c_sl.sel(sl=1) * 50 + c_sl.sel(sl=2) * 50
        ) / 100  # average over layer 0-100
        ds_c.loc[dict(layer=1)] = (
            c_sl.sel(sl=2) * 50 + c_sl.sel(sl=3) * 150 + c_sl.sel(sl=4) * 100
        ) / 300  # average over layer 100-400
        ds_c.loc[dict(layer=2)] = (
            c_sl.sel(sl=4) * 200 + c_sl.sel(sl=5) * 400 + c_sl.sel(sl=6) * 200
        ) / 800  # average over layer 400-1200
        ds_c.loc[dict(layer=3)] = c_sl.sel(sl=6)  # layer >1200
    else:
        # for soilgrids 2017, keep the direct mapping calculation.
        da_c = []
        for (i, sl_ind) in enumerate(c_sl_index):
            da_c.append(3.0 + (2.0 / lambda_sl.sel(sl=sl_ind)))
        ds_c = xr.concat(
            da_c, pd.Index(np.arange(len(c_sl_index), dtype=int), name="layer")
        ).transpose("layer", ...)
        ds_c.name = "c"
    ds_out["c"] = ds_c

    ds_out["KsatVer"] = kv_sl.sel(sl=1).astype(np.float32)

    for i, sl in enumerate(kv_sl["sl"]):
        kv = kv_sl.sel(sl=sl)
        ds_out["KsatVer_" + str(soildepth_cm_midpoint[i]) + "cm"] = kv.astype(
            np.float32
        )

    kv = kv_sl / kv_sl.sel(sl=1)
    logger.info("fit z - log(KsatVer) with numpy linalg regression (y = b*x) -> M_")
    popt_0_ = xr.apply_ufunc(
        do_linalg,
        soildepth_mm_midpoint_surface,
        kv.compute(),
        vectorize=True,
        dask="parallelized",
        input_core_dims=[["z"], ["sl"]],
        output_dtypes=[float],
        keep_attrs=True,
    )

    M_ = (thetas - thetar) / (-popt_0_)
    ds_out["M_original_"] = M_.astype(np.float32)
    M_ = constrain_M(M_, popt_0_, M_minmax)
    ds_out["M_"] = M_.astype(np.float32)
    ds_out["f_"] = ((thetas - thetar) / M_).astype(np.float32)

    logger.info("fit zi - Ksat with curve_fit (scipy.optimize) -> M")
    popt_0 = xr.apply_ufunc(
        do_curve_fit,
        soildepth_mm_midpoint_surface,
        kv.compute(),
        vectorize=True,
        dask="parallelized",
        input_core_dims=[["z"], ["sl"]],
        output_dtypes=[float],
        keep_attrs=True,
    )

    M = (thetas - thetar) / (popt_0)
    ds_out["M_original"] = M.astype(np.float32)
    M = constrain_M(M, popt_0, M_minmax)
    ds_out["M"] = M.astype(np.float32)
    ds_out["f"] = ((thetas - thetar) / M).astype(np.float32)

    # wflow soil map is based on USDA soil classification
    # soilmap = ds["tax_usda"].raster.interpolate_na()
    # soilmap = soilmap.raster.reproject_like(ds_like, method="mode")
    # ds_out["wflow_soil"] = soilmap.astype(np.float32)

    # wflow_soil map is based on soil texture calculated with percentage sand, silt, clay
    # clay, silt percentages averaged over soil thickness
    if soil_fn == "soilgrids_2020":
        clay_av = average_soillayers_block(ds["clyppt"], ds["soilthickness"])
        silt_av = average_soillayers_block(ds["sltppt"], ds["soilthickness"])
    else:
        clay_av = average_soillayers(ds["clyppt"], ds["soilthickness"])
        silt_av = average_soillayers(ds["sltppt"], ds["soilthickness"])

    # calc soil texture
    soil_texture = xr.apply_ufunc(
        ptf.soil_texture_usda,
        clay_av,
        silt_av,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )

    soil_texture = soil_texture.raster.reproject_like(ds_like, method="mode")
    # np.nan is not a valid value for array with type integer
    ds_out["wflow_soil"] = soil_texture
    ds_out["wflow_soil"].raster.set_nodata(0)

    # for writing pcraster map files a scalar nodata value is required
    dtypes = {"wflow_soil": np.int32}
    for var in ds_out:
        dtype = dtypes.get(var, np.float32)
        logger.debug(f"Interpolate nodata (NaN) values for {var}")
        ds_out[var] = ds_out[var].raster.interpolate_na("nearest")
        ds_out[var] = ds_out[var].fillna(nodata).astype(dtype)
        ds_out[var].raster.set_nodata(np.dtype(dtype).type(nodata))

    return ds_out


def soilgrids_sediment(ds, ds_like, usleK_method, logger=logger):

    """
    Returns soil parameter maps for sediment modelling at model resolution based on soil 
    properties from SoilGrids dataset.

    The following soil parameter maps are calculated:\
        - PercentClay: clay content of the topsoil [%]\
        - PercentSilt: silt content of the topsoil [%]\
        - PercentOC: organic carbon in the topsoil [%]\
        - ErosK: mean detachability of the soil (Morgan et al., 1998) [g/J]\
        - USLE_K: soil erodibility factor from the USLE equation [-]\
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing soil properties.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    usleK_method : str
        Method to use for calculation of USLE_K {"renard", "epic"}.

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing gridded soil parameters for sediment modelling.
    """

    ds_out = xr.Dataset(coords=ds_like.raster.coords)

    # set nodata values in dataset to NaN (based on soil property SLTPPT at first soil layer)
    # ds = xr.where(ds["sltppt_sl1"] == ds["sltppt_sl1"].raster.nodata, np.nan, ds)
    ds = ds.raster.mask_nodata()

    # soil properties
    pclay = ds["clyppt_sl1"]
    percentclay = pclay.raster.reproject_like(ds_like, method="average")
    ds_out["PercentClay"] = percentclay.astype(np.float32)

    psilt = ds["sltppt_sl1"]
    percentsilt = psilt.raster.reproject_like(ds_like, method="average")
    ds_out["PercentSilt"] = percentsilt.astype(np.float32)

    poc = ds["oc_sl1"]
    percentoc = poc.raster.reproject_like(ds_like, method="average")
    ds_out["PercentOC"] = percentoc.astype(np.float32)

    # Detachability of the soil
    erosK = xr.apply_ufunc(
        ptf.ErosK_texture,
        pclay,
        psilt,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )
    erosK = erosK.raster.reproject_like(ds_like, method="average")
    ds_out["ErosK"] = erosK.astype(np.float32)

    # USLE K parameter
    if usleK_method == "renard":
        usleK = xr.apply_ufunc(
            ptf.UsleK_Renard,
            pclay,
            psilt,
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
        )
    elif usleK_method == "epic":
        usleK = xr.apply_ufunc(
            ptf.UsleK_EPIC,
            pclay,
            psilt,
            poc,
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
        )
    usleK = usleK.raster.reproject_like(ds_like, method="average")
    ds_out["USLE_K"] = usleK.astype(np.float32)

    # for writing pcraster map files a scalar nodata value is required
    for var in ds_out:
        ds_out[var] = ds_out[var].raster.interpolate_na("nearest")
        logger.info(f"Interpolate NAN values for {var}")
        ds_out[var] = ds_out[var].fillna(nodata)
        ds_out[var].raster.set_nodata(nodata)

    return ds_out
