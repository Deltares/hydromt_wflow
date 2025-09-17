"""Soilgrid workflows for Wflow plugin."""

import logging

import numpy as np
import pandas as pd
import xarray as xr
from hydromt.gis import raster_utils
from scipy.optimize import curve_fit

from hydromt_wflow.workflows import ptf, soilparams

logger = logging.getLogger(f"hydromt.{__name__}")

__all__ = [
    "soilgrids",
    "soilgrids_sediment",
    "soilgrids_brooks_corey",
    "update_soil_with_paddy",
]

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
# mapping of wflow_sbm parameter soil_brooks_corey_c to soil layer SoilGrids
c_sl_index = [2, 4, 6, 7]  # v2017 direct mapping
# c_sl_index = [2, 3, 5, 6] #v2021 if direct mapping - not used,
# averages are taken instead.


def concat_layers(
    ds: xr.Dataset,
    soil_fn: str = "soilgrids",
    variables: list[str] = ["bd", "oc", "ph", "clyppt", "sltppt", "sndppt"],
):
    """
    Preprocess functions to concat soilgrids along a layer dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing soil properties.
    soil_fn : str
        soilgrids version {'soilgrids', 'soilgrids_2020'}
    variables : list
        List of soil properties to concat.
    """
    if soil_fn == "soilgrids_2020":
        nb_sl = 6
    else:
        nb_sl = 7
    ds = ds.assign_coords(sl=np.arange(1, nb_sl + 1))

    for var in variables:
        da_prop = []
        for i in np.arange(1, nb_sl + 1):
            da_prop.append(ds[f"{var}_sl{i}"])
            # remove layer from ds
            ds = ds.drop_vars(f"{var}_sl{i}")
        da = xr.concat(
            da_prop,
            pd.Index(np.arange(1, nb_sl + 1, dtype=int), name="sl"),
        ).transpose("sl", ...)
        da.name = var
        # add concat maps to ds
        ds[f"{var}"] = da

    return ds


def average_soillayers_block(ds, soilthickness):
    """
    Determine weighted average of soil property at different depths over soil thickness.

    Assuming that the properties are computed at the mid-point of the interval and
    are considered constant over the whole depth interval (Sousa et al., 2020).
    https://doi.org/10.5194/soil-2020-65
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
    Determine weighted average of soil property at different depths over soil thickness.

    Using the trapezoidal rule.
    See also: Hengl, T., Mendes de Jesus, J., Heuvelink, G. B. M., Ruiperez Gonzalez,
    M., Kilibarda, M., Blagotic, A., et al.: SoilGrids250m:
    Global gridded soil information based on machine learning,
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
        Dataset containing the weighted average of the soil property.

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


def pore_size_distribution_index_layers(ds, thetas):
    """
    Determine pore size distribution index per soil layer depth based on PTF.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing soil properties at each soil depth [sl1 - sl7].
    thetas: xarray.Dataset
        Dataset containing theta_s at each soil layer depth.

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


def brooks_corey_layers(
    thetas_sl: xr.Dataset,
    ds: xr.Dataset,
    ds_like: xr.Dataset,
    soil_fn: str = "soilgrids",
    wflow_layers: list[int] = [100, 300, 800],
    soildepth_cm: np.array = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0]),
):
    """
    Determine Brooks Corey coefficient per wflow soil layer depth.

    First pore size distribution index is computed based on theta_s and other soil
    parameters and scaled to the model resolution.

    Then the Brooks Corey coefficient is computed for each wflow soil layer depth
    by weighted averaging the pore size distribution index over the soil thickness.

    Parameters
    ----------
    thetas_sl: xarray.Dataset
        Dataset containing theta_s at each soil layer depth.
    ds: xarray.Dataset
        Dataset containing soil properties at each soil depth.
    ds_like: xarray.Dataset
        Dataset at model resolution for reprojection.
    soil_fn: str
        soilgrids version {'soilgrids', 'soilgrids_2020'}
    wflow_layers: list
        List of soil layer depths [cm] for which soil_brooks_corey_c is calculated.
    soildepth_cm: np.array
        Depth of each soil layers [cm].

    Returns
    -------
    ds_c : xarray.Dataset
        Dataset containing soil_brooks_corey_c for the wflow_sbm soil layers.
    """
    # Get pore size distribution index
    lambda_sl_hr = pore_size_distribution_index_layers(ds, thetas_sl)
    lambda_sl = np.log(lambda_sl_hr)
    lambda_sl = lambda_sl.raster.reproject_like(ds_like, method="average")
    lambda_sl = np.exp(lambda_sl)

    # Brooks Corey coefficient
    c_sl = 3.0 + (2.0 / lambda_sl)
    c_sl.name = "c_sl"

    # Resample for the wflow layers
    wflow_thickness = wflow_layers.copy()
    # Go from wflow layer thickness to soil depths (cumulative)
    wflow_depths = np.cumsum(wflow_thickness)
    # Check if the last wflow depth is less than 2000 mm (soilgrids limit)
    if wflow_depths[-1] > 2000:
        raise ValueError(
            "The total depth of the wflow soil layers should be 2000 mm, as the "
            "soilgrids data does not go deeper than 2 m."
        )
    # Add a zero for the first depth and 2000 for the last depth
    wflow_depths = np.insert(wflow_depths, 0, 0)
    wflow_depths = np.append(wflow_depths, 2000)
    # Compute the thickness of the last layer
    wflow_thickness.append(2000 - wflow_depths[-2])

    # Soil data depth
    soildepth = soildepth_cm * 10

    # make empty dataarray for soil_brooks_corey_c for the 4 sbm layers
    ds_c = raster_utils.full(
        coords=c_sl.raster.coords,
        nodata=np.nan,
        crs=c_sl.raster.crs,
        dtype="float32",
        name="soil_brooks_corey_c",
        attrs={"thickness": wflow_thickness, "soil_fn": soil_fn},
    )
    ds_c = ds_c.expand_dims(dim=dict(layer=np.arange(len(wflow_thickness)))).copy()

    # calc weighted average values of soil_brooks_corey_c over the sbm soil layers
    for nl in range(len(wflow_thickness)):
        top_depth = wflow_depths[nl]
        bottom_depth = wflow_depths[nl + 1]
        c_nl_sum = None
        for d in range(len(soildepth) - 1):
            if soil_fn == "soilgrids_2020":
                c_av = c_sl.sel(sl=d + 1)
            else:
                c_av = (c_sl.sel(sl=d + 1) + c_sl.sel(sl=d + 2)) / 2
            # wflow layer fully within soilgrid layer
            if soildepth[d] <= top_depth and soildepth[d + 1] >= bottom_depth:
                c_nl = c_av * (bottom_depth - top_depth)
            # layer fully within wflow layer
            elif soildepth[d] >= top_depth and soildepth[d + 1] <= bottom_depth:
                c_nl = c_av * (soildepth[d + 1] - soildepth[d])
            # bottom part of the layer within wflow
            elif soildepth[d] <= bottom_depth and soildepth[d + 1] >= bottom_depth:
                c_nl = c_av * (bottom_depth - soildepth[d])
            # top part of the layer within wflow
            elif soildepth[d] <= top_depth and soildepth[d + 1] >= top_depth:
                c_nl = c_av * (soildepth[d + 1] - top_depth)
            # layer outside of wflow layer
            else:
                c_nl = None

            # Add to the sum
            if c_nl is not None:
                if c_nl_sum is None:
                    c_nl_sum = c_nl
                else:
                    c_nl_sum = c_nl_sum + c_nl

        ds_c.loc[dict(layer=nl)] = c_nl_sum / wflow_thickness[nl]

    return ds_c


def kv_layers(ds, thetas, ptf_name):
    """
    Determine vertical saturated hydraulic conductivity per soil layer depth.

    Based on PTF.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset containing soil properties at each soil depth [sl1 - sl7].
    thetas: xarray.Dataset
        Dataset containing theta_s at each soil layer depth.
    ptf_name : str
        PTF to use for calculation ksat_vertical .

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing ksat_vertical  [mm/day] for each soil layer depth.
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
        "Coefficient"  matrix.
    y : array_like (float)
        dependent variable.

    Returns
    -------
    popt_0 : float
        Optimal value for the parameter fit.

    """
    idx = (~np.isinf(np.log(y))) & (~np.isnan(y))
    return np.linalg.lstsq(x[idx, np.newaxis], np.log(y[idx]), rcond=None)[0][0]


def do_curve_fit(x, y):
    """
    Apply scipy.optimize.curve_fit and return fitted parameter.

    If least-squares minimization fails with an initial guess p0 of 1e-3,
    and 1e-4, np.linalg.lstsq is used for curve fitting.

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
    idx = (~np.isinf(np.log(y))) & (~np.isnan(y))
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
    """Constrain M parameter with the value M_minmax."""
    M = xr.where((M > 0) & (popt_0 == 0), M_minmax, M)
    M = xr.where(M > M_minmax, M_minmax, M)
    M = xr.where(M < 0, M_minmax, M)
    return M


def soilgrids(
    ds: xr.Dataset,
    ds_like: xr.Dataset,
    ptfKsatVer: str = "brakensiek",
    soil_fn: str = "soilgrids",
    wflow_layers: list[int] = [100, 300, 800],
):
    """
    Return soil parameter maps at model resolution.

    Based on soil properties from SoilGrids datasets.
    Both soilgrids 2017 and 2020 are supported. Soilgrids 2017 provides soil properties
    at 7 specific depths, while soilgrids_2020 provides soil properties averaged over
    6 depth intervals.
    Ref: Hengl, T., Mendes de Jesus, J., Heuvelink, G. B. M., Ruiperez Gonzalez,
    M., Kilibarda, M., Blagotic, A., et al.: SoilGrids250m:
    Global gridded soil information based on machine learning,
    PLoS ONE, 12, https://doi.org/10.1371/journal.pone.0169748, 2017.
    Ref: de Sousa, L.M., Poggio, L., Batjes, N.H., Heuvelink, G., Kempen, B., Riberio,
    E. and Rossiter, D., 2020. SoilGrids 2.0:
    producing quality-assessed soil information for the globe. SOIL Discussions,
    pp.1-37. https://doi.org/10.5194/soil-2020-65.

    A ``soil_mapping`` table can optionally be provided to derive parameters based
    on soil texture classes. A default table *soil_mapping_default* is available
    to derive the infiltration capacity of the soil.

    The following soil parameter maps are calculated:

        - **theta_s** : average saturated soil water content [m3/m3]
        - **theta_r** : average residual water content [m3/m3]
        - **ksat_vertical** : vertical saturated hydraulic conductivity at soil
            surface [mm/day]
        - **soil_thickness** : soil thickness [mm]
        - **f** : scaling parameter controlling the decline of ksat_vertical [mm-1]
            (fitted with curve_fit (scipy.optimize)), bounds are checked
        - **soil_f_** : scaling parameter controlling the decline of ksat_vertical
            [mm-1] (fitted with numpy linalg regression), bounds are checked
        - **soil_brooks_corey_c_** map: Brooks Corey coefficients [-] based on pore
            size distribution index for the wflow_sbm soil layers.
        - **meta_{soil_fn}_ksat_vertical_[z]cm** : ksat vertical [mm/day] at soil
            depths [z] of SoilGrids data [0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0]
        - **meta_soil_texture** : USDA Soil texture based on percentage clay, silt,
            sand mapping: [1:Clay, 2:Silty Clay, 3:Silty Clay-Loam, 4:Sandy Clay,
            5:Sandy Clay-Loam, 6:Clay-Loam, 7:Silt, 8:Silt-Loam, 9:Loam, 10:Sand,
            11: Loamy Sand, 12:Sandy Loam]


    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing soil properties.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    ptfKsatVer : str
        PTF to use for calculation ksat_vertical .
    soil_fn : str
        soilgrids version {'soilgrids', 'soilgrids_2020'}
    wflow_layers : list
        List of soil layer depths [cm] for which soil_brooks_corey_c is calculated.

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
        soildepth_mm_midpoint_surface = 10.0 * soildepth_cm_midpoint_surface
    else:
        soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_cm_midpoint = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
        soildepth_mm_midpoint_surface = 10.0 * soildepth_cm

    ds_out = xr.Dataset(coords=ds_like.raster.coords)

    # set nodata values in dataset to NaN (based on soil property SLTPPT at
    # first soil layer)
    # ds = xr.where(ds["sltppt_sl1"] == ds["sltppt_sl1"].raster.nodata, np.nan, ds)
    ds = ds.raster.mask_nodata()

    # concat along a sl dimension
    ds = concat_layers(ds, soil_fn)

    logger.info("calculate and resample theta_s")
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
    ds_out["theta_s"] = thetas.astype(np.float32)

    logger.info("calculate and resample theta_r")
    thetar_sl = xr.apply_ufunc(
        ptf.thetar_rawls_brakensiek,
        ds["sndppt"],
        ds["clyppt"],
        thetas_sl,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )

    if soil_fn == "soilgrids_2020":
        thetar = average_soillayers_block(thetar_sl, ds["soilthickness"])
    else:
        thetar = average_soillayers(thetar_sl, ds["soilthickness"])
    thetar = thetar.raster.reproject_like(ds_like, method="average")
    ds_out["theta_r"] = thetar.astype(np.float32)

    soilthickness_hr = ds["soilthickness"]
    soilthickness = soilthickness_hr.raster.reproject_like(ds_like, method="average")
    # wflow_sbm cannot handle (yet) zero soil thickness
    soilthickness = soilthickness.where(soilthickness > 0.0, np.nan)
    soilthickness.raster.set_nodata(np.nan)
    soilthickness = soilthickness.astype(np.float32)
    ds_out["soil_thickness"] = soilthickness * 10.0  # from [cm] to [mm]

    logger.info("calculate and resample ksat_vertical ")
    kv_sl_hr = kv_layers(ds, thetas_sl, ptfKsatVer)
    kv_sl = np.log(kv_sl_hr)
    kv_sl = kv_sl.raster.reproject_like(ds_like, method="average")
    kv_sl = np.exp(kv_sl)

    logger.info("calculate and resample pore size distribution index")
    ds_c = brooks_corey_layers(
        thetas_sl=thetas_sl,
        ds=ds,
        ds_like=ds_like,
        soil_fn=soil_fn,
        wflow_layers=wflow_layers,
        soildepth_cm=soildepth_cm,
    )
    ds_out["soil_brooks_corey_c"] = ds_c

    ds_out["ksat_vertical"] = kv_sl.sel(sl=1).astype(np.float32)

    for i, sl in enumerate(kv_sl["sl"]):
        kv = kv_sl.sel(sl=sl)
        ds_out[
            f"meta_{soil_fn}_ksat_vertical_" + str(soildepth_cm_midpoint[i]) + "cm"
        ] = kv.astype(np.float32)

    kv = kv_sl / kv_sl.sel(sl=1)
    logger.info(
        "fit z - log(ksat_vertical ) with numpy linalg regression (y = b*x) -> M_"
    )
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
    M_ = constrain_M(M_, popt_0_, M_minmax)
    ds_out["soil_f_"] = ((thetas - thetar) / M_).astype(np.float32)

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
    M = constrain_M(M, popt_0, M_minmax)
    ds_out["f"] = ((thetas - thetar) / M).astype(np.float32)

    # wflow soil map is based on USDA soil classification
    # soilmap = ds["tax_usda"].raster.interpolate_na()
    # soilmap = soilmap.raster.reproject_like(ds_like, method="mode")
    # ds_out["wflow_soil"] = soilmap.astype(np.float32)

    # wflow_soil map is based on soil texture calculated with percentage
    # sand, silt, clay
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

    soil_texture_out = soil_texture.raster.reproject_like(ds_like, method="mode")
    # np.nan is not a valid value for array with type integer
    ds_out["meta_soil_texture"] = soil_texture_out
    ds_out["meta_soil_texture"].raster.set_nodata(0)

    dtypes = {"meta_soil_texture": np.int32}
    for var in ds_out:
        dtype = dtypes.get(var, np.float32)
        logger.debug(f"Interpolate nodata (NaN) values for {var}")
        ds_out[var] = ds_out[var].raster.interpolate_na("nearest")
        ds_out[var] = ds_out[var].fillna(nodata).astype(dtype)
        ds_out[var].raster.set_nodata(np.dtype(dtype).type(nodata))

    return ds_out


def soilgrids_brooks_corey(
    ds: xr.Dataset,
    ds_like: xr.Dataset,
    soil_fn: str = "soilgrids",
    wflow_layers: list[int] = [100, 300, 800],
):
    """
    Determine Brooks Corey coefficient per wflow soil layer depth.

    First pore size distribution index is computed based on theta_s and other soil
    parameters and scaled to the model resolution.

    Then the Brooks Corey coefficient is computed for each wflow soil layer depth
    by weighted averaging the pore size distribution index over the soil thickness.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing soil properties.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    soil_fn : str
        soilgrids version {'soilgrids', 'soilgrids_2020'}
    wflow_layers : list
        List of wflow soil layer depths [cm] over which soil_brooks_corey_c is
        calculated.

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing soil_brooks_corey_c for the wflow_sbm soil layers.
    """
    if soil_fn == "soilgrids_2020" or soil_fn == "soilgrids":
        soildepth_cm = np.array([0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0])
    else:
        raise ValueError("Only soilgrids_2020 and soilgrids are supported.")

    ds_out = xr.Dataset(coords=ds_like.raster.coords)
    ds = ds.raster.mask_nodata()
    # concat along a sl dimension
    ds = concat_layers(ds, soil_fn)

    logger.info("calculate and resample theta_s")
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

    logger.info("calculate and resample pore size distribution index")
    # Brooks Corey coefficient
    ds_c = brooks_corey_layers(
        thetas_sl=thetas_sl,
        ds=ds,
        ds_like=ds_like,
        soil_fn=soil_fn,
        wflow_layers=wflow_layers,
        soildepth_cm=soildepth_cm,
    )
    ds_out["soil_brooks_corey_c"] = ds_c

    return ds_out


def soilgrids_sediment(
    ds: xr.Dataset,
    ds_like: xr.Dataset,
    usle_k_method: str = "renard",
    add_aggregates: bool = True,
) -> xr.Dataset:
    """
    Return soil parameter maps for sediment modelling at model resolution.

    Based on soil properties from SoilGrids dataset.
    Sediment size distribution and addition of small and large aggregates can be
    estimated from primary particle size distribution with Foster et al. (1980).
    USLE K factor can be computed from the soil data using Renard or EPIC methods.
    Calculation of D50 and fraction of fine and very fine sand (fvfs) from
    Fooladmand et al, 2006.

    The following soil parameter maps are calculated:

        * soil_clay_fraction: clay content of the topsoil [g/g]
        * soil_silt_fraction: silt content of the topsoil [g/g]
        * soil_sand_fraction: sand content of the topsoil [g/g]
        * soil_sagg_fraction: small aggregate content of the topsoil [g/g]
        * soil_lagg_fraction: large aggregate content of the topsoil [g/g]
        * erosion_soil_detachability: mean detachability of the soil
            (Morgan et al., 1998) [g/J]
        * usle_k: soil erodibility factor from the USLE equation [-]
        * soil_sediment_d50: median sediment diameter of the soil [mm]
        * land_govers_c: Govers factor for overland flow transport capacity [-]
        * land_govers_n: Govers exponent for overland flow transport capacity [-]

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing soil properties.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    usle_k_method : str
        Method to use for calculation of USLE_K {"renard", "epic"}.
    add_aggregates : bool
        Add small and large aggregates to the soil properties. Default is True.

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing gridded soil parameters for sediment modelling.
    """
    ds_out = xr.Dataset(coords=ds_like.raster.coords)

    # set nodata values in dataset to NaN (based on soil property SLTPPT at
    # first soil layer)
    # ds = xr.where(ds["sltppt_sl1"] == ds["sltppt_sl1"].raster.nodata, np.nan, ds)
    ds = ds.raster.mask_nodata()

    # soil properties
    pclay = ds["clyppt_sl1"]
    psilt = ds["sltppt_sl1"]
    poc = ds["oc_sl1"]

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
    ds_out["erosion_soil_detachability"] = erosK.astype(np.float32)

    # USLE K parameter
    if usle_k_method == "renard":
        usle_k = xr.apply_ufunc(
            ptf.UsleK_Renard,
            pclay,
            psilt,
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
        )
    elif usle_k_method == "epic":
        usle_k = xr.apply_ufunc(
            ptf.UsleK_EPIC,
            pclay,
            psilt,
            poc,
            dask="parallelized",
            output_dtypes=[float],
            keep_attrs=True,
        )
    usle_k = usle_k.raster.reproject_like(ds_like, method="average")
    ds_out["usle_k"] = usle_k.astype(np.float32)

    # Mean diameter of the soil
    d50 = xr.apply_ufunc(
        ptf.mean_diameter_soil,
        pclay,
        psilt,
        dask="parallelized",
        output_dtypes=[float],
        keep_attrs=True,
    )
    ds_out["soil_sediment_d50"] = d50.raster.reproject_like(
        ds_like, method="average"
    ).astype(np.float32)

    # Govers factor and exponent for overland flow transport capacity
    c_govers = ((d50 * 1000 + 5) / 0.32) ** (-0.6)
    n_govers = ((d50 * 1000 + 5) / 300) ** (0.25)
    ds_out["land_govers_c"] = c_govers.raster.reproject_like(
        ds_like, method="average"
    ).astype(np.float32)
    ds_out["land_govers_n"] = n_govers.raster.reproject_like(
        ds_like, method="average"
    ).astype(np.float32)

    # Sediment size distribution
    if add_aggregates:
        fclay = 0.20 * pclay / 100
        fsilt = 0.13 * psilt / 100
        fsand = (1 - pclay / 100 - psilt / 100) * (1 - pclay / 100) ** 2.4
        fsagg = 0.28 * (pclay / 100 - 0.25) + 0.5
        fsagg = fsagg.where(pclay < 50, 0.57)
        fsagg = fsagg.where(pclay > 25, 2 * pclay / 100)
    else:
        fclay = pclay / 100
        fsilt = psilt / 100
        fsagg = fclay * 0.0

    # Reproject to model resolution
    fclay = fclay.raster.reproject_like(ds_like, method="average")
    ds_out["soil_clay_fraction"] = fclay.astype(np.float32)
    fsilt = fsilt.raster.reproject_like(ds_like, method="average")
    ds_out["soil_silt_fraction"] = fsilt.astype(np.float32)
    fsagg = fsagg.raster.reproject_like(ds_like, method="average")
    ds_out["soil_sagg_fraction"] = fsagg.astype(np.float32)
    # Make sure that the sum of the percentages is 100
    if add_aggregates:
        fsand = fsand.raster.reproject_like(ds_like, method="average").astype(
            np.float32
        )
        ds_out["soil_sand_fraction"] = fsand
        ds_out["soil_lagg_fraction"] = (1 - fclay - fsilt - fsand - fsagg).astype(
            np.float32
        )
    else:
        ds_out["soil_sand_fraction"] = (1 - fclay - fsilt).astype(np.float32)
        ds_out["soil_lagg_fraction"] = (fclay * 0.0).astype(np.float32)

    for var in ds_out:
        ds_out[var] = ds_out[var].raster.interpolate_na("nearest")
        logger.info(f"Interpolate NAN values for {var}")
        ds_out[var] = ds_out[var].fillna(nodata)
        ds_out[var].raster.set_nodata(nodata)

    return ds_out


def update_soil_with_paddy(
    ds: xr.Dataset,
    ds_like: xr.Dataset,
    paddy_mask: xr.DataArray,
    soil_fn: str = "soilgrids",
    update_c: bool = True,
    wflow_layers: list[int] = [50, 100, 50, 200, 800],
    target_conductivity: list[None | int | float] = [None, None, 5, None, None],
):
    """
    Update soil_brooks_corey_c and soil_ksat_vertical_factor for paddy fields.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing soil properties.
    ds_like : xarray.DataArray
        Dataset at model resolution.
        Required variables: basins, ksat_vertical, f, elevtn, soil_brooks_corey_c
    paddy_mask : xarray.DataArray
        Dataset containing paddy fields mask.
    soil_fn : str
        soilgrids version {'soilgrids', 'soilgrids_2020'}
    update_c : bool
        Update soil_brooks_corey_c based on change in wflow_layers.
    wflow_layers : list
        List of soil layer depths [cm] for which soil_brooks_corey_c is calculated.
    target_conductivity : list
        Target conductivity for each wflow layer.

    Returns
    -------
    soil_out : xarray.Dataset
        Dataset containing updated soil properties.
    """
    if len(wflow_layers) != len(target_conductivity):
        raise ValueError(
            "Lengths of wflow_thicknesslayers and target_conductivity does not match"
        )

    # Set soil_ksat_vertical_factor maps
    # Determine the fraction required to reach target_conductivity
    # Using to wflow exponential decline to determine the conductivity at the
    # required depth Find value at the bottom of the required layer and infer
    # required correction factor for that layer Values are only set for locations
    # with paddy irrigation, all other cells are set to be equal to 1
    logger.info("Adding soil_ksat_vertical_factor map")
    kv0 = ds_like["ksat_vertical"]
    f = ds_like["f"]
    kv0_mask = kv0.where(paddy_mask == 1)
    f_mask = f.where(paddy_mask == 1)

    # update soil_brooks_corey_c
    if update_c:
        ds_out = soilgrids_brooks_corey(
            ds=ds,
            ds_like=ds_like,
            soil_fn=soil_fn,
            wflow_layers=wflow_layers,
        )
        for var in ds_out:
            dtype = np.float32
            logger.debug(f"Interpolate nodata (NaN) values for {var}")
            ds_out[var] = ds_out[var].raster.interpolate_na("nearest")
            ds_out[var] = ds_out[var].where(
                ~ds_like["basins"].raster.mask_nodata().isnull()
            )
            ds_out[var] = ds_out[var].fillna(-9999).astype(dtype)
            ds_out[var].raster.set_nodata(np.dtype(dtype).type(-9999))

        # temporarily add the dem to the dataset
        ds_out["elevtn"] = ds_like["elevtn"]

    # Compute the kv_frac (should be done after updating soil_brooks_corey_c!)
    da_kvfrac = soilparams.update_kvfrac(
        ds_model=ds_like if not update_c else ds_out,
        kv0_mask=kv0_mask,
        f_mask=f_mask,
        wflow_thicknesslayers=wflow_layers,
        target_conductivity=target_conductivity,
    )
    if update_c:
        ds_out["soil_ksat_vertical_factor"] = da_kvfrac
        # Remove elevation variable
        ds_out = ds_out.drop_vars("elevtn")
    else:
        ds_out = da_kvfrac.to_dataset(name="soil_ksat_vertical_factor")

    return ds_out
