"""Groundwater workflows."""

import logging

import dask
import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import binary_erosion, convolve

METHODS = {
    "mean": np.mean,
    "max": np.max,
    "min": np.min,
}

logger = logging.getLogger(__name__)


__all__ = ["constant_boundary", "soil_adjust", "soil_layers", "soil_parameters"]


def full_like(
    da_like: xr.DataArray,
    dtype: type = np.float32,
    fill_value: float | int = np.nan,
    name: str = "data",
):
    """_summary_."""
    dsk = dask.array.full_like(
        da_like,
        dtype=dtype,
        fill_value=fill_value,
    )
    da = xr.DataArray(
        data=dsk,
        coords=da_like.coords,
        dims=da_like.dims,
    )
    da.attrs = {"_FillValue": fill_value}
    da.name = name
    return da


def constant_boundary(
    da_like: xr.DataArray,
    waterfrac: xr.DataArray,
    value: float | int = 0,
) -> xr.DataArray:
    """_summary_."""
    # First create the mask
    mask = da_like != da_like.raster.nodata
    mask = mask.astype(int)

    # Define the kernel
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int32)

    all_bounds = np.logical_xor(
        mask,
        binary_erosion(mask, structure=kernel),
    )
    all_bounds = all_bounds.astype(int)

    # TODO update this, its bad
    mask2 = waterfrac > 0
    mask2 = mask2.astype(int)
    water_buffer = convolve(mask2, kernel, mode="constant", cval=0.0)

    bounds_mask = all_bounds.where(water_buffer > 0, 0)
    bounds = full_like(da_like, fill_value=-9999, name="Bounds_gw")
    bounds = bounds.where(bounds_mask == 0, value)

    return bounds


def soil_adjust(
    da_like: xr.DataArray,
    fparam: xr.DataArray,
    mask: xr.DataArray,
):
    """_summary_."""
    # Set the new f param from the mask
    new_f = xr.where(mask, fparam / 10, np.nan)
    new_f.raster.set_crs(da_like.raster.crs)
    new_f.name = "f_gw"

    new_f.raster.set_nodata(np.nan)
    new_f = new_f.raster.interpolate_na(method="linear")
    new_f = new_f.raster.interpolate_na()  # Do again to fill gaps
    new_f = new_f.where(da_like != da_like.raster.nodata, -9999)
    new_f.raster.set_nodata(-9999)

    return new_f.raster.gdal_compliant()


def soil_layers(
    da_like: xr.DataArray,
    bounds: xr.DataArray,
    soil_thickness: xr.DataArray,
    layers: xr.Dataset,
    layer_ids: list | tuple,
    max_depth: float | int = 30,
    resampling_method: str = "linear",
) -> xr.DataArray:
    """_summary_."""
    # reproject the layers to model resolution
    layers_mr = layers.raster.reproject_like(
        da_like,
        method="average",
    )

    # Create the layer dataset
    acq = full_like(
        da_like,
        name="SoilThickness_gw",
    )

    # Fill out the new layer first with values from data
    _count = 0
    for layer in layer_ids:
        if _count == 0:
            acq = acq.where(np.isnan(layers_mr[layer]), layers_mr[layer])
            _count += 1
            continue
        acq = acq.where(~np.isnan(acq), layers_mr[layer])
        _count += 1

    # Solve the acquifer thickness in a proper manner
    dem = da_like.raster.mask_nodata()
    bot = dem - acq
    bot = xr.where(bot > bounds, bounds - 1, bot)
    bot = xr.where(np.isnan(bot) & ~np.isnan(bounds), bounds - 1, bot)
    new_acq = dem - bot

    # Clip the acquifer thickness
    new_acq = xr.where(new_acq > max_depth, max_depth, new_acq)
    new_acq = xr.where(new_acq < soil_thickness, soil_thickness, new_acq)
    new_acq.raster.set_nodata(np.nan)
    mask = ~np.isnan(new_acq)

    # Interpolate the missing values
    new_acq = new_acq.raster.interpolate_na(method=resampling_method, extrapolate=True)
    acq.values = new_acq

    # Fill out the correct nodata
    acq = acq.where(da_like != da_like.raster.nodata, -9999)
    acq.raster.set_nodata(-9999)

    # Spit it out gdal compliant
    return acq.raster.gdal_compliant(), mask


def soil_parameters(
    da_like: xr.DataArray,
    layers: xr.Dataset,
    linktable: pd.DataFrame,
    layer_ids: list | tuple,
    params_method: str = "mean",
    resampling_method: str = "linear",
) -> xr.Dataset:
    """_summary_."""
    # reproject the layers to model resolution
    layers_mr = layers.raster.reproject_like(
        da_like,
        method="average",
    )

    # resulting dataset
    kv = f"KsatVer_gw_{params_method}"
    ss = f"SpecificStorage_{params_method}"
    khf = "KsatHorFrac_gw"
    ds = xr.Dataset()
    ds[kv] = full_like(da_like)
    ds[ss] = full_like(da_like)
    ds[khf] = full_like(da_like)

    # Fill out the new layer first with values from data
    _count = 0
    for layer in layer_ids:
        if _count == 0:
            ds[kv] = ds[kv].where(
                np.isnan(layers_mr[layer]),
                METHODS[params_method](linktable.loc[layer, "kv"]),
            )
            ds[ss] = ds[ss].where(
                np.isnan(layers_mr[layer]),
                METHODS[params_method](linktable.loc[layer, "ss"]),
            )
            ds[khf] = ds[khf].where(
                np.isnan(layers_mr[layer]),
                METHODS[params_method](linktable.loc[layer, "aniv"]),
            )
            _count += 1
            continue
        ds[kv] = ds[kv].where(
            ~np.isnan(ds[kv]) | np.isnan(layers_mr[layer]),
            METHODS[params_method](linktable.loc[layer, "kv"]),
        )
        ds[ss] = ds[ss].where(
            ~np.isnan(ds[ss]) | np.isnan(layers_mr[layer]),
            METHODS[params_method](linktable.loc[layer, "ss"]),
        )
        ds[khf] = ds[khf].where(
            ~np.isnan(ds[khf]) | np.isnan(layers_mr[layer]),
            METHODS[params_method](linktable.loc[layer, "aniv"]),
        )
        _count += 1

    # Fill nodata values by extrapolating and reset the nodata value to -9999
    ds = ds.raster.interpolate_na(method=resampling_method, extrapolate=True)
    ds = ds.where(da_like != da_like.raster.nodata, -9999)
    for var in ds.data_vars:
        ds[var].raster.set_nodata(-9999)

    return ds
