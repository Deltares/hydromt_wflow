"""Generic methods to add gridded data to Wflow."""

import logging

import xarray as xr
from hydromt.gis.raster_utils import full
from hydromt.model.processes.meteo import resample_time

logger = logging.getLogger(f"hydromt.{__name__}")

__all__ = ["grid_from_geodataset"]


def _check_geodataset_type(da: xr.DataArray) -> str:
    """Check the type of geodataset dataarray (static/cyclic/forcing)."""
    if "time" in da.dims:
        if len(da.time) == 1:
            da_type = "static"
        elif len(da.time) in [12, 365, 366]:
            da_type = "cyclic"
        else:
            da_type = "forcing"
    elif len(da.dims) == 1:
        # only index dim
        da_type = "static"
    else:
        raise ValueError(
            "Cannot determine if geodataset contains static/cyclic/forcing data."
        )

    return da_type


def grid_from_geodataset(
    da: xr.DataArray,
    ds_like: xr.Dataset,
    fill_value: float | int | None = None,
    nodata_value: float | int = -9999,
    mask: str | None = None,
    freq: str = "D",
    resample_time_kwargs: dict | None = None,
) -> tuple[xr.DataArray, str]:
    """
    Regrid a geodataset dataarray to the grid of the staticmaps dataset.

    Parameters
    ----------
    da : xr.DataArray
        Data array from geodataset with vector geometry.
    ds_like : xr.Dataset
        Dataset with the grid to reproject to.
    fill_value : float | int, optional
        Fill value for the other grid cells. If not provided, the same value as the
        nodata value is used. By default None.
    nodata_value : float | int, optional
        No data value for the grid cells if not defined in the geodataset. By default
        -9999.
    mask : str, optional
        Name of the mask to apply on the grid. Should be a layer already present in
        ds_like. If None, the basin mask will be used.
    freq : str, optional
        Resampling frequency for time dimension, by default "D". Only used if da is of
        type forcing.
    resample_time_kwargs : dict, optional
        Additional keyword arguments for resample_time function, by default None. Only
        used if da is of type forcing.
    """
    # Check the type static/forcing/cyclic
    da_type = _check_geodataset_type(da)

    # Reproject to model crs
    da = da.vector.to_crs(ds_like.raster.crs)
    # Time resampling of the timeseries
    resample_time_kwargs = resample_time_kwargs or {}
    if freq is not None and da_type == "forcing":
        da = resample_time(da, freq=freq, **resample_time_kwargs)

    # Prepare output dataarray
    coords = {dim: ds_like[dim] for dim in ds_like.raster.dims}
    if da_type != "static":
        coords["time"] = da.time
        # Move time to first dimension if it exists
        coords = {"time": coords.pop("time"), **coords}

    nodata = da.vector.nodata
    da_out = full(
        coords=coords,
        nodata=nodata if nodata is not None else nodata_value,
        fill_value=fill_value,
        name=da.name,
        dtype=da.dtype,
        attrs=da.attrs,
        crs=ds_like.raster.crs,
        lazy=da_type == "forcing",
    )

    # Get locations indexes
    # idxs = ds_like.raster.xy_to_idx(da.x, da.y)
    rows, cols = ds_like.raster.rowcol(
        da.vector.geometry.x.values, da.vector.geometry.y.values
    )
    # Add values to output dataarray
    if da_type == "static":
        for r, c, v in zip(rows, cols, da.values):
            da_out[r, c] = v.item()
    else:
        # Use sel as da_out is lazy in case of forcing
        y_dim = da_out.raster.y_dim
        x_dim = da_out.raster.x_dim
        index_dim = da.vector.index_dim
        for r, c, v in zip(rows, cols, da[index_dim].values):
            val = da.sel({index_dim: v}).values
            da_out[dict(time=slice(None), **{y_dim: r, x_dim: c})] = val

    # Masking
    if mask is not None:
        if mask not in ds_like:
            raise ValueError(f"Mask {mask} not found in staticmaps.")
        da_out = da_out.where(
            ds_like[mask] != ds_like[mask].raster.nodata, da_out.raster.nodata
        )

    return da_out, da_type
