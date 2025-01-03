"""Forcing workflow for wflow."""

import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from hydromt.workflows.forcing import resample_time
from metpy.interpolate import interpolate_to_grid, remove_nan_observations

logger = logging.getLogger(__name__)

__all__ = ["pet"]


def pet(
    pet: xr.DataArray,
    ds_like: xr.Dataset,
    freq: str = "D",
    mask_name: Optional[str] = None,
    chunksize: Optional[int] = None,
    logger: Optional[logging.Logger] = logger,
) -> xr.DataArray:
    """
    Resample and reproject PET to the grid of ds_like.

    Parameters
    ----------
    pet : xr.DataArray
        PET data array with time as first dimension.
    ds_like : xr.Dataset
        Dataset with the grid to reproject to.
    freq : str, optional
        Resampling frequency, by default "D".
    mask_name : str, optional
        Name of the mask variable in ds_like, by default None.
    chunksize : int, optional
        Chunksize for the time dimension for resampling, by default None to use default
        time chunk.

    Returns
    -------
    pet_out : xr.DataArray
        Resampled and reprojected PET data array.
    """
    if chunksize is not None:
        pet = pet.chunk({"time": chunksize})

    if pet.raster.dim0 != "time":
        raise ValueError(f'First pet dim should be "time", not {pet.raster.dim0}')

    # change nodata
    pet = pet.where(pet != pet.raster.nodata, np.nan)
    pet.raster.set_nodata(np.nan)

    # Minimum is zero
    pet_out = np.fmax(pet.raster.reproject_like(ds_like, method="nearest_index"), 0)

    # resample time
    resample_kwargs = dict(label="right", closed="right")
    if freq is not None:
        resample_kwargs.update(upsampling="bfill", downsampling="sum", logger=logger)
        pet_out = resample_time(pet_out, freq, conserve_mass=True, **resample_kwargs)
        # nodata is lost in resampling, set it back
        pet_out.raster.set_nodata(np.nan)

    # Mask
    if mask_name is not None:
        mask = ds_like[mask_name].values > 0
        pet_out = pet_out.where(mask)

    # Attributes
    pet_out.name = "pet"
    pet_out.attrs.update(unit="mm")

    return pet_out


def spatial_interpolation(
    forcing: pd.DataFrame,
    stations: gpd.GeoDataFrame,
    interp_type: str,
    hres: float,
    *kwargs: Optional[dict],
) -> xr.DataArray:
    x = stations.geometry.x
    y = stations.geometry.y
    time = forcing.index
    data = []

    if np.isnan(forcing.values).any():
        logger.warning(
            """Forcing data contains NaN values.
            These will be skipped during interpolation.
            Consider replacing NaN with 0 to include missing observations."""
        )

    # TODO adding checks and logging depending on the different interpolation types?

    for timestep, observations in forcing.iterrows():
        z = observations.values
        x, y, z = remove_nan_observations(x=x, y=y, z=z)
        grid_x, grid_y, img = interpolate_to_grid(
            x=x, y=y, z=z, interp_type=interp_type, hres=hres, **kwargs
        )
        data.append(img)

    coords = {"time": time, "x": grid_x[0, :], "y": grid_y[:, 0]}
    da_forcing = xr.DataArray(data=data, coords=coords, dims=["time", "y", "x"])
    return da_forcing
