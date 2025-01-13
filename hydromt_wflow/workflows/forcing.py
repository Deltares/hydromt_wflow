"""Forcing workflow for wflow."""

import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from hydromt.workflows.forcing import resample_time

try:
    from metpy.interpolate import (
        interpolate_to_grid,
        remove_nan_observations,
    )
    from metpy.interpolate.grid import (
        generate_grid,
        get_boundary_coords,
    )

    HAS_METPY = True
except ImportError:
    HAS_METPY = False

logger = logging.getLogger(__name__)

__all__ = ["pet", "spatial_interpolation"]

interpolation_supported = {
    "nearest": None,
    "linear": None,
    "cubic": None,
    "rbf": ["rbf_func", "rbf_smooth"],
    # "natural_neighbor": None,
    # "cressman": ["minimum_neighbors", "search_radius"],
    "barnes": ["minimum_neighbors", "search_radius", "gamma", "kappa_star"],
}


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
    rbf_func: Optional[str] = "linear",
    rbf_smooth: Optional[float] = 0,
    minimum_neighbours: Optional[int] = 3,
    search_radius: Optional[float] = None,
    gamma: Optional[float] = 0.25,
    kappa_star: Optional[float] = 5.052,
    logger: Optional[logging.Logger] = logger,
) -> xr.DataArray:
    """
    Interpolate spatial forcing data from station observations to a regular grid.

    This workflow uses the hydromt_wflow.workflows.forcing.spatial_interpolation \
    function in MetPy. It supports methods from SciPy (nearest, linear, cubic, rbf), \
    and methods implemented directy in Metpy: \
    - 'barnes', based on Barnes (1964). A technique for maximizing details in \
    numerical weather map analysis. J. Appl. Meteor. Climatol., 3, \
    396-409, doi:10.1175/1520-0450(1964)003%3C0396:ATFMDI%3E2.0.CO;2.
    - 'cressman', based on Cressman (1959). An operational objective analysis \
    system. Mon. Wea. Rev., 87, 367-374,
    doi:10.1175/1520-0493(1959)087%3C0367:AOOAS%3E2.0.CO;2.
    - 'nearest_neighbor', based on Liang and Hale (2010): A Stable and Fast \
    Implementation of Natural Neighbor Interpolation. Center for Wave \
    Phenomena CWP-657, 14 pp.

    Parameters
    ----------
    forcing : pd.DataFrame
        DataFrame with the forcing data with time as index and stations as columns.
    stations : gpd.GeoDataFrame
        GeoDataFrame with the station locations with geometry column.
    interp_type : str
        Type of interpolation to use. Supported types are "nearest", "linear", "cubic",
        "rbf", "natural_neighbor", "cressman", and "barnes".
    hres : float
        Horizontal resolution of the output grid.
    rbf_func : str, optional
        Specifies which function to use for Rbf interpolation. Options include:
        'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', and
        'thin_plate'. Default is 'linear'.
    rbf_smooth : float, optional
        Smoothing value applied to rbf interpolation. Higher values result in more
        smoothing. Default is 0.
    minimum_neighbours : int, optional
        Minimum number of neighbors needed to perform Barnes or Cressman interpolation
        for a point. Default is 3.
    search_radius : float, optional
        A search radius to use for the Barnes and Cressman interpolation schemes.
        If search_radius is not specified, it will default to 5 times the average
        spacing of observations.
    gamma : float, optional
        Adjustable smoothing parameter for the barnes interpolation. Default is 0.25.
    kappa_star : float, optional
        Response parameter for barnes interpolation, specified nondimensionally
        in terms of the Nyquist. Default is 5.052.

    Returns
    -------
    xr.DataArray
        Interpolated forcing data on a regular grid with dimensions (time, y, x).

    See Also
    --------
    hydromt_wflow.workflows.forcing.spatial_interpolation
    """
    if not HAS_METPY:
        raise ModuleNotFoundError("MetPy package is required for spatial interpolation")

    if np.isnan(forcing.values).any():
        logger.warning(
            """Forcing data contains NaN values. \
These will be skipped during interpolation.""")

    if interp_type not in interpolation_supported.keys():
        raise ValueError(f"Interpolation type {interp_type} not recognized.")

    elif interpolation_supported[interp_type]:
        # Create a dictionary of the arguments to show in logging
        interp_args_dict = {
            "rbf_func": rbf_func,
            "rbf_smooth": rbf_smooth,
            "minimum_neighbors": minimum_neighbours,
            "search_radius": search_radius,
            "gamma": gamma,
            "kappa_star": kappa_star,
        }

        # Filter the arguments based on the interpolation type
        interp_args = ", ".join(
            [
                f"{key}={interp_args_dict[key]}"
                for key in interpolation_supported[interp_type]
            ]
        )
        logger.info(
            f"Starting interpolation with interp_type '{interp_type}' ({interp_args})."
        )

    else:
        logger.info(f"Starting interpolation with interp_type '{interp_type}'.")

    # Determine the dimensions of the resulting interpolated grid
    x = stations.geometry.x
    y = stations.geometry.y
    boundary_coords = get_boundary_coords(x, y)
    grid_x, grid_y = generate_grid(hres, boundary_coords)
    
    # Create empty DataArray to store the interpolated values
    coords = {"time": forcing.index, "x": grid_x[0, :], "y": grid_y[:, 0]}
    da_forcing = xr.DataArray(coords=coords, dims=["time", "y", "x"])
    da_forcing.raster.set_crs(stations.crs)
    
    empty_timesteps = []
    last_logged_progress = 0
    for i, (timestep, observations) in enumerate(forcing.iterrows()):
        
        # TODO is there a nicer way using the logger?
        progress = ((i + 1)/len(forcing))*100
        if progress // 10 > last_logged_progress // 10:
            last_logged_progress = progress
            logger.info(f"Interpolation progress: {int(progress)}%")
    
        z = observations.values
        _x, _y, _z = remove_nan_observations(x=x, y=y, z=z)
        
        if len(z) == 0:
            empty_timesteps.append(timestep)
            continue

        interp_x, interp_y, img = interpolate_to_grid(
            x=_x,
            y=_y,
            z=_z,
            interp_type=interp_type,
            hres=hres,
            rbf_func=rbf_func,
            rbf_smooth=rbf_smooth,
            minimum_neighbors=minimum_neighbours,
            search_radius=search_radius,
            kappa_star=kappa_star,
            gamma=gamma,
        )
        
        # Sometimes stations have missing data which changes the shape for that timestep
        if img.shape != grid_x.shape:
            interp_coords = { "x": interp_x[0, :], "y": interp_y[:, 0]}
            da_interp = xr.DataArray(data=img, coords=interp_coords, dims=["y", "x"])
            da_interp.raster.set_crs(stations.crs)
            # Reproject the interpolation result to fit da_forcing
            img = da_interp.raster.reproject_like(da_forcing, method='nearest').values
        
        # Assign the interpolated data to the DataArray
        da_forcing.loc[timestep, :, :] = img            

    if empty_timesteps:
        logger.info(f"{len(empty_timesteps)} timesteps contained only NaN values.")
    
    # Ensure correct metadata
    da_forcing = da_forcing.astype("float32")
    da_forcing.raster.set_nodata(np.nan)
    return da_forcing
