"""Forcing workflow for wflow."""

import logging
from typing import Optional

import numpy as np
import xarray as xr
from hydromt.workflows.forcing import resample_time

logger = logging.getLogger(__name__)

__all__ = ["pet", "spatial_interpolation", "temp"]


def precip(
    precip,
    da_like,
    freq=None,
    reproj_method="nearest_index",
    resample_kwargs=None,
    logger=logger,
):
    """Return the lazy reprojection of precipitation to model.

      Applies the projection to the grid and resampling of time dimension to frequency.

    Parameters
    ----------
    precip: xarray.DataArray
        DataArray of precipitation forcing [mm]
    da_like: xarray.DataArray or Dataset
        DataArray of the target resolution and projection.
    clim: xarray.DataArray
        DataArray of monthly precipitation climatology. If provided this is used to
        to correct the precip downscaling.
    freq: str, Timedelta
        output frequency of time dimension
    reproj_method: str, optional
        Method for spatital reprojection of precip, by default 'nearest_index'
    resample_kwargs:
        Additional key-word arguments (e.g. label, closed) for time resampling method
    kwargs:
        Additional arguments to pass through to underlying methods.
    logger:
        The logger to use

    Returns
    -------
    p_out: xarray.DataArray (lazy)
        processed precipitation forcing
    """
    resample_kwargs = resample_kwargs or {}
    if precip.raster.dim0 != "time":
        raise ValueError(f'First precip dim should be "time", not {precip.raster.dim0}')
    # downscale precip (lazy); global min of zero
    p_out = precip.raster.reproject_like(da_like, method=reproj_method)

    # Fill nodata
    p_out = p_out.interpolate_na(
        dim=p_out.raster.x_dim,
        fill_value="extrapolate",
    )
    # Minimum must be zero
    p_out = p_out.where(p_out >= 0, 0)

    # resample time
    p_out.name = "precip"
    p_out.attrs.update(unit="mm")
    if freq is not None:
        resample_kwargs.update(upsampling="bfill", downsampling="sum", logger=logger)
        p_out = resample_time(p_out, freq, conserve_mass=True, **resample_kwargs)
    return p_out


def temp(
    temp: xr.DataArray,
    ds_like: xr.Dataset,
    freq: str = "D",
    mask_name: Optional[str] = None,
    chunksize: Optional[int] = None,
    logger: Optional[logging.Logger] = logger,
) -> xr.DataArray:
    """
    Resample and reproject temp to the grid of ds_like.

    Parameters
    ----------
    Temp : xr.DataArray
        Temp data array with time as first dimension.
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
    temp_out : xr.DataArray
        Resampled and reprojected temp data array.
    """
    if chunksize is not None:
        temp = temp.chunk({"time": chunksize})

    if temp.raster.dim0 != "time":
        raise ValueError(f'First temp dim should be "time", not {temp.raster.dim0}')

    # change nodata
    temp = temp.where(temp != temp.raster.nodata, np.nan)
    temp.raster.set_nodata(np.nan)

    # Reproject the data
    temp_out = temp.raster.reproject_like(ds_like, method="nearest_index")

    # Interpolate data
    temp_out = temp_out.interpolate_na(
        dim=temp_out.raster.x_dim,
        fill_value="extrapolate",
    )

    # resample time
    resample_kwargs = dict(label="right", closed="right")
    if freq is not None:
        resample_kwargs.update(upsampling="bfill", downsampling="mean", logger=logger)
        temp_out = resample_time(temp_out, freq, conserve_mass=False, **resample_kwargs)
        # nodata is lost in resampling, set it back
        temp_out.raster.set_nodata(np.nan)

    # Mask
    if mask_name is not None:
        mask = ds_like[mask_name].values > 0
        temp_out = temp_out.where(mask)

    # Attributes
    temp_out.name = "temp"
    temp_out.attrs.update(unit="degree C.")

    return temp_out


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

    # Reproject the data
    pet_out = pet.raster.reproject_like(ds_like, method="nearest_index")

    # Fill nodata
    pet_out = pet_out.interpolate_na(
        dim=pet_out.raster.x_dim,
        fill_value="extrapolate",
    )
    # Minimum must be zero
    pet_out = pet_out.where(pet_out >= 0, 0)

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
    forcing: xr.DataArray,
    ds_like: xr.Dataset,
    interp_type: str,
    nnearest: Optional[int] = 4,
    p: Optional[float] = 2,
    remove_missing: Optional[bool] = False,
    cov: Optional[str] = "1.0 Exp(10000.)",
    src_drift: Optional[np.ndarray] = None,
    trg_drift: Optional[np.ndarray] = None,
    mask_name: Optional[str] = None,
    logger: Optional[logging.Logger] = logger,
) -> xr.DataArray:
    """
    Interpolate spatial forcing data from station observations to a regular grid.

    This workflow uses the wradlib.ipol.interpolate function in wradlib. \
    It wraps the following interpolation types into a single function:
    - "nearest": nearest-neighbour interpolation, also works with a single station.
    - "idw": inverse-distance weighing using 1 / distance ** p.
    - "linear": linear interpolation using scipy.interpolate.LinearNDInterpolator, \
    may result in missing values when station coverage is limited.
    - "ordinarykriging": interpolate using Ordinary Kriging, see wradlib documentation \
    for a full explanation: `wradlib.ipol.OrdinaryKriging <https://docs.wradlib.org/en/latest/generated/wradlib.ipol.OrdinaryKriging.html>`
    - "externaldriftkriging": Kriging interpolation including an external drift, see \
    wradlib documentation for a full explanation: `wradlib.ipol.ExternalDriftKriging <https://docs.wradlib.org/en/latest/generated/wradlib.ipol.ExternalDriftKriging.html>`

    Parameters
    ----------
    forcing : xr.DataArray
        GeoDataArray with the forcing data with time and index of the point data.
    ds_like : xr.Dataset
        Target dataset defining the grid for interpolation.
    interp_type : str
        Interpolation method. Options: "nearest", "idw", "linear", \
        "ordinarykriging", "externaldriftkriging".
    nnearest : int, optional
        Maximum number of neighbors for interpolation. Default is 4.
    p : float, optional
        Power parameter for IDW interpolation. Default is 2.
    remove_missing : bool, optional
        Whether to mask NaN values in the input data. Default is False.
    cov : str, optional
        Covariance model for Kriging. Default is '1.0 Exp(10000.)'.
    src_drift : np.ndarray, optional
        External drift values at source points (stations).
    trg_drift : np.ndarray, optional
        External drift values at target points (grid).
    mask_name : str, optional
        Name of the mask variable in ds_like, by default None.

    Returns
    -------
    xr.DataArray
        Interpolated forcing data on the targeted grid.

    See Also
    --------
    `wradlib.ipol.interpolate <https://docs.wradlib.org/en/latest/ipol.html#wradlib.ipol.interpolate>`
    """
    try:
        import wradlib as wrl
    except ImportError:
        raise ModuleNotFoundError(
            "The wradlib package is required for spatial interpolation."
        )

    # Specify wradlib interpolation classes
    interpolation_classes = {
        "nearest": {"obj": wrl.ipol.Nearest, "args": None},
        "linear": {"obj": wrl.ipol.Linear, "args": ["remove_missing"]},
        "idw": {"obj": wrl.ipol.Idw, "args": ["nnearest", "p"]},
        "ordinarykriging": {
            "obj": wrl.ipol.OrdinaryKriging,
            "args": ["cov", "nnearest"],
        },
        "externaldriftkriging": {
            "obj": wrl.ipol.ExternalDriftKriging,
            "args": ["cov", "nnearest", "src_drift", "trg_drift", "remove_missing"],
        },
    }

    # Reproject forcing data to match the target CRS
    crs = ds_like.raster.crs
    forcing = forcing.vector.to_crs(crs).astype("float32")

    # Extract station coordinates
    gdf_stations = forcing.vector.to_gdf()
    src = np.vstack((gdf_stations.geometry.x, gdf_stations.geometry.y)).T

    # Some info/checks on the station data
    nb_stations = len(gdf_stations)
    if mask_name is not None:
        basins = ds_like[mask_name].raster.vectorize()
        nb_inside = gdf_stations.within(basins.union_all()).sum()
        logger.info(
            f"Found {nb_stations} stations in the forcing data, "
            f"of which {nb_inside} are located inside the basin."
        )
    else:
        logger.info(f"Found {nb_stations} stations in the forcing data.")
    if forcing.isnull().any():
        logger.warning(
            "Forcing data contains NaN values. "
            "These will be skipped during interpolation."
        )

    # Extract grid coordinates
    x_coords = ds_like.coords[ds_like.raster.x_dim].values
    y_coords = ds_like.coords[ds_like.raster.y_dim].values
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    trg = np.vstack((grid_x.ravel(), grid_y.ravel())).T

    # Validate interpolation type
    if interp_type not in interpolation_classes:
        raise ValueError(
            f"Unsupported interpolation type '{interp_type}'. "
            f"Choose from: {', '.join(interpolation_classes.keys())}."
        )

    # Prepare interpolation arguments
    ipclass = interpolation_classes[interp_type]["obj"]
    required_args = interpolation_classes[interp_type]["args"]

    interp_args_all = {
        "nnearest": nnearest,
        "p": p,
        "remove_missing": remove_missing,
        "cov": cov,
        "src_drift": src_drift,
        "trg_drift": trg_drift,
    }

    if required_args:
        interp_args = {k: v for k, v in interp_args_all.items() if k in required_args}
        interp_args_log = ", ".join(f"{k}={v}" for k, v in interp_args.items())
        logger.info(
            f"Starting interpolation with type '{interp_type}' ({interp_args_log})."
        )
    else:
        interp_args = {}
        logger.info(f"Starting interpolation with type '{interp_type}'.")

    # Perform interpolation using wradlib
    interpolated = wrl.ipol.interpolate(
        src=src,
        trg=trg,
        vals=forcing.values,
        ipclass=ipclass,
        **interp_args,
    )

    # Reshape interpolated data and create DataArray
    interpolated_reshaped = interpolated.reshape(
        (len(y_coords), len(x_coords), len(forcing.time))
    ).transpose(2, 0, 1)
    da_forcing = xr.DataArray(
        data=interpolated_reshaped,
        coords={
            "time": forcing.time,
            ds_like.raster.y_dim: y_coords,
            ds_like.raster.x_dim: x_coords,
        },
        dims=["time", ds_like.raster.y_dim, ds_like.raster.x_dim],
    )

    # Set metadata
    da_forcing = da_forcing.astype("float32")
    da_forcing.raster.set_nodata(np.nan)
    da_forcing.raster.set_crs(crs)

    # Mask data
    if mask_name is not None:
        mask = ds_like[mask_name].values > 0
        da_forcing = da_forcing.where(mask)

    return da_forcing
