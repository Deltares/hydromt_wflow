"""Land surface temperature workflow for wflow."""

import logging
from typing import Optional, Union

import numpy as np
import xarray as xr
from hydromt.model.processes.meteo import resample_time

logger = logging.getLogger(__name__)

__all__ = ["albedo", "emissivity", "radiation", "add_var_to_forcing", 
           "solar_declination", "relative_distance", "extraterrestrial_radiation",
           "compute_net_longwave_radiation", "compute_net_radiation"]


def solar_declination(doy: int) -> float:
    """
    Calculate solar declination angle.
    
    Parameters
    ----------
    doy : int
        Day of year (1-365)
        
    Returns
    -------
    float
        Solar declination angle in radians
    """
    days_in_year = 365
    radians = np.sin((2 * np.pi * doy / days_in_year) - 1.39)
    decl = 0.409 * radians
    return decl


def relative_distance(doy: int) -> float:
    """
    Calculate relative distance between Earth and Sun.
    
    Parameters
    ----------
    doy : int
        Day of year (1-365)
        
    Returns
    -------
    float
        Relative distance in AU
    """
    days_in_year = 365
    radians = np.cos(2 * np.pi * doy / days_in_year)
    dist = radians * 0.033 + 1  # distance in AU
    return dist


def extraterrestrial_radiation(lat: float, doy: int) -> float:
    """
    Calculate extraterrestrial radiation.
    
    Parameters
    ----------
    lat : float
        Latitude in degrees
    doy : int
        Day of year (1-365)
        
    Returns
    -------
    float
        Extraterrestrial radiation in MJ/m²/day
    """
    gsc = 118.08  # MJ/m²/day
    lat_rad = np.radians(lat)
    decl = solar_declination(doy)
    dist = relative_distance(doy)
    sha = np.arccos(-np.tan(lat_rad) * np.tan(decl))
    Ra = (dist * gsc / np.pi * 
          (np.cos(lat_rad) * np.cos(decl) * np.sin(sha) + 
           sha * np.sin(lat_rad) * np.sin(decl)))
    return Ra


def compute_net_longwave_radiation(
    air_temperature: xr.DataArray,
    shortwave_radiation_in: xr.DataArray,
    latitude: xr.DataArray,
    time_coord: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate net longwave radiation.
    
    Formula: RLN = (σTa^4)(0.34 - 0.14√ea)(1.35(Rins/Rso) - 0.35)
    
    Parameters
    ----------
    air_temperature : xr.DataArray
        Air temperature [°C]
    shortwave_radiation_in : xr.DataArray
        Incoming shortwave radiation [W m-2]
    latitude : xr.DataArray
        Latitude [degrees]
    time_coord : xr.DataArray
        Time coordinate for day of year calculation
        
    Returns
    -------
    xr.DataArray
        Net longwave radiation [W m-2]
    """
    # Stefan-Boltzmann constant in MJK-4m-2day-1
    sigma = 4.903e-9
    
    # Convert temperature to Kelvin and calculate Ta^4
    temp_kelvin = air_temperature + 273.15
    temp_kelvin_4 = temp_kelvin ** 4
    
    # Calculate vapor pressure (ea)
    # ea = 0.611 * exp(17.27 * Ta / (237.3 + Ta)^2)
    vapor_pressure = 0.611 * np.exp(17.27 * air_temperature / (237.3 + air_temperature) ** 2)
    
    # Calculate (0.34 - 0.14√ea)
    b_term = 0.34 - 0.14 * np.sqrt(vapor_pressure)
    
    # Calculate extraterrestrial radiation for each time step
    doy = time_coord.dt.dayofyear
    Rso = xr.apply_ufunc(
        extraterrestrial_radiation,
        latitude,
        doy,
        input_core_dims=[[], []],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float]
    ) * 0.75  # Clear sky solar radiation
    
    # Convert shortwave from W m-2 to MJ/m²/day for ratio calculation
    # 1 W m-2 = 0.0864 MJ/m²/day
    shortwave_mj = shortwave_radiation_in * 0.0864
    
    # Calculate (1.35(Rins/Rso) - 0.35)
    ratio = shortwave_mj / Rso
    c_term = (1.35 * ratio) - 0.35
    
    # Calculate net longwave radiation
    net_longwave = sigma * temp_kelvin_4 * b_term * c_term
    
    # Convert back to W m-2
    net_longwave_w = net_longwave / 0.0864
    
    # Set attributes
    net_longwave_w.name = "net_longwave_radiation"
    net_longwave_w.attrs.update({
        "unit": "W m-2",
        "long_name": "Net longwave radiation",
        "description": "Calculated using Stefan-Boltzmann law and atmospheric correction"
    })
    
    return net_longwave_w


def compute_net_radiation(
    albedo: xr.DataArray,
    shortwave_radiation_in: xr.DataArray,
    air_temperature: xr.DataArray,
    latitude: xr.DataArray,
    time_coord: xr.DataArray,
) -> xr.DataArray:
    """
    Calculate net radiation.
    
    Formula: Rn = Rins - Routs + Rinl - Routl
    Simplified: Rn = (1-α)Rins - RLN
    
    Parameters
    ----------
    albedo : xr.DataArray
        Surface albedo [-]
    shortwave_radiation_in : xr.DataArray
        Incoming shortwave radiation [W m-2]
    air_temperature : xr.DataArray
        Air temperature [°C]
    latitude : xr.DataArray
        Latitude [degrees]
    time_coord : xr.DataArray
        Time coordinate for day of year calculation
        
    Returns
    -------
    xr.DataArray
        Net radiation [W m-2]
    """
    # Calculate net shortwave radiation: (1-α)Rins
    net_shortwave = (1 - albedo) * shortwave_radiation_in
    
    # Calculate net longwave radiation
    net_longwave = compute_net_longwave_radiation(
        air_temperature, shortwave_radiation_in, latitude, time_coord
    )
    
    # Calculate net radiation: Rn = RSNet - RLN
    net_radiation = net_shortwave - net_longwave
    
    # Set attributes
    net_radiation.name = "net_radiation"
    net_radiation.attrs.update({
        "unit": "W m-2",
        "long_name": "Net radiation",
        "description": "Net radiation = net shortwave - net longwave"
    })
    
    return net_radiation


def add_var_to_forcing(
    mod: "WflowModel",
    ds: Union[xr.Dataset, xr.DataArray],
    var: str,
    freq: Optional[str] = None,
    reproj_method: str = "nearest_index",
    resample_kwargs: Optional[dict] = None,
) -> "WflowModel":
    """
    Add variable to model forcing with proper reprojection and masking.
    
    Parameters
    ----------
    mod : WflowModel
        Wflow model instance
    ds : xr.Dataset or xr.DataArray
        Dataset or DataArray containing the variable to add
    var : str
        Variable name to add to forcing
    freq : str, optional
        Resampling frequency, by default None
    reproj_method : str, optional
        Method for spatial reprojection, by default "nearest_index"
    resample_kwargs : dict, optional
        Additional arguments for time resampling, by default None
        
    Returns
    -------
    WflowModel
        Updated model with new forcing variable
    """
    resample_kwargs = resample_kwargs or {}
    
    # get reference grid and fill value
    ex_grid = mod.grid["wflow_dem"]
    ex_fillval = ex_grid.attrs["_FillValue"]
    ex_mask = ex_grid.values == ex_fillval
    
    # reproject data
    if isinstance(ds, xr.Dataset):
        da = ds[var].raster.reproject_like(ex_grid, method=reproj_method)
    else:
        da = ds.raster.reproject_like(ex_grid, method=reproj_method)
    
    # apply mask and set nodata
    da = da.where(~ex_mask, ex_fillval)
    da.raster.set_nodata(ex_fillval)
    da.raster.attrs["_FillValue"] = ex_fillval
    
    # resample time if requested
    if freq is not None and "time" in da.dims:
        resample_kwargs.update(upsampling="bfill", downsampling="mean")
        da = resample_time(da, freq, conserve_mass=False, **resample_kwargs)
        da.raster.set_nodata(ex_fillval)
    
    # set variable name and add to forcing
    da.name = var
    mod.forcing[var] = da
    
    return mod


def albedo(
    albedo: xr.DataArray,
    da_model: Union[xr.DataArray, xr.Dataset],
    freq: Optional[str] = None,
    reproj_method: str = "nearest_index",
    resample_kwargs: Optional[dict] = None,
) -> xr.DataArray:
    """
    Process albedo data for land surface temperature calculations.
    
    Parameters
    ----------
    albedo : xr.DataArray
        Albedo data array
    da_model : xr.DataArray or xr.Dataset
        Target grid for reprojection
    freq : str, optional
        Resampling frequency, by default None
    reproj_method : str, optional
        Reprojection method, by default "nearest_index"
    resample_kwargs : dict, optional
        Additional resampling arguments, by default None
        
    Returns
    -------
    xr.DataArray
        Processed albedo data
    """
    resample_kwargs = resample_kwargs or {}
    
    if albedo.raster.dim0 != "time":
        raise ValueError(f'First albedo dim should be "time", not {albedo.raster.dim0}')
    
    # reproject to model grid
    albedo_out = albedo.raster.reproject_like(da_model, method=reproj_method)
    
    # ensure values are between 0 and 1
    albedo_out = np.clip(albedo_out, 0, 1)
    
    # resample time if requested
    albedo_out.name = "albedo"
    albedo_out.attrs.update(unit="1")
    if freq is not None:
        resample_kwargs.update(upsampling="bfill", downsampling="mean")
        albedo_out = resample_time(albedo_out, freq, conserve_mass=False, **resample_kwargs)
    
    return albedo_out


def emissivity(
    emissivity: xr.DataArray,
    da_model: Union[xr.DataArray, xr.Dataset],
    freq: Optional[str] = None,
    reproj_method: str = "nearest_index",
    resample_kwargs: Optional[dict] = None,
) -> xr.DataArray:
    """
    Process emissivity data for land surface temperature calculations.
    
    Parameters
    ----------
    emissivity : xr.DataArray
        Emissivity data array
    da_model : xr.DataArray or xr.Dataset
        Target grid for reprojection
    freq : str, optional
        Resampling frequency, by default None
    reproj_method : str, optional
        Reprojection method, by default "nearest_index"
    resample_kwargs : dict, optional
        Additional resampling arguments, by default None
        
    Returns
    -------
    xr.DataArray
        Processed emissivity data
    """
    resample_kwargs = resample_kwargs or {}
    
    if emissivity.raster.dim0 != "time":
        raise ValueError(f'First emissivity dim should be "time", not {emissivity.raster.dim0}')
    
    # reproject to model grid
    emissivity_out = emissivity.raster.reproject_like(da_model, method=reproj_method)
    
    # ensure values are between 0 and 1
    emissivity_out = np.clip(emissivity_out, 0, 1)
    
    # resample time if requested
    emissivity_out.name = "emissivity"
    emissivity_out.attrs.update(unit="1")
    if freq is not None:
        resample_kwargs.update(upsampling="bfill", downsampling="mean")
        emissivity_out = resample_time(emissivity_out, freq, conserve_mass=False, **resample_kwargs)
    
    return emissivity_out


def radiation(
    radiation: xr.DataArray,
    da_model: Union[xr.DataArray, xr.Dataset],
    var_name: str = "radiation",
    freq: Optional[str] = None,
    reproj_method: str = "nearest_index",
    resample_kwargs: Optional[dict] = None,
) -> xr.DataArray:
    """
    Process radiation data for land surface temperature calculations.
    
    Parameters
    ----------
    radiation : xr.DataArray
        Radiation data array [W m-2]
    da_model : xr.DataArray or xr.Dataset
        Target grid for reprojection
    var_name : str, optional
        Variable name for output, by default "radiation"
    freq : str, optional
        Resampling frequency, by default None
    reproj_method : str, optional
        Reprojection method, by default "nearest_index"
    resample_kwargs : dict, optional
        Additional resampling arguments, by default None
        
    Returns
    -------
    xr.DataArray
        Processed radiation data
    """
    resample_kwargs = resample_kwargs or {}
    
    if radiation.raster.dim0 != "time":
        raise ValueError(f'First radiation dim should be "time", not {radiation.raster.dim0}')
    
    # reproject to model grid
    radiation_out = radiation.raster.reproject_like(da_model, method=reproj_method)
    
    # ensure non-negative values
    radiation_out = np.fmax(radiation_out, 0)
    
    # resample time if requested
    radiation_out.name = var_name
    radiation_out.attrs.update(unit="W m-2")
    if freq is not None:
        resample_kwargs.update(upsampling="bfill", downsampling="mean")
        radiation_out = resample_time(radiation_out, freq, conserve_mass=False, **resample_kwargs)
    
    return radiation_out

