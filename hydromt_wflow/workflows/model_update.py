"""Model update workflows for wflow."""

import logging
from typing import Optional, Union, Dict, Any

import numpy as np
import xarray as xr
from hydromt.model.processes.meteo import resample_time

logger = logging.getLogger(__name__)

__all__ = [
    "update_forcing_variable",
    "update_forcing_from_catalog",
    "update_model_config",
    "standardize_forcing_encoding",
    "rename_forcing_variables"
]


def update_forcing_variable(
    mod: "WflowModel",
    ds: Union[xr.Dataset, xr.DataArray],
    var: str,
    freq: Optional[str] = None,
    reproj_method: str = "nearest_index",
    resample_kwargs: Optional[dict] = None,
    enforce_nodata: bool = True,
    enforce_time_range: Optional[tuple] = None,
) -> "WflowModel":
    """
    Update model forcing with a new variable using standardized processing.
    
    Parameters
    ----------
    mod : WflowModel
        Wflow model instance
    ds : xr.Dataset or xr.DataArray
        Dataset or DataArray containing the variable
    var : str
        Variable name to add to forcing
    freq : str, optional
        Resampling frequency, by default None
    reproj_method : str, optional
        Method for spatial reprojection, by default "nearest_index"
    resample_kwargs : dict, optional
        Additional arguments for time resampling, by default None
    enforce_nodata : bool, optional
        Apply standard nodata masking, by default True
    enforce_time_range : tuple, optional
        Time range to enforce (start, end), by default None
        
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
    
    # apply mask and set nodata if requested
    if enforce_nodata:
        da = da.where(~ex_mask, ex_fillval)
        da.raster.set_nodata(ex_fillval)
        da.raster.attrs["_FillValue"] = ex_fillval
    
    # resample time if requested
    if freq is not None and "time" in da.dims:
        resample_kwargs.update(upsampling="bfill", downsampling="mean")
        da = resample_time(da, freq, conserve_mass=False, **resample_kwargs)
        if enforce_nodata:
            da.raster.set_nodata(ex_fillval)
    
    # enforce time range if requested
    if enforce_time_range is not None and "time" in da.dims:
        start_time, end_time = enforce_time_range
        da = da.sel(time=slice(start_time, end_time))
    
    # set variable name and add to forcing
    da.name = var
    mod.forcing[var] = da
    
    logger.info(f"Added variable '{var}' to model forcing")
    return mod


def update_forcing_from_catalog(
    mod: "WflowModel",
    catalog: "DataCatalog",
    variables: Dict[str, str],
    freq: Optional[str] = None,
    reproj_method: str = "nearest_index",
    resample_kwargs: Optional[dict] = None,
    enforce_nodata: bool = True,
    enforce_time_range: Optional[tuple] = None,
) -> "WflowModel":
    """
    Update model forcing with multiple variables from a data catalog.
    
    Parameters
    ----------
    mod : WflowModel
        Wflow model instance
    catalog : DataCatalog
        Data catalog containing the variables
    variables : dict
        Dictionary mapping variable names to catalog keys
    freq : str, optional
        Resampling frequency, by default None
    reproj_method : str, optional
        Method for spatial reprojection, by default "nearest_index"
    resample_kwargs : dict, optional
        Additional arguments for time resampling, by default None
    enforce_nodata : bool, optional
        Apply standard nodata masking, by default True
    enforce_time_range : tuple, optional
        Time range to enforce (start, end), by default None
        
    Returns
    -------
    WflowModel
        Updated model with new forcing variables
    """
    for var_name, catalog_key in variables.items():
        try:
            ds = catalog.get_rasterdataset(catalog_key)
            mod = update_forcing_variable(
                mod=mod,
                ds=ds,
                var=var_name,
                freq=freq,
                reproj_method=reproj_method,
                resample_kwargs=resample_kwargs,
                enforce_nodata=enforce_nodata,
                enforce_time_range=enforce_time_range,
            )
        except Exception as e:
            logger.warning(f"Failed to add variable '{var_name}' from catalog: {e}")
    
    return mod


def update_model_config(
    mod: "WflowModel",
    starttime: Optional[str] = None,
    endtime: Optional[str] = None,
    forcing_path: Optional[str] = None,
    **kwargs: Any,
) -> "WflowModel":
    """
    Update model configuration parameters.
    
    Parameters
    ----------
    mod : WflowModel
        Wflow model instance
    starttime : str, optional
        Model start time, by default None
    endtime : str, optional
        Model end time, by default None
    forcing_path : str, optional
        Path to forcing file, by default None
    **kwargs : Any
        Additional configuration parameters
        
    Returns
    -------
    WflowModel
        Updated model with new configuration
    """
    if starttime is not None:
        mod.config["starttime"] = starttime
    if endtime is not None:
        mod.config["endtime"] = endtime
    if forcing_path is not None:
        mod.config["input"]["path_forcing"] = forcing_path
    
    # update any additional config parameters
    for key, value in kwargs.items():
        if key in mod.config:
            mod.config[key] = value
        else:
            logger.warning(f"Config key '{key}' not found in model config")
    
    return mod


def standardize_forcing_encoding(
    mod: "WflowModel",
    reference_var: str = "precip",
    enforce_nodata: bool = True,
    enforce_time_range: Optional[tuple] = None,
) -> "WflowModel":
    """
    Standardize encoding and nodata values across all forcing variables.
    
    Parameters
    ----------
    mod : WflowModel
        Wflow model instance
    reference_var : str, optional
        Reference variable for encoding, by default "precip"
    enforce_nodata : bool, optional
        Apply standard nodata masking, by default True
    enforce_time_range : tuple, optional
        Time range to enforce (start, end), by default None
        
    Returns
    -------
    WflowModel
        Updated model with standardized forcing
    """
    if reference_var not in mod.forcing:
        logger.warning(f"Reference variable '{reference_var}' not found in forcing")
        return mod
    
    # get reference encoding and fill value
    encoding = mod.forcing[reference_var].encoding
    fillval = encoding.get("_FillValue")
    
    if fillval is None:
        logger.warning(f"No _FillValue found in reference variable '{reference_var}'")
        return mod
    
    # get reference grid mask
    ex_grid = mod.grid["wflow_dem"]
    ex_fillval = ex_grid.attrs["_FillValue"]
    ex_mask = ex_grid.values == ex_fillval
    
    for key, item in mod.forcing.items():
        # enforce nodata if requested
        if enforce_nodata:
            if "_FillValue" not in item.attrs:
                logger.info(f"Adding _FillValue to '{key}'")
                remasked = item.where(~ex_mask, ex_fillval)
                remasked.raster.set_nodata(ex_fillval)
                remasked.raster.attrs["_FillValue"] = ex_fillval
                mod.forcing[key] = remasked
        
        # enforce time range if requested
        if enforce_time_range is not None and "time" in item.dims:
            start_time, end_time = enforce_time_range
            mod.forcing[key] = item.sel(time=slice(start_time, end_time))
        
        # copy encoding from reference variable
        for enc_key in list(encoding.keys())[:10]:  # limit to first 10 encoding keys
            if enc_key in encoding:
                da = mod.forcing[key]
                da.encoding[enc_key] = encoding[enc_key]
                mod.forcing[key] = da
    
    logger.info("Standardized forcing encoding across all variables")
    return mod


def rename_forcing_variables(
    mod: "WflowModel",
    rename_dict: Dict[str, str],
) -> "WflowModel":
    """
    Rename forcing variables in the model.
    
    Parameters
    ----------
    mod : WflowModel
        Wflow model instance
    rename_dict : dict
        Dictionary mapping old names to new names
        
    Returns
    -------
    WflowModel
        Updated model with renamed forcing variables
    """
    for old_name, new_name in rename_dict.items():
        if old_name in mod.forcing:
            mod.forcing[new_name] = mod.forcing[old_name]
            mod.forcing.pop(old_name)
            logger.info(f"Renamed forcing variable '{old_name}' to '{new_name}'")
        else:
            logger.warning(f"Forcing variable '{old_name}' not found for renaming")
    
    return mod 