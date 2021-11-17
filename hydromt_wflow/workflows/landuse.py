# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xarray as xr
import logging


logger = logging.getLogger(__name__)


__all__ = ["landuse", "lai"]


RESAMPLING = {"landuse": "nearest", "lai": "average"}
DTYPES = {"landuse": np.int16}


def landuse(da, ds_like, fn_map, logger=logger, params=None):
    """Returns landuse map and related parameter maps.
    The parameter maps are prepared based on landuse map and
    mapping table as provided in the generic data folder of hydromt.

    The following topography maps are calculated:\
    - TODO
    
    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing LULC classes.
    ds_like : xarray.DataArray
        Dataset at model resolution.

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing gridded landuse based maps
    """
    # read csv with remapping values
    df = pd.read_csv(fn_map, index_col=0, sep=",|;", engine="python", dtype=DTYPES)
    # limit dtypes to avoid gdal errors downstream
    ddict = {"float64": np.float32, "int64": np.int32}
    dtypes = {c: ddict.get(str(df[c].dtype), df[c].dtype) for c in df.columns}
    df = pd.read_csv(fn_map, index_col=0, sep=",|;", engine="python", dtype=dtypes)
    keys = df.index.values
    if params is None:
        params = [p for p in df.columns if p != "description"]
    elif not np.all(np.isin(params, df.columns)):
        missing = [p for p in params if p not in df.columns]
        raise ValueError(f"Parameter(s) missing in mapping file: {missing}")
    # setup ds out
    ds_out = xr.Dataset(coords=ds_like.raster.coords)
    # setup reclass method
    def reclass(x):
        return np.vectorize(d.get)(x, nodata)

    da = da.raster.interpolate_na(method="nearest")
    # apply for each parameter
    for param in params:
        method = RESAMPLING.get(param, "average")
        values = df[param].values
        nodata = values[-1]  # NOTE values is set in last row
        d = dict(zip(keys, values))  # NOTE global param in reclass method
        logger.info(f"Deriving {param} using {method} resampling (nodata={nodata}).")
        da_param = xr.apply_ufunc(
            reclass, da, dask="parallelized", output_dtypes=[values.dtype]
        )
        da_param.attrs.update(_FillValue=nodata)  # first set new nodata values
        ds_out[param] = da_param.raster.reproject_like(
            ds_like, method=method
        )  # then resample

    return ds_out


def lai(da, ds_like, logger=logger):
    """Returns climatology of Leaf Area Index (LAI).

    The following topography maps are calculated:\
    - LAI\
    
    Parameters
    ----------
    da : xarray.DataArray or xarray.Dataset
        DataArray or Dataset with LAI array containing LAI values.
    ds_like : xarray.DataArray
        Dataset at model resolution.

    Returns
    -------
    da_out : xarray.DataArray
        Dataset containing resampled LAI maps
    """
    if isinstance(da, xr.Dataset) and "LAI" in da:
        da = da["LAI"]
    elif not isinstance(da, xr.DataArray):
        raise ValueError("lai method requires a DataArray or Dataset with LAI array")
    method = RESAMPLING.get(da.name, "average")
    nodata = da.raster.nodata
    logger.info(f"Deriving {da.name} using {method} resampling (nodata={nodata}).")
    da = da.astype(np.float32)
    da = da.where(da.values != nodata).fillna(
        0.0
    )  # Assuming missing values correspond to bare soil, urban and snow (LAI=0.0)
    da_out = da.raster.reproject_like(ds_like, method=method)
    da_out.attrs.update(_FillValue=nodata)
    return da_out
