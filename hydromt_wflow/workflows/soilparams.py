"""Functions for individual soil parameters."""

import numpy as np
import xarray as xr


def ksathorfrac(
    da: xr.DataArray,
    ds_like: xr.Dataset,
    scale_method: str,
):
    """_summary_.

    _extended_summary_

    Parameters
    ----------
    da : xr.DataArray
        _description_
    ds_like : xr.Dataset
        _description_
    scale_method : str
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # Transfer data to a logaritmic scale
    da.values = np.log10(da.values)
    # Reproject the data
    da = da.raster.reproject_like(
        ds_like,
        method=scale_method,
    )
    # Scale the data back to normal values
    da.values = 10 ** (da.values)

    # Fill all nodata holes in the map
    da = da.interpolate_na(dim=da.raster.x_dim, method="linear")
    da = da.interpolate_na(
        dim=da.raster.x_dim, method="nearest", fill_value="extrapolate"
    )

    # Set outgoing name
    da.name = "KsatHorFrac"
    # Set the default no fill value for doubles
    da = da.fillna(-9999.0)
    da.raster.set_nodata(-9999.0)
    # Return as a dataset to be used for 'set_grid'
    return da.to_dataset()
