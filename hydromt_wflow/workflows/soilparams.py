"""Functions for individual soil parameters."""

import numpy as np
import xarray as xr


def ksathorfrac(
    da: xr.DataArray,
    ds_like: xr.Dataset,
    scale_method: str,
) -> xr.Dataset:
    """Create KsatHorfrac map.

    Based on the data properties of the WflowModel.

    Parameters
    ----------
    da : xr.DataArray
        KsathorFrac values, i.e. an xarray.DataArray containing the values from \
the predefined KsatHorFrac map.
    ds_like : xr.Dataset
        Dataset at model resolution.
    scale_method : str
        Scale method when up- or downscaling.

    Returns
    -------
    xr.Dataset
        A xarray Dataset containing the scaled KsatHorFrac values.
    """
    # Transfer data to a logaritmic scale
    da.values = np.log10(da)
    # Reproject the data
    da = da.raster.reproject_like(
        ds_like,
        method=scale_method,
    )
    # Scale the data back to normal values
    da.values = np.power(10, da)

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
