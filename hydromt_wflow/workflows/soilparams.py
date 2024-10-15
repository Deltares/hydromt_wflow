"""Functions for individual soil parameters."""

import numpy as np
import xarray as xr
from hydromt import raster


def ksathorfrac(
    da: xr.DataArray,
    ds_like: xr.Dataset,
    resampling_method: str,
) -> xr.DataArray:
    """Create KsatHorfrac map.

    Based on the data properties of the WflowModel.

    Parameters
    ----------
    da : xr.DataArray
        KsathorFrac values, i.e. an xarray.DataArray containing the values from \
the predefined KsatHorFrac map.
    ds_like : xr.Dataset
        Dataset at model resolution.
    resampling_method : str
        Scale method when up- or downscaling.

    Returns
    -------
    xr.DataArray
        A xarray DataArray containing the scaled KsatHorFrac values.
    """
    # Transfer data to a logaritmic scale
    da = np.log10(da)
    # Reproject the data
    da = da.raster.reproject_like(
        ds_like,
        method=resampling_method,
    )
    # Scale the data back to normal values
    da = np.power(10, da)

    # Fill all nodata holes in the map
    da = da.interpolate_na(dim=da.raster.x_dim, method="linear")
    da = da.interpolate_na(
        dim=da.raster.x_dim, method="nearest", fill_value="extrapolate"
    )

    # Set the default no fill value for doubles
    da = da.fillna(-9999.0)
    da.raster.set_nodata(-9999.0)
    # Return as a dataset to be used for 'set_grid'
    return da


def calc_kv_at_depth(depth, kv_0, f):
    """
    Calculate the kv value at a certain depth.

    Value is based on the kv at the surface, the f parameter that describes the
    exponential decline and the depth.

    Parameters
    ----------
    depth
        Depth at which kv needs to be calculated
    kv_0
        The vertical conductivity at the surface
    f
        The value describing the exponential decline

    Returns
    -------
    kv_z
        The kv value at the requested depth
    """
    kv_z = kv_0 * np.exp(-f * depth)
    return kv_z


def calc_kvfrac(kv_depth, target):
    """Calculate the kvfrac.

    Based on the kv value at a certain depth and the target value.

    Parameters
    ----------
    kv_depth:
        Value of kv at a certain depth
    target:
        Target kv value

    Returns
    -------
    kvfrac:
        The value which kv_depths needs to be multiplied with to reach the target value
    """
    kvfrac = target / kv_depth
    return kvfrac


def update_kvfrac(
    ds_model, kv0_mask, f_mask, wflow_thicknesslayers, target_conductivity
):
    """
    Calculate kvfrac values for each layer.

    Done such that the bottom of the layer equals to the target_conductivity.
    Calculation assumes exponentially declining vertical conductivities, based on the f
    parameter. If no target_conductivity is specified, kvfrac is set to be equal to 1.

    Parameters
    ----------
    ds_model
        Dataset of the wflow model
    kv0_mask
        Values of vertical conductivity at the surface, masked to paddy locations
    f_mask
        Values of the f parameter, masked to paddy locations
    wflow_thicknesslayers
        List of requested layers in the wflow model
    target_conductivity
        List of target conductivities for each layer (None if no target value is
        requested)

    Returns
    -------
    da_kvfrac
        Maps for each layer with the required kvfrac value
    """
    # Convert to np.array
    wflow_thicknesslayers = np.array(wflow_thicknesslayers)
    target_conductivity = np.array(target_conductivity)

    # Prepare emtpy dataarray
    da_kvfrac = raster.full_like(ds_model["c"])
    # Set all values to 1
    da_kvfrac = da_kvfrac.where(
        ds_model["wflow_dem"].raster.mask_nodata().isnull(), 1.0
    )

    # Get the actual depths
    wflow_depths = np.cumsum(wflow_thicknesslayers)
    # Find the index of the layers where a kvfrac should be set
    idx = np.where(target_conductivity is not None)[0]

    # Loop through the target_conductivity values
    for idx, target in enumerate(target_conductivity):
        if target is not None:
            depth = wflow_depths[idx]
            # Calculate the kv at that depth (only for the pixels that have paddy
            # fields)
            kv_depth = calc_kv_at_depth(depth=depth, kv_0=kv0_mask, f=f_mask)
            paddy_values = calc_kvfrac(kv_depth=kv_depth, target=target)
            # Set the values in the correct places
            kvfrac = xr.where(
                paddy_values.raster.mask_nodata().isnull(),
                da_kvfrac.loc[dict(layer=idx)],
                paddy_values,
            )
            kvfrac = kvfrac.fillna(-9999)
            kvfrac.raster.set_nodata(-9999)
            # Update layer in dataarray
            da_kvfrac.loc[dict(layer=idx)] = kvfrac

    return da_kvfrac
