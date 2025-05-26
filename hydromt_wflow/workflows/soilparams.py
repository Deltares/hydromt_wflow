"""Functions for individual soil parameters."""

import numpy as np
import xarray as xr
from hydromt import raster

__all__ = [
    "ksat_horizontal_ratio",
    "ksatver_vegetation",
]


def ksat_horizontal_ratio(
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
    # Transfer data to a logarithmic scale
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
    """Calculate the soil_ksat_vertical_factor.

    Based on the kv value at a certain depth and the target value.

    Parameters
    ----------
    kv_depth:
        Value of kv at a certain depth
    target:
        Target kv value

    Returns
    -------
    soil_ksat_vertical_factor:
        The value which kv_depths needs to be multiplied with to reach the target value
    """
    soil_ksat_vertical_factor = target / kv_depth
    return soil_ksat_vertical_factor


def update_kvfrac(
    ds_model, kv0_mask, f_mask, wflow_thicknesslayers, target_conductivity
):
    """
    Calculate soil_ksat_vertical_factor values for each layer.

    Done such that the bottom of the layer equals to the target_conductivity.
    Calculation assumes exponentially declining vertical conductivities, based on the f
    parameter. If no target_conductivity is specified, soil_ksat_vertical_factor is set
    to be equal to 1.

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
        Maps for each layer with the required soil_ksat_vertical_factor value
    """
    # Convert to np.array
    wflow_thicknesslayers = np.array(wflow_thicknesslayers)
    target_conductivity = np.array(target_conductivity)

    # Prepare empty dataarray
    da_kvfrac = raster.full_like(ds_model["soil_brooks_corey_c"])
    # Set all values to 1
    da_kvfrac = da_kvfrac.where(ds_model["elevtn"].raster.mask_nodata().isnull(), 1.0)

    # Get the actual depths
    wflow_depths = np.cumsum(wflow_thicknesslayers)
    # Find the index of the layers where a soil_ksat_vertical_factor should be set
    idx = target_conductivity.nonzero()[0]

    # Loop through the target_conductivity values
    for idx, target in enumerate(target_conductivity):
        if target is not None:
            depth = wflow_depths[idx]
            # Calculate the kv at that depth (only for the pixels that have paddy
            # fields)
            kv_depth = calc_kv_at_depth(depth=depth, kv_0=kv0_mask, f=f_mask)
            paddy_values = calc_kvfrac(kv_depth=kv_depth, target=target)
            # Set the values in the correct places
            soil_ksat_vertical_factor = xr.where(
                paddy_values.raster.mask_nodata().isnull(),
                da_kvfrac.loc[dict(layer=idx)],
                paddy_values,
            )
            soil_ksat_vertical_factor = soil_ksat_vertical_factor.fillna(-9999)
            soil_ksat_vertical_factor.raster.set_nodata(-9999)
            # Update layer in dataarray
            da_kvfrac.loc[dict(layer=idx)] = soil_ksat_vertical_factor

    return da_kvfrac


def get_ks_veg(
    soil_ksat_vertical, sndppt, vegetation_leaf_area_index, alfa=4.5, beta=5
):
    """
    Based on Bonetti et al. (2021) https://www.nature.com/articles/s43247-021-00180-0.

    Parameters
    ----------
    soil_ksat_vertical  : [xr.DataSet, float]
        saturated hydraulic conductivity from PTF based on soil properties [cm/d].
    sndppt : [xr.DataSet, float]
        percentage sand [%].
    vegetation_leaf_area_index : [xr.DataSet, float]
        Mean annual leaf area index [-].
    alfa : float, optional
        Shape parameter. The default is 4.5 when using vegetation_leaf_area_index.
    beta : float, optional
        Shape parameter. The default is 5 when using vegetation_leaf_area_index.

    Returns
    -------
    ks : [xr.DataSet, float]
        saturated hydraulic conductivity based on soil and vegetation [cm/d].

    """
    # get the saturated hydraulic conductivity with fully developed vegetation.
    ksmax = 10 ** (3.5 - 1.5 * sndppt**0.13 + np.log10(soil_ksat_vertical))
    # get the saturated hydraulic conductivity based on soil and
    # vegetation mean vegetation_leaf_area_index
    ks = ksmax - (ksmax - soil_ksat_vertical) / (
        1 + (vegetation_leaf_area_index / alfa) ** beta
    )
    return ks


def ksatver_vegetation(
    ds_like: xr.Dataset,
    sndppt: xr.Dataset,
    alfa: float = 4.5,
    beta: float = 5,
) -> xr.DataArray:
    """
    Calculate saturated hydraulic conductivity based on soil and vegetation [mm/d].

    based on Bonetti et al. (2021) https://www.nature.com/articles/s43247-021-00180-0.

    Parameters
    ----------
    ds_like : xr.Dataset
        Dataset at model resolution.
        The required variables in ds_like are vegetation_leaf_area_index [-],
        KSatVer [mm/d] and subcatchment
    sndppt : [xr.DataSet, float]
        percentage sand [%].
    alfa : float, optional
        Shape parameter. The default is 4.5 when using vegetation_leaf_area_index.
    beta : float, optional
        Shape parameter. The default is 5 when using vegetation_leaf_area_index.

    Returns
    -------
    ks : [xr.DataSet, float]
        saturated hydraulic conductivity based on soil and vegetation [mm/d].

    """
    sndppt = sndppt.where(sndppt != sndppt._FillValue, np.nan)
    # reproject to model resolution
    sndppt = sndppt.raster.reproject_like(ds_like, method="average")
    sndppt.raster.set_nodata(np.nan)
    # interpolate to fill missing values
    sndppt = sndppt.raster.interpolate_na("rio_idw")
    # mask outside basin
    sndppt = sndppt.where(ds_like["basins"] > 0)

    # mean annual lai is required (see fig 1 in Bonetti et al. 2021)
    LAI_mean = ds_like["vegetation_leaf_area_index"].mean("time")
    LAI_mean.raster.set_nodata(255.0)

    # in this function, Ksatver should be provided in cm/d
    KSatVer_vegetation = get_ks_veg(
        ds_like["ksat_vertical"] / 10, sndppt, LAI_mean, alfa, beta
    )

    # convert back from cm/d to mm/d
    KSatVer_vegetation = KSatVer_vegetation * 10
    return KSatVer_vegetation
