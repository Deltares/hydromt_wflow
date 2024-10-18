"""Workflow for water demand."""

import logging
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import xarray as xr
from hydromt import raster
from hydromt.workflows.grid import grid_from_constant

logger = logging.getLogger(__name__)

__all__ = [
    "allocation_areas",
    "domestic",
    "other_demand",
    "surfacewaterfrac_used",
    "irrigation",
]

map_vars = {
    "dom": "domestic",
    "ind": "industry",
    "lsk": "livestock",
}


def create_grid_from_bbox(
    bbox: List[float],
    res: float,
    crs: int,
    align: True,
    y_name: str = "y",
    x_name: str = "x",
):
    """Create grid from bounding box and target resolution."""
    xmin, ymin, xmax, ymax = bbox
    res = abs(res)
    if align:
        xmin = round(xmin / res) * res
        ymin = round(ymin / res) * res
        xmax = round(xmax / res) * res
        ymax = round(ymax / res) * res
    xcoords = np.linspace(
        xmin + res / 2,
        xmax - res / 2,
        num=round((xmax - xmin) / res),
        endpoint=True,
    )
    ycoords = np.flip(
        np.linspace(
            ymin + res / 2,
            ymax - res / 2,
            num=round((ymax - ymin) / res),
            endpoint=True,
        )
    )
    coords = {y_name: ycoords, x_name: xcoords}
    grid = raster.full(
        coords=coords,
        nodata=np.nan,
        dtype=np.float32,
        name="data",
        attrs={},
        crs=crs,
        lazy=True,
    )

    return grid


def domestic(
    ds: xr.Dataset,
    ds_like: xr.Dataset,
    popu: Optional[xr.Dataset] = None,
    original_res: Optional[float] = None,
) -> tuple:
    """Create domestic water demand maps.

    Parameters
    ----------
    ds : xr.Dataset
        Raw domestic data dataset.
    ds_like : xr.Dataset
        Dataset at wflow model domain and resolution.
    popu : xr.Dataset
        Population dataset (number of people per gridcell).
    original_res : float
        Original resolution of ds. If provided, ds will be upscaled before
        downsampling with population.

    Returns
    -------
    tuple
        Domestic data at model resolution, Population data at model resolution.
    """
    # Mask no data values
    ds = ds.raster.mask_nodata()
    # Convert population to density and reproject to model
    if popu is not None:
        popu = popu.raster.mask_nodata().fillna(0)
        popu_density = popu / popu.raster.area_grid()
        popu_density.name = "population_density"
        popu_scaled = popu_density.raster.reproject_like(
            ds_like,
            method="average",
        )
        # Back to cap per cell
        popu_scaled = popu_scaled * popu_scaled.raster.area_grid()
        popu_scaled.name = "population"
    else:
        popu_scaled = None

    # Reproject to model resolution
    # Simple reprojection
    ds_scaled = ds.raster.reproject_like(
        ds_like,
        method="average",
    )
    # Downscale with population if available
    if popu is not None:
        # Reproject ds to original resolution if needed before downscaling
        if original_res is not None:
            # Create empty grid at the original resolution
            dda = create_grid_from_bbox(
                bbox=ds.raster.bounds,
                res=original_res,
                crs=ds_like.raster.crs,
                align=True,
                y_name=ds.raster.y_dim,
                x_name=ds.raster.x_dim,
            )
            # Use (weighted) average to scale demands in mm!
            ds = ds.raster.reproject_like(
                dda,
                method="average",
            )
        # Check if population is of higher res than ds (else no need to downscale)
        if abs(popu.raster.res[0]) < abs(ds.raster.res[0]):
            # Convert ds from mm to m3
            ds_m3 = ds * ds.raster.area_grid() * 1000
            # Get the number of capita per cell
            popu_ds = popu.raster.reproject_like(
                ds_m3,
                method="sum",
            )
            # Get m3 per capita
            ds_m3_per_cap = ds_m3 / popu_ds
            ds_m3_per_cap = ds_m3_per_cap.where(popu_ds > 0, 0)
            # Downscale to model resolution
            ds_m3_per_cap_model = ds_m3_per_cap.raster.reproject_like(
                ds_like,
                method="average",
            )
            # Get m3 per cell
            ds_m3_model = ds_m3_per_cap_model * popu_scaled
            # Get back to mm
            ds_scaled = ds_m3_model / ds_m3_model.raster.area_grid() / 1000

    return ds_scaled, popu_scaled


def other_demand(
    ds: xr.Dataset,
    ds_like: xr.Dataset,
    ds_method: str = "average",
) -> xr.Dataset:
    """Create non-irrigation water demand maps.

    Parameters
    ----------
    ds : xr.Dataset
        Raw demand data dataset.
    ds_like : xr.Dataset
        Dataset at wflow model domain and resolution.
    ds_method : str
        Demand data resampling method. Default is 'average'.

    Returns
    -------
    xr.Dataset
        Demand data at model resolution.
    """
    # Reproject to up or downscale
    non_irri_scaled = ds.raster.reproject_like(
        ds_like,
        method=ds_method,
    )

    # Mask data to fill nodata gaps with 0
    non_irri_scaled = non_irri_scaled.raster.mask_nodata()
    for var in non_irri_scaled.data_vars:
        non_irri_scaled[var] = non_irri_scaled[var].where(
            ~np.isnan(non_irri_scaled[var]), 0
        )

    return non_irri_scaled


def allocation_areas(
    ds_like: xr.Dataset,
    waterareas: gpd.GeoDataFrame,
    basins: gpd.GeoDataFrame,
    priority_basins: bool = True,
) -> Tuple[xr.DataArray, gpd.GeoDataFrame]:
    """Create water allocation area.

    Based on current wflow model domain and resolution and making use of
    the model basins and administrative boundaries.

    Parameters
    ----------
    ds_like : xr.DataArray
        A grid covering the wflow model domain and the rivers.

        * Required variables: ['wflow_river', 'wflow_subcatch']
    waterareas : gpd.GeoDataFrame
        Administrative boundaries, e.g. sovereign nations.
    basins : gpd.GeoDataFrame
        The wflow model basins.
    priority_basins : bool
        For basins that do not contain river cells after intersection with the admin
        boundaries, the priority_basins flag can be used to decide if these basins
        should be merged with the closest downstream basin (True, default) or with any
        large enough basin in the same administrative area (False).

    Returns
    -------
    xr.DataArray
        The water demand allocation areas.
    gpd.GeoDataFrame
        The water demand allocation areas as geodataframe.
    """
    # Intersect the basins with the admin boundaries
    subbasins = basins.overlay(waterareas, how="intersection")
    # Create a unique index
    subbasins.index = np.arange(1, len(subbasins) + 1)
    subbasins.index.name = "uid"

    # After intersection, some cells at the coast are not included in the subbasins
    # convert to raster to fill na with nearest value
    subbasins_raster = ds_like.raster.rasterize(subbasins, col_name="uid", nodata=0)
    subbasins_raster = subbasins_raster.raster.interpolate_na(
        method="nearest", extrapolate=True
    )
    # Mask the wflow subcatch after extrapolation
    subbasins_raster = subbasins_raster.where(
        ds_like["wflow_subcatch"] != ds_like["wflow_subcatch"].raster.nodata,
        subbasins_raster.raster.nodata,
    )

    # Convert back to geodataframe and prepare the new unique index for the subbasins
    subbasins = subbasins_raster.raster.vectorize()
    if priority_basins:
        # Redefine the uid, equivalent to exploding the geometries
        subbasins.index = np.arange(1, len(subbasins) + 1)
    else:
        # Keep the uid of the intersection with the admin bounds
        subbasins.index = subbasins["value"].astype(int)
    subbasins.index.name = "uid"

    # Rasterize the subbasins
    da_subbasins = ds_like.raster.rasterize(subbasins, col_name="uid", nodata=0)
    # Mask with river cells
    da_subbasins_to_keep = np.unique(
        da_subbasins.raster.mask_nodata()
        .where(ds_like["wflow_river"] > 0, np.nan)
        .values
    )
    # Remove the nodata value from the list
    da_subbasins_to_keep = np.int32(
        da_subbasins_to_keep[~np.isnan(da_subbasins_to_keep)]
    )

    # Create the water allocation map starting with the subbasins that contain a river
    allocation_areas = da_subbasins.where(
        da_subbasins.isin(da_subbasins_to_keep), da_subbasins.raster.nodata
    )
    # Use nearest to fill the nodata values for subbasins without rivers
    allocation_areas = allocation_areas.raster.interpolate_na(
        method="nearest", extrapolate=True
    )
    allocation_areas = allocation_areas.where(
        ds_like["wflow_subcatch"] != ds_like["wflow_subcatch"].raster.nodata,
        allocation_areas.raster.nodata,
    )
    allocation_areas.name = "allocation_areas"

    # Create the equivalent geodataframe
    allocation_areas_gdf = allocation_areas.raster.vectorize()
    allocation_areas_gdf.index = allocation_areas_gdf["value"].astype(int)
    allocation_areas_gdf.index.name = "uid"

    return allocation_areas, allocation_areas_gdf


def surfacewaterfrac_used(
    gwfrac_raw: xr.DataArray,
    da_like: xr.DataArray,
    waterareas: xr.DataArray,
    gwbodies: Optional[xr.DataArray] = None,
    ncfrac: Optional[xr.DataArray] = None,
    interpolate: bool = False,
    mask_and_scale_gwfrac: bool = True,
) -> xr.DataArray:
    """Create surface water fraction map.

    Parameters
    ----------
    gwfrac_raw : xr.DataArray
        Raw groundwater fraction data map.
    da_like : xr.DataArray
        Wflow model like grid.
    waterareas : xr.DataArray
        Water source areas.
    gwbodies : xr.DataArray, optional
        Groundwater bodies map. Either 0 or 1. If None, assumes 1.
    ncfrac : xr.DataArray, optional
        Non-conventional water fraction. If None, assumes 0.
    interpolate : bool
        Interpolate missing data values within wflow model domain.
    mask_and_scale_gwfrac : bool, optional
        If True, gwfrac will be masked for areas with no groundwater bodies. To keep
        the average gwfrac used over waterareas similar after the masking, gwfrac
        for areas with groundwater bodies can increase. If False, gwfrac will be
        used as is. By default True.

    Returns
    -------
    xr.DataArray
        Surface water fraction.
    """
    # Resample the data to model resolution
    gwfrac_raw_mr = gwfrac_raw.raster.reproject_like(
        da_like,
        method="average",
    )
    # Only reproject waterareas if needed
    if not waterareas.raster.identical_grid(da_like):
        waterareas_mr = waterareas.raster.reproject_like(
            da_like,
            method="mode",
        )
    else:
        waterareas_mr = waterareas
    # Set nodata values to zeros
    waterareas_mr = waterareas_mr.where(
        waterareas_mr != waterareas_mr.raster.nodata,
        0,
    )

    if gwbodies is not None:
        gwbodies_mr = gwbodies.raster.reproject_like(
            da_like,
            method="mode",
        )
    else:
        gwbodies_mr = grid_from_constant(
            grid_like=da_like,
            constant=1,
            name="gwbodies",
            nodata=-9999,
            dtype=np.int32,
        )
    if ncfrac is not None:
        ncfrac_mr = ncfrac.raster.reproject_like(
            da_like,
            method="average",
        )
    else:
        ncfrac_mr = grid_from_constant(
            grid_like=da_like,
            constant=0,
            name="ncfrac",
            nodata=-9999,
            dtype=np.float32,
        )

    # Get the fractions based on area count
    x, y = np.where(~np.isnan(gwfrac_raw_mr))
    if mask_and_scale_gwfrac:
        gw_pixels = np.take(
            np.bincount(
                waterareas_mr.values[x, y],
                weights=gwbodies_mr.values[x, y],
            ),
            waterareas_mr.values[x, y],
        )
        a_pixels = np.take(
            np.bincount(
                waterareas_mr.values[x, y],
                weights=gwbodies_mr.values[x, y] * 0.0 + 1.0,
            ),
            waterareas_mr.values[x, y],
        )
        scale_factor = a_pixels / (gw_pixels + 0.01)
    else:
        scale_factor = 1.0

    # Determine the groundwater fraction
    gwfrac_val = np.minimum(
        gwfrac_raw_mr.values[x, y] * scale_factor,
        1 - ncfrac_mr.values[x, y],
    )
    if mask_and_scale_gwfrac:
        gwfrac_val[np.where(gwbodies_mr.values[x, y] == 0)] = 0
    # invert to get surface water frac
    swfrac_val = np.maximum(np.minimum(1 - gwfrac_val - ncfrac_mr.values[x, y], 1), 0)

    # create the dataarray for the fraction
    swfrac = xr.full_like(
        gwfrac_raw_mr,
        fill_value=np.nan,
        dtype=np.float32,
    ).load()
    swfrac.name = "frac_sw_used"
    swfrac.attrs = {"_FillValue": -9999}
    swfrac.raster.set_crs(da_like.raster.crs)

    # Set and interpolate the values
    swfrac = swfrac.copy()  # to avoid read-only error
    swfrac.values[x, y] = swfrac_val
    if interpolate:
        swfrac = swfrac.raster.interpolate_na(method="linear", extrapolate=True)
    else:
        # Fill na with 1 (no groundwater or nc sources used)
        swfrac = swfrac.fillna(1.0)

    # Set the nodata values based on the dem of the model (da_like)
    swfrac = swfrac.where(da_like != da_like.raster.nodata, -9999)

    # Return surface water frac
    return swfrac


def classify_pixels(
    da_irr: xr.DataArray,
    da_crop_model: xr.DataArray,
    threshold: float,
    nodata_value: float | int = -9999,
):
    """Classifies pixels based on a (fractional) threshold.

    Pixels with a value above this threshold are set to be irrigated, and pixels below
    this value as rainfed.

    Parameters
    ----------
    da_irr: xr.DataArray
        Data with irrigation mask
    da_crop_model: xr.DataArray
        Layer of the masked cropland in wflow model
    threshold: float
        Threshold above which pixels are classified as irrigated
    nodata_value: float | int = -9999
        Value to be used for nodata

    Returns
    -------
    irr_map: xr.DataArray
        Mask with classified pixels
    """
    # Check if the resolution of the irrigation map is higher than the model
    if abs(da_irr.raster.res[0]) < abs(da_crop_model.raster.res[0]):
        # Use area to resample
        # Convert irr map to an area map
        da_area = da_irr * da_irr.raster.area_grid()
        # Resample to model grid and sum areas
        da_area2model = da_area.raster.reproject_like(da_crop_model, method="sum")

        # Calculate relative area
        relative_area = da_area2model / da_area2model.raster.area_grid()

        # Classify pixels with a 1 where it exceeds the threshold, and 0 where it doesnt
        da_irr_model = xr.where(relative_area >= threshold, 1, 0)

    # Else resample using nearest
    else:
        da_irr_model = da_irr.raster.reproject_like(da_crop_model, method="nearest")

    # Find cropland pixels that are irrigated
    irr_map = xr.where(
        np.logical_and(da_irr_model == 1, da_crop_model != da_crop_model.raster.nodata),
        1,
        0,
    )
    # Fix nodata values (will be used as catchment mask)
    irr_map.raster.set_nodata(nodata_value)

    return irr_map


def calc_lai_threshold(da_lai, threshold, dtype=np.int32, na_value=-9999):
    """
    Calculate irrigation trigger based on LAI threshold.

    Trigger is set to 1 when the LAI is bigger than 20% of the variation (set by the
    threshold value).

    Parameters
    ----------
    da_lai
        Dataarray with LAI values
    threshold
        Value to be used as threshold
    dtype
        Datatype for the resulting map
    na_value
        Value for nodata

    Returns
    -------
    trigger:
        Maps with a value of 1 where the LAI indicates growing season, and 0 for all
        other pixels
    """
    # Compute min and max of LAI
    lai_min = da_lai.min(dim="time")
    lai_max = da_lai.max(dim="time")
    # Determine critical threshold
    ct = lai_min + threshold * (lai_max - lai_min)
    # Set missing value and dtype
    trigger = xr.where(da_lai >= ct, 1, 0)
    trigger = trigger.where(~ct.isnull())
    trigger = trigger.fillna(na_value).astype(dtype)
    trigger.raster.set_nodata(np.dtype(dtype).type(na_value))

    return trigger


def irrigation(
    da_irrigation: xr.DataArray,
    ds_like: xr.Dataset,
    irrigation_value: List[int],
    cropland_class: List[int],
    paddy_class: List[int] = [],
    area_threshold: float = 0.6,
    lai_threshold: float = 0.2,
    logger=logger,
):
    """
    Prepare irrigation maps for paddy and non paddy.

    Parameters
    ----------
    da_irrigation: xr.DataArray
        Irrigation map
    ds_like: xr.Dataset
        Dataset at wflow model domain and resolution.

        * Required variables: ['wflow_landuse', 'LAI']
    irrigation_value: List[int]
        Values that indicate irrigation in da_irrigation.
    cropland_class: List[int]
        Values that indicate cropland in landuse map.
    paddy_class: List[int]
        Values that indicate paddy fields in landuse map.
    area_threshold: float
        Threshold for the area of a pixel to be classified as irrigated.
    lai_threshold: float
        Threshold for the LAI value to be classified as growing season.

    Returns
    -------
    ds_irrigation: xr.Dataset
        Dataset with paddy and non-paddy irrigation maps: ['paddy_irrigation_areas',
        'nonpaddy_irrigation_areas', 'paddy_irrigation_trigger',
        'nonpaddy_irrigation_trigger']
    """
    # Check that the landuse map is available
    if "wflow_landuse" not in ds_like:
        raise ValueError("Landuse map is required in ds_like.")
    landuse = ds_like["wflow_landuse"].copy()

    # Create the irrigated areas mask
    irrigation_mask = da_irrigation.isin(irrigation_value).astype(float)

    # Get cropland mask from landuse
    logger.info("Preparing irrigated areas map for non paddy.")
    nonpaddy = landuse.where(landuse.isin(cropland_class), landuse.raster.nodata)
    nonpaddy_areas = classify_pixels(irrigation_mask, nonpaddy, area_threshold)
    ds_irrigation = nonpaddy_areas.to_dataset(name="nonpaddy_irrigation_areas")

    # Get paddy mask from landuse
    if len(paddy_class) > 0:
        logger.info("Preparing irrigated areas map for paddy.")
        paddy = landuse.where(landuse.isin(paddy_class), landuse.raster.nodata)
        paddy_areas = classify_pixels(irrigation_mask, paddy, area_threshold)
        ds_irrigation["paddy_irrigation_areas"] = paddy_areas

    # Calculate irrigation trigger based on LAI
    logger.info("Calculating irrigation trigger.")
    if "LAI" not in ds_like:
        raise ValueError("LAI map is required in ds_like.")
    lai = ds_like["LAI"].copy()
    trigger = calc_lai_threshold(lai, lai_threshold)

    # Mask trigger with paddy and nonpaddy
    ds_irrigation["nonpaddy_irrigation_trigger"] = trigger.where(nonpaddy_areas == 1, 0)
    if len(paddy_class) > 0:
        ds_irrigation["paddy_irrigation_trigger"] = trigger.where(paddy_areas == 1, 0)

    return ds_irrigation
