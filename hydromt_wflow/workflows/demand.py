"""Workflow for water demand."""

import logging
from typing import List, Optional

import dask
import geopandas as gpd
import numpy as np
import xarray as xr
from hydromt import raster
from scipy.ndimage import convolve

logger = logging.getLogger(__name__)

__all__ = [
    "allocation_areas",
    "domestic",
    "other_demand",
    "surfacewaterfrac",
    "irrigation",
]

map_vars = {
    "dom": "domestic",
    "ind": "industry",
    "lsk": "livestock",
}


def full(
    da_like: xr.DataArray,
    dtype: type = np.float32,
    fill_value: float | int = np.nan,
    crs: int = 4326,
    name: str = "data",
    extra: dict = None,
):
    """_summary_."""
    # Setup some basic information from da_like
    dims = da_like.dims
    coords = {dim: da_like[dim].values for dim in dims}
    size = [da_like.coords.sizes[dim] for dim in dims]

    # if not extra dimension is given, default to da_like
    if extra is None:
        dsk = dask.array.full(
            shape=size,
            dtype=dtype,
            fill_value=fill_value,
        )
        da = xr.DataArray(
            data=dsk,
            coords=da_like.coords,
            dims=da_like.dims,
        )
        da.attrs = {"_FillValue": fill_value}
        da.name = name
        return da

    # If extra dimensions are given, construct from there
    coords.update(extra)
    extra_size = [len(item) for item in extra.values()]

    dsk = dask.array.full(
        shape=extra_size + size,
        dtype=dtype,
        fill_value=fill_value,
    )
    da = xr.DataArray(
        data=dsk,
        coords=coords,
        dims=tuple(extra.keys()) + dims,
    )
    da.attrs = {"_FillValue": fill_value}
    da.name = name

    if crs is not None:
        da.raster.set_crs(crs)

    return da


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
    da_like: xr.DataArray,
    min_area: float | int,
    admin_bounds: gpd.GeoDataFrame,
    basins: gpd.GeoDataFrame,
) -> xr.DataArray:
    """Create water allocation area.

    Based on current wflow model domain and resolution and making use of
    the model basins and optional administrative boundaries.

    Parameters
    ----------
    da_like : xr.DataArray
        A grid covering the wflow model domain.
    min_area : float | int
        The minimum area an allocation area should have.
    admin_bounds : gpd.GeoDataFrame
        Administrative boundaries, e.g. sovereign nations.
    basins : gpd.GeoDataFrame
        The wflow model basins.

    Returns
    -------
    xr.DataArray
        The water demand allocation areas.
    """
    # Set variables
    nodata = -9999
    # Add a unique identifier as a means for dissolving later on, bit pro forma here
    sub_basins = basins.copy()
    sub_basins["uid"] = range(len(sub_basins))

    # Split based on administrative boundaries
    if admin_bounds is not None:
        sub_basins = basins.overlay(
            admin_bounds,
            how="union",
        )
        sub_basins = sub_basins[~sub_basins["value"].isna()]

        # Remove unneccessary columns
        cols = sub_basins.columns.drop(["value", "geometry", "admin_id"]).tolist()
        sub_basins.drop(cols, axis=1, inplace=True)
        # Use this uid to dissolve on later
        sub_basins["uid"] = range(len(sub_basins))

        # Dissolve cut pieces back
        for _, row in sub_basins.iterrows():
            if not str(row.admin_id).lower() == "nan":
                continue
            touched = sub_basins[sub_basins.touches(row.geometry)]
            uid = touched[touched["value"] == row.value].uid.values[0]
            sub_basins.loc[sub_basins["uid"] == row.uid, "uid"] = uid
        sub_basins = sub_basins.dissolve("uid", sort=False, as_index=False)

    # Set the contain flag per geom
    # TODO figure out the code below for more precise allocation areas
    sub_basins = sub_basins.explode(index_parts=False, ignore_index=True)
    sub_basins["uid"] = range(len(sub_basins))
    admin_bounds = None

    # Setup the allocation grid based on exploded geometries
    # TODO get exploded geometries from something raster based
    alloc = full(da_like, fill_value=nodata, dtype=np.int32, name="allocation_areas")
    alloc = alloc.raster.rasterize(sub_basins, col_name="uid", nodata=nodata)

    # Get the areas per exploded area
    alloc_area = alloc.to_dataset()
    alloc_area["area"] = da_like.raster.area_grid()
    area = alloc_area.groupby("uid").sum().area
    del alloc_area
    # To km2
    area = area.drop_sel(uid=nodata) / 1e6

    _count = 0
    old_no_riv = None

    # Solve the area iteratively
    while True:
        # Break if cannot be solved
        if _count == 100:
            break

        # Define the surround matrix for convolution
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

        # Get ID's of basins containing a river and those that do not
        riv = np.setdiff1d(
            np.unique(alloc.where(da_like == 1, nodata)),
            [-9999],
        )
        no_riv = np.setdiff1d(area.uid.values, riv)

        # Also look at minimum areas
        area_riv = area.sel(uid=riv)
        area_mask = area_riv < min_area
        no_riv = np.append(no_riv, area_riv.uid[area_mask].values)

        # Solved, as there are not more areas with no river, so break
        if no_riv.size == 0:
            break

        # Solve with corners included, as the areas remaining touch diagonally
        if np.array_equal(np.sort(no_riv), old_no_riv):
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        # Loop through all the area that have no river and merge those
        # with the areas that lie besides it
        for val in no_riv:
            # Mask based on the value of the subbasin without a river
            mask = alloc == val
            # Find the ids of surrouding basins based on the kernel
            rnd = convolve(mask.astype(int).values, kernel, mode="constant")
            nbs = np.setdiff1d(
                np.unique(alloc.values[np.where(rnd > 0)]), [nodata, val]
            )
            del rnd
            # If none are found, continue to the next
            if nbs.size == 0:
                continue
            # Set the new id of this subbasin based on the largest next to it
            area_nbs = area.sel(uid=nbs)
            new_uid = area_nbs.uid[area_nbs.argmax(dim="uid")].values
            alloc = alloc.where(alloc != val, new_uid)

        # Take out the merged areas and append counter
        area = area.sel(uid=np.setdiff1d(np.unique(alloc.values), [nodata]))
        _count += 1
        old_no_riv = np.sort(no_riv)

    alloc.name = "allocation_areas"
    return alloc


def surfacewaterfrac(
    da_like: xr.DataArray,
    gwfrac_raw: xr.DataArray,
    gwbodies: xr.DataArray,
    ncfrac: xr.DataArray,
    waterareas: xr.DataArray,
    interpolate: bool = True,
) -> xr.DataArray:
    """Create surface water fraction map.

    Parameters
    ----------
    da_like : xr.DataArray
        Wflow model like grid.
    gwfrac_raw : xr.DataArray
        Raw groundwater fraction data map.
    gwbodies : xr.DataArray
        Groundwater bodies map. Either 0 or 1.
    ncfrac : xr.DataArray
        Non-conventional water fraction.
    waterareas : xr.DataArray
        Water source areas.
    interpolate : bool
        Interpolate missing data values within wflow model domain.

    Returns
    -------
    xr.DataArray
        Surface water fraction.
    """
    # Mask to int value for later use
    da_like = da_like.raster.mask_nodata(-9999)

    # Resample the data to model resolution
    gwfrac_raw_mr = gwfrac_raw.raster.reproject_like(
        da_like,
        method="nearest",
    )
    x, y = np.where(~np.isnan(gwfrac_raw_mr))
    gwbodies_mr = gwbodies.raster.reproject_like(
        da_like,
        method="nearest",
    )
    ncfrac_mr = ncfrac.raster.reproject_like(
        da_like,
        method="nearest",
    )
    # Prepare the nodata a little as lisflood data has none set by default
    waterareas = waterareas.raster.mask_nodata(-9999)
    if "_FillValue" not in waterareas.attrs:
        waterareas = waterareas.assign_attrs({"_FillValue": -9999})
    # Reproject
    waterareas_mr = waterareas.raster.reproject_like(
        da_like,
        method="nearest",
    )
    # waterareas_mr = waterareas_mr.raster.mask_nodata(-9999)
    # Set nodata values to zeros
    waterareas_mr = waterareas_mr.where(
        waterareas_mr != waterareas_mr.raster.nodata,
        0,
    )

    # Get the fractions based on area count
    w_pixels = np.take(
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

    # Determine the groundwater fraction
    gwfrac_val = np.minimum(
        gwfrac_raw_mr.values[x, y] * (a_pixels / (w_pixels + 0.01)),
        1 - ncfrac_mr.values[x, y],
    )
    gwfrac_val[np.where(gwbodies_mr.values[x, y] == 0)] = 0
    # invert to get surface water frac
    # TODO fix with line from listflood
    gwfrac_val = 1 - gwfrac_val

    # create the dataarray for the fraction
    swfrac = xr.full_like(
        gwfrac_raw_mr,
        fill_value=np.nan,
        dtype=np.float32,
    ).load()
    swfrac.name = "SurfaceWaterFrac"
    swfrac.attrs = {"_FillValue": -9999}
    swfrac = swfrac.copy()

    # Set and interpolate the values
    swfrac.values[x, y] = gwfrac_val
    if interpolate:
        swfrac = swfrac.interpolate_na(dim=swfrac.raster.x_dim, method="linear")
        swfrac = swfrac.interpolate_na(
            dim=swfrac.raster.x_dim, method="linear", fill_value="extrapolate"
        )

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
