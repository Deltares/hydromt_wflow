"""Workflow for water demand."""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from hydromt.raster import full_from_transform, full_like
from scipy.ndimage import convolve

__all__ = ["allocation_areas", "non_irrigation"]

map_vars = {
    "dom": "domestic",
    "ind": "industry",
    "lsk": "livestock",
}


def transform_half_degree(
    bbox: tuple | list,
) -> tuple:
    """Transform a bbox into a covering box with a 0.5 degree resolution.

    Parameters
    ----------
    bbox : tuple | list
        A bounding box in the format of (minx, miny, maxx, maxy).

    Returns
    -------
    tuple
        Affine matrix (geospatial transform), width and height
    """
    left = round(math.floor(bbox[0] * 2) / 2, 1)
    bottom = round(math.floor(bbox[1] * 2) / 2, 1)
    top = round(math.ceil(bbox[3] * 2) / 2, 1)
    right = round(math.ceil(bbox[2] * 2) / 2, 1)

    affine = Affine(0.5, 0.0, left, 0.0, -0.5, top)
    h = round((top - bottom) / 0.5)
    w = round((right - left) / 0.5)

    return affine, w, h


def touch_intersect(
    row: pd.Series,
    vector: gpd.GeoDataFrame,
) -> pd.Series:
    """Find if a geometry (row) has unequal intersects and touches.

    Parameters
    ----------
    row : pd.Series
        Row from a GeoDataFrame.
    vector : gpd.GeoDataFrame
        GeoDataFrame to test touches and intersects with.

    Returns
    -------
    pd.Series
        A row containing the boolean of equal touches and intersects.
    """
    contain = True
    _t = sum(vector.touches(row.geometry))
    _i = sum(vector.intersects(row.geometry))
    diff = abs(_i - _t)
    if diff == 0:
        contain = False
    row["contain"] = contain
    return row


def non_irrigation(
    ds: xr.Dataset,
    ds_like: xr.Dataset,
    ds_method: str,
    popu: xr.Dataset,
    popu_method: str,
) -> tuple:
    """Create non-irrigation water demand maps.

    Parameters
    ----------
    ds : xr.Dataset
        Raw non-irrigation data dataset.
    ds_like : xr.Dataset
        Dataset at wflow model domain and resolution.
    ds_method : str
        Non-irrigation data resampling method.
    popu : xr.Dataset
        Population dataset (number of people per gridcell).
    popu_method : str
        Population data resampling method.

    Returns
    -------
    tuple
        Non-irrigation data at model resolution, Population data at model resolution.
    """
    # Reproject to up or downscale
    non_irri_scaled = ds.raster.reproject_like(
        ds_like,
        method=ds_method,
    )

    # Focus on population data
    popu_shape = popu.raster.shape
    if any([_ex < _ey for _ex, _ey in zip(popu_shape, ds_like.raster.shape)]):
        popu_scaled = popu.raster.reproject_like(
            ds_like,
            method="nearest",
        )
        scaling = [_ey / _ex for _ex, _ey in zip(popu_shape, ds_like.raster.shape)]
        scaling = scaling[0] * scaling[1]
        popu_scaled = popu_scaled * scaling
    else:
        popu_scaled = popu.raster.reproject_like(
            ds_like,
            method=popu_method,
        )

    popu_scaled.name = "population"

    # Get transform at half degree resolution
    transform, width, height = transform_half_degree(
        ds_like.raster.bounds,
    )

    # Create dummy dataset from this info
    dda = full_from_transform(
        transform=transform,
        shape=(height, width),
        crs=4326,
    )

    # Some info regarding the original ds
    ds.raster.clip_bbox(ds_like.raster.bounds)

    # Some extra info from ds_like

    # Scale the polulation data based on global glob cells
    popu_down = popu_scaled.raster.reproject_like(
        dda,
        method="sum",
    )
    popu_redist = popu_down.raster.reproject_like(
        popu_scaled,
        method="nearest",
    )
    popu_redist = popu_scaled / popu_redist

    # Setup downscaled non irigation domestic data
    non_irri_down = non_irri_scaled.raster.reproject_like(
        dda,
        method="sum",
    )

    # Loop through the domestic variables
    for var in ["dom_gross", "dom_net"]:
        attrs = non_irri_scaled[var].attrs
        new = non_irri_down[var].raster.reproject_like(
            non_irri_scaled,
            method="nearest",
        )
        new = new * popu_redist
        new = new.fillna(0.0)
        non_irri_scaled[var] = new
        non_irri_scaled[var] = non_irri_scaled[var].assign_attrs(attrs)

    return non_irri_scaled, popu_scaled


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
    # Split based on admin bounds
    sub_basins = basins.copy()
    sub_basins["uid"] = range(len(sub_basins))

    if admin_bounds is not None:
        sub_basins = basins.overlay(
            admin_bounds,
            how="union",
        )
        sub_basins = sub_basins[~sub_basins["value"].isna()]

        # Remove unneccessary stuff
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
    alloc = full_like(da_like, nodata=nodata, lazy=True).astype(int)
    alloc = alloc.raster.rasterize(sub_basins, col_name="uid", nodata=nodata)

    # Get the areas per exploded area
    alloc_area = alloc.to_dataset()
    alloc_area["area"] = da_like.raster.area_grid()
    area = alloc_area.groupby("uid").sum().area
    del alloc_area
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

        riv = np.setdiff1d(
            np.unique(alloc.where(da_like == 1, nodata)),
            [-9999],
        )
        no_riv = np.setdiff1d(area.uid.values, riv)

        # Also look at minimum areas
        area_riv = area.sel(uid=riv)
        area_mask = area_riv < min_area
        no_riv = np.append(no_riv, area_riv.uid[area_mask].values)

        # Solved, so break
        if no_riv.size == 0:
            break

        if np.array_equal(np.sort(no_riv), old_no_riv):
            kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

        # Loop through all 'loose' area and merge those
        for val in no_riv:
            mask = alloc == val
            rnd = convolve(mask.astype(int).values, kernel, mode="constant")
            nbs = np.setdiff1d(
                np.unique(alloc.values[np.where(rnd > 0)]), [nodata, val]
            )
            del rnd
            if nbs.size == 0:
                continue
            area_nbs = area.sel(uid=nbs)
            new_uid = area_nbs.uid[area_nbs.argmax(dim="uid")].values
            alloc = alloc.where(alloc != val, new_uid)

        # Take out the merged areas and append counter
        area = area.sel(uid=np.setdiff1d(np.unique(alloc.values), [nodata]))
        _count += 1
        old_no_riv = np.sort(no_riv)

    alloc.name = "allocation_areas"
    return alloc


def classify_pixels(
    da_crop: xr.DataArray,
    da_model: xr.DataArray,
    threshold: float,
    nodata_value: float | int = -9999,
):
    """Classifies pixels based on a (fractional) threshold.

    Pixels with a value above this threshold are set to be irrigated, and pixels below
    this value as rainfed.

    Parameters
    ----------
    da_crop: xr.DataArray
        Data with crop mask
    da_model: xr.DataArray
        Layer of the wflow model (used for reprojecting)
    threshold: float
        Threshold above which pixels are classified as irrigated
    nodata_value: float | int = -9999
        Value to be used for nodata

    Returns
    -------
    crop_map: xr.DataArray
        Mask with classified pixels
    """
    # Convert crop map to an area map
    da_area = da_crop * da_crop.raster.area_grid()
    # Resample to model grid and sum areas
    da_area2model = da_area.raster.reproject_like(da_model, method="sum")

    # Calculate relative area
    relative_area = da_area2model / da_area2model.raster.area_grid()

    # Classify pixels with a 1 where it exceeds the threshold, and 0 where it doesnt.
    crop_map = xr.where(relative_area >= threshold, 1, 0)
    crop_map = xr.where(da_model.isnull(), nodata_value, crop_map)

    # Fix nodata values
    crop_map.raster.set_nodata(nodata_value)

    return crop_map


def find_paddy(
    landuse_da: xr.DataArray,
    irrigated_area: xr.DataArray,
    paddy_class: int,
    nodata_value: float | int = -9999,
):
    """Find the location of paddy crops.

    Done by finding the overlap between the irrigated area
    grid, and a landcover map that has a separate class for rice/paddy fields.

    Parameters
    ----------
    landuse_da: xr.DataArray
        Map with landuse classes (must contain a class for paddy)
    irrigated_area: xr.DataArray
        Mask with irrigated areas
    paddy_class: int
        Class that matches the rice or paddy land cover type
    nodata_value: float | int = -9999
        Value to be used as nodata value

    Returns
    -------
    paddy: xr.DataArray
        Mask with pixels classified as paddy or rice fields
    nonpaddy: xr.DataArray
        Maks with pixels classified as non paddy irrigation
    """
    # Resample irrigated area to landuse datasets
    irr2lu = irrigated_area.raster.reproject_like(landuse_da)
    # Mask pixels with paddies
    paddy = xr.where((irr2lu != 0) & (landuse_da == paddy_class), 1, 0)
    # Mask pixels that are irrigated, but not paddy
    nonpaddy = xr.where((irr2lu != 0) & (landuse_da != paddy_class), 1, 0)
    # Fix nodata values
    paddy.raster.set_nodata(nodata_value)
    nonpaddy.raster.set_nodata(nodata_value)
    # Return two rasters
    return paddy, nonpaddy


def add_crop_maps(
    ds_rain: xr.Dataset,
    ds_irri: xr.Dataset,
    mod,
    do_paddy: bool,
    do_nonpaddy: bool,
    paddy_value: int,
    default_value: float,
    map_type: str = "crop_factor",
):
    """Add crop_factor maps to the model.

    Based on (MIRCA) datasets with rainfed and irrigated crop information. First adds
    the value for the paddy fields, followed by the non paddy irrigated crops. All other
    cells are treated as rainfed.

    Parameters
    ----------
    ds_rain: xr.Dataset
        Dataset with crop information for rainfed crops
    ds_irri: xr.Dataset
        Dataset with crop information for irrigated crops
    mod: WflowModel
        WflowModel object with grid information, used to find the model domain, and
        locations of paddy and nonpaddy irrigation
    paddy_value: int
        Value to be used for locations with paddy irrigation
    default_value: float
        Fallback value for rainfed data
    map_type: str = "crop_factor"
        Type of data requested, currently supports {"crop_factor" and "rootingdepth"}

    Returns
    -------
    crop_map: xr.DataArray
        Map with the values for the requested map_type
    """
    if map_type == "crop_factor":
        ds_layer_name = "crop_factor"
    elif map_type == "rootingdepth":
        ds_layer_name = "rootingdepth"

    # Start with rainfed (all pixels, and fill missing values with default value)
    rainfed_highres = ds_rain[ds_layer_name].raster.reproject_like(
        mod.grid.wflow_subcatch
    )
    # Fill missing values with default values
    rainfed_highres = rainfed_highres.where(~rainfed_highres.isnull(), default_value)
    # Mask to model domain
    crop_map = xr.where(
        mod.grid["wflow_dem"].raster.mask_nodata().isnull(), np.nan, rainfed_highres
    )

    # Add paddy values
    if do_paddy:
        crop_map = xr.where(
            mod.grid["paddy_irrigation_areas"] == 1, paddy_value, crop_map
        )

    if do_nonpaddy:
        # Resample to model resolution
        irrigated_highres = ds_irri[ds_layer_name].raster.reproject_like(
            mod.grid.wflow_subcatch
        )
        # Map values to the correct mask
        tmp = xr.where(mod.grid["nonpaddy_irrigation_areas"] == 1, irrigated_highres, 0)
        # Fill missing values with the default crop factor (as it can happen that not all
        # cells are covered in this data)
        tmp = tmp.where(~tmp.isnull(), default_value)
        # Add data to crop_factop map
        crop_map = xr.where(mod.grid["nonpaddy_irrigation_areas"] == 1, tmp, crop_map)

    return crop_map


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
        The vertical conductitivity at the surface
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
    parameter. If no target_conducitivity is specified, kvfrac is set to be equal to 1.

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
    da_kvfrac = full_like(ds_model["c"])
    # Set all values to 1
    da_kvfrac = da_kvfrac.where(ds_model.wflow_dem.raster.mask_nodata().isnull(), 1.0)

    # Get the actual depths
    wflow_depths = np.cumsum(wflow_thicknesslayers)
    # Find the index of the layers where a kvfrac should be set
    idx = np.where(target_conductivity is not None)[0]
    # Find the corresponding depths
    [wflow_depths[i] for i in idx]

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
            # Update layer in dataarray
            da_kvfrac.loc[dict(layer=idx)] = kvfrac

    return da_kvfrac


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
