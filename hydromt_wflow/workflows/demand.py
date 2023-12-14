"""Workflow for water demand."""

import math

import pandas as pd
import xarray as xr
from affine import Affine
from hydromt.raster import full_from_transform

map_vars = {
    "dom": "domestic",
    "ind": "industry",
    "lsk": "livestock",
}


def transform_half_degree(
    bbox: tuple | list,
):
    """_summary_.

    _extended_summary_

    Parameters
    ----------
    bbox : tuple | list
        _description_
    """
    left = round(math.floor(bbox[0] * 2) / 2, 1)
    bottom = round(math.floor(bbox[1] * 2) / 2, 1)
    top = round(math.ceil(bbox[3] * 2) / 2, 1)
    right = round(math.ceil(bbox[2] * 2) / 2, 1)

    affine = Affine(0.5, 0.0, left, 0.0, -0.5, top)
    h = round((top - bottom) / 0.5)
    w = round((right - left) / 0.5)

    return affine, w, h


def touch_intersect(row, vector):
    """_summary_.

    _extended_summary_

    Parameters
    ----------
    row : _type_
        _description_
    vector : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    contain = True
    _t = sum(vector.touches(row.geometry))
    _i = sum(vector.intersects(row.geometry))
    diff = abs(_i - _t)
    if diff == 0:
        contain = False
    row["contain"] = contain
    return row


def non_irigation(
    ds: xr.Dataset,
    ds_like: xr.Dataset,
    ds_method: str,
    popu: xr.Dataset,
    popu_method: str,
):
    """_summary_.

    _extended_summary_

    Parameters
    ----------
    ds : xr.Dataset
        _description_
    """
    # Reproject to up or downscale
    non_iri_scaled = ds.raster.reproject_like(
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

    popu_scaled.name = "Population_scaled"

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
    non_iri_down = non_iri_scaled.raster.reproject_like(
        dda,
        method="sum",
    )

    # Loop through the domestic variables
    for var in ["dom_gross", "dom_net"]:
        attrs = non_iri_scaled[var].attrs
        new = non_iri_down[var].raster.reproject_like(
            non_iri_scaled,
            method="nearest",
        )
        new = new * popu_redist
        new = new.fillna(0.0)
        non_iri_scaled[var] = new
        non_iri_scaled[var] = non_iri_scaled[var].assign_attrs(attrs)

    return non_iri_scaled, popu_scaled


def allocate(
    ds_like: xr.Dataset,
    min_area: float | int,
    admin_bounds: object,
    basins: xr.Dataset,
    rivers: xr.Dataset,
):
    """_summary_.

    _extended_summary_

    Parameters
    ----------
    ds_like : xr.Dataset
        _description_
    min_area : float | int
        _description_
    admin_bounds : object
        _description_
    basins : xr.Dataset
        _description_
    rivers : xr.Dataset
        _description_
    """
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
    sub_basins = sub_basins.apply(lambda row: touch_intersect(row, rivers), axis=1)
    sub_basins["sqkm"] = sub_basins.geometry.to_crs(3857).area / 1000**2
    _count = 0

    # Create touched and not touched by rivers datasets
    while True:
        # Ensure a break if it cannot be solved
        if _count == 100:
            break

        # Everything touched by river based on difference
        # (is not what we want, yet)
        if _count != 0:
            sub_basins = sub_basins.apply(
                lambda row: touch_intersect(row, rivers), axis=1
            )
            sub_basins["sqkm"] = sub_basins.geometry.to_crs(3857).area / 1000**2

        # Set no_riv and riv (what's touched and what's not)
        no_riv = sub_basins[~sub_basins["contain"]]
        riv = sub_basins[sub_basins["contain"]]

        # Include minimal area option
        min_basins = sub_basins[sub_basins["sqkm"] < min_area]
        min_basins = min_basins[~min_basins.uid.isin(no_riv.uid)]
        # Only concatenate if there are different small basins
        # compared to basins that do not touch a river
        if not min_basins.empty:
            no_riv = pd.concat(
                [no_riv, min_basins],
                ignore_index=True,
            )

        _n = 0

        if no_riv.empty:
            break

        for _, row in no_riv.iterrows():
            touched = riv[riv.touches(row.geometry)]
            if touched.empty:
                continue
            if row.value in touched.value.values:
                uid = touched[touched["value"] == row.value].uid.values[0]
            else:
                touched["area"] = touched.area
                uid = touched[touched["area"] == touched["area"].max()].uid.values[0]
            # Set the identifier to the new value
            # (i.e. the touched basin)
            sub_basins.loc[sub_basins["uid"] == row.uid, "uid"] = uid
            _n += 1

        # Ensure a break if nothing is touched
        # This means that it cannot be solved
        # TODO maybe look at iteratively buffering...
        if _n == 0:
            break

        sub_basins = sub_basins.dissolve("uid", sort=False, as_index=False)
        _count += 1

    alloc = full_from_transform(
        ds_like.raster.transform,
        ds_like.raster.shape,
        crs=ds_like.raster.crs,
        lazy=True,
    )
    alloc = alloc.raster.rasterize(sub_basins, col_name="uid", nodata=-9999)
    alloc.name = "Allocation_id"
    alloc = alloc.rename(dict(zip(alloc.dims, list(ds_like.dims)[:2])))
    return alloc
