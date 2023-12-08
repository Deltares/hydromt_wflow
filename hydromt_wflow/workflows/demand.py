"""Workflow for water demand."""

import math

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
        new["spatial_ref"] = non_iri_scaled.spatial_ref
        non_iri_scaled[var] = new
        non_iri_scaled[var] = non_iri_scaled[var].assign_attrs(attrs)

    return non_iri_scaled, popu_scaled


def allocate(
    ds_like: xr.Dataset,
    admin_bounds: xr.DataArray,
    basins: xr.Dataset,
    rivers: xr.Dataset,
):
    """_summary_.

    _extended_summary_
    """
    pass
