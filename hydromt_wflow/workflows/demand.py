"""Workflow for water demand."""

import xarray as xr

map_vars = {
    "dom": "domestic",
    "ind": "industry",
    "lsk": "livestock",
}


def static(
    ds: xr.Dataset,
    ds_like: xr.Dataset,
    method: str = "nearest",
):
    """_summary_.

    _extended_summary_

    Parameters
    ----------
    ds : xr.Dataset
        _description_
    """
    # Reproject to up or downscale
    static = ds.raster.reproject_like(
        ds_like,
        method=method,
    )

    return static


def allocate():
    """_summary_.

    _extended_summary_
    """
    pass
