"""Utility functions for HydroMT-wflow."""

import logging

import geopandas as gpd
import hydromt  # noqa : F401
import xarray as xr

logger = logging.getLogger(f"hydromt.{__name__}")


def vectorize(
    ds: xr.Dataset,
    var: str,
) -> gpd.GeoDataFrame:
    """Vectorize a layer from a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        _description_
    var : str
        _description_

    Returns
    -------
    gpd.GeoDataFrame
        _description_
    """
    gdf = ds[var].raster.vectorize().set_index("value").sort_index()
    return gdf
