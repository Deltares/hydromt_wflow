import logging

import geopandas as gpd
import numpy as np
import pyflwdir
import xarray as xr
from hydromt.gis import flw

logger = logging.getLogger(f"hydromt.{__name__}")

__all__ = ["subbasin_map"]


def subbasin_map(
    ds: xr.Dataset,
    flwdir: pyflwdir.FlwdirRaster,
    method: str,
    threshold: int,
    add_outlets_map: bool = False,
) -> tuple[xr.DataArray, xr.DataArray | None, gpd.GeoDataFrame | None]:
    """
    Create a subbasin map based on the specified method and threshold.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset of the hydrological model.
    flwdir : pyflwdir.FlwdirRaster
        Flow direction raster used for delineation.
    method : str
        Method for subbasin delineation. One of ['streamorder', 'pfafstetter', 'area']
    threshold : int
        Threshold value for the selected method.
        For 'streamorder', minimum Strahler order of the streams to
        delineate subbasins.
        For 'pfafstetter', the Pfafstetter level to delineate subbasins.
        For 'area', the minimum upstream area [km2] to delineate subbasins.
    add_outlets_map : bool, optional
        If True, also derive an outlets map for the subbasins, by default False.

    Returns
    -------
    da_subbas : xr.DataArray
        DataArray containing the subbasin map.
    da_outlets : xr.DataArray | None
        DataArray containing the outlets map if `add_outlets_map` is True, else None.
    gdf_outlets : gpd.GeoDataFrame | None
        GeoDataFrame containing the outlets points if `add_outlets_map` is True, else
        None.
    --------
    pyflwdir.FlwdirRaster.subbasins_streamorder
    pyflwdir.FlwdirRaster.subbasins_pfafstetter
    pyflwdir.FlwdirRaster.subbasins_area
    """
    # Delineate subbasins
    if method == "streamorder":
        subbas, idxs_out = flwdir.subbasins_streamorder(min_sto=threshold)
    elif method == "pfafstetter":
        subbas, idxs_out = flwdir.subbasins_pfafstetter(depth=threshold)
    elif method == "area":
        subbas, idxs_out = flwdir.subbasins_area(area_min=threshold)
    else:
        valid_methods = ["streamorder", "pfafstetter", "area"]
        raise ValueError(f"Invalid method '{method}'. Must be one of {valid_methods}.")

    # Convert to xarray DataArray
    attrs = {
        "description": f"Subbasin map delineated using method '{method}' with threshold {threshold}",  # noqa: E501
        "_FillValue": 0,
    }
    da_subbas = xr.DataArray(
        subbas.astype(np.int32),
        dims=ds.raster.dims,
        coords=ds.raster.coords,
        attrs=attrs,
    )

    da_outlets = None
    gdf_outlets = None

    if add_outlets_map:
        # Derive gauge map
        da_outlets, idxs, ids = flw.gauge_map(
            ds,
            idxs=idxs_out,
        )
        # Get geom
        points = gpd.points_from_xy(*ds.raster.idx_to_xy(idxs))
        gdf_outlets = gpd.GeoDataFrame(
            index=ids.astype(np.int32), geometry=points, crs=ds.raster.crs
        )
        # Add value column
        gdf_outlets["value"] = ids.astype(np.int32)

    return da_subbas, da_outlets, gdf_outlets
