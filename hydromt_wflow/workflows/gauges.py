"""Gauges workflows for Wflow plugin."""

import logging
from typing import Optional

import geopandas as gpd
import numpy as np
import xarray as xr
from hydromt.gis import flw

logger = logging.getLogger(f"hydromt.{__name__}")


__all__ = ["gauge_map_uparea"]


def gauge_map_uparea(
    ds: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    uparea_name: Optional[str] = "meta_upstream_area",
    mask: Optional[np.ndarray] = None,
    wdw: Optional[int] = 1,
    rel_error: float = 0.05,
    abs_error: float = 50,
    fillna: bool = False,
):
    """
    Snap point locations to grid cell.

    With smallest difference in upstream area within `wdw` around the
    original location if the local cell does not meet the error criteria.

    Both the upstream area variable named ``uparea_name`` in ``ds`` and ``gdf`` as
    well as ``abs_error`` should have the same unit (typically km2).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with upstream area variable.
    gdf : gpd.GeoDataFrame
        GeoDataFrame with gauge points and uparea column.
    uparea_name : str, optional
        Name of the upstream area variable in ``ds``, by default "meta_upstream_area".
    mask : np.ndarray, optional
        Mask cells to apply the uparea snapping, by default None.
    wdw : int, optional
        Window size around the original location to search for the best matching cell,
        by default 1.
    rel_error : float, optional
        Relative error threshold to accept the best matching cell, by default 0.05.
    abs_error : float, optional
        Absolute error threshold to accept the best matching cell, by default (50 km2).
    fillna : bool, optional
        Fill NaN values in gdf["uparea"] with uparea from ds, by default False.

    Returns
    -------
    da : xr.DataArray
        Gauge map with gauge points snapped to the best matching cell.
    idxs_out : np.ndarray
        Array of indices of the best matching cell.
    ids_out : np.ndarray
        Array of gauge point ids.

    """
    if uparea_name not in ds:
        raise ValueError(f"uparea_name {uparea_name} not found in ds.")
    if gdf["uparea"].isna().all():
        raise ValueError(
            "All gauges have NaN values for uparea. Use another method for snapping."
        )

    # Original number of gauges
    nb_gauges_before_snapping = gdf.index.size

    # Find if there are any nodata in gdf["uparea"]
    if gdf["uparea"].isna().any():
        # Index of gauges with NaN values
        nodata_gauges = list(gdf.index[gdf["uparea"].isna()])
        if fillna:
            logger.warning(
                f"Gauges with ID {nodata_gauges} have NaN values for uparea."
                "Uparea from wflow will be used and they won't be snapped."
            )
            # Replace nan values in gdf with uparea from wflow
            uparea_wflow = ds[uparea_name].raster.sample(gdf, wdw=0)
            gdf["uparea"] = gdf["uparea"].fillna(uparea_wflow.to_pandas())
        else:
            logger.warning(
                f"Gauges with ID {nodata_gauges} have NaN values for uparea."
                "They will be ignored."
            )
            gdf = gdf[gdf["uparea"].notna()]

    ds = ds.copy()
    # Add mask to ds
    if mask is not None:
        ds["mask"] = xr.DataArray(mask, dims=(ds.raster.y_dim, ds.raster.x_dim))

    ds_wdw = ds.raster.sample(gdf, wdw=wdw)
    # Mask valid cells in ds_wdw
    if mask is not None:
        ds_wdw[uparea_name] = ds_wdw[uparea_name].where(
            ds_wdw["mask"], ds_wdw[uparea_name].raster.nodata
        )
    logger.debug(
        f"Snapping gauges points to best matching uparea cell within wdw (size={wdw})."
    )
    upa0 = xr.DataArray(gdf["uparea"], dims=("index"))
    upa_dff = np.abs(ds_wdw[uparea_name].where(ds_wdw[uparea_name] > 0).load() - upa0)
    upa_check = np.logical_and((upa_dff / upa0) <= rel_error, upa_dff <= abs_error)

    # drop index will all nan values
    upa_dff = upa_dff.dropna(dim="index", how="all")
    upa_check = upa_check.sel(index=upa_dff.index)
    ds_wdw = ds_wdw.sel(index=upa_dff.index)

    # find best matching uparea cell in window
    i_wdw = upa_dff.argmin("wdw").load()
    idx_valid = np.nonzero(upa_check.isel(wdw=i_wdw).values)[0]
    if idx_valid.size < gdf.index.size:
        logger.warning(
            f"{idx_valid.size}/{gdf.index.size} gauge points successfully snapped."
        )
    i_wdw = i_wdw.isel(index=idx_valid)
    ds_out = ds_wdw.isel(wdw=i_wdw.load(), index=idx_valid)

    idxs_out = ds.raster.xy_to_idx(
        xs=ds_out[ds.raster.x_dim].values, ys=ds_out[ds.raster.y_dim].values
    )
    ids_out = ds_out.index.values

    # Derive gauge map
    da, idxs_out, ids_out = flw.gauge_map(
        ds,
        idxs=idxs_out,
        ids=ids_out,
        stream=None,
        flwdir=None,
    )

    # Final message
    logger.info(
        f"Snapped {idxs_out.size}/{nb_gauges_before_snapping} gauge points "
        "to best matching uparea cell."
    )

    return da, idxs_out, ids_out
