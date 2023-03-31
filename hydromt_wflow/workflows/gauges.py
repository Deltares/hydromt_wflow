import numpy as np
import xarray as xr
import geopandas as gpd
from hydromt import flw
from typing import Optional
import logging


logger = logging.getLogger(__name__)


__all__ = ["gauge_map_uparea"]


def gauge_map_uparea(
    ds: xr.Dataset,
    gdf: gpd.GeoDataFrame,
    uparea_name: Optional[str] = "wflow_uparea",
    wdw: Optional[int] = 1,
    rel_error: float = 0.05,
    abs_error: float = 50,
    logger=logger,
):
    """ """
    if uparea_name not in ds:
        return

    ds_wdw = ds.raster.sample(gdf, wdw=wdw)
    logger.debug(
        f"Snapping gauges points to best matching uparea cell within wdw (size={wdw})."
    )
    upa0 = xr.DataArray(gdf["uparea"], dims=("index"))
    upa_dff = np.abs(ds_wdw[uparea_name].where(ds_wdw[uparea_name] > 0).load() - upa0)
    upa_check = np.logical_or((upa_dff / upa0) <= rel_error, upa_dff <= abs_error)
    # find best matching uparea cell in window
    i_wdw = upa_dff.argmin("wdw").load()

    idx_valid = np.where(upa_check.isel(wdw=i_wdw).values)[0]
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
        logger=logger,
    )

    return da, idxs_out, ids_out
