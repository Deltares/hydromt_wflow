"""Some pcraster functions to support older models."""
import glob
import logging
from os.path import join
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from hydromt.io import open_mfraster

logger = logging.getLogger(__name__)

# specify pcraster map types
# NOTE non scalar (float) data types only
PCR_VS_MAP = {
    "wflow_ldd": "ldd",
    "wflow_river": "bool",
    "wflow_streamorder": "ordinal",
    "wflow_gauges": "nominal",  # to avoid large memory usage in pcraster.aguila
    "wflow_subcatch": "nominal",  # idem.
    "wflow_landuse": "nominal",
    "wflow_soil": "nominal",
    "wflow_reservoirareas": "nominal",
    "wflow_reservoirlocs": "nominal",
    "wflow_lakeareas": "nominal",
    "wflow_lakelocs": "nominal",
    "wflow_glacierareas": "nominal",
}


def read_staticmaps_pcr(
    root: Path | str,
    crs: int = 4326,
    obj: object = None,
    **kwargs
):
    """Read and staticmaps at <root/staticmaps> and parse to xarray."""
    da_lai = None
    da = None

    fn = join(root, "staticmaps", "*.map")
    fns = glob.glob(fn)
    if len(fns) == 0:
        logger.warning(f"No staticmaps found at {fn}")
        return
    _staticmaps = open_mfraster(fns, **kwargs)
    path = join(root, "staticmaps", "clim", "LAI*")
    if len(glob.glob(path)) > 0:
        da_lai = open_mfraster(
            path, concat=True, concat_dim="time", logger=logger, **kwargs
        )
        if obj is not None:
            obj.set_grid(da_lai, "LAI")

    # reorganize c_0 etc maps
    da_c = []
    list_c = [v for v in _staticmaps if str(v).startswith("c_")]
    if len(list_c) > 0:
        for i, v in enumerate(list_c):
            da_c.append(_staticmaps[f"c_{i:d}"])
        da = xr.concat(
            da_c, pd.Index(np.arange(len(list_c), dtype=int), name="layer")
        ).transpose("layer", ...)
        if obj is not None:
            obj.set_grid(da, "c")

    if obj is not None:
        if obj.crs is None:
            if crs is None:
                crs = 4326  # default to 4326
            obj.set_crs(crs)

    return da, da_lai


def write_staticmaps_pcr(
    staticmaps: xr.Dataset,
    root: Path | str,
):
    """Write staticmaps at <root/staticmaps> in PCRaster maps format."""
    ds_out = staticmaps
    if "LAI" in ds_out.data_vars:
        ds_out = ds_out.rename_vars({"LAI": "clim/LAI"})
    if "c" in ds_out.data_vars:
        for layer in ds_out["layer"]:
            ds_out[f"c_{layer.item():d}"] = ds_out["c"].sel(layer=layer)
            ds_out[f"c_{layer.item():d}"].raster.set_nodata(
                ds_out["c"].raster.nodata
            )
        ds_out = ds_out.drop_vars(["c", "layer"])
    logger.info("Writing (updated) staticmap files.")
    # add datatypes for maps with same basenames, e.g. wflow_gauges_grdc
    pcr_vs_map = PCR_VS_MAP.copy()
    for var_name in ds_out.raster.vars:
        base_name = "_".join(var_name.split("_")[:-1])  # clip _<postfix>
        if base_name in PCR_VS_MAP:
            pcr_vs_map.update({var_name: PCR_VS_MAP[base_name]})
    ds_out.raster.to_mapstack(
        root=join(root, "staticmaps"),
        mask=True,
        driver="PCRaster",
        pcr_vs_map=pcr_vs_map,
        logger=logger,
    )