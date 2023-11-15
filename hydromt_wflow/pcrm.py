"""Some pcraster functions to support older models."""
import glob
import logging
import os
import tempfile
from os.path import basename, dirname, isdir, isfile, join
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from hydromt.io import open_mfraster
from pyflwdir import core_conversion, core_d8, core_ldd
from pyproj import CRS

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

GDAL_DRIVER_CODE_MAP = {
    "asc": "AAIGrid",
    "blx": "BLX",
    "bmp": "BMP",
    "bt": "BT",
    "dat": "ZMap",
    "dem": "USGSDEM",
    "gen": "ADRG",
    "gif": "GIF",
    "gpkg": "GPKG",
    "grd": "NWT_GRD",
    "gsb": "NTv2",
    "gtx": "GTX",
    "hdr": "MFF",
    "hf2": "HF2",
    "hgt": "SRTMHGT",
    "img": "HFA",
    "jpg": "JPEG",
    "kro": "KRO",
    "lcp": "LCP",
    "map": "PCRaster",
    "mbtiles": "MBTiles",
    "mpr/mpl": "ILWIS",
    "ntf": "NITF",
    "pix": "PCIDSK",
    "png": "PNG",
    "pnm": "PNM",
    "rda": "R",
    "rgb": "SGI",
    "rst": "RST",
    "rsw": "RMF",
    "sdat": "SAGA",
    "sqlite": "Rasterlite",
    "ter": "Terragen",
    "tif": "GTiff",
    "vrt": "VRT",
    "xpm": "XPM",
    "xyz": "XYZ",
}
GDAL_EXT_CODE_MAP = {v: k for k, v in GDAL_DRIVER_CODE_MAP.items()}


def write_clone(tmpdir, gdal_transform, wkt_projection, shape):
    """Write pcraster clone file to a tmpdir using gdal."""
    from osgeo import gdal

    gdal.AllRegister()
    driver1 = gdal.GetDriverByName("GTiff")
    driver2 = gdal.GetDriverByName("PCRaster")
    fn = join(tmpdir, "clone.map")
    # create temp tif file
    fn_temp = join(tmpdir, "clone.tif")
    TempDataset = driver1.Create(fn_temp, shape[1], shape[0], 1, gdal.GDT_Float32)
    TempDataset.SetGeoTransform(gdal_transform)
    if wkt_projection is not None:
        TempDataset.SetProjection(wkt_projection)
    # TODO set csr
    # copy to pcraster format
    driver2.CreateCopy(fn, TempDataset, 0)
    # close and cleanup
    TempDataset = None
    return fn


def write_map(
    data,
    raster_path,
    nodata,
    transform,
    crs=None,
    clone_path=None,
    pcr_vs="scalar",
    **kwargs,
):
    """Write pcraster map files using pcr.report functionality.

    A PCRaster clone map is written to a temporary directory if not provided.
    For PCRaster types see https://www.gdal.org/frmt_various.html#PCRaster

    Parameters
    ----------
    data : ndarray
        Raster data
    raster_path : str
        Path to output map
    nodata : int, float
        no data value
    transform : affine transform
        Two dimensional affine transform for 2D linear mapping
    clone_path : str, optional
        Path to PCRaster clone map, by default None
    pcr_vs : str, optional
        pcraster type, by default "scalar"
    **kwargs:
        not used in this function, mainly here for compatability reasons.
    crs:
        The coordinate reference system of the data.


    Raises
    ------
    ImportError
        pcraster package is required
    ValueError
        if invalid ldd
    """
    import tempfile

    import pcraster as pcr

    with tempfile.TemporaryDirectory() as tmpdir:
        # deal with pcr clone map
        if clone_path is None:
            clone_path = write_clone(
                tmpdir,
                gdal_transform=transform.to_gdal(),
                wkt_projection=None if crs is None else CRS.from_user_input(crs).wkt,
                shape=data.shape,
            )
        elif not isfile(clone_path):
            raise IOError(f'clone_path: "{clone_path}" does not exist')
        pcr.setclone(clone_path)
        if nodata is None and pcr_vs != "ldd":
            raise ValueError("nodata value required to write PCR map")
        # write to pcrmap
        if pcr_vs == "ldd":
            # if d8 convert to ldd
            data = data.astype(np.uint8)  # force dtype
            if core_d8.isvalid(data):
                data = core_conversion.d8_to_ldd(data)
            elif not core_ldd.isvalid(data):
                raise ValueError("LDD data not understood")
            mv = int(core_ldd._mv)
            ldd = pcr.numpy2pcr(pcr.Ldd, data.astype(int), mv)
            # make sure it is pcr sound
            # NOTE this should not be necessary
            pcrmap = pcr.lddrepair(ldd)
        elif pcr_vs == "bool":
            pcrmap = pcr.numpy2pcr(pcr.Boolean, data.astype(bool), np.bool_(nodata))
        elif pcr_vs == "scalar":
            pcrmap = pcr.numpy2pcr(pcr.Scalar, data.astype(float), float(nodata))
        elif pcr_vs == "ordinal":
            pcrmap = pcr.numpy2pcr(pcr.Ordinal, data.astype(int), int(nodata))
        elif pcr_vs == "nominal":
            pcrmap = pcr.numpy2pcr(pcr.Nominal, data.astype(int), int(nodata))
        pcr.report(pcrmap, raster_path)
        # set crs (pcrmap ignores this info from clone ??)
        if crs is not None:
            with rasterio.open(raster_path, "r+") as dst:
                dst.crs = crs


def read_staticmaps_pcr(
    root: Path | str, crs: int = 4326, obj: object = None, **kwargs
):
    """Read and staticmaps at <root/staticmaps> and parse to xarray."""
    da = None

    fn = join(root, "staticmaps", "*.map")
    fns = glob.glob(fn)
    if len(fns) == 0:
        logger.warning(f"No staticmaps found at {fn}")
        return
    _staticmaps = open_mfraster(fns, **kwargs)

    path = join(root, "staticmaps", "clim", "LAI*")
    if len(glob.glob(path)) > 0:
        ds_lai = open_mfraster(
            path, concat=True, concat_dim="time", logger=logger, **kwargs
        )
        lai_key = list(ds_lai.data_vars)[0]
        _staticmaps["LAI"] = ds_lai[lai_key]

    # reorganize c_0 etc maps
    da_c = []
    list_c = [v for v in _staticmaps if str(v).startswith("c_")]
    if len(list_c) > 0:
        for i, v in enumerate(list_c):
            da_c.append(_staticmaps[f"c_{i:d}"])
        da = xr.concat(
            da_c, pd.Index(np.arange(len(list_c), dtype=int), name="layer")
        ).transpose("layer", ...)
        _staticmaps = _staticmaps.drop_vars(list_c)
        _staticmaps["c"] = da

    _staticmaps = _staticmaps.rename({"x": "lon", "y": "lat"})

    # add maps to staticmaps
    if obj is not None:
        obj.set_grid(_staticmaps)
        if obj.crs is None:
            if crs is None:
                crs = 4326  # default to 4326
            obj.set_crs(crs)

    return _staticmaps


def write_staticmaps_pcr(
    staticmaps: xr.Dataset,
    root: Path | str,
):
    """Write staticmaps at <root/staticmaps> in PCRaster maps format."""
    root = os.path.join(root, "staticmaps")
    if not isdir(root):
        os.makedirs(root)
    profile_kwargs = {}
    mask = True
    ds_out = staticmaps
    if "LAI" in ds_out.data_vars:
        ds_out = ds_out.rename_vars({"LAI": "clim/LAI"})
    if "c" in ds_out.data_vars:
        for layer in ds_out["layer"]:
            ds_out[f"c_{layer.item():d}"] = ds_out["c"].sel(layer=layer)
            ds_out[f"c_{layer.item():d}"].raster.set_nodata(ds_out["c"].raster.nodata)
        ds_out = ds_out.drop_vars(["c", "layer"])
    logger.info("Writing (updated) staticmap files.")
    # add datatypes for maps with same basenames, e.g. wflow_gauges_grdc
    pcr_vs_map = PCR_VS_MAP.copy()
    for var_name in ds_out.raster.vars:
        base_name = "_".join(var_name.split("_")[:-1])  # clip _<postfix>
        if base_name in PCR_VS_MAP:
            pcr_vs_map.update({var_name: PCR_VS_MAP[base_name]})
    # ds_out.raster.to_mapstack(
    #     root=join(root, "staticmaps"),
    #     mask=True,
    #     driver="PCRaster",
    #     pcr_vs_map=pcr_vs_map,
    #     logger=logger,
    # )
    ext = GDAL_EXT_CODE_MAP.get("PCRaster")
    with tempfile.TemporaryDirectory() as tmpdir:
        clone_path = write_clone(
            tmpdir,
            gdal_transform=ds_out.raster.transform.to_gdal(),
            wkt_projection=None
            if ds_out.raster.crs is None
            else ds_out.raster.crs.to_wkt(),
            shape=ds_out.raster.shape,
        )
        profile_kwargs.update({"clone_path": clone_path})
        for var in ds_out.raster.vars:
            if "/" in var:
                # variables with in subfolders
                folders = "/".join(var.split("/")[:-1])
                if not isdir(join(root, folders)):
                    os.makedirs(join(root, folders))
                var0 = var.split("/")[-1]
                raster_path = join(root, folders, f"{var0}.{ext}")
            else:
                raster_path = join(root, f"{var}.{ext}")
            profile_kwargs.update({"pcr_vs": pcr_vs_map.get(var, "scalar")})
            da_out = ds_out[var].copy()
            for k in ["height", "width", "count", "transform"]:
                if k in profile_kwargs:
                    msg = f"{k} will be set based on the DataArray, remove the argument"
                    raise ValueError(msg)
                if "nodata" in profile_kwargs:
                    da_out.raster.set_nodata(profile_kwargs.pop("nodata"))
                nodata = da_out.raster.nodata
                if nodata is not None and not np.isnan(nodata):
                    da_out = da_out.fillna(nodata)
                elif nodata is None:
                    logger.warning(f"nodata value missing for {raster_path}")
                if mask and "mask" in da_out.coords and nodata is not None:
                    da_out = da_out.where(da_out.coords["mask"] != 0, nodata)
                if "crs" in profile_kwargs:
                    da_out.raster.set_crs(profile_kwargs.pop("crs"))
                # check dimensionality
                dim0 = da_out.raster.dim0
                count = 1
                if dim0 is not None:
                    count = da_out[dim0].size
                    da_out = da_out.sortby(dim0)
                # write
                for i in range(count):
                    if dim0:
                        bname = basename(raster_path).split(".")[0]
                        bname = f"{bname[:8]:8s}".replace(" ", "0")
                        raster_path = join(dirname(raster_path), f"{bname}.{i+1:03d}")
                        data = da_out.isel({dim0: i}).load().squeeze().data
                    else:
                        data = da_out.load().data
                    write_map(
                        data,
                        raster_path,
                        crs=da_out.raster.crs,
                        transform=da_out.raster.transform,
                        nodata=nodata,
                        **profile_kwargs,
                    )
