"""Workflow functions for setting up wflow basemaps."""

import logging

import geopandas as gpd
import numpy as np
import pyflwdir
import xarray as xr
from hydromt.gis import _raster_utils, flw
from hydromt.model.processes import basin_mask
from pyflwdir import core_conversion, core_d8, core_ldd

logger = logging.getLogger(f"hydromt.{__name__}")

__all__ = [
    "convert_flow_direction",
    "hydrography",
    "prep_raw_basemaps_data",
    "topography",
]


def prep_raw_basemaps_data(
    hydro_data: xr.Dataset,
    region: gpd.GeoDataFrame,
    res: float | int,
    basin_index_data: gpd.GeoDataFrame | None = None,
    derive_region: bool = True,
) -> tuple[xr.Dataset, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """First stage of prepping the basemaps data.

    Parameters
    ----------
    hydro_data : xr.Dataset
        _description_
    region : gpd.GeoDataFrame
        _description_
    res : float | int
        _description_
    basin_index_data : gpd.GeoDataFrame, optional
        Basin indices
    derive_region : bool, optional
        _description_, by default True

    Returns
    -------
    tuple[xr.Dataset, gpd.GeoDataFrame]
        _description_
    """
    # Check on resolution (degree vs meter) depending on ds_org res/crs
    scale_ratio = int(np.round(res / hydro_data.raster.res[0]))
    if scale_ratio < 1:
        raise ValueError(
            f"The model resolution {res} should be \
larger than the hydrography resolution {hydro_data.raster.res[0]}"
        )
    if hydro_data.raster.crs.is_geographic:
        if res > 1:  # 111 km
            raise ValueError(
                f"The model resolution {res} should be smaller than 1 degree \
(111km) for geographic coordinate systems. "
                "Make sure you provided res in degree rather than in meters."
            )

    # Set the geom to the current region in case derive_region is false
    geom = region
    # xy are pits in the terrain
    xy = None
    # Derive the region based on the
    if derive_region:
        geom, xy = basin_mask.get_basin_geometry(
            ds=hydro_data.copy(),
            kind="subbasin",
            basin_index=basin_index_data,
            geom=region,
        )

    # Clip the data based on the region (whether derived or not)
    hydro_data = hydro_data.raster.clip_geom(geom, align=res, buffer=10, mask=True)

    return hydro_data, geom, xy


def convert_flow_direction(
    hydro_grid: xr.Dataset,
) -> xr.Dataset:
    """Convert the flow directions.

    From d8 to ldd if the data is currently in d8.

    Parameters
    ----------
    hydro_grid : xr.Dataset
        _description_

    Returns
    -------
    xr.Dataset
        _description_
    """
    # Take the flow direction
    flwdir_data = hydro_grid["flwdir"].values.astype(np.uint8)  # force dtype

    # if d8 convert to ldd
    if core_d8.isvalid(flwdir_data):
        data = core_conversion.d8_to_ldd(flwdir_data)
        da_flwdir = xr.DataArray(
            name="flwdir",
            data=data,
            coords=hydro_grid.raster.coords,
            dims=hydro_grid.raster.dims,
            attrs=dict(
                long_name="ldd flow direction",
                _FillValue=core_ldd._mv,
            ),
        )
        hydro_grid["flwdir"] = da_flwdir

    return hydro_grid


def hydrography(
    hydro_data: xr.Dataset,
    res: float,
    xy: gpd.GeoDataFrame | None = None,
    upscale_method: str = "ihu",
    ftype: str = "infer",
):
    """Return hydrography maps (see list below) and FlwdirRaster object.

    Based on gridded flow direction and elevation data input.

    The output maps are:

    - flwdir : flow direction [-]
    - basins : basin map [-]
    - uparea : upstream area [km2]
    - strord : stream order [-]

    If the resolution is lower than the source resolution, the flow direction data is
    upscaled and river length and slope are based on subgrid flow paths and
    the following maps are added:

    - subare : contributing area to each subgrid outlet pixel \
(unit catchment area) [km2]
    - subelv : elevation at subgrid outlet pixel [m+REF]

    Parameters
    ----------
    hydro_data : xarray.DataArray
        Dataset containing gridded flow direction and elevation data.
    res : float
        Output resolution
    xy : geopandas.GeoDataFrame, optional
        Subbasin pits. Only required when upscaling a subbasin.
    upscale_method : {'ihu', 'eam', 'dmm'}
        Upscaling method for flow direction data, by default 'ihu', see [1]_
    ftype : str, optional
        Name of flow direction type, chose from either 'd8', 'ldd', 'nextxy',
        'nextidx' or 'infer'. Will be inferred from the data if 'infer',
        by default is 'infer'

    Returns
    -------
    ds_out : xarray.DataArray
        Dataset containing gridded hydrography data
    flwdir_out : pyflwdir.FlwdirRaster
        Flow direction raster object.

    References
    ----------
    .. [1] Eilander et al. (2021). A hydrography upscaling method for scale-invariant \
parametrization of distributed hydrological models.
           Hydrology and Earth System Sciences, 25(9), 5287–5313. \
https://doi.org/10.5194/hess-25-5287-2021

    See Also
    --------
    pyflwdir.FlwdirRaster.upscale_flwdir

    """
    # TODO add check if flwdir in ds, calculate if not
    flwdir = None
    basins = None
    outidx = None
    flwdir_name: str = "flwdir"
    uparea_name: str = "uparea"
    basins_name: str = "basins"
    strord_name: str = "strord"
    if "mask" not in hydro_data.coords and xy is None:
        hydro_data.coords["mask"] = xr.Variable(
            dims=hydro_data.raster.dims,
            data=np.ones(hydro_data.raster.shape, dtype=bool),
        )
    elif "mask" not in hydro_data.coords:
        # NOTE if no subbasin mask is provided calculate it here
        logger.debug(f"Delineate {xy[0].size} subbasin(s).")
        flwdir = flw.flwdir_from_da(hydro_data[flwdir_name], ftype=ftype)
        basins = flwdir.basins(xy=xy).astype(np.int32)
        hydro_data.coords["mask"].data = basins != 0
        if not np.any(hydro_data.coords["mask"]):
            raise ValueError("Delineating subbasins not successful.")
    elif xy is not None:
        # NOTE: this mask is passed on from get_basin_geometry method
        logger.debug("Mask in dataset assumed to represent subbasins.")
    ncells = np.sum(hydro_data["mask"].values)
    scale_ratio = int(np.round(res / hydro_data.raster.res[0]))

    if ncells < 4:
        raise ValueError(
            "(Sub)basin at original resolution should at least consist of two cells on "
            f"each axis and the total number of cells is {ncells}. "
            "Consider using a larger domain or higher spatial resolution. "
            "For subbasin models, consider a (higher) threshold on for example "
            "upstream area or stream order to snap the outlet."
        )
    elif ncells < 100 and scale_ratio > 1:
        logger.warning(
            f"(Sub)basin at original resolution is small and has {ncells} cells. "
            "This may results in errors later when upscaling flow directions. "
            "If so, consider using a larger domain or higher spatial resolution. "
            "For subbasin models, consider a (higher) threshold on for example "
            "upstream area or stream order to snap the outlet."
        )
    else:
        logger.debug(f"(Sub)basin at original resolution has {ncells} cells.")

    if scale_ratio > 1:  # upscale flwdir
        if flwdir is None:
            # NOTE initialize with mask is FALSE
            flwdir = flw.flwdir_from_da(
                hydro_data[flwdir_name], ftype=ftype, mask=False
            )
        if xy is not None:
            logger.debug("Burn subbasin outlet in upstream area data.")
            if isinstance(xy, gpd.GeoDataFrame):
                assert xy.crs == hydro_data.raster.crs
                xy = xy.geometry.x, xy.geometry.y
            idxs_pit = flwdir.index(*xy)
            flwdir.add_pits(idxs=idxs_pit)
            uparea = hydro_data[uparea_name].values
            uparea.flat[idxs_pit] = uparea.max() + 1.0
            hydro_data[uparea_name].data = uparea
        logger.info(
            f"Upscale flow direction data: {scale_ratio:d}x, {upscale_method} method."
        )
        da_flw, flwdir_out = flw.upscale_flwdir(
            hydro_data,
            flwdir=flwdir,
            scale_ratio=scale_ratio,
            method=upscale_method,
            uparea_name=uparea_name,
            flwdir_name=flwdir_name,
        )
        da_flw.raster.set_crs(hydro_data.raster.crs)
        # make sure x_out and y_out get saved
        ds_out = da_flw.to_dataset().reset_coords(["x_out", "y_out"])
        dims = ds_out.raster.dims
        # find pits within basin mask
        idxs_pit0 = flwdir_out.idxs_pit
        outlon = ds_out["x_out"].values.ravel()
        outlat = ds_out["y_out"].values.ravel()
        sel = {
            hydro_data.raster.x_dim: xr.Variable("yx", outlon[idxs_pit0]),
            hydro_data.raster.y_dim: xr.Variable("yx", outlat[idxs_pit0]),
        }
        outbas_pit = hydro_data.coords["mask"].sel(sel, method="nearest").values
        # derive basins
        if np.any(outbas_pit != 0):
            idxs_pit = idxs_pit0[outbas_pit != 0]
            basins = flwdir_out.basins(idxs=idxs_pit).astype(np.int32)
            ds_out.coords["mask"] = xr.Variable(
                dims=ds_out.raster.dims, data=basins != 0, attrs=dict(_FillValue=0)
            )
        else:
            # This is a patch for basins which are clipped based on bbox or wrong geom
            mask_int = hydro_data["mask"].astype(np.int8)
            mask_int.raster.set_nodata(-1)  # change nodata value
            ds_out.coords["mask"] = mask_int.raster.reproject_like(
                da_flw, method="nearest"
            ).astype(bool)
            basins = ds_out["mask"].values.astype(np.int32)
            logger.warning(
                "The basin delineation might be wrong as no original resolution outlets"
                " are found in the upscaled map."
            )
        ds_out[basins_name] = xr.Variable(dims, basins, attrs=dict(_FillValue=0))
        # calculate upstream area using subgrid ucat cell areas
        outidx = np.where(
            ds_out["mask"], da_flw.coords["idx_out"].values, flwdir_out._mv
        )
        subare = flwdir.ucat_area(outidx, unit="km2")[1]
        uparea = flwdir_out.accuflux(subare)
        attrs = dict(_FillValue=-9999, unit="km2")
        ds_out[uparea_name] = xr.Variable(dims, uparea, attrs=attrs).astype(np.float32)
        # NOTE: subgrid cella area is currently not used in wflow
        ds_out["subare"] = xr.Variable(dims, subare, attrs=attrs).astype(np.float32)
        if "elevtn" in hydro_data:
            subelv = hydro_data["elevtn"].values.flat[outidx]
            subelv = np.where(outidx >= 0, subelv, -9999)
            attrs = dict(_FillValue=-9999, unit="m+REF")
            ds_out["subelv"] = xr.Variable(dims, subelv, attrs=attrs)
        # initiate masked flow dir
        flwdir_out = flw.flwdir_from_da(
            ds_out[flwdir_name], ftype=flwdir.ftype, mask=True
        )
    else:
        # NO upscaling : source resolution equals target resolution
        # NOTE (re-)initialize with mask is TRUE
        ftype = flwdir.ftype if flwdir is not None and ftype == "infer" else ftype
        flwdir = flw.flwdir_from_da(hydro_data[flwdir_name], ftype=ftype, mask=True)
        flwdir_out = flwdir
        ds_out = xr.DataArray(
            name=flwdir_name,
            data=flwdir_out.to_array(),
            coords=hydro_data.raster.coords,
            dims=hydro_data.raster.dims,
            attrs=dict(
                long_name=f"{ftype} flow direction",
                _FillValue=flwdir_out._core._mv,
            ),
        ).to_dataset()
        dims = ds_out.raster.dims
        ds_out.coords["mask"] = xr.Variable(
            dims=dims, data=flwdir_out.mask.reshape(flwdir_out.shape)
        )
        # copy data variables from source if available
        for dvar in [basins_name, uparea_name, strord_name]:
            if dvar in hydro_data.data_vars:
                ds_out[dvar] = xr.where(
                    ds_out["mask"],
                    hydro_data[dvar],
                    hydro_data[dvar].dtype.type(hydro_data[dvar].raster.nodata),
                )
                ds_out[dvar].attrs.update(hydro_data[dvar].attrs)
        # basins
        if basins_name not in ds_out.data_vars:
            if basins is None:
                basins = flwdir_out.basins(idxs=flwdir_out.idxs_pit).astype(np.int32)
            ds_out[basins_name] = xr.Variable(dims, basins, attrs=dict(_FillValue=0))
        else:
            # make sure dtype in ds_out is np.int32
            ds_out[basins_name] = ds_out[basins_name].astype(np.int32)
        # upstream area
        if uparea_name not in ds_out.data_vars:
            uparea = flwdir_out.upstream_area("km2")  # km2
            attrs = dict(_FillValue=-9999, unit="km2")
            ds_out[uparea_name] = xr.Variable(dims, uparea, attrs=attrs).astype(
                np.float32
            )
        # cell area
        # NOTE: subgrid cella area is currently not used in wflow
        ys, xs = hydro_data.raster.ycoords.values, hydro_data.raster.xcoords.values
        subare = _raster_utils._reggrid_area(ys, xs) / 1e6  # km2
        attrs = dict(_FillValue=-9999, unit="km2")
        ds_out["subare"] = xr.Variable(dims, subare, attrs=attrs).astype(np.float32)
    # logging
    npits = flwdir_out.idxs_pit.size
    xy_pit = flwdir_out.xy(flwdir_out.idxs_pit[:5])
    xy_pit_str = ", ".join([f"({x:.5f},{y:.5f})" for x, y in zip(*xy_pit)])
    # stream order
    if strord_name not in ds_out.data_vars:
        logger.debug("Derive stream order.")
        strord = flwdir_out.stream_order()
        ds_out[strord_name] = xr.Variable(dims, strord)
        ds_out[strord_name].raster.set_nodata(255)

    # clip to basin extent
    ds_out = ds_out.raster.clip_mask(da_mask=ds_out[basins_name])

    ds_out.raster.set_crs(hydro_data.raster.crs)
    logger.debug(
        f"Map shape: {ds_out.raster.shape}; active cells: {flwdir_out.ncells}."
    )
    logger.debug(f"Outlet coordinates ({len(xy_pit[0])}/{npits}): {xy_pit_str}.")
    if np.any(np.asarray(ds_out.raster.shape) == 1):
        raise ValueError(
            "The output extent at model resolution should at least consist of two "
            "cells on each axis. Consider using a larger domain or higher spatial "
            "resolution. For subbasin models, consider a (higher) threshold to snap "
            "the outlet."
        )
    return ds_out, flwdir_out


def topography(
    ds: xr.Dataset,
    ds_like: xr.Dataset,
    elevtn_name: str = "elevtn",
    lndslp_name: str = "lndslp",
    method: str = "average",
):
    """Return topography maps (see list below) at model resolution.

    Based on gridded elevation data input.

    The following topography maps are calculated:

    - elevtn : average elevation [m]
    - lndslp : average land surface slope [m/m]

    Parameters
    ----------
    ds : xarray.DataArray
        Dataset containing gridded flow direction and elevation data.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    elevtn_name, lndslp_name : str, optional
        Name of elevation [m] and land surface slope [m/m] variables in ds
    method: str, optional
        Resample method used to reproject the input data, by default "average"

    Returns
    -------
    ds_out : xarray.DataArray
        Dataset containing gridded hydrography data
    flwdir_out : pyflwdir.FlwdirRaster
        Flow direction raster object.

    See Also
    --------
    pyflwdir.dem.slope
    """
    if lndslp_name not in ds.data_vars:
        logger.debug(f"Slope map {lndslp_name} not found: derive from elevation map.")
        crs = ds[elevtn_name].raster.crs
        nodata = ds[elevtn_name].raster.nodata
        ds[lndslp_name] = xr.Variable(
            dims=ds.raster.dims,
            data=pyflwdir.dem.slope(
                elevtn=ds[elevtn_name].values,
                nodata=nodata,
                latlon=crs is not None and crs.to_epsg() == 4326,
                transform=ds[elevtn_name].raster.transform,
            ),
        )
        ds[lndslp_name].raster.set_nodata(nodata)
    # clip or reproject if non-identical grids
    ds_out = ds[[elevtn_name, lndslp_name]].raster.reproject_like(ds_like, method)
    ds_out[elevtn_name].attrs.update(unit="m")
    ds_out[lndslp_name].attrs.update(unit="m.m-1")
    ds_out["lndslp"] = np.maximum(ds_out["lndslp"], 0.0)
    return ds_out
