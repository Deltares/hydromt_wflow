"""Nature-Based Solutions (NBS) workflows for Wflow plugin."""

import logging

import numpy as np
import pyflwdir
import xarray as xr
from hydromt.gis import flw
from hydromt.gis.raster_utils import full_like

logger = logging.getLogger(f"hydromt.{__name__}")


__all__ = [
    "nbs_suitability_from_thresholds",
    "ponding_level_from_suitability",
]


def _compute_landslope(elevtn: xr.DataArray) -> xr.DataArray:
    """Compute land slope from elevation data."""
    crs = elevtn.raster.crs
    nodata = elevtn.raster.nodata
    slope = xr.DataArray(
        name="lndslp",
        data=pyflwdir.dem.slope(
            elevtn=elevtn.values,
            nodata=nodata,
            latlon=crs is not None and crs.to_epsg() == 4326,
            transform=elevtn.raster.transform,
        ),
        coords=elevtn.raster.coords,
        dims=elevtn.raster.dims,
        attrs={
            "long_name": "land slope",
            "unit": "m.m-1",
            "_FillValue": nodata,
        },
    )
    # Mask possible negative values for slope
    slope = slope.where(slope >= 0, 0)

    return slope


def _compute_hand(
    hydro_data: xr.Dataset,
    river_upa: float = 30,
) -> xr.DataArray:
    """Compute hand from elevation and flow direction data."""
    # Get flwdir
    if "flwdir" not in hydro_data:
        raise ValueError("Flow direction data ('flwdir') is required to compute hand.")
    flwdir = flw.flwdir_from_da(hydro_data["flwdir"], ftype="infer")

    # Get drain mask based on river_upa
    if "uparea" not in hydro_data:
        uparea = flwdir.upstream_area("km2")
        attrs = {"_FillValue": -9999, "unit": "km2"}
        uparea = xr.DataArray(
            name="uparea",
            data=uparea,
            coords=hydro_data.coords,
            dims=hydro_data.raster.dims,
            attrs=attrs,
        )
    else:
        uparea = hydro_data["uparea"]

    # Compute hand
    hand = flwdir.hand(
        drain=uparea.values > river_upa,
        elevtn=hydro_data["elevtn"].values,
    )
    attrs = {
        "_FillValue": -9999,
        "long_name": "height above nearest drainage",
        "units": "m",
    }

    hand = xr.DataArray(
        name="hand",
        data=hand,
        coords=hydro_data.coords,
        dims=hydro_data.raster.dims,
        attrs=attrs,
    )

    return hand


def _compute_nbs_coverage(
    nbs_map: xr.DataArray,
    basin: xr.DataArray,
    min_value: float = 0.0,
):
    """
    Compute NBS coverage of the basin.

    Parameters
    ----------
    nbs_map : xr.DataArray
        Map with NBS suitability or measures (e.g. ponding level).
    basin : xr.DataArray
        Map of basin areas (non-NaN values indicate basin area).
    min_value : float, optional
        Minimum value to consider as NBS coverage, by default 0.0.
    """
    # Reproject basin to nbs_map grid if needed
    basin_mask = basin.raster.reproject_like(nbs_map, method="nearest")

    basin_mask = basin_mask.where(basin_mask != basin_mask.raster.nodata, -9999)
    basin_mask.raster.set_nodata(-9999)

    area = basin_mask.raster.area_grid()
    area_basin = area.where(basin_mask == 1, np.nan).sum().values / 1e6
    area_nbs = round(
        area.where((nbs_map > min_value) & (basin_mask == 1), np.nan).sum().values / 1e6
    )

    area_nbs_perc = round(area_nbs / area_basin * 100, 2)
    logger.info(
        f"NBS coverage: {round(area_nbs, 2)} km2 ({area_nbs_perc} % of basin area)"
    )


def _landuse_suitability(
    landuse: xr.DataArray, lulc_classes: list[int]
) -> xr.DataArray:
    """Derive land use suitability based on specified land use classes."""
    # need -1 as nodata and not 0 for mode resampling later on
    suitability = full_like(landuse, nodata=-1, fill_value=0, dtype=np.int8)
    for lulc_class in lulc_classes:
        suitability = suitability.where(landuse != lulc_class, 1)
    return suitability


def _hydrography_suitability(
    hydro_data: xr.Dataset,
    elevtn_range: tuple[float, float] | None = None,
    slope_range: tuple[float, float] | None = None,
    hand_range: tuple[float, float] | None = None,
    river_upa: float = 30,
) -> xr.DataArray:
    """Derive hydrography suitability based on elevation, slope, and hand criteria."""
    suitability_hydro = full_like(
        hydro_data["elevtn"], nodata=-1, fill_value=1, dtype=np.int8
    )
    if elevtn_range is not None:
        suitability_hydro = xr.where(
            (
                (hydro_data["elevtn"] >= elevtn_range[0])
                & (hydro_data["elevtn"] <= elevtn_range[1])
                & (suitability_hydro == 1)
            ),
            1,
            0,
        )
    if slope_range is not None:
        if "lndslp" not in hydro_data:
            logger.info("Slope data ('lndslp') not found. Derive from elevation map.")
            hydro_data["lndslp"] = _compute_landslope(hydro_data["elevtn"])

        suitability_hydro = xr.where(
            (
                (hydro_data["lndslp"] >= slope_range[0])
                & (hydro_data["lndslp"] <= slope_range[1])
                & (suitability_hydro == 1)
            ),
            1,
            0,
        )
    if hand_range is not None:
        if "hand" not in hydro_data:
            logger.info(
                "Hand data ('hand') not found. "
                "Derive from elevation and flow direction maps."
            )
            hydro_data["hand"] = _compute_hand(hydro_data, river_upa=river_upa)
        suitability_hydro = xr.where(
            (
                (hydro_data["hand"] >= hand_range[0])
                & (hydro_data["hand"] <= hand_range[1])
                & (suitability_hydro == 1)
            ),
            1,
            0,
        )
    suitability_hydro.raster.set_crs(hydro_data.raster.crs)
    suitability_hydro.raster.set_nodata(-1)
    suitability_hydro = suitability_hydro.astype(np.int8)

    return suitability_hydro


def nbs_suitability_from_thresholds(
    landuse: xr.DataArray | None = None,
    hydro_data: xr.Dataset | None = None,
    lulc_classes: list[int] | None = None,
    elevtn_range: tuple[float, float] | None = None,
    slope_range: tuple[float, float] | None = None,
    hand_range: tuple[float, float] | None = None,
    river_upa: float = 30,
    basin_mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Derive NBS suitability based on threshold criteria.

    Parameters
    ----------
    landuse : xr.DataArray, optional
        Land use/cover map with a "landuse" variable, by default None.
    hydro_data : xr.Dataset, optional
        Dataset containing hydrography data arrays ('elevtn', 'lndslp', 'hand'),
        by default None. Requires `flwdir` and optionally `uparea` for hand calculation.
    lulc_classes : list[int], optional
        List of land use/cover classes to consider for NBS suitability,
        by default None.
    elevtn_range : tuple[float, float], optional
        Elevation range for NBS suitability, by default None.
    slope_range : tuple[float, float], optional
        Slope range for NBS suitability, by default None.
    hand_range : tuple[float, float], optional
        Height above nearest drainage range for NBS suitability, by default None.
    river_upa : float, optional
        Minimum upstream area threshold for the river map [km2]. By default 30. Used
        to define drains for hand calculation.

    Returns
    -------
    xr.DataArray
        Boolean array indicating NBS suitability based on the specified criteria.
    """
    # Start with masking landuse
    if landuse is not None and lulc_classes is not None:
        suitability_lulc = _landuse_suitability(landuse, lulc_classes)
    else:
        suitability_lulc = None

    # Now elevation related criteria
    if any([elevtn_range, slope_range, hand_range]) and hydro_data is not None:
        suitability_hydro = _hydrography_suitability(
            hydro_data,
            elevtn_range=elevtn_range,
            slope_range=slope_range,
            hand_range=hand_range,
            river_upa=river_upa,
        )
    else:
        suitability_hydro = None

    # Combine suitability criteria
    if suitability_lulc is not None and suitability_hydro is not None:
        # Reproject landuse suitability to hydro data grid if needed
        suitability_lulc = suitability_lulc.raster.reproject_like(
            suitability_hydro, method="mode"
        )
        suitability = ((suitability_lulc == 1) & (suitability_hydro == 1)).astype(
            np.int8
        )
    elif suitability_lulc is not None:
        suitability = suitability_lulc
    elif suitability_hydro is not None:
        suitability = suitability_hydro
    else:
        raise ValueError(
            "At least one of landuse or hydro_data with criteria must be provided."
        )

    suitability.name = "nbs_suitability"
    suitability.attrs = {
        "long_name": (
            f"NBS suitability based on threshold criteria: "
            f"elevation range {elevtn_range}, "
            f"slope range {slope_range}, "
            f"HAND range {hand_range}, "
            f"land use classes {lulc_classes}."
        )
    }
    suitability.raster.set_crs(hydro_data.raster.crs)
    suitability.raster.set_nodata(0)

    if basin_mask is not None:
        logger.info("Calculating NBS suitability coverage at high resolution.")
        _compute_nbs_coverage(suitability, basin_mask, min_value=0)

    return suitability


def ponding_level_from_suitability(
    suitability: xr.DataArray,
    ds_like: xr.Dataset,
    pond_level: float = 0.2,
    basin_mask_name: str = "basins",
) -> xr.DataArray:
    """Assign ponding level based on NBS suitability map.

    Parameters
    ----------
    suitability : xr.DataArray
        Boolean array indicating NBS suitability.
    ds_like : xr.Dataset
        Wflow staticmaps for reprojection.
    pond_level : float, optional
        Ponding level to assign to suitable areas, by default 0.2

    Returns
    -------
    xr.DataArray
        Array with assigned ponding levels based on suitability.
    """
    # Create ponding level map based on suitability
    ponding_level = xr.where(suitability == 1, pond_level, 0.0)
    ponding_level.name = "ponding_level"
    ponding_level.attrs["long_name"] = "ponding level"
    if "long_name" in suitability.attrs:
        ponding_level.attrs["description"] = suitability.attrs["long_name"]
    ponding_level.raster.set_crs(ds_like.raster.crs)
    ponding_level.raster.set_nodata(-9999)

    # Reproject to match Wflow staticmaps grid
    ponding_level = ponding_level.raster.reproject_like(ds_like, method="average")

    # Ponding coverage
    # calculate extent of the area where changes occur
    if basin_mask_name in ds_like:
        logger.info("Calculating ponding coverage at model resolution.")
        basin_mask = ds_like[basin_mask_name].copy()
        _compute_nbs_coverage(ponding_level, basin_mask, min_value=0.0)

    return ponding_level
