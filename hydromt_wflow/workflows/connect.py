"""Workflows to connect a wflow model to a 1D model."""

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from hydromt.gis import flw
from pyflwdir import FlwdirRaster
from scipy.ndimage import binary_erosion
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import snap

from hydromt_wflow.utils import planar_operation_in_utm

logger = logging.getLogger(f"hydromt.{__name__}")


__all__ = ["wflow_1dmodel_connection"]


def flwdir_mask_to_subbasins(
    flwdir_mask: FlwdirRaster,
    area_max: float,
    ds_model: xr.Dataset,
) -> xr.DataArray:
    """
    Split the common basin between wflow model and 1D model into subbasins.

    The basin is split into subbasins with an area equivalent to area_max [km2].
    """
    logger.info(
        f"Deriving lateral subbasins based on subbasin area threshold: {area_max} km2"
    )
    # calculate subbasins based on area_max threshold
    subbas, _ = flwdir_mask.subbasins_area(area_max)
    da_subbasins = xr.DataArray(
        data=subbas.astype(np.int32),
        dims=ds_model.raster.dims,
        coords=ds_model.raster.coords,
    )
    da_subbasins.raster.set_nodata(0)
    da_subbasins.raster.set_crs(ds_model.raster.crs)
    return da_subbasins


def derive_riv1d_edges(
    riv1d: gpd.GeoDataFrame, ds_model: xr.Dataset, flwdir: FlwdirRaster, **kwargs
) -> gpd.GeoDataFrame:
    """
    Derive the edges of riv1d and snap them to the wflow river.

    First derive the edges of the riv1d (start and end points of each linestring).
    Then snap them to the closest downstream wflow river cell using the
    :py:meth:`hydromt.gis.flw.gauge_map` method.
    Finally, clean up the edges that were snapped to the main river (eg tributary
    starting to close to the main river) by keeping only the most downstream edge
    and the most upstream one.

    Note: this function restricts the use to one river 1d and 1 wflow subcatchment as
    we only consider one downstream edge/edge point for 1d river.
    """
    # get the edges of the riv1d
    riv1d_edges = riv1d.geometry.apply(lambda x: Point(x.coords[0]))
    riv1d_edges = pd.concat(
        [riv1d_edges, riv1d.geometry.apply(lambda x: Point(x.coords[-1]))]
    )
    # find geometry that are unique in riv1d_edges
    riv1d_edges = gpd.GeoDataFrame(
        riv1d_edges[~riv1d_edges.duplicated(keep=False)],
        crs=riv1d.crs,
        geometry="geometry",
    )

    _, idxs, ids = flw.gauge_map(
        ds_model,
        xy=(riv1d_edges.geometry.x, riv1d_edges.geometry.y),
        stream=ds_model["rivmsk"].values,
        flwdir=flwdir,
        **kwargs,
    )
    points = gpd.points_from_xy(*ds_model.raster.idx_to_xy(idxs))
    riv1d_edges = gpd.GeoDataFrame(
        index=ids.astype(np.int32), geometry=points, crs=ds_model.raster.crs
    )

    # Some of the river edges may have been snapped to the main river
    # (eg tributary starting to close to the main river).
    riv1d_edges["uparea"] = ds_model["uparea"].raster.sample(riv1d_edges).values
    # Sort by upstream area descending
    riv1d_edges = riv1d_edges.sort_values(by="uparea", ascending=False)
    # Find the ID of the most downstream edge
    riv1d_edges_outlet_id = riv1d_edges.index[0]
    # Rasterize the riv1d_edges
    riv1d_edges_raster = ds_model.raster.rasterize(riv1d_edges, col_name="index")
    mask = flwdir.downstream(riv1d_edges_raster.values)
    # Find the downstream ID of each edge
    idxs_edge, _ = flwdir.snap(
        xy=(riv1d_edges.geometry.x, riv1d_edges.geometry.y),
        mask=mask,
    )
    edges_on_main_river = mask.flat[idxs_edge]

    # Keep only the edges that are not on the main river and the most downstream edge
    riv1d_edges = riv1d_edges[
        ~riv1d_edges.index.isin(edges_on_main_river)
        | (riv1d_edges.index == riv1d_edges_outlet_id)
    ]

    return riv1d_edges


def connect_subbasin_area(
    ds_model: xr.Dataset,
    riv1d: gpd.GeoDataFrame,
    flwdir: FlwdirRaster,
    area_max: float,
    include_river_boundaries: bool,
    **kwargs,
) -> tuple[FlwdirRaster, gpd.GeoDataFrame]:
    """
    Connect 1d river to Wflow by creating subbasins based on area_max threshold.

    First derive the river edges of the 1d river and snap them to the wflow river.
    Then derive the subbasins corresponding to the river edges.
    Then filter which subbasins are the upstream ones (tributaries) and which ones
    are the downstream ones (main river) and should be split into subbasins.

    The common basin between wflow and the 1d river is then split into subbasins of
    area_max size. Tributaries within the common basin that are larger than area_max are
    derived separately and will results into a tributary inflow point rather than a
    lateral inflow subbasin.

    This function results into a geodataframe of tributaries including the ones upstream
    of the common basin if include_river_boundaries is True and the ones within the
    common basin that have an area larger than area_max. The second output is the rest
    of the common basin where the tributaries have been masked out.
    """
    logger.info("Linking 1D river to wflow river")

    # 1. Derive the river edges / boundaries and snap to the wflow river
    riv1d_edges = derive_riv1d_edges(riv1d, ds_model, flwdir, **kwargs)

    # 2. Derive the subbasins corresponding to the river edges
    da_edges_subbas, _ = flw.basin_map(
        ds_model,
        flwdir=flwdir,
        xy=(riv1d_edges.geometry.x, riv1d_edges.geometry.y),
        ids=riv1d_edges.index,
    )
    da_edges_subbas.raster.set_crs(ds_model.raster.crs)
    # convert to gdf
    gdf_edges_subbas = da_edges_subbas.raster.vectorize()

    # 3. Filter which subbasins are the upstream ones (tributaries)
    # and which ones are the downstream ones (main river)
    # and should be split into subbasins
    # First intersect riv1d with gdf_edges_subbas
    rivmerge = (
        gpd.overlay(riv1d, gdf_edges_subbas)
        .explode(index_parts=True)
        .reset_index(drop=True)
    )
    # Compute len of river
    if rivmerge.crs.is_geographic:
        rivmerge["len"] = rivmerge.geometry.to_crs(3857).length
    else:
        rivmerge["len"] = rivmerge.geometry.length
    # Groupby value and sum length
    rivmerge = rivmerge.groupby("value")["len"].sum()
    # Select the subcatch where rivlength is more than 5times rivlen_avg
    riv_mask = ds_model["rivmsk"].values == 1
    rivlen_avg = ds_model["rivlen"].values[riv_mask].mean()
    subids = rivmerge.index[rivmerge > rivlen_avg * 5].values
    subcatch_to_split = gdf_edges_subbas[gdf_edges_subbas["value"].isin(subids)]
    subcatch_to_split = subcatch_to_split.to_crs(ds_model.raster.crs)
    da_subcatch_to_split = ds_model.raster.rasterize(
        subcatch_to_split, col_name="value"
    ).astype(np.float32)

    # First tributaries are the edges that are not included in the subcatch_to_split
    gdf_tributaries = riv1d_edges[~riv1d_edges.index.isin(subids)]

    # 4. Derive a mask of gdf_riv in the subcatch_to_split wflow rivers
    # compute flw_path to mask out the river included in dfm 1d network
    xy = (
        gdf_tributaries.geometry.x.values.tolist(),
        gdf_tributaries.geometry.y.values.tolist(),
    )

    # Get paths
    flowpaths, _ = flwdir.path(xy=xy, max_length=None, unit="m", direction="down")
    feats = flwdir.geofeatures(flowpaths)
    gdf_paths = gpd.GeoDataFrame.from_features(feats, crs=ds_model.raster.crs)
    gdf_paths.index = np.arange(1, len(gdf_paths) + 1)
    # Create a mask with subcatch id in subcatch_to_split and flowpath is nodata
    da_flwpaths = ds_model.raster.rasterize(gdf_paths)
    da_flwpaths = da_flwpaths.where(
        da_subcatch_to_split != da_subcatch_to_split.raster.nodata,
        da_flwpaths.raster.nodata,
    )

    # 5. Derive the tributaries
    # Find tributaries
    logger.info("Deriving tributaries")
    trib_msk = da_subcatch_to_split.where(
        da_flwpaths == da_flwpaths.raster.nodata, da_subcatch_to_split.raster.nodata
    )
    trib_msk = trib_msk.where(
        (trib_msk != trib_msk.raster.nodata)
        & (ds_model["uparea"] > area_max)
        & (flwdir.downstream(da_flwpaths) != da_flwpaths.raster.nodata),
        trib_msk.raster.nodata,
    )
    # Make sure we vectorize to single cells and not to polygons for adjacent cells
    trib_msk = trib_msk.stack(z=(trib_msk.raster.y_dim, trib_msk.raster.x_dim))
    nodata = trib_msk.raster.nodata
    trib_msk = trib_msk.where(trib_msk != nodata, drop=True)
    trib_msk.values = np.arange(1, len(trib_msk) + 1)
    trib_msk = trib_msk.unstack(fill_value=nodata)
    trib_msk = trib_msk.reindex_like(da_subcatch_to_split, fill_value=nodata).astype(
        np.int32
    )

    gdf_trib = trib_msk.raster.vectorize()
    # Test if there gdf_trib is empty
    if gdf_trib.empty:
        logger.info("No tributaries found")
        if not include_river_boundaries:
            gdf_tributaries = gpd.GeoDataFrame()
    else:
        gdf_trib = gdf_trib.to_crs(ds_model.raster.crs)
        centroid = planar_operation_in_utm(gdf_trib, lambda geom: geom.centroid)
        gdf_trib["geometry"] = centroid

        # Merge with gdf_tributary if include_river_boundaries
        # else only keep intersecting tributaries
        if include_river_boundaries:
            gdf_tributaries = pd.concat(
                [gdf_tributaries, gdf_trib.drop(["value"], axis=1)]
            )
        else:
            gdf_tributaries = gdf_trib.drop(["value"], axis=1)
        gdf_tributaries.index = np.arange(1, len(gdf_tributaries) + 1)

    # 6. Mask the tributaries out of the subatch_to_split map
    if not gdf_tributaries.empty:
        # Derive the tributary basin map
        da_trib_subbas, _ = flw.basin_map(
            ds_model,
            flwdir=flwdir,
            xy=(gdf_tributaries.geometry.x, gdf_tributaries.geometry.y),
        )
        da_trib_subbas.raster.set_crs(ds_model.raster.crs)
        # Mask tributaries
        da_flwdir_mask = ds_model["flwdir"].where(
            (da_subcatch_to_split != da_subcatch_to_split.raster.nodata)
            & (da_trib_subbas == da_trib_subbas.raster.nodata),
            ds_model["flwdir"].raster.nodata,
        )
    else:
        # Mask subcatch to split only
        da_flwdir_mask = ds_model["flwdir"].where(
            da_subcatch_to_split != da_subcatch_to_split.raster.nodata,
            ds_model["flwdir"].raster.nodata,
        )
    flwdir_mask = flw.flwdir_from_da(da_flwdir_mask)

    return (flwdir_mask, gdf_tributaries)


def connect_nodes(
    ds_model: xr.Dataset,
    gdf_riv: gpd.GeoDataFrame,
    flwdir: xr.DataArray,
    flwdir_mask: xr.DataArray,
    **kwargs,
) -> xr.DataArray:
    """Derive wflow subbasins for each 1d river node."""
    # Get the nodes from gdf_riv
    logger.info("Deriving subbasins based on 1D river nodes snapped to wflow river")
    # from multiline to line
    gdf_riv = gdf_riv.explode(ignore_index=True, index_parts=False)
    nodes = []
    for bi, branch in gdf_riv.iterrows():
        nodes.append([Point(branch.geometry.coords[0]), bi])  # start
        nodes.append([Point(branch.geometry.coords[-1]), bi])  # end
    gdf_nodes = gpd.GeoDataFrame(
        nodes, columns=["geometry", "river_id"], crs=gdf_riv.crs
    )
    # Drop duplicates geometry
    gdf_nodes = gdf_nodes[~gdf_nodes.geometry.duplicated(keep="first")]
    gdf_nodes.index = np.arange(1, len(gdf_nodes) + 1)
    # Snap the nodes to the wflow river
    _, idxs, ids = flw.gauge_map(
        ds_model,
        xy=(gdf_nodes.geometry.x, gdf_nodes.geometry.y),
        stream=ds_model["rivmsk"].values,
        flwdir=flwdir,
        **kwargs,
    )
    # Derive subbasins
    da_subbasins, _ = flw.basin_map(
        ds_model,
        flwdir=flwdir_mask,
        idxs=idxs,
        ids=ids,
        stream=ds_model["rivmsk"].values,
    )
    da_subbasins.raster.set_crs(ds_model.raster.crs)
    return da_subbasins


def buffer_basin_mask(
    basin_mask: xr.DataArray, basin_buffer_cells: int
) -> xr.DataArray:
    """Buffer the basin mask raster by a specified number of cells."""
    # Retain raster shape using binary erosion
    basin_mask_values = binary_erosion(
        input=basin_mask.values,
        structure=np.ones((3, 3)),
        iterations=basin_buffer_cells,
    ).astype(basin_mask.dtype)

    # Replace values with shrinked mask
    basin_mask = basin_mask.copy()
    basin_mask.values = basin_mask_values
    if np.sum(basin_mask_values) == 0:
        raise ValueError(
            f"Basin mask is empty after buffering with {basin_buffer_cells} cells. "
            "Consider using a smaller value for 'basin_buffer_cells'."
        )
    return basin_mask


def snap_river_endpoints(
    gdf_riv: gpd.GeoDataFrame,
    geom_snapping_tolerance: float,
) -> gpd.GeoDataFrame:
    """Snap river endpoints that are close to each other."""
    logger.info(
        f"Snapping river segments using geom_snapping_tolerance = "
        f"{geom_snapping_tolerance}."
    )

    # Extract endpoints of each line geometry
    endpoints = []
    for geom in gdf_riv.geometry:
        if isinstance(geom, LineString):
            endpoints.append(Point(geom.coords[0]))  # start point
            endpoints.append(Point(geom.coords[-1]))  # end point

    # Create a GeoSeries of endpoints
    endpoint_gs = gpd.GeoSeries(endpoints)

    # Snap each line geometry to nearby endpoints
    snapped_geometries = []
    snapped_count = 0
    for geom in gdf_riv.geometry:
        snapped_geom = geom
        for point in endpoint_gs:
            new_geom = snap(snapped_geom, point, geom_snapping_tolerance)
            if new_geom != snapped_geom:
                snapped_geom = new_geom
                snapped_count += 1
        snapped_geometries.append(snapped_geom)

    # Update the geometry in the GeoDataFrame
    gdf_riv.geometry = snapped_geometries
    logger.info(
        f"{snapped_count} river segments were snapped. "
        "Update `geom_snapping_tolerance` if this leads to issues."
    )

    return gdf_riv


def subbasin_preprocess_river_geometry(
    gdf_riv: gpd.GeoDataFrame,
    basin_buffer_cells: int,
    geom_snapping_tolerance: float,
) -> gpd.GeoDataFrame:
    """Preprocess the river geometry by snapping and merging segments."""
    # 0. Preprocess the river geometry
    # Check for MultiLineStrings
    if (gdf_riv.geom_type == "MultiLineString").any():
        logger.warning(
            "The river geometry contains one or more MultiLineStrings after "
            "clipping by the basin extent and will now be snapped. Consider increasing "
            f"'basin_buffer_cells' (currently {basin_buffer_cells} cells) if issues "
            "arise later in the process."
        )
        logger.info("Connecting rivers after basin clipping.")
        # we turn these into LineStrings again using its points
        gdf_riv["geometry"] = gdf_riv.geometry.apply(
            lambda geom: LineString(
                [coord for line in geom.geoms for coord in line.coords]
            )
            if isinstance(geom, MultiLineString)
            else geom
        )

        assert not (gdf_riv.geom_type == "MultiLineString").any()

    # Snapping
    if geom_snapping_tolerance > 0:
        gdf_riv = snap_river_endpoints(
            gdf_riv=gdf_riv,
            geom_snapping_tolerance=geom_snapping_tolerance,
        )

    if gdf_riv.empty:
        raise ValueError(
            "No river segments remaining in 'gdf_riv' after preprocessing. "
            "Consider using less strict requirements or check 'gdf_riv'."
        )
    return gdf_riv


def wflow_1dmodel_connection(
    gdf_riv: gpd.GeoDataFrame,
    ds_model: xr.Dataset,
    connection_method: str = "subbasin_area",
    area_max: float = 30.0,
    basin_buffer_cells: int = 0,
    geom_snapping_tolerance: float = 0.001,
    add_tributaries: bool = True,
    include_river_boundaries: bool = True,
    **kwargs,
) -> xr.Dataset:
    """
    Connect wflow to a 1D model by deriving linked subcatchs (and tributaries).

    There are two methods to connect models:

    - `subbasin_area`:
        creates subcatchments linked to the 1d river based on an
        area threshold (area_max) for the subbasin size. With this method, if a
        tributary is larger than the area_max, it will be connected to the 1d river
        directly.
    - `nodes`:
        subcatchments are derived based on the 1driver nodes (used as gauges
        locations). With this method, large tributaries can also be derived separately
        using the add_tributaries option and adding a area_max threshold for the
        tributaries.

    If `add_tributary` option is on, you can decide to include or exclude the upstream
    boundary of the 1d river as an additional tributary using the
    `include_river_boundaries` option.

    River edges or river nodes are snapped to the closest downstream wflow river
    cell using the :py:meth:`hydromt.gis.flw.gauge_map` method.

    Parameters
    ----------
    gdf_riv : gpd.GeoDataFrame
        River geometry.
    ds_model : xr.Dataset
        Model dataset with 'flwdir', 'rivmsk', 'rivlen', 'basins' and 'uparea'
        (optional).
    connection_method : str, default subbasin_area
        Method to connect wflow to the 1D model. Available methods are
        {'subbasin_area', 'nodes'}.
    area_max : float, default 10.0
        Maximum area [km2] of the subbasins to connect to the 1D model in km2 with
        connection_method **subbasin_area** or
        **nodes** with add_tributaries set to True.
    basin_buffer_cells : int, default 0
        Number of cells to use when clipping the 1d river geometry to the basin extent.
        This can be used to not include river geometries near the basin border.
    geom_snapping_tolerance : float, default 0.1
        Distance used to determine whether to snap parts of the 1d river geometry that
        are close to each other. This can be useful if some of the tributaries of the
        1D river are not perfectly connected to the main river.
    add_tributaries : bool, default True
        If True, derive tributaries for the subbasins larger than area_max. Always True
        for **subbasin_area** method.
    include_river_boundaries : bool, default True
        If True, include the upstream boundary(ies) of the 1d river as an additional
        tributary(ies).
    **kwargs
        Additional keyword arguments passed to the snapping method
        hydromt.gis.flw.gauge_map. See its documentation for more information.

    Returns
    -------
    ds_out: xr.Dataset
        Dataset with variables 'subcatch' for the subbasin map, 'subcatch_riv' for the
        subbasin map masked with river cells to be able to save river output with wflow
        and 'gauges' for the tributaries outflow locations (add_tributaries True or
        subbasin_area method).

    See Also
    --------
    hydromt.gis.flw.gauge_map
    """
    # Check variables in 'ds_model'
    dvars_model = ["flwdir", "rivmsk", "rivlen", "basins"]
    if not np.all([v in ds_model for v in dvars_model]):
        raise ValueError(f"One or more variables missing from ds_model: {dvars_model}")
    # Derive flwdir
    flwdir = flw.flwdir_from_da(ds_model["flwdir"])
    # Check for uparea and derive on the fly if needed
    if "uparea" not in ds_model:
        logger.info(
            "upstream area map not found in 'ds_model' and will be derived on the fly."
        )
        uparea = flwdir.upstream_area(unit="km2")
        ds_model["uparea"] = xr.Variable(ds_model.raster.dims, uparea)

    # Check that the max_area is larger than the wflow river threshold
    # Get from attrs if available for newer wflow models built with hydromt
    riv_upa = ds_model["rivmsk"].attrs.get("river_upa", None)
    if riv_upa is None:
        # Derive from the uparea and rivmsk
        riv_upa = xr.where(ds_model["rivmsk"] > 0, ds_model["uparea"], np.nan)
        riv_upa = float(riv_upa.min())
    if area_max < riv_upa:
        new_area_max = np.ceil(riv_upa / 0.5) * 0.5
        # Add 5% buffer to avoid getting subbasins slightly smaller than riv_upa
        # and therefore which won't have river cells
        new_area_max = new_area_max * 1.05
        logger.warning(
            f"The area_max {area_max} is smaller than the minimum upstream area of "
            f"the wflow river {riv_upa} which means tributaries will "
            "not be connected to the wflow river. Changing and setting area_max to "
            f"{new_area_max} km2. "
            f"To keep {area_max} km2 threshold, please update the wflow model to "
            "include more detailed rivers."
        )
        area_max = new_area_max

    # Reproject
    if gdf_riv.crs != ds_model.raster.crs:
        gdf_riv = gdf_riv.to_crs(ds_model.raster.crs)
    # merge multilinestrings in gdf_riv to linestrings to ease processing later on
    if any(gdf_riv.geometry.apply(lambda geom: isinstance(geom, MultiLineString))):
        logger.debug(
            "'gdf_riv' contains MultiLineStrings which will be converted into"
            "LineStrings."
        )
        gdf_riv = gdf_riv.explode(index_parts=True).reset_index(drop=True)

    # Basin mask and clip river geometry
    basin_mask = ds_model["basins"]
    if basin_buffer_cells > 0:
        basin_mask = buffer_basin_mask(
            basin_mask=basin_mask, basin_buffer_cells=basin_buffer_cells
        )
    gdf_riv = gdf_riv.clip(basin_mask.raster.vectorize())

    if connection_method == "subbasin_area" or add_tributaries == True:
        riv1d = subbasin_preprocess_river_geometry(
            gdf_riv=gdf_riv,
            basin_buffer_cells=basin_buffer_cells,
            geom_snapping_tolerance=geom_snapping_tolerance,
        )

        # Obtain flwdir_mask from 'riv1d'
        flwdir_mask, gdf_tributaries = connect_subbasin_area(
            ds_model=ds_model,
            riv1d=riv1d,
            flwdir=flwdir,
            area_max=area_max,
            include_river_boundaries=include_river_boundaries,
            **kwargs,
        )
    else:
        # Use the entire model as flwdir_mask
        flwdir_mask, gdf_tributaries = flwdir, gpd.GeoDataFrame()

    # Derive subbasins
    if connection_method == "subbasin_area":
        da_subbasins = flwdir_mask_to_subbasins(
            flwdir_mask=flwdir_mask, area_max=area_max, ds_model=ds_model
        )
    elif connection_method == "nodes":
        da_subbasins = connect_nodes(
            ds_model=ds_model,
            gdf_riv=gdf_riv,
            flwdir=flwdir,
            flwdir_mask=flwdir_mask,
            **kwargs,
        )

    # Check if the output is not covering the entire model domain
    valid_subbasins = (da_subbasins != da_subbasins.raster.nodata).sum()
    valid_basinmask = (ds_model["flwdir"] != ds_model["flwdir"].raster.nodata).sum()

    if valid_subbasins >= valid_basinmask:
        raise ValueError(
            "The entire model domain is assigned to subbasins. If this should not be "
            "the case, try setting other parameter values or modifying 'gdf_riv'."
        )

    # Check if all subbasins have a wflow river cell
    gdf_subcatch = da_subbasins.raster.vectorize()
    has_river = ds_model["rivmsk"].raster.zonal_stats(gdf_subcatch, stats="count")
    # Count how many subcatchments have no river
    no_river = np.sum(has_river["rivmsk_count"].values == 0)
    if no_river > 0:
        raise ValueError(
            f"{no_river} subbasin(s) do not contain a wflow river cell. "
            "Consider refining the river network in the Wflow model or increasing "
            f"the area_max threshold (currently {area_max} km2)."
        )

    da_subbasins.name = "subcatch"
    ds_out = da_subbasins.to_dataset()

    # Subcatchment map for river cells only (to be able to save river outputs in wflow)
    ds_out["subcatch_riv"] = da_subbasins.where(
        ds_model["rivmsk"] > 0, da_subbasins.raster.nodata
    )

    # Add tributaries
    if not gdf_tributaries.empty:
        ds_out["gauges"] = ds_out.raster.rasterize(
            gdf_tributaries, col_name="index", nodata=0
        )

    return ds_out
