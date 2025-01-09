"""Workflows to connect a wflow model to a 1D model."""

import logging

import geopandas as gpd
import hydromt
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

logger = logging.getLogger(__name__)


__all__ = ["wflow_1dmodel_connection"]


def wflow_1dmodel_connection(
    gdf_riv: gpd.GeoDataFrame,
    ds_model: xr.Dataset,
    connection_method: str = "subbasin_area",
    area_max: float = 30.0,
    add_tributaries: bool = True,
    include_river_boundaries: bool = True,
    logger=logger,
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
    boundary of the 1d river as an additionnal tributary using the
    `include_river_boundaries` option.

    River edges or river nodes are snapped to the closest downstream wflow river
    cell using the :py:meth:`hydromt.flw.gauge_map` method.

    Parameters
    ----------
    gdf_riv : gpd.GeoDataFrame
        River geometry.
    ds_model : xr.Dataset
        Model dataset with 'flwdir', 'rivmsk', 'rivlen', 'uparea'.
    connection_method : str, default subbasin_area
        Method to connect wflow to the 1D model. Available methods are
        {'subbasin_area', 'nodes'}.
    area_max : float, default 10.0
        Maximum area [km2] of the subbasins to connect to the 1D model in km2 with
        connection_method **subbasin_area** or
        **nodes** with add_tributaries set to True.
    add_tributaries : bool, default True
        If True, derive tributaries for the subbasins larger than area_max. Always True
        for **subbasin_area** method.
    include_river_boundaries : bool, default True
        If True, include the upstream boundary(ies) of the 1d river as an additional
        tributary(ies).
    logger : logging.Logger, optional
        Logger object, by default logger
    **kwargs
        Additional keyword arguments passed to the snapping method
        hydromt.flw.gauge_map. See its documentation for more information.

    Returns
    -------
    ds_out: xr.Dataset
        Dataset with variables 'subcatch' for the subbasin map, 'subcatch_riv' for the
        subbasin map masked with river cells to be able to save river output with wflow
        and 'gauges' for the tributaries outflow locations (add_tributaries True or
        subbasin_area method).

    See Also
    --------
    hydromt.flw.gauge_map
    """
    # Checks
    dvars_model = ["flwdir", "rivmsk", "rivlen", "uparea"]
    if not np.all([v in ds_model for v in dvars_model]):
        raise ValueError(f"One or more variables missing from ds_model: {dvars_model}")

    # Reproject
    if gdf_riv.crs != ds_model.raster.crs:
        gdf_riv = gdf_riv.to_crs(ds_model.raster.crs)
    # Derive flwdir
    flwdir = hydromt.flw.flwdir_from_da(ds_model["flwdir"])
    # Basin mask
    basin_mask = ds_model["basins"].raster.vectorize()
    basin_mask = basin_mask.buffer(-2 * max(ds_model.raster.res, key=abs))

    # If tributaries or subbasins area method,
    # need to derive the tributaries areas first
    if connection_method == "subbasin_area" or add_tributaries:
        # 0. Check that the max_area is not too small
        # Should be bigger than the wflow river threshold
        # Get from attrs if available for newer wflow models built with hydromt
        riv_upa = ds_model["rivmsk"].attrs.get("river_upa", None)
        if riv_upa is None:
            # Derive from the uparea and rivmsk
            riv_upa = xr.where(ds_model["rivmsk"] > 0, ds_model["uparea"], np.nan)
            riv_upa = float(riv_upa.min())
        if area_max < riv_upa:
            new_area_max = np.ceil(riv_upa / 0.5) * 0.5
            logger.warning(
                f"The area_max {area_max} is smaller than the minimum upstream area of "
                f"the wflow river {riv_upa} which means tributaries will "
                "not be connected to the wflow river. Changing and setting area_max to "
                f"{new_area_max} km2. "
                f"To keep {area_max} km2 threshold, please update the wflow model to "
                "include more detailed rivers."
            )
            area_max = new_area_max

        logger.info("Linking 1D river to wflow river")
        # 1. Derive the river edges / boundaries
        # merge multilinestrings in gdf_riv to linestrings
        riv1d = gdf_riv.explode(index_parts=True).reset_index(drop=True)
        # clip to basins
        riv1d = riv1d.clip(basin_mask)
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

        # 2. snap edges to wflow river
        da_edges, idxs, ids = hydromt.flw.gauge_map(
            ds_model,
            xy=(riv1d_edges.geometry.x, riv1d_edges.geometry.y),
            stream=ds_model["rivmsk"].values,
            flwdir=flwdir,
            logger=logger,
            **kwargs,
        )
        points = gpd.points_from_xy(*ds_model.raster.idx_to_xy(idxs))
        # if csv contains additional columns, these are also written in the staticgeoms
        riv1d_edges = gpd.GeoDataFrame(
            index=ids.astype(np.int32), geometry=points, crs=ds_model.raster.crs
        )

        # 3. Derive the subbasins corresponding to the river edges
        da_edges_subbas, _ = hydromt.flw.basin_map(
            ds_model, flwdir=flwdir, xy=(riv1d_edges.geometry.x, riv1d_edges.geometry.y)
        )
        da_edges_subbas.raster.set_crs(ds_model.raster.crs)
        # convert to gdf
        gdf_edges_subbas = da_edges_subbas.raster.vectorize()

        # 4. Filter which subbasins are the upstream ones (tributaries)
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

        # 5. Derive a mask of gdf_riv in the subcatch_to_split wflow rivers
        # compute flw_path to mask out the river included in dfm 1d network
        xy = (
            gdf_tributaries.geometry.x.values.tolist(),
            gdf_tributaries.geometry.y.values.tolist(),
        )

        # Get paths
        flowpaths, dists = flwdir.path(
            xy=xy, max_length=None, unit="m", direction="down"
        )
        feats = flwdir.geofeatures(flowpaths)
        gdf_paths = gpd.GeoDataFrame.from_features(feats, crs=ds_model.raster.crs)
        gdf_paths.index = np.arange(1, len(gdf_paths) + 1)
        # Create a mask with subcatch id in subcatch_to_split and flowpath is nodata
        da_flwpaths = ds_model.raster.rasterize(gdf_paths)
        da_flwpaths = da_flwpaths.where(
            da_subcatch_to_split != da_subcatch_to_split.raster.nodata,
            da_flwpaths.raster.nodata,
        )

        # 6. Derive the tributaries
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
        trib_msk = trib_msk.reindex_like(
            da_subcatch_to_split, fill_value=nodata
        ).astype(np.int32)

        gdf_trib = trib_msk.raster.vectorize()
        # Test if there gdf_trib is empty
        if gdf_trib.empty:
            logger.info("No tributaries found")
            if not include_river_boundaries:
                gdf_tributaries = gpd.GeoDataFrame()
        else:
            gdf_trib["geometry"] = gdf_trib.centroid

            # Merge with gdf_tributary if include_river_boundaries
            # else only keep intersecting tributaries
            if include_river_boundaries:
                gdf_tributaries = pd.concat(
                    [gdf_tributaries, gdf_trib.drop(["value"], axis=1)]
                )
            else:
                gdf_tributaries = gdf_trib.drop(["value"], axis=1)
            gdf_tributaries.index = np.arange(1, len(gdf_tributaries) + 1)

        # 7. Mask the tributaries out of the subatch_to_split map
        if not gdf_tributaries.empty:
            # Derive the tributary basin map
            da_trib_subbas, _ = hydromt.flw.basin_map(
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
        flwdir_mask = hydromt.flw.flwdir_from_da(da_flwdir_mask)

    else:
        # The mask for deriving subbasins is the whole wflow model
        flwdir_mask = flwdir
        gdf_tributaries = gpd.GeoDataFrame()

    # 8. Derive the subbasins
    if connection_method == "subbasin_area":
        logger.info(
            "Deriving lateral subbasins based on"
            f"subbasin area threshold: {area_max} km2"
        )
        # calculate subbasins with a minimum stream order 7 and its outlets
        subbas, idxs_out = flwdir_mask.subbasins_area(area_max)
        da_subbasins = xr.DataArray(
            data=subbas.astype(np.int32),
            dims=ds_model.raster.dims,
            coords=ds_model.raster.coords,
        )
        da_subbasins.raster.set_nodata(0)
        da_subbasins.raster.set_crs(ds_model.raster.crs)
    else:
        # Get the nodes from gdf_riv
        logger.info("Deriving subbasins based on 1D river nodes snapped to wflow river")
        # from multiline to line
        gdf_riv = gdf_riv.explode(ignore_index=True, index_parts=False)
        # Clip to basin
        gdf_riv = gdf_riv.clip(basin_mask)
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
        da_nodes, idxs, ids = hydromt.flw.gauge_map(
            ds_model,
            xy=(gdf_nodes.geometry.x, gdf_nodes.geometry.y),
            stream=ds_model["rivmsk"].values,
            flwdir=flwdir,
            logger=logger,
            **kwargs,
        )
        # Derive subbasins
        da_subbasins, _ = hydromt.flw.basin_map(
            ds_model,
            flwdir=flwdir_mask,
            idxs=idxs,
            ids=ids,
            stream=ds_model["rivmsk"].values,
        )
        da_subbasins.raster.set_crs(ds_model.raster.crs)

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
