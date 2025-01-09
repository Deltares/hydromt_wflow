"""Glaciers workflows for Wflow plugin."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


__all__ = ["glaciermaps", "glacierattrs"]


def glaciermaps(
    gdf,
    ds_like,
    id_column="simple_id",
    elevtn_name="elevtn",
    logger=logger,
):
    """Return glacier maps (see list below) at model resolution.

    The following glacier maps are calculated:

    - wflow_glacierareas: glacier IDs [ID]
    - wflow_glacierfrac: area fraction of glacier per cell [-]
    - wflow_glacierstore: storage (volume) of glacier per cell [mm]

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing glacier geometries and attributes.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    id_column : str, optional, one of "simple_id", "C3S_id", "RGI_id", or "GLIMS_id"
        Column used for the glacier IDs, see data/data_sources.yml.

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing gridded glacier data
    """
    # Rasterize the GeoDataFrame to get the areas mask of glaciers with their ids
    ds_out = ds_like.raster.rasterize(
        gdf,
        col_name=id_column,
        nodata=0,
        all_touched=True,
        dtype=None,
        sindex=False,
    ).astype("int32")
    ds_out = ds_out.rename("glacareas")
    ds_out = ds_out.to_dataset()

    # Calculate glacier storage for each glacier
    # 'old' version to calculate storage, by calculating area from the glacier geometry
    # proj = partial(pyproj.transform, pyproj.Proj('epsg:4326'),
    # pyproj.Proj('epsg:3857'))
    # def calcStore(polygon, k=0.2055, gamma=1.375, proj=proj):
    #    return np.int(k * transform(proj, polygon).area ** gamma)
    # gdf['glaciervolume'] = gdf.apply(lambda row: calcStore(row['geometry']), axis=1)
    # def convStoremm(polygon, volm3, rhoice=0.9, proj=proj):
    #    return np.int(rhoice * volm3 / transform(proj, polygon).area * 1000)
    # gdf['glacierstore'] = gdf.apply(lambda row: convStoremm(row['geometry'],
    # row['glaciervolume']), axis=1)
    # 'new' version to calculate storage, because the 'old' version is now returning
    # inf/nan values, using the area listed in the dataset

    gdf["geometry"] = gdf.geometry.buffer(0)  # fix potential geometry errors
    # TODO: use alternative to calucate area? There seems to be a factor 2 difference
    # with the AREA column
    # gdf["AREA2"] = gdf.to_crs(3857).area / 1e6  # km2, area calculation
    # needs projected crs

    def calcStore(areakm2, k=0.2055, gamma=1.375):
        return int(k * (areakm2 * 1e6) ** gamma)

    def convStoremm(areakm2, volm3, rhoice=0.9):
        return int(rhoice * volm3 / (areakm2 * 1e6) * 1000)

    gdf["glaciervolume"] = gdf.apply(lambda row: calcStore(row["AREA"]), axis=1)
    gdf["glacierstore"] = gdf.apply(
        lambda row: convStoremm(row["AREA"], row["glaciervolume"]), axis=1
    )

    # Create vector grid (for calculating fraction and storage per grid cell)
    logger.debug(
        "Creating vector grid for calculating glacier fraction and \
storage per grid cell"
    )
    elevtn = ds_like[elevtn_name]
    idx_valid = np.where(elevtn.values.flatten() != elevtn.raster.nodata)[0]
    gdf_grid = ds_like.raster.vector_grid().loc[idx_valid]
    gdf_grid["glacierfrac"] = np.zeros(len(idx_valid), dtype=np.float32)
    gdf_grid["glacierstore"] = np.zeros(len(idx_valid), dtype=np.float32)
    gdf_grid["area"] = gdf_grid.to_crs(3857).area  # area calculation in projected crs

    # Calculate fraction and storage (i.e. volume) per (vector) grid cell
    # Looping over each vector GLACIER
    logger.debug("Setting glacierfrac and store values per glacier.")
    for i in range(len(gdf)):
        glacier = gdf.iloc[i]
        gridded_glacier = gdf_grid.intersection(glacier.geometry)
        gridded_glacier = gridded_glacier.loc[~gridded_glacier.is_empty]
        idxs = gridded_glacier.index
        # logger.debug(f"index: {i}, ID: {glacier[id_column]} // {len(idxs)} cells")
        if np.any(idxs):
            garea_cell = gridded_glacier.to_crs(
                3857
            ).area  # area calculation needs projected crs
            gfrac = garea_cell / np.sum(garea_cell)
            gdf_grid.loc[idxs, "glacierfrac"] += garea_cell / gdf_grid.loc[idxs, "area"]
            gdf_grid.loc[idxs, "glacierstore"] += gfrac * glacier["glacierstore"]

    # reproject back to original projection
    # Create the rasterized glacier storage map
    ds_out["glacstore"] = ds_like.raster.rasterize(
        gdf_grid,
        col_name="glacierstore",
        nodata=0,
        all_touched=False,
        dtype=None,
        sindex=False,
    ).astype("float32")

    # Create the rasterized glacier fraction map
    ds_out["glacfracs"] = ds_like.raster.rasterize(
        gdf_grid,
        col_name="glacierfrac",
        nodata=0,
        all_touched=False,
        dtype=None,
        sindex=False,
    ).astype("float32")

    ds_out["glacareas"].raster.set_nodata(0)
    ds_out["glacstore"].raster.set_nodata(0)
    ds_out["glacfracs"].raster.set_nodata(0)
    return ds_out


def glacierattrs(
    gdf,
    TT=1.3,
    Cfmax=5.3,
    SIfrac=0.002,
    id_column="simple_id",
    logger=logger,
):
    """Return glacier intbls (see list below).

    The following glacier intbls are calculated:

    - glacTempThresh: glacier temperature threshold [°C]
    - glacCfmax: glacier melting factor [mm/(°C*day)]
    - glacSIfrac: fraction of snowpack converted into ice and added to \
glacier storage [-]

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing glacier geometries and attributes.
    TT : float, optional
        Default value for glacier temperature threshold.
    Cfmax : float, optional
        Default value for glacier melting factor.
    SIfrac : float, optional
        Default value for fraction of snowpack converted into ice and added to \
glacier storage.
    id_column : str, optional, one of "simple_id", "C3S_id", "RGI_id", or "GLIMS_id"
        Column used for the glacier IDs, see data/data_sources.yml.

    Returns
    -------
    df_out : pandas.DataFrame
        DataFrame containing glacier attributes.
    """
    # Initialize output DataFrame with empty values and glacier ID
    df_out = pd.DataFrame(
        index=range(len(gdf[id_column])),
        columns=(
            [
                "glacId",
                "glacTempThresh",
                "glacCfmax",
                "glacSIfrac",
            ]
        ),
    )
    df_out["glacId"] = gdf[id_column].values
    # Fill in other attributes
    df_out["glacTempThresh"] = list(np.full(len(gdf.index), TT))
    df_out["glacCfmax"] = list(np.full(len(gdf.index), Cfmax))
    df_out["glacSIfrac"] = list(np.full(len(gdf.index), SIfrac))

    return df_out
