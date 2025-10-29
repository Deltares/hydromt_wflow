"""Landuse workflows for Wflow plugin."""

import logging
import os
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

from hydromt_wflow.workflows.demand import create_grid_from_bbox

logger = logging.getLogger(f"hydromt.{__name__}")


__all__ = [
    "landuse",
    "landuse_from_vector",
    "lai",
    "create_lulc_lai_mapping_table",
    "lai_from_lulc_mapping",
    "add_paddy_to_landuse",
    "add_planted_forest_to_landuse",
]


RESAMPLING = {
    "landuse": "nearest",
    "lai": "average",
    "vegetation_feddes_alpha_h1": "mode",
}
DTYPES = {"landuse": np.int16, "vegetation_feddes_alpha_h1": np.int16}


def landuse(
    da: xr.DataArray,
    ds_like: xr.Dataset,
    df: pd.DataFrame,
    params: Optional[list] = None,
):
    """Return landuse map and related parameter maps.

    The parameter maps are prepared based on landuse map and
    mapping table as provided in the generic data folder of hydromt.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray containing LULC classes.
    ds_like : xarray.DataArray
        Dataset at model resolution.

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing gridded landuse based maps
    """
    keys = df.index.values
    if params is None:
        params = [p for p in df.columns if p != "description"]
    elif not np.all(np.isin(params, df.columns)):
        missing = [p for p in params if p not in df.columns]
        raise ValueError(f"Parameter(s) missing in mapping file: {missing}")
    # setup ds out
    ds_out = xr.Dataset(coords=ds_like.raster.coords)

    # setup reclass method
    def reclass(x):
        return np.vectorize(d.get)(x, nodata)

    da = da.raster.interpolate_na(method="nearest")
    # apply for each parameter
    for param in params:
        method = RESAMPLING.get(param, "average")
        values = df[param].values
        nodata = values[-1]  # NOTE values is set in last row
        d = dict(zip(keys, values))  # NOTE global param in reclass method
        logger.info(f"Deriving {param} using {method} resampling (nodata={nodata}).")
        da_param = xr.apply_ufunc(
            reclass, da, dask="parallelized", output_dtypes=[values.dtype]
        )
        da_param.attrs.update(_FillValue=nodata)  # first set new nodata values
        ds_out[param] = da_param.raster.reproject_like(
            ds_like, method=method
        )  # then resample

    return ds_out


def landuse_from_vector(
    gdf: gpd.GeoDataFrame,
    ds_like: xr.Dataset,
    df: pd.DataFrame,
    params: Optional[list] = None,
    lulc_res: float | int | None = None,
    all_touched: bool = False,
    buffer: int = 1000,
    lulc_out: Optional[str] = None,
):
    """
    Derive several wflow maps based on vector landuse-landcover (LULC) data.

    The vector lulc data is first rasterized to a raster map at the model resolution
    or at a higher resolution specified in ``lulc_res``.

    Lookup table `df` columns are converted to lulc classes model
    parameters based on literature. The data is remapped at its rasterized resolution
    and then resampled to the model resolution using the average value, unless noted
    differently.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing LULC classes.
    ds_like : xarray.Dataset
        Dataset at model resolution.
    df : pd.DataFrame
        Mapping table with landuse values.
    params : list of str, optional
        List of parameters to derive, by default None
    lulc_res : float or int, optional
        Resolution of the rasterized LULC data, by default None (use model resolution)
    all_touched : bool, optional
        If True, all pixels touched by the polygon will be burned in, by default False
    buffer : int, optional
        Buffer in meters to add around the bounding box of the vector data, by default
        1000.
    lulc_out : str, optional
        Path to save the rasterised original landuse map to file, by default None.

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing gridded landuse based maps
    """
    # intersect with bbox
    bounds = gpd.GeoDataFrame(
        geometry=[box(*ds_like.raster.bounds)], crs=ds_like.raster.crs
    )
    bounds = bounds.to_crs(3857).buffer(buffer).to_crs(gdf.crs)
    gdf = gdf.overlay(gpd.GeoDataFrame(geometry=bounds), how="intersection")

    # rasterize the vector data
    logger.info("Rasterizing landuse map")
    if lulc_res is None:
        gdf_reproj = gdf.to_crs(ds_like.raster.crs)
        grid_like = create_grid_from_bbox(
            gdf_reproj.total_bounds,
            res=max(np.abs(ds_like.raster.res)),
            crs=ds_like.raster.crs,
            align=True,
        )
    else:
        grid_like = create_grid_from_bbox(
            gdf.total_bounds,
            res=lulc_res,
            crs=gdf.crs,
            align=True,
        )
    # get the nodata values of the landuse (last row in the df)
    nodata = df["landuse"].values[-1]
    da = grid_like.raster.rasterize(
        gdf,
        col_name="landuse",
        nodata=nodata,
        all_touched=all_touched,
        dtype="int32",
    )
    if lulc_out is not None:
        logger.info(f"Saving rasterized landuse map to {lulc_out}")
        os.makedirs(os.path.dirname(lulc_out), exist_ok=True)
        da.raster.to_raster(lulc_out)

    # derive the landuse maps
    ds_out = landuse(da, ds_like, df, params=params)

    return ds_out


def lai(da: xr.DataArray, ds_like: xr.Dataset):
    """Return climatology of Leaf Area Index (LAI).

    The following maps are calculated:
    - LAI

    Parameters
    ----------
    da : xarray.DataArray or xarray.Dataset
        LAI array containing LAI values.
    ds_like : xarray.DataArray
        Dataset at model resolution.

    Returns
    -------
    da_out : xarray.DataArray
        Dataset containing resampled LAI maps
    """
    if isinstance(da, xr.Dataset) and "LAI" in da:
        da = da["LAI"]
    elif not isinstance(da, xr.DataArray):
        raise ValueError("lai method requires a DataArray or Dataset with LAI array")
    method = RESAMPLING.get(da.name, "average")
    nodata = da.raster.nodata
    logger.info(f"Deriving {da.name} using {method} resampling (nodata={nodata}).")
    da = da.astype(np.float32)
    # Assuming missing values correspond to: bare soil, urban and snow (LAI=0.0)
    da = da.where(da.values != nodata).fillna(0.0)
    da_out = da.raster.reproject_like(ds_like, method=method)
    da_out.attrs.update(_FillValue=nodata)
    return da_out


def create_lulc_lai_mapping_table(
    da_lulc: xr.DataArray,
    da_lai: xr.DataArray,
    sampling_method: str = "any",
    lulc_zero_classes: list[int] = [],
) -> pd.DataFrame:
    """
    Derive LAI values per landuse class.

    Parameters
    ----------
    da_lulc : xr.DataArray
        Landuse map.
    da_lai : xr.DataArray
        Cyclic LAI map.
    sampling_method : str, optional
        Resampling method for the LULC data to LAI resolution.
        Two methods are supported:

        * 'any' (default): if any cell of the desired landuse class is present in the
            resampling window (even just one), it will be used to derive LAI values.
            This method is less exact but will provide LAI values for all landuse
            classes for the high resolution landuse map.
        * 'mode': the most frequent value in the resampling window is
            used. This method is less precise as for cells with a lot of different
            landuse classes, the most frequent value might still be only a small
            fraction of the cell. More landuse classes should however be covered and
            it can always be used with the landuse map of the wflow model instead of
            the original high resolution one.
        * 'q3': only cells with the most frequent value (mode) and that cover 75%
            (q3) of the resampling window will be used. This method is more exact but
            for small basins, you may have less or no samples to derive LAI values
            for some classes.
    lulc_zero_classes : list of int, optional
        List of landuse classes that should have zero for leaf area index values
        for example waterbodies, open ocean etc. For very high resolution landuse
        maps, urban surfaces and bare areas can be included here as well.
        By default empty.

    Returns
    -------
    df_lai_mapping : pd.DataFrame
        Mapping table with LAI values per landuse class. One column for each month and
        one line per landuse class. The number of samples used to derive the mapping
        values is also added to a `samples` column in the dataframe.
    """
    # check the method values
    if sampling_method not in ["any", "mode", "q3"]:
        raise ValueError(f"Unsupported resampling method: {sampling_method}")

    # process the lai da
    if "dim0" in da_lai.dims:
        da_lai = da_lai.rename({"dim0": "time"})
    da_lai = da_lai.raster.mask_nodata()
    da_lai = da_lai.fillna(
        0
    )  # use zeros to better represent city and open water surfaces

    # landuse
    da_lulc.name = "landuse"
    lulc_classes = np.unique(da_lulc.values)

    # Initialise the outputs
    df_lai_mapping = None

    if sampling_method != "any":
        # The data can already be resampled to the LAI resolution
        da_lulc_mode = da_lulc.raster.reproject_like(da_lai, method="mode")
        if sampling_method == "q3":
            # Filter mode cells that cover less than 75% of the resampling window
            da_lulc_q3 = da_lulc.raster.reproject_like(da_lai, method="q3")
            da_lulc = da_lulc_mode.where(
                da_lulc_q3 == da_lulc_mode, da_lulc_mode.raster.nodata
            )
        else:
            da_lulc = da_lulc_mode

    # Loop over the landuse classes
    for lulc_id in lulc_classes:
        logger.info(f"Processing landuse class {lulc_id}")

        if lulc_id in lulc_zero_classes:
            logger.info(f"Using zeros for landuse class {lulc_id}")
            df_lai = pd.DataFrame(
                columns=da_lai.time.values,
                data=[[0] * 12],
                index=[lulc_id],
            )
            df_lai.index.name = "landuse"
            n_samples = 0

        else:
            # Select for a specific landuse class
            lu = da_lulc.where(da_lulc == lulc_id, da_lulc.raster.nodata)
            lu = lu.raster.mask_nodata()

            if sampling_method == "any":
                # Resample the landuse data to the LAI resolution
                lu = lu.raster.reproject_like(da_lai, method="mode")

            # Add lai
            lu = lu.to_dataset()
            lu["lai"] = da_lai

            # Stack and remove the nodata values
            lu = lu.stack(z=(lu.raster.y_dim, lu.raster.x_dim)).dropna(
                dim="z", how="all", subset=["landuse"]
            )

            # Count the number of samples
            n_samples = len(lu["z"])
            if n_samples == 0:
                logger.info(
                    f"No samples found for landuse class {lulc_id}. "
                    "Try using a different resampling method."
                )
                df_lai = pd.DataFrame(
                    columns=da_lai.time.values,
                    data=[[0] * 12],
                    index=[lulc_id],
                )
                df_lai.index.name = "landuse"
            else:
                # Compute the mean
                lai_mean_per_lu = np.round(lu["lai"].load().mean(dim="z"), 3)
                # Add the landuse id as an extra dimension
                lai_mean_per_lu = lai_mean_per_lu.expand_dims("landuse")
                lai_mean_per_lu["landuse"] = [lulc_id]

                # Convert to dataframe
                df_lai = lai_mean_per_lu.drop_vars(
                    "spatial_ref", errors="ignore"
                ).to_pandas()

        # Add number of samples in the first column
        df_lai.insert(0, "samples", n_samples)

        # Append to the output
        if df_lai_mapping is None:
            df_lai_mapping = df_lai
        else:
            df_lai_mapping = pd.concat([df_lai_mapping, df_lai])

    return df_lai_mapping


def lai_from_lulc_mapping(
    da: xr.DataArray,
    ds_like: xr.Dataset,
    df: pd.DataFrame,
) -> xr.Dataset:
    """
    Derive LAI values from a landuse map and a mapping table.

    Parameters
    ----------
    da : xr.DataArray
        Landuse map.
    ds_like : xr.Dataset
        Dataset at model resolution.
    df : pd.DataFrame
        Mapping table with LAI values per landuse class. One column for each month and
        one line per landuse class.

    Returns
    -------
    ds_lai : xr.Dataset
        Dataset with LAI values for each month.
    """
    months = np.arange(1, 13)
    df.columns = [int(col) if str(col).isdigit() else col for col in df.columns]
    # Map the monthly LAI values to the landuse map
    ds_lai = landuse(
        da=da,
        ds_like=ds_like,
        df=df,
        params=months,
    )
    # Re-organise the dataset to have a time dimension
    da_lai = ds_lai.to_array(dim="time", name="LAI")

    return da_lai


def add_paddy_to_landuse(
    landuse: xr.DataArray,
    paddy: xr.DataArray,
    paddy_class: int,
    df_mapping: pd.DataFrame,
    df_paddy_mapping: pd.DataFrame,
    output_paddy_class: Optional[int] = None,
) -> tuple[xr.DataArray, pd.DataFrame]:
    """
    Burn paddy fields into landuse map and update mapping table.

    The resulting paddy class in the landuse map will have ID output_paddy_class
    if provided and paddy_class otherwise. The mapping table will be updated with
    the values from the df_paddy_mapping table.

    Parameters
    ----------
    landuse : xr.DataArray
        Landuse map.
    paddy : xr.DataArray
        Paddy fields map.
    paddy_class : int
        ID of the paddy class in the paddy map.
    df_mapping : pd.DataFrame
        Mapping table with landuse values.
    df_paddy_mapping : pd.DataFrame
        Mapping table with paddy values.
    output_paddy_class : int, optional
        ID of the paddy class in the output landuse map. If not provided, the
        paddy_class will be used.

    Returns
    -------
    landuse : xr.DataArray
        Updated landuse map.
    df_mapping : pd.DataFrame
        Updated mapping table.
    """
    # Get output paddy class
    if output_paddy_class is None:
        output_paddy_class = paddy_class

    # Reproject paddy map to landuse resolution
    # if paddy has lower res than landuse, use nearest resampling
    if abs(paddy.raster.res[0]) >= abs(landuse.raster.res[0]):
        paddy = paddy.raster.reproject_like(landuse, method="nearest")
    # else use mode resampling
    else:
        paddy = paddy.raster.reproject_like(landuse, method="mode")

    # Burn in the rice fields in the landuse map
    landuse = landuse.where(paddy != paddy_class, output_paddy_class)

    # Update the mapping table
    df_paddy_mapping.index = [output_paddy_class]
    df_paddy_mapping["landuse"] = output_paddy_class
    # Add the paddy class to the first line of the mapping table
    df_mapping = pd.concat([df_paddy_mapping, df_mapping])

    return landuse, df_mapping


def add_planted_forest_to_landuse(
    planted_forest: gpd.GeoDataFrame,
    ds_like: xr.Dataset,
    planted_forest_c: float = 0.0881,
    orchard_name: str = "Orchard",
    orchard_c: float = 0.2188,
) -> xr.DataArray:
    """
    Update USLE C map with planted forest and orchard data.

    Default USLE C values for planted forest and orchards are derived from Panagos et
    al., 2015 (10.1016/j.landusepol.2015.05.021).
    For harvested forest at different regrowth stages see also Borrelli and Schutt, 2014
    (10.1016/j.geomorph.2013.08.022).

    Parameters
    ----------
    planted_forest : geopandas.GeoDataFrame
        GeoDataFrame containing planted forest data. Required columns are: 'geometry',
        and optionally 'forest_type' to find orchards.
    ds_like : xr.Dataset
        Dataset at model resolution. Required variables are 'usle_c'.
    planted_forest_c : float, optional
        USLE C value for planted forest, by default 0.0881.
    orchard_name : str, optional
        Name of the orchard landuse class, by default "Orchard".
    orchard_c : float, optional
        USLE C value for orchards, by default 0.2188.

    Returns
    -------
    usle_c : xr.DataArray
        Updated USLE C map.
    """
    # Add a usle_c column with default value
    logger.info(
        "Correcting usle_c with planted forest and orchards using {planted_forest_fn}."  # noqa: E501
    )
    planted_forest["usle_c"] = planted_forest_c
    # If forest_type column is available, update usle_c value for orchards
    if "forest_type" in planted_forest.columns:
        planted_forest.loc[planted_forest["forest_type"] == orchard_name, "usle_c"] = (
            orchard_c
        )
    # Rasterize forest data
    usle_c = ds_like.raster.rasterize(
        gdf=planted_forest,
        col_name="usle_c",
        nodata=ds_like["usle_c"].raster.nodata,
        all_touched=False,
    )
    # Cover nodata with the usle_c map from all landuse classes
    usle_c = usle_c.where(
        usle_c != usle_c.raster.nodata,
        ds_like["usle_c"],
    )

    return usle_c
