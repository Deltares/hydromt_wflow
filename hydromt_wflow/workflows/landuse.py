"""Landuse workflows for Wflow plugin."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


__all__ = ["landuse", "lai", "create_lulc_lai_mapping_table", "lai_from_lulc_mapping"]


RESAMPLING = {"landuse": "nearest", "lai": "average", "alpha_h1": "mode"}
DTYPES = {"landuse": np.int16, "alpha_h1": np.int16}


def landuse(
    da: xr.DataArray,
    ds_like: xr.Dataset,
    df: pd.DataFrame,
    params: Optional[List] = None,
    logger=logger,
):
    """Return landuse map and related parameter maps.

    The parameter maps are prepared based on landuse map and
    mapping table as provided in the generic data folder of hydromt.

    The following topography maps are calculated:

    - TODO

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


def lai(da: xr.DataArray, ds_like: xr.Dataset, logger=logger):
    """Return climatology of Leaf Area Index (LAI).

    The following topography maps are calculated:
    - LAI

    Parameters
    ----------
    da : xarray.DataArray or xarray.Dataset
        DataArray or Dataset with LAI array containing LAI values.
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
    da = da.where(da.values != nodata).fillna(
        0.0
    )  # Assuming missing values correspond to bare soil, urban and snow (LAI=0.0)
    da_out = da.raster.reproject_like(ds_like, method=method)
    da_out.attrs.update(_FillValue=nodata)
    return da_out


def create_lulc_lai_mapping_table(
    da_lulc: xr.DataArray,
    da_lai: xr.DataArray,
    sampling_method: str = "any",
    lulc_zero_classes: List[int] = [],
    logger=logger,
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
        Resampling method for the LULC data to the LAI resolution. Two methods are
        supported:

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
                # Resample only now the landuse data to the LAI resolution
                lu = lu.raster.reproject_like(da_lai, method="mode")

            # Add lai
            lu = lu.to_dataset()
            lu["lai"] = da_lai

            # Stack and remove the nodata values
            lu = lu.stack(z=("y", "x")).dropna(dim="z", how="all", subset=["landuse"])

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
                lai_mean_per_lu = np.round(lu["lai"].load().mean(dim="z") / 10, 3)
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
    logger=logger,
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
    logger : logging.Logger, optional
        Logger object.

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
        logger=logger,
    )
    # Re-organise the dataset to have a time dimension
    da_lai = ds_lai.to_array(dim="time", name="LAI")

    return da_lai
