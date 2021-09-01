import numpy as np
import xarray as xr
from typing import Dict, Union
from pathlib import Path

from hydromt.io import open_timeseries_from_table
from hydromt.vector import GeoDataArray


__all__ = ["read_csv_results"]


def read_csv_results(fn: Union[str, Path], config: Dict, maps: xr.Dataset) -> Dict:
    """Read wflow results csv timeseries and parse to dictionnary

    Parses the wflow csv results file into different ``hydromt.GeoDataArrays``, one per
    column (csv section and csv.column sections of the TOML). The xy coordinates are the
    coordinates of the station or of the representative point of the subcatch/area. The
    variable name in the ``GeoDataArray`` corresponds to the csv header attribute or
    header_map when available.

    Parameters
    ----------
    fn: str
        Path to the wflow csv results file.
    config: dict
        wflow.toml configuration.
    maps: xr.Dataset
        wflow staticmaps.nc dataset

    Returns
    -------
    csv_dict: dict
        Dictionnary of hydromt.GeoDataArrays for the different csv.column section of the config.
    """
    # Count items by csv.column
    count = 1
    csv_dict = dict()
    # Loop over csv.column
    for col in config["csv"].get("column"):
        header = col["header"]
        # Column based on map
        if "map" in col.keys():
            # Read the corresponding map and derive the different locations
            # The centroid of the geometry is used as coordinates for the timeseries
            map_name = config["input"].get(f"{col['map']}")
            da = maps[map_name]
            gdf = da.raster.vectorize()
            gdf.geometry = gdf.geometry.representative_point()
            gdf.index = gdf.value.astype(da.dtype)
            gdf.index.name = "index"
            # Read the timeseries
            usecols = [0]
            usecols = np.append(usecols, np.arange(count, count + len(gdf.index)))
            count += len(gdf.index)
            da_ts = open_timeseries_from_table(
                fn, name=f'{header}_{col["map"]}', usecols=usecols
            )
            da = GeoDataArray.from_gdf(gdf, da_ts, index_dim="index")
        # Column based on xy coordinates or reducer for the full model domain domain
        else:
            # Read the timeseries
            usecols = [0]
            usecols = np.append(usecols, np.arange(count, count + 1))
            count += 1
            try:
                da_ts = open_timeseries_from_table(fn, name=header, usecols=usecols)
            except:
                colnames = ["time", "0"]
                da_ts = open_timeseries_from_table(
                    fn,
                    name=header,
                    usecols=usecols,
                    header=0,
                    names=colnames,
                )
            # Add point coordinates
            # Column based on xy coordinates
            if "coordinate" in col.keys():
                scoords = {
                    "x": xr.IndexVariable("index", [col["coordinate"]["x"]]),
                    "y": xr.IndexVariable("index", [col["coordinate"]["y"]]),
                }
            # Column based on index
            elif "index" in col.keys():
                # x and y index, works on the full 2D grid
                if isinstance(col["index"], dict):
                    # index in julia starts at 1
                    # coordinates are always ascending
                    xi = maps.raster.xcoords.values[col["index"]["x"] - 1]
                    yi = np.sort(maps.raster.ycoords.values)[col["index"]["y"] - 1]
                    scoords = {
                        "x": xr.IndexVariable("index", [xi]),
                        "y": xr.IndexVariable("index", [yi]),
                    }
                # index of the full array
                else:
                    # Create grid with full 2D Julia indices
                    # Dimensions are ascending and ordered as (x,y,layer,time)
                    # Indices are created before ordering for compatibility with raster.idx_to_xy
                    full_index = maps[f'{config["input"].get("subcatchment")}'].copy()
                    res_x, res_y = full_index.raster.res
                    if res_y < 0:
                        full_index = full_index.reindex(
                            {
                                full_index.raster.y_dim: full_index[
                                    full_index.raster.y_dim
                                ][::-1]
                            }
                        )
                    data = np.arange(0, np.size(full_index)).reshape(
                        np.size(full_index, 0), np.size(full_index, 1)
                    )
                    full_index[:, :] = data
                    full_index = full_index.transpose(
                        full_index.raster.x_dim, full_index.raster.y_dim
                    )
                    # Index depends on the struct
                    # For land uses the active subcatch IDs
                    if (
                        "vertical" in col["parameter"]
                        or "lateral.land" in col["parameter"]
                    ):
                        mask = maps[f'{config["input"].get("subcatchment")}'].copy()
                    elif "reservoir" in col["parameter"]:
                        mask = maps[
                            f'{config["input"]["lateral"]["river"]["reservoir"].get("locs")}'
                        ].copy()
                    elif "lake" in col["parameter"]:
                        mask = maps[
                            f'{config["input"]["lateral"]["river"]["lake"].get("locs")}'
                        ].copy()
                    # Else lateral.river
                    else:
                        mask = maps[f'{config["input"].get("river_location")}'].copy()
                    # Rearrange the mask
                    res_x, res_y = mask.raster.res
                    if res_y < 0:
                        mask = mask.reindex(
                            {mask.raster.y_dim: mask[mask.raster.y_dim][::-1]}
                        )
                    mask = mask.transpose(mask.raster.x_dim, mask.raster.y_dim)
                    # Filter and reduce full_index based on mask
                    full_index = full_index.where(mask != mask.raster.nodata, 0)
                    full_index.attrs.update(_FillValue=0)
                    mask_index = full_index.values.flatten()
                    mask_index = mask_index[mask_index != 0]
                    # idx corresponding to the wflow index
                    idx = mask_index[col["index"] - 1]
                    # Reorder full_index as (y,x) to use raster.idx_to_xy method
                    xi, yi = full_index.transpose(
                        full_index.raster.y_dim, full_index.raster.x_dim
                    ).raster.idx_to_xy(idx)
                    scoords = {
                        "x": xr.IndexVariable("index", xi),
                        "y": xr.IndexVariable("index", yi),
                    }
            # Based on model bbox center for column based on reducer for the full model domain
            else:
                xmin, ymin, xmax, ymax = maps.raster.bounds
                scoords = {
                    "x": xr.IndexVariable("index", [(xmax + xmin) / 2]),
                    "y": xr.IndexVariable("index", [(ymax + ymin) / 2]),
                }
            da = da_ts.assign_coords(scoords)

        csv_dict[f"{da.name}"] = da

    return csv_dict
