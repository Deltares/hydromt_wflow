"""A component to write configuration files for model simulations/kernels."""

from logging import Logger, getLogger
from os import makedirs
from os.path import dirname, isdir, isfile, join
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import geopandas as gpd
import hydromt
import numpy as np
import pyproj
import xarray as xr
from hydromt.git import _axes_attrs
from hydromt.model.components.grid import GridComponent

if TYPE_CHECKING:
    from hydromt.model import Model

logger: Logger = getLogger(__name__)


class WflowGridComponent(GridComponent):
    """implement the grid functionality from wflow."""

    def __init__(
        self,
        model: "Model",
        *,
        filename: str = "wflow_sbm.toml",
        default_template_filename: Optional[str] = None,
    ):
        """Initialize a ConfigComponent.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        filename: str
            A path relative to the root where the configuration file will
            be read and written if user does not provide a path themselves.
            By default 'config.yml'
        default_template_filename: Optional[Path]
            A path to a template file that will be used as default in the ``create``
            method to initialize the configuration file if the user does not provide
            their own template file. This can be used by model plugins to provide a
            default configuration template. By default None.
        """
        super().__init__(
            model=model,
            filename=filename,
            default_template_filename=default_template_filename,
        )

    def set(
        self,
        data: Union[xr.DataArray, xr.Dataset, np.ndarray],
        name: Optional[str] = None,
    ):
        """Add data to grid.

        All layers of grid must have identical spatial coordinates. This is an inherited
        method from HydroMT-core's GridModel.grid.set with some fixes.

        The first fix is when data with a time axis is being added. Since Wflow.jl
        v0.7.3, cyclic data at different lengths (12, 365, 366) is supported, as long as
        the dimension name starts with "time". In this function, a check is done if a
        time axis with that exact shape is already present in the grid object, and will
        use that dimension (and its name) to set the data. If a time dimension does not
        yet exist with that shape, it is created following the format
        "time_{length_data}".

        The other fix is that when the model is updated with a different number of
        layers, this is not automatically updated correctly. With this fix, the old
        layer dimension is removed (including all associated data), and the new data is
        added with the correct "layer" dimension.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to grid
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray and
            ignored if data is a Dataset
        """
        if "time" in data.dims:
            # Raise error if the dimension does not have a supported length
            if len(data.time) not in [12, 365, 366]:
                raise ValueError(
                    f"Length of cyclic dataset ({len(data)}) is not supported by "
                    "Wflow.jl. Ensure the data has length 12, 365, or 366"
                )
            tname = "time"
            time_axes = {
                k: v for k, v in dict(self.grid.dims).items() if k.startswith("time")
            }
            if data["time"].size not in time_axes.values():
                tname = f"time_{data['time'].size}" if "time" in time_axes else tname
            else:
                k = list(
                    filter(lambda x: time_axes[x] == data["time"].size, time_axes)
                )[0]
                tname = k

            if tname != "time":
                data = data.rename_dims({"time": tname})
        if "layer" in data.dims and "layer" in self.grid:
            if len(data["layer"]) != len(self.grid["layer"]):
                vars_to_drop = [
                    var for var in self.grid.variables if "layer" in self.grid[var].dims
                ]
                # Drop variables
                self.logger.info(
                    "Dropping these variables, as they depend on the layer "
                    f"dimension: {vars_to_drop}"
                )
                # Use `_grid` as `grid` cannot be set
                self._grid = self.grid.data.drop_vars(vars_to_drop)

        # fall back on default grid.set behaviour
        # GridModel.grid.set(self, data, name)

    def read(self, **kwargs):
        """
        Read wflow static input and add to ``grid``.

        Checks the path of the file in the config toml using both ``input.path_static``
        and ``dir_input``. If not found uses the default path ``staticmaps.nc`` in the
        root folder.

        For reading old PCRaster maps, see the pcrm submodule.

        See Also
        --------
        pcrm.read_staticmaps_pcr
        """
        fn_default = "staticmaps.nc"
        fn = self.config.get_value(
            "input.path_static", abs_path=True, fallback=join(self.root, fn_default)
        )

        if self.config.get_value("dir_input") is not None:
            input_dir = self.config.get_value("dir_input", abs_path=True)
            fn = join(
                input_dir,
                self.config.get_value("input.path_static", fallback=fn_default),
            )
            self.logger.info(f"Input directory found {input_dir}")

        if not self._write:
            # start fresh in read-only mode
            self._grid = xr.Dataset()
        if fn is not None and isfile(fn):
            self.logger.info(f"Read grid from {fn}")
            # FIXME: we need a smarter (lazy) solution for big models which also
            # works when overwriting / appending data in the same source!
            ds = xr.open_dataset(
                fn, mask_and_scale=False, decode_coords="all", **kwargs
            ).load()
            ds.close()
            # make sure internally maps are always North -> South oriented
            if ds.raster.res[1] > 0:
                ds = ds.raster.flipud()
            self.grid.set(ds)

    def write(
        self,
        fn_out: Optional[Union[Path, str]] = None,
    ):
        """
        Write grid to wflow static data file.

        Checks the path of the file in the config toml using both ``input.path_static``
        and ``dir_input``. If not found uses the default path ``staticmaps.nc`` in the
        root folder.

        Parameters
        ----------
        fn_out : Path, str, optional
            Name or path to the outgoing grid file (including extension). This is the
            path/name relative to the root folder and if present the ``dir_input``
            folder.
        """
        if not self._write:
            raise IOError("Model opened in read-only mode")
        # clean-up grid and write CRS according to CF-conventions
        # TODO replace later with hydromt.raster.gdal_compliant method
        # after core release
        crs = self.grid.data.raster.crs
        ds_out = self.grid.data.reset_coords()
        # TODO?!
        # if ds_out.raster.res[1] < 0: # write data with South -> North orientation
        #     ds_out = ds_out.raster.flipud()
        x_dim, y_dim, x_attrs, y_attrs = _axes_attrs(crs)
        ds_out = ds_out.rename({ds_out.raster.x_dim: x_dim, ds_out.raster.y_dim: y_dim})
        ds_out[x_dim].attrs.update(x_attrs)
        ds_out[y_dim].attrs.update(y_attrs)
        ds_out = ds_out.drop_vars(["mask", "spatial_ref", "ls"], errors="ignore")
        ds_out.rio.write_crs(crs, inplace=True)
        ds_out.rio.write_transform(self.grid.data.raster.transform, inplace=True)
        ds_out.raster.set_spatial_dims()

        # Remove FillValue Nan for x_dim, y_dim
        encoding = dict()
        for v in [ds_out.raster.x_dim, ds_out.raster.y_dim]:
            ds_out[v].attrs.pop("_FillValue", None)
            encoding[v] = {"_FillValue": None}

        # filename
        if fn_out is not None:
            fn = join(self.root, fn_out)
            self.config.set("input.path_static", fn_out)
        else:
            fn_out = "staticmaps.nc"
            fn = self.config.get_value(
                "input.path_static", abs_path=True, fallback=join(self.root, fn_out)
            )
        # Append inputdir if required
        if self.config.get_value("dir_input") is not None:
            input_dir = self.config.get_value("dir_input", abs_path=True)
            fn = join(
                input_dir,
                self.config.get_value("input.path_static", fallback=fn_out),
            )
        # Check if all sub-folders in fn exists and if not create them
        if not isdir(dirname(fn)):
            makedirs(dirname(fn))
        self.logger.info(f"Write grid to {fn}")
        mask = ds_out[self._MAPS["basins"]] > 0
        for v in ds_out.data_vars:
            # nodata is required for all but boolean fields
            if ds_out[v].dtype != "bool":
                ds_out[v] = ds_out[v].where(mask, ds_out[v].raster.nodata)
        ds_out.to_netcdf(fn, encoding=encoding)

    def setup_grid_from_raster(
        self,
        raster_fn: Union[str, xr.Dataset],
        reproject_method: str,
        variables: Optional[List[str]] = None,
        wflow_variables: Optional[List[str]] = None,
        fill_method: Optional[str] = None,
    ) -> List[str]:
        """
        Add data variable(s) from ``raster_fn`` to grid object.

        If raster is a dataset, all variables will be added unless ``variables``
        list is specified. The config toml can also be updated to include
        the new maps using ``wflow_variables``.

        Adds model layers:

        * **raster.name** or **variables** grid: data from raster_fn

        Parameters
        ----------
        raster_fn: str
            Source name of RasterDataset in data_catalog.
        reproject_method: str
            Reprojection method from rasterio.enums.Resampling.
            Available methods: ['nearest', 'bilinear', 'cubic', 'cubic_spline', \
'lanczos', 'average', 'mode', 'gauss', 'max', 'min', 'med', 'q1', 'q3', \
'sum', 'rms']
        variables: list, optional
            List of variables to add to grid from raster_fn. By default all.
        wflow_variables: list, optional
            List of corresponding wflow variables to update the config toml
            (e.g: ["input.vertical.altitude"]).
            Should match the variables list. variables list should be provided unless
            raster_fn contains a single variable (len 1).
        fill_method : str, optional
            If specified, fills nodata values using fill_nodata method.
            Available methods are {'linear', 'nearest', 'cubic', 'rio_idw'}.

        Returns
        -------
        list
            Names of added model staticmap layers.
        """
        self.logger.info(f"Preparing grid data from raster source {raster_fn}")
        # Read raster data and select variables
        ds = self.data_catalog.get_rasterdataset(
            raster_fn,
            geom=self.region,
            buffer=2,
            variables=variables,
            single_var_as_array=False,
        )
        # Fill nodata
        if fill_method is not None:
            ds = ds.raster.interpolate_na(method=fill_method)
        # Reprojection
        ds_out = ds.raster.reproject_like(self.grid, method=reproject_method)
        # Add to grid
        self.set_grid(ds_out)

        # Update config
        if wflow_variables is not None:
            self.logger.info(
                f"Updating the config for wflow_variables: {wflow_variables}"
            )
            if variables is None:
                if len(ds_out.data_vars) == 1:
                    variables = list(ds_out.data_vars.keys())
                else:
                    raise ValueError(
                        "Cannot update the toml if raster_fn has more than \
one variable and variables list is not provided."
                    )

            # Check on len
            if len(wflow_variables) != len(variables):
                raise ValueError(
                    f"Length of variables {variables} do not match wflow_variables \
{wflow_variables}. Cannot update the toml."
                )
            else:
                for i in range(len(variables)):
                    self.config.set(wflow_variables[i], variables[i])

    def clip(
        self,
        region,
        buffer=0,
        align=None,
        crs=4326,
    ):
        """Clip grid to subbasin.

        Parameters
        ----------
        region : dict
            See :meth:`models.wflow.WflowModel.setup_basemaps`
        buffer : int, optional
            Buffer around subbasin in number of pixels, by default 0
        align : float, optional
            Align bounds of region to raster with resolution <align>, by default None
        crs: int, optional
            Default crs of the grid to clip.

        Returns
        -------
        xarray.DataSet
            Clipped grid.
        """
        basins_name = self._MAPS["basins"]
        flwdir_name = self._MAPS["flwdir"]

        kind, region = hydromt.workflows.parse_region(region, logger=self.logger)
        # translate basin and outlet kinds to geom
        geom = region.get("geom", None)
        bbox = region.get("bbox", None)
        if kind in ["basin", "outlet", "subbasin"]:
            # supply bbox to avoid getting basin bounds first when clipping subbasins
            if kind == "subbasin" and bbox is None:
                region.update(bbox=self.bounds)
            geom, _ = hydromt.workflows.get_basin_geometry(
                ds=self.grid.data,
                logger=self.logger,
                kind=kind,
                basins_name=basins_name,
                flwdir_name=flwdir_name,
                **region,
            )
        # clip based on subbasin args, geom or bbox
        if geom is not None:
            ds_grid = self.grid.data.raster.clip_geom(geom, align=align, buffer=buffer)
            ds_grid.coords["mask"] = ds_grid.raster.geometry_mask(geom)
            ds_grid[basins_name] = ds_grid[basins_name].where(
                ds_grid.coords["mask"], self.grid.data[basins_name].raster.nodata
            )
            ds_grid[basins_name].attrs.update(
                _FillValue=self.grid.data[basins_name].raster.nodata
            )
        elif bbox is not None:
            ds_grid = self.grid.data.raster.clip_bbox(bbox, align=align, buffer=buffer)

        # Update flwdir grid and geoms
        if self.crs is None and crs is not None:
            self.set_crs(crs)

        self._grid = xr.Dataset()
        self.grid.set(ds_grid)

        # add pits at edges after clipping
        self._flwdir = None  # make sure old flwdir object is removed
        self.grid.data[self._MAPS["flwdir"]].data = self.flwdir.to_array("ldd")

        # Reinitiliase geoms and re-create basins/rivers
        self._geoms = dict()
        # self.basins
        # self.rivers
        # now geoms links to geoms which does not exist in every hydromt version
        # remove when updating wflow to new objects
        # Basin shape
        basins = (
            self.grid.data[basins_name]
            .raster.vectorize()
            .set_index("value")
            .sort_index()
        )
        basins.index.name = basins_name
        self.set_geoms(basins, name="basins")

        rivmsk = self.grid.data[self._MAPS["rivmsk"]].values != 0
        # Check if there are river cells in the model before continuing
        if np.any(rivmsk):
            # add stream order 'strord' column
            strord = self.flwdir.stream_order(mask=rivmsk)
            feats = self.flwdir.streams(mask=rivmsk, strord=strord)
            gdf = gpd.GeoDataFrame.from_features(feats)
            gdf.crs = pyproj.CRS.from_user_input(self.crs)
            self.set_geoms(gdf, name="rivers")

        # Update reservoir and lakes
        remove_reservoir = False
        if self._MAPS["resareas"] in self.grid.data:
            reservoir = self.grid.data[self._MAPS["resareas"]]
            if not np.any(reservoir > 0):
                remove_reservoir = True
                remove_maps = [
                    self._MAPS["resareas"],
                    self._MAPS["reslocs"],
                    "ResSimpleArea",
                    "ResDemand",
                    "ResTargetFullFrac",
                    "ResTargetMinFrac",
                    "ResMaxRelease",
                    "ResMaxVolume",
                ]
                self._grid = self.grid.data.drop_vars(remove_maps)

        remove_lake = False
        if self._MAPS["lakeareas"] in self.grid.data:
            lake = self.grid.data[self._MAPS["lakeareas"]]
            if not np.any(lake > 0):
                remove_lake = True
                remove_maps = [
                    self._MAPS["lakeareas"],
                    self._MAPS["lakelocs"],
                    "LinkedLakeLocs",
                    "LakeStorFunc",
                    "LakeOutflowFunc",
                    "LakeArea",
                    "LakeAvgLevel",
                    "LakeAvgOut",
                    "LakeThreshold",
                    "Lake_b",
                    "Lake_e",
                ]
                self._grid = self.grid.data.drop_vars(remove_maps)

            # Update tables
            ids = np.unique(lake)
            self._tables = {
                k: v
                for k, v in self.tables.items()
                if not any([str(x) in k for x in ids])
            }

        # Update config
        # Remove the absolute path and if needed remove lakes and reservoirs
        if remove_reservoir:
            # change reservoirs = true to false
            self.model.config.set("model.reservoirs", False)
            # remove states
            self.model.config.delete_key("state.lateral.river.reservoir")

        if remove_lake:
            # change lakes = true to false
            self.model.config.set("model.lakes", False)
            # remove states
            self.model.config.delete_key("state.lateral.river.lake")
