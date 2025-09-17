"""Staticmaps component module."""

import logging
from pathlib import Path

import hydromt
import numpy as np
import xarray as xr
from hydromt.gis import flw
from hydromt.model import Model
from hydromt.model.components import GridComponent, ModelComponent
from hydromt.model.processes.basin_mask import get_basin_geometry
from hydromt.model.processes.region import (
    _parse_region_value,
)
from hydromt.model.steps import hydromt_step

from hydromt_wflow.components.utils import get_mask_layer, test_equal_grid_data

__all__ = ["WflowStaticmapsComponent"]

logger = logging.getLogger(f"hydromt.{__name__}")


class WflowStaticmapsComponent(GridComponent):
    """Wflow staticmaps component.

    Inherits from the HydroMT-core GridComponent model-component.
    It is used for setting, creating, writing, and reading static and cyclic data for a
    Wflow model on a regular grid. The component data, stored in the ``data``
    property of this class, is of the hydromt.gis.raster.RasterDataset type which
    is an extension of xarray.Dataset for regular grid.
    """

    def __init__(
        self,
        model: Model,
        *,
        filename: str = "staticmaps.nc",
        region_filename: str = "staticgeoms/region.geojson",
    ):
        """Initialize a WflowStaticmapsComponent.

        Parameters
        ----------
        model : Model
            HydroMT model instance
        filename : str
            The path to use for reading and writing of component data by default.
            By default "staticmaps.nc".
        region_filename : str
            The path to use for reading and writing of the region data by default.
            By default "staticgeoms/region.geojson".
        """
        super().__init__(
            model,
            filename=filename,
            region_component=None,
            region_filename=region_filename,
        )

    ## I/O methods
    @hydromt_step
    def read(
        self,
        **kwargs,
    ):
        """Read staticmaps model data.

        Checks the path of the file in the config toml using both ``input.path_static``
        and ``dir_input``. If not found uses the default path ``staticmaps.nc`` in the
        root folder.
        Key-word arguments are passed to :py:meth:`~hydromt._io.readers._read_nc`

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to the `read_nc` method.
        """
        # Sort which path/ filename is actually the one used
        # Hierarchy is: 1: config, 2: default
        p = self.model.config.get_value("input.path_static") or self._filename
        # Check for input dir
        p_input = Path(self.model.config.get_value("dir_input", fallback=""), p)

        # Supercharge with parent method
        super().read(
            filename=p_input,
            mask_and_scale=False,
            **kwargs,
        )

    @hydromt_step
    def write(
        self,
        filename: str | None = None,
        **kwargs,
    ):
        """Write staticmaps model data.

        Checks the path of the file in the config toml using both ``input.path_static``
        and ``dir_input``. If not found uses the default path ``staticmaps.nc`` in the
        root folder.

        If filename is supplied, the config will be updated.

        Key-word arguments are passed to :py:meth:`~hydromt._io.writers._write_nc`

        Parameters
        ----------
        filename : Path, str, optional
            Name or path to the outgoing staticmaps file (including extension).
            This is the path/name relative to the root folder and if present the
            ``dir_input`` folder. By default None.
        **kwargs : dict
            Additional keyword arguments to be passed to the `write_nc` method.
        """
        # Solve pathing same as read
        # Hierarchy is: 1: signature, 2: config, 3: default
        p = (
            filename
            or self.model.config.get_value("input.path_static")
            or self._filename
        )
        # Check for input dir
        p_input = Path(self.model.config.get_value("dir_input", fallback=""), p)

        # Supercharge with the base grid component write method
        super().write(
            str(p_input),
            gdal_compliant=True,
            rename_dims=True,
            force_sn=False,
            **kwargs,
        )

        # Set the config entry to the correct path
        self.model.config.set(
            "input.path_static",
            Path(self.root.path, p_input).as_posix(),
        )

    ## Mutating methods
    def set(
        self,
        data: xr.DataArray | xr.Dataset,
        name: str | None = None,
    ):
        """Add data to the staticmaps.

        All layers of grid must have identical spatial coordinates. If basin data is
        available the grid will be masked to that upon setting.

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
        data : xr.DataArray | xr.Dataset
            new map layer to add to grid
        name : str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray
            and ignored if data is a Dataset
        """
        self._initialize_grid()
        assert self._data is not None

        name_required = isinstance(data, xr.DataArray) and data.name is None
        if name is None and name_required:
            raise ValueError(f"Unable to set {type(data).__name__} data without a name")
        elif isinstance(data, xr.DataArray):
            if name is not None:
                data.name = name
            data = data.to_dataset()
        elif not isinstance(data, xr.Dataset):
            raise ValueError(f"Cannot set data of type {type(data).__name__}")

        # Check for cyclic data
        if "time" in data.dims:
            # Raise error if the dimension does not have a supported length
            if len(data.time) not in [12, 365, 366]:
                raise ValueError(
                    f"Length of cyclic dataset ({len(data.time)}) is not supported by "
                    "Wflow.jl. Ensure the data has length 12, 365, or 366"
                )
            tname = "time"
            time_axes = {
                k: v for k, v in dict(self.data.sizes).items() if k.startswith("time")
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
        if "layer" in data.dims and "layer" in self.data:
            if len(data["layer"]) != len(self.data["layer"]):
                vars_to_drop = [
                    var for var in self.data.data_vars if "layer" in self.data[var].dims
                ]
                # Drop variables
                logger.warning(
                    f"Replacing 'layer' coordinate, dropping variables \
({vars_to_drop}) associated with old coordinate"
                )
                # Use `_data` as `data` cannot be set
                self.drop_vars(vars_to_drop + ["layer"])

        # Check if north is really up and south therefore is down
        if data.raster.res[1] > 0:
            data = data.raster.flipud()

        # Determine the masking layer
        mask = get_mask_layer(self.model._MAPS.get("basins"), self.data, data)

        # Set the data per layer
        for dvar in data.data_vars:
            if dvar in self._data:
                logger.info(f"Replacing grid map: {dvar}")
            if mask is not None:
                if data[dvar].dtype != np.bool:
                    data[dvar] = data[dvar].where(mask, data[dvar].raster.nodata)
                else:
                    data[dvar] = data[dvar].where(mask, False)
            self._data[dvar] = data[dvar]

    def drop_vars(self, names: list[str], errors: str = "raise"):
        """
        Drop variables from the grid.

        This method is a wrapper around the xarray.Dataset.drop_vars method.

        Parameters
        ----------
        names : list of str
            List of variable names to drop from the grid.
        errors : str, optional {raise, ignore}
            How to handle errors. If 'raise', raises a ValueError error if any of the
            variable passed are not in the dataset. If 'ignore', any given names that
            are in the dataset are dropped and no error is raised.
        """
        self._data = self.data.drop_vars(names, errors=errors)

    def clip(
        self,
        region: dict,
        inverse_clip: bool = False,
        crs: int = 4326,
        basins_name: str = "subcatchment",
        flwdir_name: str = "local_drain_direction",
        reservoir_name: str = "reservoir_area_id",
        reservoir_maps: list[str] = [],
        **kwargs,
    ):
        """
        Clip staticmaps to region.

        Staticmaps are clipped to the region defined in the `region` argument. If
        `inverse_clip` is True, the upstream part of the model is removed instead of
        the subbasin itself. After clipping, the flow direction map is re-derived to
        ensure that all edges of the clipped model are pits. If reservoir maps are
        present in the staticmaps but no reservoirs are present in the clipped model,
        these maps are removed from the staticmaps.

        Note that the outlets are not re-derived in this function and should be done
        separately using `WflowBaseModel.setup_outlets`.

        Parameters
        ----------
        region : dict
            The region to clip to.
        inverse_clip: bool, optional
            Flag to perform "inverse clipping": removing an upstream part of the model
            instead of the subbasin itself, by default False
        crs: int, optional
            Default crs of the model in case it cannot be read.
        basins_name: str, optional
            Name of the basin/subbasin variable in the staticmaps data, by default
            "subcatchment"
        flwdir_name: str, optional
            Name of the flow direction variable in the staticmaps data, by default
            "local_drain_direction"
        reservoir_name: str, optional
            Name of the reservoir id variable in the staticmaps data, by default
            "reservoir_area_id"
        reservoir_maps: list of str, optional
            List of map names in the wflow model grid to be treated as reservoirs.
            These maps are removed if empty after clipping.
        **kwargs: dict
            Additional keyword arguments passed to
            :py:meth:`~hydromt.raster.Raster.clip_geom`
        """
        # translate basin and outlet kinds to geom
        # get basin geometry and clip data
        logger.debug(f"Clipping staticmaps to region: {region}")
        kind = next(iter(region))
        if kind in ["basin", "subbasin"]:
            # parse_region_basin does not return xy, only geom...
            # should be fixed in core
            region_kwargs = _parse_region_value(
                region.pop(kind),
                data_catalog=self.data_catalog,
            )
            region_kwargs.update(region)
            geom, _ = get_basin_geometry(
                ds=self.data,
                kind=kind,
                basins_name=basins_name,
                flwdir_name=flwdir_name,
                **region_kwargs,
            )
        elif kind == "bbox":
            logger.warning(
                "Kind 'bbox' for the region is not recommended as it can lead "
                "to mistakes in the catchment delineation. Use carefully."
            )
            geom = hydromt.processes.region.parse_region_bbox(region)
        elif kind == "geom":
            logger.warning(
                "Kind 'geom' for the region is not recommended as it can lead "
                "to mistakes in the catchment delineation. Use carefully."
            )
            geom = hydromt.processes.region.parse_region_geom(region)
        else:
            raise ValueError(
                f"wflow region kind not understood or supported: {kind}. "
                "Use 'basin', 'subbasin', 'bbox' or 'geom'."
            )

        # Remove upstream part from model if inverse_clip
        if inverse_clip:
            logger.debug("Performing inverse clipping of staticmaps")
            basins = self.data[basins_name].raster.vectorize()
            geom = basins.overlay(geom, how="difference")

        # clip based on subbasin args, geom or bbox
        ds_grid = self.data.raster.clip_geom(geom, **kwargs)
        ds_grid.coords["mask"] = ds_grid.raster.geometry_mask(geom)
        ds_grid[basins_name] = ds_grid[basins_name].where(
            ds_grid.coords["mask"], self.data[basins_name].raster.nodata
        )
        ds_grid[basins_name].attrs.update(
            _FillValue=self.data[basins_name].raster.nodata
        )

        # Check reservoirs and remove maps if empty
        if reservoir_name in ds_grid:
            reservoir = ds_grid[reservoir_name]
            if not np.any(reservoir > 0):
                logger.info(
                    "No reservoirs present in the clipped model, removing them from "
                    "staticmaps."
                )
                ds_grid = ds_grid.drop_vars(reservoir_maps)

        # Re-derive flwdir after clipping (add pits at edges)
        _flwdir = flw.flwdir_from_da(
            ds_grid[flwdir_name],
            ftype="infer",
            check_ftype=True,
            mask=(ds_grid[basins_name] > 0),
        )
        ds_grid[flwdir_name].data = _flwdir.to_array("ldd")

        # Check CRS
        if self.data.raster.crs is None and crs is not None:
            ds_grid.raster.set_crs(crs)

        # Update staticmaps data
        self._data = xr.Dataset()
        self.set(ds_grid)

    ## Setup and update methods
    @hydromt_step
    def update_names(
        self,
        **rename,
    ):
        """Map the names of the data variables to new ones.

        This method however does not change the new names in the config file.
        To update config file entries, you can use it together
        with the `WflowBaseModel.setup_config` method.

        Parameters
        ----------
        rename : dict, optional
            Keyword arguments that map the old names to the new names.
            So < old-name > = < new-name >.
        """
        # Check whether they are in the maps
        not_found = []
        for key in list(rename.keys()):
            if key not in self.data.data_vars:
                not_found.append(key)
                _ = rename.pop(key)
        if len(not_found) != 0:
            logger.warning(f"Could not rename {not_found}, not found in data")
        self._data = self.data.rename(**rename)

    def test_equal(self, other: ModelComponent) -> tuple[bool, dict[str, str]]:
        """Test if two staticmaps components are equal.

        Checks the model component type as well as the data variables and their values.

        Parameters
        ----------
        other : ModelComponent
            The component to compare against.

        Returns
        -------
        tuple[bool, Dict[str, str]]
            True if the components are equal, and a dict with the associated errors per
            property checked.
        """
        errors: dict[str, str] = {}
        if not isinstance(other, self.__class__):
            errors["__class__"] = f"other does not inherit from {self.__class__}."

        # Check dimensions
        _, data_errors = test_equal_grid_data(self.data, other.data)
        errors.update(data_errors)

        return len(errors) == 0, errors
