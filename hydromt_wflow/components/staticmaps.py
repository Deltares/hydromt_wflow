"""Custom staticmaps component."""

import logging
from pathlib import Path

import xarray as xr
from hydromt._io.readers import _read_nc
from hydromt.model import Model
from hydromt.model.components import GridComponent
from hydromt.model.steps import hydromt_step

__all__ = ["StaticmapsComponent"]

logger = logging.getLogger(f"hydromt.{__name__}")


class StaticmapsComponent(GridComponent):
    """Custom staticmaps component.

    Inherits from the HydroMT-core GridComponent model-component.

    Parameters
    ----------
    model : Model
        HydroMT model instance
    filename : str
        The path to use for reading and writing of component data by default.
        By default "staticmaps.nc".
    region_component : str, optional
        The name of the region component to use as reference
        for this component's region. If None, the region will be set to the grid extent.
        Note that the create method only works if the region_component is None.
        For add_data_from_* methods, the other region_component should be
        a reference to another grid component for correct reprojection, by default None
    region_filename : str
        The path to use for reading and writing of the region data by default.
        By default "region.geojson".
    """

    def __init__(
        self,
        model: Model,
        *,
        filename: str = "staticmaps_blep.nc",
        region_component: str | None = None,
        region_filename: str = "region.geojson",
    ):
        GridComponent.__init__(
            self,
            model,
            filename=filename,
            region_component=region_component,
            region_filename=region_filename,
        )

    ## I/O methods
    @hydromt_step
    def read(
        self,
        filename: Path | str | None = None,
        *,
        mask_and_scale: bool = False,
        **kwargs,
    ):
        """Read staticmaps model data.

        Key-word arguments are passed to :py:meth:`~hydromt.model.Model.read_nc`

        Parameters
        ----------
        filename : str, optional
            filename relative to model root, by default 'grid/grid.nc'
        mask_and_scale : bool, optional
            If True, replace array values equal to _FillValue with NA and scale values
            according to the formula original_values * scale_factor + add_offset, where
            _FillValue, scale_factor and add_offset are taken from variable attributes
            (if they exist).
        **kwargs : dict
            Additional keyword arguments to be passed to the `read_nc` method.
        """
        self.root._assert_read_mode()
        self._initialize_grid(skip_read=True)

        # Sort which path/ filename is actually the one used
        # Hierarchy is: 1: signature, 2: config, 3: default
        p = filename or self.model.config.get("input.path_static") or self._filename
        # Check for input dir
        p_input = self.model.config.get("dir_input", fallback="")

        # Load grid data in r+ mode to allow overwriting netcdf files
        if self.root.is_reading_mode() and self.root.is_writing_mode():
            kwargs["load"] = True
        loaded_nc_files = _read_nc(
            Path(p_input, p),
            self.root.path,
            single_var_as_array=False,
            mask_and_scale=mask_and_scale,
            **kwargs,
        )
        for ds in loaded_nc_files.values():
            if ds.raster.res[1] > 0:  # Make sure its north-south oriented
                ds = ds.raster.flipud()
            self.set(ds)

    ## Mutating methods
    def set(
        self,
        data: xr.DataArray | xr.Dataset,
        name: str | None = None,
    ):
        """Add data to the staticmaps.

        All layers of grid must have identical spatial coordinates.

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
            raise ValueError(f"cannot set data of type {type(data).__name__}")

        # Check for cyclic data
        if "time" in data.dims:
            # Raise error if the dimension does not have a supported length
            if len(data.time) not in [12, 365, 366]:
                raise ValueError(
                    f"Length of cyclic dataset ({len(data)}) is not supported by "
                    "Wflow.jl. Ensure the data has length 12, 365, or 366"
                )
            tname = "time"
            time_axes = {
                k: v for k, v in self.data.sizes.items() if k.startswith("time")
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
                    var for var in self.data.variables if "layer" in self.data[var].dims
                ]
                # Drop variables
                logger.info(
                    "Dropping these variables, as they depend on the layer "
                    f"dimension: {vars_to_drop}"
                )
                # Use `_grid` as `grid` cannot be set
                self._data = self.data.drop_vars(vars_to_drop)

        # Really set the data
        if len(self._data) == 0:  # empty grid
            self._data = data
        else:
            for dvar in data.data_vars:
                if dvar in self._data and self.root.is_reading_mode():
                    logger.warning(f"Replacing grid map: {dvar}")
                self._data[dvar] = data[dvar]

    ## Setup and update methods
    def update_names(**select):
        """_summary_."""
        pass
