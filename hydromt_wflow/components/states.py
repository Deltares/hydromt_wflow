"""Wflow states component."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr
from hydromt.model import Model
from hydromt.model.components import GridComponent

__all__ = ["WflowStatesComponent"]

logger = logging.getLogger(f"hydromt.{__name__}")


class WflowStatesComponent(GridComponent):
    """ModelComponent class for Wflow states.

    This class is used for setting, creating, writing, and reading the Wflow states.
    The states component data stored in the ``data`` property of this class is of the
    hydromt.gis.raster.RasterDataset type which is an extension of xarray.Dataset for
    regular grid.
    """

    def __init__(
        self,
        model: Model,
        *,
        filename: str = "instate/instates.nc",
        region_component: str | None = None,
        region_filename: str | None = "staticgeoms/states_region.geojson",
    ):
        """
        Initialize the Wflow states component.

        Parameters
        ----------
        model : Model
            HydroMT model instance.
        filename : str, optional
            Default path relative to the root where the states file will be read and
            written. By default 'instate/instates.nc'.
        region_component : str, optional
            Name of the region component to align the states data with. If provided,
            the states data will be aligned with the grid of this component.
            By default None.
        region_filename : str, optional
            A path relative to the root where the region file will be written.
            By default 'staticgeoms/states_region.geojson'.
        """
        super().__init__(
            model=model,
            filename=filename,
            region_component=region_component,
            region_filename=region_filename,
        )

    def set(
        self,
        data: xr.Dataset | xr.DataArray,
        name: str | None = None,
    ):
        """Set the states data.

        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The states data to set/add.
        name : str, optional
            Name of new data layer, this is used to overwrite the name of a DataArray
            and ignored if data is a Dataset.

        See Also
        --------
        hydromt.model.components.GridComponent.set
        """
        if not isinstance(data, (xr.Dataset, xr.DataArray)):
            raise TypeError("Data must be an xarray Dataset or DataArray")

        if self._region_component is not None:
            # Ensure the data is aligned with the region component (staticmaps)
            region_grid = self._get_grid_data()
            if not data.raster.identical_grid(region_grid):
                y_dim = region_grid.raster.y_dim
                x_dim = region_grid.raster.x_dim
                # First try to rename dimensions
                data = data.rename(
                    {
                        data.raster.x_dim: x_dim,
                        data.raster.y_dim: y_dim,
                    }
                )
                # Flip latitude if needed
                if (
                    np.diff(data[y_dim].values)[0] > 0
                    and np.diff(region_grid[y_dim].values)[0] < 0
                ):
                    data = data.reindex({y_dim: data[y_dim][::-1]})
            # Check again, this time the grid is really different if not True
            if not data.raster.identical_grid(region_grid):
                raise ValueError(
                    f"Data grid must be identical to {self._region_component} component"
                )

        super().set(
            data=data,
            name=name,
        )

    # I/O methods
    def read(self):
        """
        Read states at <root/dir_input/state.path_input>.

        Checks the path of the file in the config toml using both ``state.path_input``
        and ``dir_input``. If not found uses the default path ``instate/instates.nc``
        in the root folder.

        Parameters
        ----------
        filename : str or Path, optional
            A path relative to the root where the states file will be read from.
            By default 'instate/instates.nc'.

        See Also
        --------
        hydromt.model.components.GridComponent.read
        """
        # Sort which path/ filename is actually the one used
        # Hierarchy is: 1: signature, 2: config, 3: default
        p = self.model.config.get_value("state.path_input") or self._filename
        # Check for input dir
        p_input = Path(self.model.config.get_value("dir_input", fallback=""), p)
        super().read(
            filename=p_input,
            mask_and_scale=False,
        )

    def write(self, filename: str | None = None):
        """
        Write states at <root/dir_input/state.path_input> in model ready format.

        Checks the path of the file in the config toml using both ``state.path_input``
        and ``dir_input``. If not found uses the default path ``instate/instates.nc``
        in the root folder.
        If filename is provided, it will be used and config ``state.path_input``
        will be updated accordingly.

        Parameters
        ----------
        filename : str or Path, optional
            Name of the states file, relative to model root and ``dir_input`` if any.
            By default None to use the name as defined in the model config file.

        See Also
        --------
        hydromt.model.components.GridComponent.write
        """
        # Sort which path/ filename is actually the one used
        # Hierarchy is: 1: signature, 2: config, 3: default
        p = (
            filename
            or self.model.config.get_value("state.path_input")
            or self._filename
        )
        # Check for output dir
        p_output = Path(self.model.config.get_value("dir_input", fallback=""), p)

        # make sure no _FillValue is written to the time dimension
        if "time" in self.data:
            self._data["time"].attrs.pop("_FillValue", None)

        super().write(
            filename=p_output,
            gdal_compliant=True,
            rename_dims=True,
            force_sn=False,
        )

        # Update the config
        self.model.config.set("state.path_input", p)
