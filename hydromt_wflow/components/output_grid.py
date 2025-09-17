"""Wflow netcdf_grid output component."""

import logging
from pathlib import Path

import xarray as xr
from hydromt.model import Model
from hydromt.model.components import GridComponent

from hydromt_wflow.components import utils

__all__ = ["WflowOutputGridComponent"]

logger = logging.getLogger(f"hydromt.{__name__}")


class WflowOutputGridComponent(GridComponent):
    """ModelComponent class for Wflow netcdf_grid output.

    This class is used for reading the Wflow netcdf_grid output.

    The overall output netcdf_grid component data stored in the ``data`` property of
    this class is of the hydromt.gis.raster.RasterDataset type which is an extension of
    xarray.Dataset for regular grid.
    """

    def __init__(
        self,
        model: Model,
        *,
        filename: str = "output_grid.nc",
        region_component: str | None = None,
    ):
        """
        Initialize the Wflow netcdf_grid output component.

        Parameters
        ----------
        model : Model
            HydroMT model instance.
        filename : str, optional
            Default path relative to the root where the netcdf_grid output file will be
            read and written. By default 'output_grid.nc'.
        region_component : str, optional
            Name of the region component to align the states data with. If provided,
            the states data will be aligned with the grid of this component.
            By default None.
        """
        super().__init__(
            model=model,
            filename=filename,
            region_component=region_component,
            region_filename=None,
        )

    ## I/O methods
    def read(self):
        """Read netcdf_grid model output at root/dir_output/filename.

        Checks the path of the file in the config toml using both
        ``output.netcdf_grid.path`` and ``dir_output``. If not found uses the default
        path ``output_grid.nc`` in the root folder.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to the `read_nc` method.
        """
        # Sort which path/ filename is actually the one used
        # Hierarchy is: 1: config, 2: default from component
        p = self.model.config.get_value("output.netcdf_grid.path") or self._filename
        # Check for input dir
        p_input = Path(self.model.config.get_value("dir_output", fallback=""), p)
        # Add root
        p_input = Path(self.root.path, p_input)

        if not p_input.is_file():
            logger.warning(
                f"Netcdf grid output file {p_input} not found, skip reading."
            )
            return

        logger.info(f"Read netcdf_grid output from {p_input}")
        with xr.open_dataset(p_input, chunks={"time": 30}, decode_coords="all") as ds:
            self.set(ds)

    def write(self):
        """
        Skip writing output files.

        Output files are model results and are therefore not written by HydroMT.
        """
        logger.warning("netcdf_grid is an output of Wflow and will not be written.")

    ## Set methods
    def set(
        self,
        data: xr.DataArray | xr.Dataset,
        name: str | None = None,
    ):
        """Add data to netcdf_grid output.

        All layers in data must have identical spatial coordinates to existing
        staticmaps.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to netcdf_grid output
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray
            and ignored if data is a Dataset
        """
        # Check the type
        if not isinstance(data, (xr.DataArray, xr.Dataset)):
            raise TypeError("Data must be an xarray Dataset or DataArray")

        # Check that time dimension is present
        if "time" not in data.dims:
            raise ValueError("'time' dimension not found in data")

        if self._region_component is not None:
            # Ensure the data is aligned with the region component (staticmaps)
            data = utils.align_grid_with_region(
                data,
                region_grid=self._get_grid_data(),
                region_component_name=self._region_component,
            )

        # Call set of parent class
        super().set(
            data=data,
            name=name,
        )
