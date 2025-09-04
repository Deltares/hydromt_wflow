"""Wflow netcdf_scalar output component."""

import logging
from pathlib import Path
from typing import cast

import xarray as xr
from hydromt.model import Model
from hydromt.model.components import ModelComponent

__all__ = ["WflowOutputScalarComponent"]

logger = logging.getLogger(f"hydromt.{__name__}")


class WflowOutputScalarComponent(ModelComponent):
    """ModelComponent class for Wflow netcdf_scalar output.

    This class is used for reading the Wflow netcdf_scalar output.

    The overall output netcdf_scalar component data stored in the ``data`` property of
    this class is a xarray.Dataset.
    """

    def __init__(
        self,
        model: Model,
        *,
        filename: str = "output_scalar.nc",
    ):
        """
        Initialize the Wflow netcdf_scalar output component.

        Parameters
        ----------
        model : Model
            HydroMT model instance.
        filename : str, optional
            Default path relative to the root where the netcdf_scalar output file will
            be read and written. By default 'output_scalar.nc'.
        """
        self._data: xr.Dataset | None = None
        self._filename: str = filename

        super().__init__(
            model=model,
        )

    ## I/O methods
    def read(self):
        """Read netcdf_scalar model output at root/dir_output/filename.

        Checks the path of the file in the config toml using both
        ``output.netcdf_scalar.path`` and ``dir_output``. If not found uses the default
        path ``output_scalar.nc`` in the root folder.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to the `read_nc` method.
        """
        # Sort which path/ filename is actually the one used
        # Hierarchy is: 1: config, 2: default from component
        p = self.model.config.get_value("output.netcdf_scalar.path") or self._filename
        # Check for input dir
        p_input = Path(self.model.config.get_value("dir_output", fallback=""), p)
        # Add root
        p_input = Path(self.root.path, p_input)

        if not p_input.is_file():
            logger.warning(
                f"Netcdf scalar output file {p_input} not found, skip reading."
            )
            return

        logger.info(f"Read netcdf_scalar output from {p_input}")
        with xr.open_dataset(p_input, chunks={"time": 30}) as ds:
            self.set(ds)

    def write(self):
        """
        Skip writing output files.

        Output files are model results and are therefore not written by HydroMT.
        """
        logger.warning("netcdf_scalar is an output of Wflow and will not be written.")

    ## Set methods
    def _initialize(self, skip_read: bool = False) -> None:
        """Initialize netcdf_scalar output object."""
        if self._data is None:
            self._data = xr.Dataset()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    def set(
        self,
        data: xr.DataArray | xr.Dataset,
        name: str | None = None,
    ):
        """Add data to netcdf_scalar output.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to netcdf_scalar output
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray
            and ignored if data is a Dataset
        """
        self._initialize()
        assert self._data is not None

        if isinstance(data, xr.DataArray):
            # NOTE _name can be different from _data.name !
            if name is not None:
                data.name = name
            elif name is None and data.name is not None:
                name = data.name
            elif data.name is None and name is None:
                raise ValueError("Name required for DataArray.")
            data = data.to_dataset()

        if not isinstance(data, xr.Dataset):
            raise TypeError(f"cannot set data of type {type(data).__name__}")

        if len(self._data) == 0:  # empty grid
            self._data = data
        else:
            for dvar in data.data_vars:
                if dvar in self._data:
                    logger.warning(f"Replacing netcdf_scalar output: {dvar}")
                self._data[dvar] = data[dvar]

    ## Properties
    @property
    def data(self) -> xr.Dataset:
        """Model netcdf_scalar output data as xarray.Dataset."""
        if self._data is None:
            self._initialize()
        assert self._data is not None
        return self._data

    ## Testing
    def test_equal(self, other: ModelComponent) -> tuple[bool, dict[str, str]]:
        """Test if two components are equal.

        Parameters
        ----------
        other : ModelComponent
            The component to compare against.

        Returns
        -------
        Tuple[bool, Dict[str, str]]
            True if the components are equal, and a dict with the associated errors per
            property checked.
        """
        eq, errors = super().test_equal(other)
        if not eq:
            return eq, errors
        other_grid = cast(WflowOutputScalarComponent, other)
        try:
            xr.testing.assert_allclose(self.data, other_grid.data)
        except AssertionError as e:
            errors["data"] = str(e)

        return len(errors) == 0, errors
