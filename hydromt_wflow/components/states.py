"""Wflow states component."""

import logging
from pathlib import Path

import xarray as xr
from hydromt.model import Model
from hydromt.model.components import GridComponent, ModelComponent

from hydromt_wflow.components import utils

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
            data = utils.align_grid_with_region(
                data,
                region_grid=self._get_grid_data(),
                region_component_name=self._region_component,
            )

        super().set(
            data=data,
            name=name,
        )

    def drop_vars(self, names: list[str], errors: str = "raise"):
        """
        Drop variables from the states.

        This method is a wrapper around the xarray.Dataset.drop_vars method.

        Parameters
        ----------
        names : list of str
            List of variable names to drop from the states.
        errors : str, optional {raise, ignore}
            How to handle errors. If 'raise', raises a ValueError error if any of the
            variable passed are not in the dataset. If 'ignore', any given names that
            are in the dataset are dropped and no error is raised.
        """
        self._data = self.data.drop_vars(names, errors=errors)

    def clip(
        self,
        reservoir_name: str = "reservoir_area_id",
        reservoir_states: list[str] = [],
    ):
        """Clip the states component to the region of the region component.

        If no region component is set, the states component is clipped to its own
        extent.

        Parameters
        ----------
        reservoir_name : str, optional
            Name of the reservoir id variable in the region component, by default
            "reservoir_area_id"
        reservoir_states : list of str, optional
            List of state names in the wflow model states to be treated as reservoirs.
            These states are removed if empty after clipping.
        """
        if self._region_component is None:
            logger.info("No region component set, state component will not be clipped.")
            return

        if len(self.data) > 0:
            logger.info("Clipping state...")
            # Clip states to region component extent
            region_grid = self._get_grid_data()
            ds_states = self.data.raster.clip_bbox(region_grid.raster.bounds)

            # Check reservoirs and remove states if empty
            if len(reservoir_states) > 0 and reservoir_name not in region_grid:
                # no reservoirs in the clipped or original model
                ds_states = ds_states.drop_vars(reservoir_states, errors="ignore")

            # Update states data
            self._data = xr.Dataset()  # clear existing states
            self.set(ds_states)

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
        _, data_errors = utils.test_equal_grid_data(self.data, other.data)
        errors.update(data_errors)

        return len(errors) == 0, errors
