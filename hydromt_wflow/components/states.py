"""Wflow states component."""

import logging
from pathlib import Path

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
                raise ValueError(
                    f"Data grid must be identical to {self._region_component} component"
                )

        super().set(
            data=data,
            name=name,
        )

    # I/O methods
    def read(self, filename: Path | str | None = None):
        """
        Read the wflow states at root/filename.

        Parameters
        ----------
        filename : str or Path, optional
            A path relative to the root where the states file will be read from.
            By default 'instate/instates.nc'.

        See Also
        --------
        hydromt.model.components.GridComponent.read
        """
        super().read(
            filename=filename,
            mask_and_scale=False,
        )

    def write(self, filename: Path | str | None = None):
        """
        Write the wflow states to root/filename.

        Parameters
        ----------
        filename : str or Path, optional
            A path relative to the root where the states file will be written to.
            By default 'instate/instates.nc'.

        See Also
        --------
        hydromt.model.components.GridComponent.write
        """
        # make sure no _FillValue is written to the time dimension
        if "time" in self.data:
            self._data["time"].attrs.pop("_FillValue", None)

        super().write(
            filename=filename,
            gdal_compliant=True,
            rename_dims=True,
            force_sn=False,
        )
