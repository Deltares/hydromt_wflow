"""Staticmaps component module."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr
from hydromt.model import Model
from hydromt.model.components import GridComponent
from hydromt.model.steps import hydromt_step

from hydromt_wflow.components.utils import get_mask_layer

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
        filename: Path | str | None = None,
        **kwargs,
    ):
        """Read staticmaps model data.

        Key-word arguments are passed to :py:meth:`~hydromt._io.readers._read_nc`

        Parameters
        ----------
        filename : str, optional
            Filename relative to model root, by default None
        **kwargs : dict
            Additional keyword arguments to be passed to the `read_nc` method.
        """
        # Supercharge with parent method
        super().read(
            filename=filename,
            mask_and_scale=False,
            **kwargs,
        )

    @hydromt_step
    def write(
        self,
        filename: Path | str | None = None,
        **kwargs,
    ):
        """Write staticmaps model data.

        Key-word arguments are passed to :py:meth:`~hydromt._io.writers._write_nc`

        Parameters
        ----------
        filename : str, optional
            Filename relative to model root, by default None
        **kwargs : dict
            Additional keyword arguments to be passed to the `write_nc` method.
        """
        # Supercharge with the base grid component write method
        super().write(
            str(filename) if filename is not None else None,
            gdal_compliant=True,
            rename_dims=True,
            force_sn=False,
            **kwargs,
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
                    var for var in self.data.data_vars if "layer" in self.data[var].dims
                ]
                # Drop variables
                logger.warning(
                    f"Replacing 'layer' coordinate, dropping variables \
({vars_to_drop}) associated with old coordinate"
                )
                # Use `_data` as `data` cannot be set
                self._data = self.data.drop_vars(vars_to_drop + ["layer"])

        # Check if noth is really up and south therefore is down
        if data.raster.res[1] > 0:
            data = data.raster.flipud()

        # Determine the masking layer
        mask = get_mask_layer(self.model._MAPS.get("basins"), self.data, data)

        # Set the data per layer
        for dvar in data.data_vars:
            if dvar in self._data:
                logger.warning(f"Replacing grid map: {dvar}")
            if mask is not None:
                if data[dvar].dtype != np.bool:
                    data[dvar] = data[dvar].where(mask, data[dvar].raster.nodata)
                else:
                    data[dvar] = data[dvar].where(mask, False)
            self._data[dvar] = data[dvar]

    ## Setup and update methods
    @hydromt_step
    def update_names(
        self,
        **rename,
    ):
        """Map the names of the data variables to new ones.

        This method however does not change the new names in the config file.
        To update config file entries, you can use it together
        with the `WflowModel.setup_config` method.

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
