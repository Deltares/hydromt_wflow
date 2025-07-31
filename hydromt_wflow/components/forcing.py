"""Forcing component module."""

import logging
import re
from pathlib import Path

import numpy as np
import xarray as xr
from hydromt._io import _write_nc
from hydromt.model import Model
from hydromt.model.components import GridComponent

__all__ = ["WflowForcingComponent"]

logger = logging.getLogger(f"hydromt.{__name__}")


class WflowForcingComponent(GridComponent):
    """Wflow forcing component."""

    def __init__(
        self,
        model: Model,
        *,
        filename: str = "inmaps.nc",
        region_component: str | None = None,
        region_filename: str = "geoms/forcing_region.geojson",
    ):
        """
        Initialize a WflowForcingComponent.

        Parameters
        ----------
        model : Model
            HydroMT model instance
        filename : str, optional
            The path to use for reading and writing of component data by default.
            By default "inmaps.nc".
        region_component : str, optional
            The name of the region component to use as reference for this component's
            region. If None, the region will be set to the grid extent. Note that the
            create method only works if the region_component is None.
        region_filename : str, optional
            The path to use for reading and writing of the region data by default.
            By default "geoms/forcing_region.geojson".
        """
        super().__init__(
            model=model,
            filename=filename,
            region_component=region_component,
            region_filename=region_filename,
        )

    ## I/O methods
    def read(
        self,
        filename: Path | str | None = None,
        **kwargs,
    ):
        """Read forcing model data.

        Key-word arguments are passed to :py:meth:`~hydromt._io.writers._write_nc`

        Parameters
        ----------
        filename : str, optional
            Filename relative to model root, by default None
        **kwargs : dict
            Additional keyword arguments to be passed to the `write_nc` method.
        """
        super().read(
            filename=filename,
            mask_and_scale=False,
            **kwargs,
        )

    def write(
        self,
        filename: Path | str | None = None,
        output_frequency: str | None = None,
        starttime: str | None = None,
        endtime: str | None = None,
        time_chunk: int = 1,
        time_units="days since 1900-01-01T00:00:00",
        **kwargs,
    ) -> tuple[Path, str, str]:
        """Write forcing model data.

        Key-word arguments are passed to :py:meth:`~hydromt._io.writers._write_nc`

        Parameters
        ----------
        filename : str, optional
            Filename relative to model root, by default None
        output_frequency : str, optional
            The frequency of the files being written. If e.g. '3M' (3 months) is
            specified then files are written with a maximum of 3 months worth of data.
            By default None
        time_units : str, optional
            Common time units when writing several netcdf forcing files.
            By default "days since 1900-01-01T00:00:00".
        **kwargs : dict
            Additional keyword arguments to be passed to the `write_nc` method.
        """
        self.root._assert_write_mode()
        # Make absolute
        filename = Path(self.root.path, filename)
        # Clean the filename when an asterisk is present
        filename = Path(filename.parent, re.sub(r"_?\*", "", filename.name))

        # Logging the output
        logger.info("Write forcing file")

        # Check if all dates between (starttime, endtime) are in all da forcing
        # Check if starttime and endtime timestamps are correct
        starttime = np.datetime64(
            starttime or np.datetime_as_string(min(self.data.time.values), unit="s")
        )
        if (
            starttime < min(self.data.time.values)
            or starttime not in self.data.time.values
        ):
            starttime = min(self.data.time.values)
        starttime = str(np.datetime_as_string(starttime, unit="s"))
        # Same for endtime
        endtime = np.datetime64(
            endtime or np.datetime_as_string(max(self.data.time.values), unit="s")
        )
        if endtime > max(self.data.time.values) or endtime not in self.data.time.values:
            endtime = max(self.data.time.values)
        endtime = str(np.datetime_as_string(endtime, unit="s"))

        # Clean-up forcing and write
        self._data = self.data.drop_vars(["mask", "idx_out"], errors="ignore")

        # Write with output chunksizes with single timestep and complete
        # spatial grid to speed up the reading from wflow.jl
        # Dims are always ordered (time, y, x)
        self.data.raster._check_dimensions()
        chunksizes = (
            time_chunk,
            self.data.raster.ycoords.size,
            self.data.raster.xcoords.size,
        )

        # Set the encoding for the outgoing dataset
        encoding = kwargs.pop("encoding", {})
        encoding.update(
            {
                v: {"dtype": "float32", "chunksizes": chunksizes}
                for v in self.data.data_vars.keys()
            }
        )
        encoding["time"] = {"units": time_units}

        # Write the file either in one go
        if output_frequency is None:
            # TODO remove logging message when core goes from debug to info
            logger.info(f"Writing file {filename.as_posix()}")
            _write_nc(
                self.data,
                filepath=filename,
                compress=True,
                gdal_compliant=True,
                compute=False,
                force_overwrite=True,
                encoding=encoding,
                **kwargs,
            )
        # Or per period
        else:
            logger.info(f"Writing several forcing with freq {output_frequency}")
            # Updating path forcing in config
            for _, data_freq in self.data.resample(time=output_frequency):
                # Sort out the outgoing filename
                start = data_freq["time"].dt.strftime("%Y%m%d")[0].item()
                filename_freq = Path(filename.parent, f"{filename.stem}_{start}.nc")
                # TODO Remove when core logging goes from debug to info
                logger.info(f"Writing file {filename_freq.as_posix()}")
                # Write to file
                _write_nc(
                    data_freq,
                    filepath=filename_freq,
                    compress=True,
                    gdal_compliant=True,
                    compute=False,
                    force_overwrite=True,
                    encoding=encoding,
                    **kwargs,
                )
            filename = Path(filename.parent, f"{filename.stem}_*{filename.suffix}")

        return filename, starttime, endtime

    ## Mutating methods
    def set(
        self,
        data: xr.DataArray | xr.Dataset,
        name: str | None = None,
    ):
        """Add data to grid.

        All layers of grid must have identical spatial coordinates.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to grid
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray
            and ignored if data is a Dataset
        """
        # Rename the temporal dimension to time
        if "time" not in data.dims:
            raise ValueError("'time' dimension not found in data")

        # Check spatial extend
        if not data.raster.identical_grid(self._get_grid_data()):
            raise ValueError(
                "Forcing data doesn't match the spatial extend of the staticmaps."
            )

        # Call set of parent class
        super().set(
            data=data,
            name=name,
        )
