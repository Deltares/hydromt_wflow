"""Wflow forcing component."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from hydromt.io import write_nc
from hydromt.model import Model
from hydromt.model.components import GridComponent

from hydromt_wflow.components import utils

__all__ = ["WflowForcingComponent"]

logger = logging.getLogger(f"hydromt.{__name__}")


class WflowForcingComponent(GridComponent):
    """Wflow forcing component.

    This class is used for setting, creating, writing, and reading the Wflow forcing.
    The forcing component data stored in the ``data`` property of this class is of the
    hydromt.gis.raster.RasterDataset type which is an extension of xarray.Dataset for
    regular grid.
    """

    def __init__(
        self,
        model: Model,
        *,
        filename: str = "inmaps.nc",
        region_component: str | None = None,
        region_filename: str = "staticgeoms/forcing_region.geojson",
    ):
        """
        Initialize a WflowForcingComponent.

        Parameters
        ----------
        model : Model
            HydroMT model instance
        filename : str, optional
            Default path relative to the root where the forcing file(s) will be read
            and written. By default 'inmaps.nc'.
        region_component : str, optional
            The name of the region component to use as reference for this component's
            region. If None, the region will be set to the grid extent.
        region_filename : str, optional
            A path relative to the root where the region file will be written.
            By default 'staticgeoms/forcing_region.geojson'.
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
        **kwargs,
    ):
        """Read forcing model data at root/dir_input/filename.

        Checks the path of the file in the config toml using both ``input.path_forcing``
        and ``dir_input``. If not found uses the default path ``inmaps.nc`` in the
        root folder.

        If several files are used using '*' in ``input.path_forcing``, all corresponding
        files are read and merged into one xarray dataset before being split to one
        xarray DataArray per forcing variable in the hydromt ``forcing`` dictionary.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to the `read_nc` method.
        """
        # Sort which path/ filename is actually the one used
        # Hierarchy is: 1: config, 2: default from component
        p = self.model.config.get_value("input.path_forcing") or self._filename
        # Check for input dir
        p_input = Path(self.model.config.get_value("dir_input", fallback=""), p)

        super().read(
            filename=p_input,
            mask_and_scale=False,
            **kwargs,
        )

    def write(
        self,
        *,
        filename: str | None = None,
        output_frequency: str | None = None,
        time_chunk: int = 1,
        time_units="days since 1900-01-01T00:00:00",
        decimals: int = 2,
        overwrite: bool = False,
        **kwargs,
    ):
        """Write forcing model data.

        If no ``filename` path is provided and path_forcing from the wflow toml exists,
        the following default filenames are used:

            * Default name format (with downscaling):
              inmaps_sourcePd_sourceTd_methodPET_freq_startyear_endyear.nc
            * Default name format (no downscaling):
              inmaps_sourceP_sourceT_methodPET_freq_startyear_endyear.nc

        Key-word arguments are passed to :py:meth:`~hydromt._io.writers.write_nc`

        Parameters
        ----------
        filename : str, optional
            Filepath to write the forcing data to, either absolute or relative to the
            model root. If None, the default filename will be used.
        output_frequency : str, optional
            The frequency of the files being written. If e.g. '3M' (3 months) is
            specified then files are written with a maximum of 3 months worth of data.
            By default None
        time_chunk: int, optional
            The chunk size for the time dimension when writing the forcing data.
            By default 1.
        time_units : str, optional
            Common time units when writing several netcdf forcing files.
            By default "days since 1900-01-01T00:00:00".
        decimals : int, optional
            Number of decimals to use when writing the forcing data.
            By default 2.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is ``False`` unless the model
            is in w+ mode (FORCED_WRITE).
        **kwargs : dict
            Additional keyword arguments to be passed to the `write_nc` method.
        """
        self.root._assert_write_mode()
        overwrite = overwrite or self.root.mode.value == "w+"

        # Dont write if data is empty
        if self._data_is_empty():
            logger.warning(
                "Write forcing skipped: dataset is empty (no variables or data)."
            )
            return

        # Sort out the path
        # Hierarchy is: 1: signature, 2: config, 3: default
        filename = (
            filename
            or self.model.config.get_value("input.path_forcing")
            or self._filename
        )
        # Check for input dir
        p_input = Path(self.model.config.get_value("dir_input", fallback=""), filename)

        # Get the start and endtime
        starttime = self.model.config.get_value("time.starttime", fallback=None)
        endtime = self.model.config.get_value("time.endtime", fallback=None)

        # Logging the output
        logger.info("Write forcing file")
        start_time, end_time = self._validate_timespan(starttime, endtime)

        if Path(p_input).is_absolute():
            filepath = Path(p_input)
        else:
            filepath = (self.root.path / p_input).resolve()

        if filepath.exists():
            if overwrite:
                logger.warning(f"Deleting existing forcing file {filepath.as_posix()}")
                filepath.unlink()
            else:
                logger.warning(
                    f"Netcdf forcing file `{filepath}` already exists and overwriting "
                    "is not enabled. To overwrite netcdf forcing file: change name "
                    "`input.path_forcing` in setup_config section of the build "
                    "inifile or allow overwriting with `overwrite` flag. A default "
                    "name will be generated."
                )
                filepath = self._create_new_filename(
                    filepath=filepath,
                    start_time=start_time,
                    end_time=end_time,
                    frequency=output_frequency,
                )
                if filepath is None:  # should skip writing
                    return

        # Clean-up forcing and write
        ds = self.data.drop_vars(["mask", "idx_out"], errors="ignore")

        if decimals is not None:
            ds = ds.round(decimals)

        # Write with output chunksizes with single timestep and complete
        # spatial grid to speed up the reading from wflow.jl
        # Dims are always ordered (time, y, x)
        ds.raster._check_dimensions()
        chunksizes = (
            time_chunk,
            ds.raster.ycoords.size,
            ds.raster.xcoords.size,
        )

        # Set the encoding for the outgoing dataset
        encoding = kwargs.pop("encoding", {})
        encoding.update(
            {
                v: {"dtype": "float32", "chunksizes": chunksizes}
                for v in ds.data_vars.keys()
            }
        )
        encoding["time"] = {"units": time_units}

        # Write the file either in one go
        if output_frequency is None:
            logger.info(f"Writing file {filepath.as_posix()}")
            write_nc(
                ds,
                file_path=filepath,
                compress=True,
                gdal_compliant=True,
                rename_dims=True,
                force_sn=False,
                compute=False,
                force_overwrite=True,
                encoding=encoding,
                **kwargs,
            )
        # Or per period
        else:
            logger.info(f"Writing several forcing with freq {output_frequency}")
            # Updating path forcing in config
            for _, data_freq in ds.resample(time=output_frequency):
                # Sort out the outgoing filename
                start = data_freq["time"].dt.strftime("%Y%m%d")[0].item()
                filepath_freq = Path(filepath.parent, f"{filepath.stem}_{start}.nc")
                logger.info(f"Writing file {filepath_freq.as_posix()}")
                # Write to file
                write_nc(
                    data_freq,
                    file_path=filepath_freq,
                    compress=True,
                    gdal_compliant=True,
                    rename_dims=True,
                    force_sn=False,
                    compute=False,
                    force_overwrite=True,
                    encoding=encoding,
                    **kwargs,
                )
            filepath = Path(filepath.parent, f"{filepath.stem}_*{filepath.suffix}")

        # Set back to the config
        self.model.set_config("input.path_forcing", filepath.as_posix())
        self.model.set_config(
            "time.starttime", start_time.strftime("%Y-%m-%dT%H:%M:%S")
        )
        self.model.set_config("time.endtime", end_time.strftime("%Y-%m-%dT%H:%M:%S"))

        return

    ## Mutating methods
    def set(
        self,
        data: xr.DataArray | xr.Dataset,
        name: str | None = None,
    ):
        """Add data to forcing.

        All layers in data must have identical spatial coordinates to existing
        forcing and staticmaps.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to forcing
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray
            and ignored if data is a Dataset
        """
        # Rename the temporal dimension to time
        if "time" not in data.dims:
            raise ValueError("'time' dimension not found in data")

        if self._region_component is not None:
            # Ensure the data is aligned with the region component (staticmaps)
            data = utils.align_grid_with_region(
                data,
                region_grid=self._get_grid_data(),
                region_component_name=self._region_component,
            )

            # Check for CRS - eg sediment forcing is wflow.jl output and has no crs
            if data.raster.crs is None:
                data.raster.set_crs(self._get_grid_data().raster.crs)

        # Call set of parent class
        super().set(
            data=data,
            name=name,
        )

    def clip(self):
        """Clip the forcing component to the region of the region component.

        If no region component is set, the forcing component is clipped to its own
        extent.

        """
        if self._region_component is None:
            logger.info(
                "No region component set, forcing component will not be clipped."
            )
            return

        if len(self.data) > 0:
            logger.info("Clipping forcing...")
            region_grid = self._get_grid_data()
            ds_forcing = self.data.raster.clip_bbox(region_grid.raster.bounds)
            self._data = xr.Dataset()  # clear existing forcing
            self.set(ds_forcing)

    ### Internal methods
    def _create_new_filename(
        self, filepath: Path, start_time: datetime, end_time: datetime, frequency: str
    ) -> Path:
        """Determine the filename to use for writing.

        Parameters
        ----------
        filepath : Path
            The filepath to use.

        Returns
        -------
        Path
            The determined filepath as a Path object.
        """
        if "*" in filepath.name:
            # get rid of * in case model had multiple forcing files and
            # write to single nc file.
            logger.warning("Writing multiple forcing files to one file")
            filepath = filepath.parent / filepath.name.replace("*", "")

        if not filepath.exists():
            return filepath

        # Create filename based on attributes
        sourceP = ""
        sourceT = ""
        methodPET = ""
        if "precip" in self.data:
            val = self.data["precip"].attrs.get("precip_clim_fn", None)
            Pdown = "d" if val is not None else ""
            val = self.data["precip"].attrs.get("precip_fn", None)
            if val is not None:
                sourceP = f"_{val}{Pdown}"
        if "temp" in self.data:
            val = self.data["temp"].attrs.get("temp_correction", "False")
            Tdown = "d" if val == "True" else ""
            val = self.data["temp"].attrs.get("temp_fn", None)
            if val is not None:
                sourceT = f"_{val}{Tdown}"
        if "pet" in self.data:
            val = self.data["pet"].attrs.get("pet_method", None)
            if val is not None:
                methodPET = f"_{val}"

        fn_default = f"inmaps{sourceP}{sourceT}{methodPET}_{frequency}_{start_time.year}_{end_time.year}.nc"  # noqa: E501
        filepath = filepath.parent / fn_default

        if not filepath.exists():
            return filepath

        logger.warning(
            f"Netcdf generated forcing file name `{filepath}` already exists, "
            "skipping write_forcing. "
        )
        return None

    def _validate_timespan(
        self, starttime: Optional[str] = None, endtime: Optional[str] = None
    ) -> tuple[datetime, datetime]:
        """Validate the timespan of the forcing data."""
        # Check if all dates between (starttime, endtime) are in all da forcing
        # Check if starttime and endtime timestamps are correct
        starttime = np.datetime64(
            starttime or np.datetime_as_string(min(self.data.time.values), unit="s")
        )
        if (
            starttime < min(self.data.time.values)
            or starttime not in self.data.time.values
        ):
            logger.warning(
                f"Start time {starttime} does not match the beginning of the data. "
                f"Changing to start of the data: {min(self.data.time.values)}."
            )
            starttime = min(self.data.time.values)
        starttime = pd.Timestamp(starttime).to_pydatetime()

        endtime = np.datetime64(
            endtime or np.datetime_as_string(max(self.data.time.values), unit="s")
        )
        if endtime > max(self.data.time.values) or endtime not in self.data.time.values:
            logger.warning(
                f"End time {endtime} does not match the end of the data. "
                f"Changing to end of the data: {max(self.data.time.values)}."
            )
            endtime = max(self.data.time.values)
        endtime = pd.Timestamp(endtime).to_pydatetime()

        if self._data_is_empty():
            DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
            logger.warning(
                "Write forcing skipped: dataset is empty (no variables or data). Ensure"
                " that forcing data is loaded before calling 'write'."
            )
            return (
                None,
                datetime.strptime(starttime, format=DATETIME_FORMAT),
                datetime.strptime(endtime, format=DATETIME_FORMAT),
            )

        return starttime, endtime

    def _data_is_empty(self) -> bool:
        """Check if the forcing data is empty."""
        return len(self.data.data_vars) == 0 or all(
            var.size == 0 for var in self.data.data_vars.values()
        )
