"""Forcing component module."""

import logging
from pathlib import Path

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
        filename : str
            The path to use for reading and writing of component data by default.
            By default "inmaps.nc".
        region_component : str, optional
            The name of the region component to use as reference for this component's
            region. If None, the region will be set to the grid extent. Note that the
            create method only works if the region_component is None.
        region_filename : str
            The path to use for reading and writing of the region data by default.
            By default "grid/grid_region.geojson".
        """
        pass

    ## I/O methods
    def write(
        self,
        filename: Path | str | None = None,
        freq_out: str | None = None,
        **kwargs,
    ):
        """Write staticmaps model data.

        Key-word arguments are passed to :py:meth:`~hydromt._io.writers._write_nc`

        Parameters
        ----------
        filename : str, optional
            filename relative to model root, by default None
        **kwargs : dict
            Additional keyword arguments to be passed to the `write_nc` method.
        """
        self.root._assert_write_mode()
        # Solve pathing same as read
        # Hierarchy is: 1: signature, 2: config, 3: default
        p = filename or self.model.config.get("input.path_static") or self._filename
        # Check for input dir
        p_input = Path(self.model.config.get("dir_input", fallback=""), p)

        logger.info("Write forcing file")

        # Get default forcing name from forcing attrs
        yr0 = pd.to_datetime(self.get_config("time.starttime")).year
        yr1 = pd.to_datetime(self.get_config("time.endtime")).year
        freq = self.get_config("time.timestepsecs")
        # get output filename
        if fn_out is not None:
            self.set_config("input.path_forcing", fn_out)
        else:
            fn_name = self.get_config("input.path_forcing", abs_path=False)
            if fn_name is not None:
                if "*" in basename(fn_name):
                    # get rid of * in case model had multiple forcing files and
                    # write to single nc file.
                    logger.warning("Writing multiple forcing files to one file")
                    fn_name = join(dirname(fn_name), basename(fn_name).replace("*", ""))
                if self.get_config("dir_input") is not None:
                    input_dir = self.get_config("dir_input", abs_path=True)
                    fn_out = join(input_dir, fn_name)
                else:
                    fn_out = join(self.root, fn_name)
            else:
                fn_out = None

            # get default filename if file exists
            if fn_out is None or isfile(fn_out):
                logger.warning(
                    "Netcdf forcing file from input.path_forcing in the TOML  "
                    "already exists, using default name."
                )
                sourceP = ""
                sourceT = ""
                methodPET = ""
                if "precip" in self.forcing:
                    val = self.forcing["precip"].attrs.get("precip_clim_fn", None)
                    Pdown = "d" if val is not None else ""
                    val = self.forcing["precip"].attrs.get("precip_fn", None)
                    if val is not None:
                        sourceP = f"_{val}{Pdown}"
                if "temp" in self.forcing:
                    val = self.forcing["temp"].attrs.get("temp_correction", "False")
                    Tdown = "d" if val == "True" else ""
                    val = self.forcing["temp"].attrs.get("temp_fn", None)
                    if val is not None:
                        sourceT = f"_{val}{Tdown}"
                if "pet" in self.forcing:
                    val = self.forcing["pet"].attrs.get("pet_method", None)
                    if val is not None:
                        methodPET = f"_{val}"
                fn_default = (
                    f"inmaps{sourceP}{sourceT}{methodPET}_{freq}_{yr0}_{yr1}.nc"
                )
                if self.get_config("dir_input") is not None:
                    input_dir = self.get_config("dir_input", abs_path=True)
                    fn_default_path = join(input_dir, fn_default)
                else:
                    fn_default_path = join(self.root, fn_default)
                if isfile(fn_default_path):
                    logger.warning(
                        "Netcdf default forcing file already exists, \
skipping write_forcing. "
                        "To overwrite netcdf forcing file: \
change name input.path_forcing "
                        "in setup_config section of the build inifile."
                    )
                    return
                else:
                    self.set_config("input.path_forcing", fn_default)
                    fn_out = fn_default_path

        # Check if all dates between (starttime, endtime) are in all da forcing
        # Check if starttime and endtime timestamps are correct
        start = pd.to_datetime(self.get_config("time.starttime"))
        end = pd.to_datetime(self.get_config("time.endtime"))
        correct_times = False
        for da in self.forcing.values():
            if "time" in da.coords:
                # only correct dates in toml for standard calendars:
                if not hasattr(da.indexes["time"], "to_datetimeindex"):
                    times = da.time.values
                    if (start < pd.to_datetime(times[0])) or (start not in times):
                        start = pd.to_datetime(times[0])
                        correct_times = True
                    if (end > pd.to_datetime(times[-1])) or (end not in times):
                        end = pd.to_datetime(times[-1])
                        correct_times = True
        # merge, process and write forcing
        ds = xr.merge([da.reset_coords(drop=True) for da in self.forcing.values()])
        ds.raster.set_crs(self.crs)
        # Send warning, and update config with new start and end time
        if correct_times:
            logger.warning(
                f"Not all dates found in precip_fn changing starttime to \
{start} and endtime to {end} in the toml."
            )
            # Set the strings first
            self.set_config("time.starttime", start.strftime("%Y-%m-%dT%H:%M:%S"))
            self.set_config("time.endtime", end.strftime("%Y-%m-%dT%H:%M:%S"))

        if decimals is not None:
            ds = ds.round(decimals)
        # clean-up forcing and write CRS according to CF-conventions
        ds = ds.raster.gdal_compliant(rename_dims=True, force_sn=False)
        ds = ds.drop_vars(["mask", "idx_out"], errors="ignore")

        # write with output chunksizes with single timestep and complete
        # spatial grid to speed up the reading from wflow.jl
        # dims are always ordered (time, y, x)
        ds.raster._check_dimensions()
        chunksizes = (chunksize, ds.raster.ycoords.size, ds.raster.xcoords.size)
        encoding = {
            v: {"zlib": True, "dtype": "float32", "chunksizes": chunksizes}
            for v in ds.data_vars.keys()
        }
        # make sure no _FillValue is written to the time / x_dim / y_dim dimension
        # For several forcing files add common units attributes to time
        for v in ["time", ds.raster.x_dim, ds.raster.y_dim]:
            ds[v].attrs.pop("_FillValue", None)
            encoding[v] = {"_FillValue": None}

        # Check if all sub-folders in fn_out exists and if not create them
        if not isdir(dirname(fn_out)):
            os.makedirs(dirname(fn_out))

        forcing_list = []

        if freq_out is None:
            # with compute=False we get a delayed object which is executed when
            # calling .compute where we can pass more arguments to
            # the dask.compute method
            forcing_list.append([fn_out, ds])
        else:
            logger.info(f"Writing several forcing with freq {freq_out}")
            # For several forcing files add common units attributes to time
            encoding["time"] = {"_FillValue": None, "units": time_units}
            # Updating path forcing in config
            fns_out = os.path.relpath(fn_out, self.root)
            fns_out = f"{str(fns_out)[0:-3]}_*.nc"
            self.set_config("input.path_forcing", fns_out)
            for label, ds_gr in ds.resample(time=freq_out):
                # ds_gr = group[1]
                start = ds_gr["time"].dt.strftime("%Y%m%d")[0].item()
                fn_out_gr = f"{str(fn_out)[0:-3]}_{start}.nc"
                forcing_list.append([fn_out_gr, ds_gr])

        for fn_out_gr, ds_gr in forcing_list:
            logger.info(f"Process forcing; saving to {fn_out_gr}")
            delayed_obj = ds_gr.to_netcdf(
                fn_out_gr, encoding=encoding, mode="w", compute=False
            )
            with ProgressBar():
                delayed_obj.compute(**kwargs)
