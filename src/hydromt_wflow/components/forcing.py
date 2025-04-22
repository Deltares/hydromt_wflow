"""Custom forcing component."""

from logging import Logger, getLogger
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import shapely.geometry as sg
import xarray as xr
from affine import Affine
from hydromt._io.readers import _read_nc
from hydromt._io.writers import _write_nc
from hydromt._typing.error import NoDataStrategy, exec_nodata_strat
from hydromt._typing.type_def import DeferedFileClose
from hydromt.model import Model
from hydromt.model.components import SpatialModelComponent
from hydromt.model.steps import hydromt_step
from pyproj import CRS

__all__ = ["ForcingComponent"]

logger: Logger = getLogger(f"hydromt.{__name__}")


class ForcingComponent(SpatialModelComponent):
    """
    Custom forcing component for wflow model.

    Parameters
    ----------
    model : Model
        HydroMT model instance
    filename : str
        The path to use for reading and writing of component data by default.
        By default "inmaps/forcing.nc".
    region_component : str, optional
        The name of the region component to use as reference for
        this component's region. If None, the region will be set to the grid extent.
        Note that the create method only works if the region_component is None.
        For add_data_from_* methods, the other region_component should be a
        reference to another grid component for correct reprojection.
    region_filename : str
        The path to use for reading and writing of the region data by default.
        By default "region.geojson".
    """

    def __init__(
        self,
        model: "Model",
        *,
        filename: str = "inmaps/forcing.nc",
        region_component: str | None = None,
        region_filename: str = "region.geojson",
    ):
        # region_component referencing is not possible for grids.
        # The region should be passed via create().
        SpatialModelComponent.__init__(
            self,
            model=model,
            region_component=region_component,
            region_filename=region_filename,
        )
        self._data: Optional[xr.Dataset] = None
        self._filename: str = filename

    def set(
        self,
        data: xr.DataArray | xr.Dataset,
        name: Optional[str] = None,
    ):
        """Add data to grid.

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
            raise ValueError(f"Cannot set data of type {type(data).__name__}")

        if len(self._data) == 0:  # empty grid
            self._data = data
        else:
            for dvar in data.data_vars:
                if dvar in self._data:
                    logger.warning(f"Replacing grid map: {dvar}")
                self._data[dvar] = data[dvar]

    @hydromt_step
    def write(
        self,
        filename: Optional[str] = None,
        *,
        gdal_compliant: bool = False,
        rename_dims: bool = False,
        force_sn: bool = False,
        region_options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[DeferedFileClose]:
        """Write model grid data to netcdf file at <root>/<fn>.

        key-word arguments are passed to :py:meth:`~hydromt.model.Model.write_nc`

        Parameters
        ----------
        filename : str, optional
            filename relative to model root, by default 'grid/grid.nc'
        gdal_compliant : bool, optional
            If True, write grid data in a way that is compatible with GDAL,
            by default False
        rename_dims : bool, optional
            If True and gdal_compliant, rename x_dim and y_dim to standard names
            depending on the CRS (x/y for projected and lat/lon for geographic).
        force_sn : bool, optional
            If True and gdal_compliant, forces the dataset to have
            South -> North orientation.
        region_options : dict, optional
            Options to pass to the write_region method.
            Can contain `filename`, `to_wgs84`, and anything that will be passed
            to `GeoDataFrame.to_file`. If `filename` is not provided,
            self.region_filename will be used.
        **kwargs : dict
            Additional keyword arguments to be passed to the `write_nc` method.
        """
        self.root._assert_write_mode()
        region_options = region_options or {}
        self.write_region(**region_options)

        if len(self.data) == 0:
            exec_nodata_strat(
                msg="No grid data found, skip writing.",
                strategy=NoDataStrategy.WARN,
            )
            return None
        # write_nc requires dict - use dummy 'grid' key
        return _write_nc(
            {"grid": self.data},
            filename or self._filename,
            root=self.model.root.path,
            gdal_compliant=gdal_compliant,
            rename_dims=rename_dims,
            force_overwrite=self.root.mode.is_override_mode(),
            force_sn=force_sn,
            **kwargs,
        )

    @hydromt_step
    def read(
        self,
        filename: Optional[str] = None,
        *,
        mask_and_scale: bool = False,
        **kwargs,
    ) -> None:
        """Read model grid data at <root>/<fn> and add to grid property.

        key-word arguments are passed to :py:meth:`~hydromt.model.Model.read_nc`

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

        # Load grid data in r+ mode to allow overwriting netcdf files
        if self.root.is_reading_mode() and self.root.is_writing_mode():
            kwargs["load"] = True
        loaded_nc_files = _read_nc(
            filename or self._filename,
            self.root.path,
            single_var_as_array=False,
            mask_and_scale=mask_and_scale,
            **kwargs,
        )
        for ds in loaded_nc_files.values():
            self.set(ds)

    @property
    def res(self) -> Optional[Tuple[float, float]]:
        """Returns the resolution of the model grid."""
        if len(self.data) > 0:
            return self.data.raster.res
        exec_nodata_strat(
            msg="No grid data found for deriving resolution",
            strategy=NoDataStrategy.WARN,
        )
        return None

    @property
    def transform(self) -> Optional[Affine]:
        """Returns spatial transform of the model grid."""
        if len(self.data) > 0:
            return self.data.raster.transform
        exec_nodata_strat(
            msg="No grid data found for deriving transform",
            strategy=NoDataStrategy.WARN,
        )
        return None

    @property
    def crs(self) -> Optional[CRS]:
        """Returns coordinate reference system embedded in the model grid."""
        if self.data.raster is None:
            exec_nodata_strat(
                msg="No grid data found for deriving crs",
                strategy=NoDataStrategy.WARN,
            )
            return None
        if self.data.raster.crs is None:
            exec_nodata_strat(
                msg="No crs found in grid data",
                strategy=NoDataStrategy.WARN,
            )
            return None
        return CRS(self.data.raster.crs)

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Returns the bounding box of the model grid."""
        if len(self.data) > 0:
            return self.data.raster.bounds
        exec_nodata_strat(
            msg="No grid data found for deriving bounds",
            strategy=NoDataStrategy.WARN,
        )
        return None

    @property
    def _region_data(self) -> Optional[gpd.GeoDataFrame]:
        """Returns the geometry of the model area of interest."""
        if len(self.data) > 0:
            crs: Optional[int | CRS] = self.crs
            if crs is not None and hasattr(crs, "to_epsg"):
                crs = crs.to_epsg()  # not all CRS have an EPSG code
            return gpd.GeoDataFrame(geometry=[sg.box(*self.bounds)], crs=crs)
        exec_nodata_strat(
            msg="No grid data found for deriving region", strategy=NoDataStrategy.WARN
        )
        return None

    @property
    def data(self) -> xr.Dataset:
        """Model static gridded data as xarray.Dataset."""
        if self._data is None:
            self._initialize_grid()
        assert self._data is not None
        return self._data

    def _initialize_grid(self, skip_read: bool = False) -> None:
        """Initialize grid object."""
        if self._data is None:
            self._data = xr.Dataset()
            if self.root.is_reading_mode() and not skip_read:
                self.read()
