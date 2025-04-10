"""Custom region components."""

import os
from logging import Logger, getLogger
from os.path import dirname, isdir, join
from pathlib import Path
from typing import Dict, Optional, Union, cast

import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from hydromt.model import Model
from hydromt.model.components.spatial import SpatialModelComponent
from hydromt.model.steps import hydromt_step

__all__ = ["RegionComponent"]

logger: Logger = getLogger(f"hydromt.{__name__}")


class RegionComponent(SpatialModelComponent):
    """Custom component for region.

    Parameters
    ----------
    model : Model
        HydroMT model instance
    filename : str
        The path to use for reading and writing of component data by default.
        by default "region.geojson" i.e. one file.
    region_component : str, optional
        The name of the region component to use as reference for this component's
        region. If None, the region will be set to the union of all geometries in
        the data dictionary.
    region_filename : str, optional
        The path to use for writing the region data to a file. By default
        "region.geojson".
    """

    def __init__(
        self,
        model: Model,
        *,
        filename: str = "region.geojson",
        region_component: Optional[str] = None,
        region_filename: str = "region.geojson",
    ):
        self._data: Optional[Dict[str, Union[GeoDataFrame, GeoSeries]]] = None
        self._filename: str = filename
        super().__init__(
            model=model,
            region_component=region_component,
            region_filename=region_filename,
        )

    @property
    def data(self) -> Dict[str, Union[GeoDataFrame, GeoSeries]]:
        """Model geometries.

        Return dict of geopandas.GeoDataFrame or geopandas.GeoSeries
        """
        if self._data is None:
            self._initialize()

        return self._data

    def _initialize(self, skip_read=False) -> None:
        """Initialize region."""
        if self._data is None:
            self._data = dict()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    @property
    def _region_data(self) -> Optional[GeoDataFrame]:
        # Use the total bounds of all geometries as region
        if len(self.data) == 0:
            return None
        return self.data["region"]

    def set(self, geom: Union[GeoDataFrame, GeoSeries]):
        """Add data to the region component.

        If a region is already present

        Parameters
        ----------
        geom : geopandas.GeoDataFrame or geopandas.GeoSeries
            New geometry data to add
        """
        self._initialize()
        if len(self.data) != 0:
            logger.warning("Replacing/ updating region")

        if isinstance(geom, GeoSeries):
            geom = cast(GeoDataFrame, geom.to_frame())

        # Verify if a geom is set to model crs and if not sets geom to model crs
        model_crs = self.model.crs
        if model_crs and model_crs != geom.crs:
            geom.to_crs(model_crs.to_epsg(), inplace=True)

        # Get rid of columns that aren't geometry
        geom = geom["geometry"].to_frame()

        # Make a union with the current region geodataframe
        cur = self._data.get("region")
        if cur is not None and not geom.equals(cur):
            geom = geom.union(cur)

        self._data["region"] = geom

    @hydromt_step
    def read(self, filename: Optional[str] = None, **kwargs) -> None:
        r"""Read model geometries files at <root>/<filename>.

        key-word arguments are passed to :py:func:`geopandas.read_file`

        Parameters
        ----------
        filename : str, optional
            filename relative to model root.
            If None, the path that was provided at init will be used.
        **kwargs : dict
            Additional keyword arguments that are passed to the
            `geopandas.read_file` function.
        """
        self.root._assert_read_mode()
        self._initialize(skip_read=True)
        f = filename or self._filename
        read_path = self.root.path / f
        if not read_path.is_file():
            return
        geom = cast(GeoDataFrame, gpd.read_file(read_path, **kwargs))
        self.set(geom=geom)

    @hydromt_step
    def write(
        self,
        filename: Optional[str] = None,
        to_wgs84: bool = False,
        **kwargs,
    ) -> None:
        r"""Write model geometries to a vector file at <root>/<filename>.

        Key-word arguments are passed to :py:meth:`geopandas.GeoDataFrame.to_file`

        Parameters
        ----------
        filename : str, optional
            filename relative to model root.
            If None, the path that was provided at init will be used.
        to_wgs84 : bool, optional
            If True, the geoms will be reprojected to WGS84(EPSG:4326)
            before they are written.
        **kwargs : dict
            Additional keyword arguments that are passed to the
            `geopandas.to_file` function.
        """
        self.root._assert_write_mode()

        if len(self.data) == 0:
            logger.debug("No geoms data found, skip writing.")
            return

        for name, gdf in self.data.items():
            if len(gdf) == 0:
                logger.warning(f"{name} is empty. Skipping...")
                continue

            geom_filename = filename or self._filename

            write_path = Path(
                join(
                    self.root.path,
                    geom_filename.format(name=name),
                )
            )

            logger.debug(f"Writing file {write_path}")

            write_folder = dirname(write_path)
            if not isdir(write_folder):
                os.makedirs(write_folder, exist_ok=True)

            if to_wgs84 and (
                kwargs.get("driver") == "GeoJSON"
                or str(write_path).lower().endswith(".geojson")
            ):
                gdf.to_crs(epsg=4326, inplace=True)

            gdf.to_file(write_path, **kwargs)
