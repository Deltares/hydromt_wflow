import glob
import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
from hydromt.model import Model
from hydromt.model.components import GeomsComponent

logger = logging.getLogger(__name__)


class WflowGeomsComponent(GeomsComponent):
    """Wflow Geoms Component to manage spatial geometries.

    It extends the base GeomsComponent from hydromt and consists of a dictionary of
    geopandas GeoDataFrames.
    """

    def __init__(
        self,
        model: Model,
        *,
        filename: str = "geoms/{name}.geojson",
        region_component: Optional[str] = None,
        region_filename: str = "geoms/geoms_region.geojson",
    ):
        """Initialize a GeomsComponent.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        filename: str
            The path to use for reading and writing of component data by default.
            by default "geoms/{name}.geojson", i.e. one file per geodataframe in
            the data dictionary.
        region_component: str, optional
            The name of the region component to use as reference for this component's
            region. If None, the region will be set to the union of all geometries in
            the data dictionary.
        region_filename: str
            The path to use for writing the region data to a file. By default
            "geoms/geoms_region.geojson".
        """
        super().__init__(
            model=model,
            filename=filename,
            region_component=region_component,
            region_filename=region_filename,
        )

    def read(
        self,
        read_dir: Path,
        merge_data: bool = False,
        **kwargs: dict,
    ):
        r"""Read model geometries files at <read_dir>/*.geojson.

        key-word arguments are passed to :py:func:`geopandas.read_file`

        Parameters
        ----------
        read_dir : Path
            The directory to read the geometry files from.
        merge_data : bool, optional
            If True, the data will be merged into the existing data dictionary.
            If False, the existing data will be cleared before reading.
        **kwargs:
            Additional keyword arguments that are passed to the
            `geopandas.read_file` function.
        """
        if not merge_data:
            self._data = {}

        fns = glob.glob(str(read_dir / "*.geojson"))
        if fns:
            logger.info("Reading model staticgeom files.")
        for fn in fns:
            name = Path(fn).stem
            if name != "region":
                self.set(gpd.read_file(fn, **kwargs), name=name)

    def write(
        self,
        dir_out: Path,
        to_wgs84: bool = False,
        precision: Optional[int] = None,
        **kwargs,
    ) -> None:
        r"""Write model geometries to a vector file (by default GeoJSON) at <dir_out>/*.geojson.

        key-word arguments are passed to :py:meth:`geopandas.GeoDataFrame.to_file`

        Parameters
        ----------
        dir_out: Path, optional
            The directory to write the geometry files to. If it does not exist, it will
            be created.
        to_wgs84: bool, optional
            If True, the geoms will be reprojected to WGS84(EPSG:4326) before being
            written.
        precision: int, optional
            The precision to use for writing the geometries. If None, it will be set to 1
            for projected CRS and 6 for geographic CRS.
        **kwargs:
            Additional keyword arguments that are passed to the
            `geopandas.to_file` function.
        """  # noqa: E501
        self.root._assert_write_mode()
        if len(self.data) == 0:
            logger.debug("No geoms data found, skip writing.")
            return

        if precision is None:
            if self.crs.is_projected:
                _precision = 1
            else:
                _precision = 6
        else:
            _precision = precision

        grid_size = 10 ** (-_precision)
        dir_out.parent.mkdir(parents=True, exist_ok=True)

        for name, gdf in self.data.items():
            if len(gdf) == 0:
                logger.warning(f"{name} is empty. Skipping...")
                continue
            fn_out = dir_out / f"{name}.geojson"
            logger.debug(f"Writing file {fn_out}")

            gdf.geometry.set_precision(
                grid_size=grid_size,
            )
            if to_wgs84:
                gdf.to_crs(epsg=4326, inplace=True)
            gdf.to_file(fn_out, **kwargs)

    def get(self, name: str) -> Optional[gpd.GeoDataFrame | gpd.GeoSeries]:
        """Get geometry by name."""
        geom = self.data.get(name, None)
        if geom is None:
            logger.warning(f"Geometry '{name}' not found in geoms.")
        return geom

    def pop(
        self, name: str, default=None
    ) -> Optional[gpd.GeoDataFrame | gpd.GeoSeries]:
        """Remove and return geometry by name."""
        geom = self.data.pop(name, default)
        if geom is default:
            logger.warning(
                f"Geometry '{name}' not found in geoms, returning default value : {default}."  # noqa: E501
            )
        else:
            logger.info(f"Removed geometry '{name}' from geoms.")
        return geom
