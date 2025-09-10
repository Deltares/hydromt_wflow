import logging
from pathlib import Path
from typing import cast

import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
from hydromt.model import Model
from hydromt.model.components import GeomsComponent, ModelComponent

logger = logging.getLogger(f"hydromt.{__name__}")


class WflowGeomsComponent(GeomsComponent):
    """Wflow Geoms Component to manage spatial geometries.

    It extends the base GeomsComponent from hydromt and consists of a dictionary of
    geopandas GeoDataFrames.
    """

    def __init__(
        self,
        model: Model,
        *,
        filename: str = "staticgeoms/{name}.geojson",
        region_component: str | None = None,
        region_filename: str = "staticgeoms/geoms_region.geojson",
    ):
        """Initialize a WflowGeomsComponent.

        Parameters
        ----------
        model : Model
            HydroMT model instance
        filename : str
            The path to use for reading and writing of component data by default.
            by default "staticgeoms/{name}.geojson", i.e. one file per geodataframe in
            the data dictionary.
        region_component : str, optional
            The name of the region component to use as reference for this component's
            region. If None, the region will be set to the union of all geometries in
            the data dictionary.
        region_filename : str
            The path to use for writing the region data to a file. By default
            "staticgeoms/geoms_region.geojson".
        """
        super().__init__(
            model=model,
            filename=filename,
            region_component=region_component,
            region_filename=region_filename,
        )

    def read(
        self,
        folder: str = "staticgeoms",
    ):
        """
        Read static geometries and adds to ``geoms``.

        If ``dir_input`` is set in the config, the path where all static geometries are
        read, will be constructed as ``<model_root>/<dir_input>/<geoms_fn>``.
        Where <dir_input> is relative to the model root. Depending on the config value
        ``dir_input``, the path will be constructed differently.

        Parameters
        ----------
        folder : str, optional
            Folder name/path where the static geometries are stored relative to the
            model root and ``dir_input`` if any. By default "staticgeoms".
        """
        # Check for input dir
        p_input = Path(self.model.config.get_value("dir_input", fallback=""), folder)
        pattern = Path(p_input, "{name}.geojson")

        super().read(filename=str(pattern))

    def write(
        self,
        folder: str = "staticgeoms",
        to_wgs84: bool = False,
        precision: int | None = None,
        **kwargs,
    ) -> None:
        r"""
        Write model geometries to vector file(s) (by default GeoJSON) at <dir_out>/\*.geojson.

        Checks the path of ``geoms_fn`` using both model root and
        ``dir_input``. If not found uses the default path ``staticgeoms`` in the root
        folder.

        Key-word arguments are passed to :py:meth:`geopandas.GeoDataFrame.to_file`

        Parameters
        ----------
        folder : Path, optional
            The directory to write the geometry files to. If it does not exist, it will
            be created.
        to_wgs84 : bool, optional
            If True, the geoms will be reprojected to WGS84(EPSG:4326) before being
            written.
        precision : int, optional
            The precision to use for writing the geometries. If None, it will be set to 1
            for projected CRS and 6 for geographic CRS.
        **kwargs : dict
            Additional keyword arguments that are passed to the
            `geopandas.to_file` function.
        """  # noqa: E501
        # Check for dir_input
        p_input = Path(self.model.config.get_value("dir_input", fallback=""), folder)
        pattern = Path(p_input, "{name}.geojson")

        # Set precision
        if precision is None:
            if self.crs.is_projected:
                _precision = 1
            else:
                _precision = 6
        else:
            _precision = precision

        grid_size = 10 ** (-_precision)
        for gdf in self.data.values():
            gdf.geometry = gdf.geometry.set_precision(
                grid_size=grid_size,
            )

        super().write(
            filename=str(pattern),
            to_wgs84=to_wgs84,
            **kwargs,
        )

    def get(self, name: str) -> gpd.GeoDataFrame | gpd.GeoSeries:
        """Get geometry by name.

        Parameters
        ----------
        name : str
            The name of the geometry to retrieve.

        Returns
        -------
        gpd.GeoDataFrame | gpd.GeoSeries
            The geometry associated with the given name.

        Raises
        ------
        KeyError
            If the geometry with the specified name does not exist.
        """
        if name not in self.data:
            raise KeyError(
                f"Geometry '{name}' not found in geoms. Available geometries: {list(self.data.keys())}"  # noqa: E501
            )
        return self.data[name]

    def pop(self, name: str) -> gpd.GeoDataFrame | gpd.GeoSeries:
        """Remove and return geometry by name.

        Parameters
        ----------
        name : str
            The name of the geometry to remove and return.

        Returns
        -------
        gpd.GeoDataFrame | gpd.GeoSeries
            The geometry associated with the given name.

        Raises
        ------
        KeyError
            If the geometry with the specified name does not exist.
        """
        if name not in self.data:
            raise KeyError(
                f"Geometry '{name}' not found in geoms. Available geometries: {list(self.data.keys())}"  # noqa: E501
            )
        geom = self.data.pop(name)
        logger.debug(f"Removed geometry '{name}' from geoms.")
        return geom

    def clear(self) -> None:
        """Clear all geometries."""
        self.data.clear()
        logger.debug("Cleared all geometries from geoms.")

    def test_equal(self, other: ModelComponent) -> tuple[bool, dict[str, str]]:
        """Test if two GeomsComponents are equal.

        Parameters
        ----------
        other: GeomsComponent
            The other GeomsComponent to compare with.

        Returns
        -------
        tuple[bool, dict[str, str]]
            True if the components are equal, and a dict with the associated errors per
            property checked.
        """
        errors: dict[str, str] = {}
        if not isinstance(other, self.__class__):
            errors["__class__"] = f"other does not inherit from {self.__class__}."

        other_geoms = cast(WflowGeomsComponent, other)
        for name, gdf in self.data.items():
            if name not in other_geoms.data:
                errors[name] = "Geom not found in other component."

            # Geometries need to be normalized before comparison
            # (eg polygon points defined in the exact same order)
            gdf = gdf.copy()
            gdf["geometry"] = gdf["geometry"].apply(lambda geom: geom.normalize())
            other_gdf = other_geoms.data[name].copy()
            other_gdf["geometry"] = other_gdf["geometry"].apply(
                lambda geom: geom.normalize()
            )

            try:
                assert_geodataframe_equal(
                    gdf,
                    other_gdf,
                    check_like=True,
                    check_less_precise=True,
                )
            except AssertionError as e:
                errors[name] = str(e)

        return len(errors) == 0, errors
