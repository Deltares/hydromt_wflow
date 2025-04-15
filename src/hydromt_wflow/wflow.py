"""Main wflow module."""

import logging
from pathlib import Path

import geopandas as gpd
from hydromt.model import Model
from hydromt.model.components import (
    GeomsComponent,
    GridComponent,
    TablesComponent,
)
from hydromt.model.steps import hydromt_step

from hydromt_wflow import workflows
from hydromt_wflow.components import (
    ForcingComponent,
    RegionComponent,
    WflowConfigComponent,
)
from hydromt_wflow.utils import vectorize

# Set some global variables
__all__ = ["WflowModel"]
__hydromt_eps__ = ["WflowModel"]  # core entrypoints

# Create a logger
logger = logging.getLogger(f"hydromt.{__name__}")


class WflowModel(Model):
    """Read or Write a wflow model.

    Parameters
    ----------
    root : str, optional
        Model root, by default None
    mode : {'r','r+','w'}, optional
        read/append/write mode, by default "w"
    data_libs : list[str] | str, optional
        List of data catalog configuration files, by default None
    logger:
        The logger to be used.
    **catalog_keys:
        Additional keyword arguments to be passed down to the DataCatalog.
    """

    name: str = "wflow_model"
    # supported model version should be filled by the plugins
    # e.g. _MODEL_VERSION = ">=1.0, <1.1"
    _MODEL_VERSION = None

    def __init__(
        self,
        root: str | None = None,
        mode: str = "r",
        data_libs: list[str] | str | None = None,
        **catalog_keys,
    ):
        Model.__init__(
            self,
            root,
            components={"region": RegionComponent(model=self)},
            mode=mode,
            region_component="region",
            data_libs=data_libs,
            **catalog_keys,
        )

        ## Setup components
        self.add_component(
            "config",
            WflowConfigComponent(model=self, filename="wflow_sbm.toml"),
        )
        self.add_component(
            "forcing",
            ForcingComponent(model=self),
        )
        self.add_component(
            "lake_tables",
            TablesComponent(model=self, filename="lakes/{name}.csv"),
        )
        self.add_component(
            "states",
            GridComponent(
                model=self,
                filename="instate/instates.nc",
                region_component="region",
            ),
        )
        self.add_component(
            "staticmaps",
            GridComponent(
                model=self,
                filename="staticmaps.nc",
                region_component="region",
            ),
        )
        self.add_component(
            "staticgeoms",
            GeomsComponent(
                model=self,
                filename="staticgeoms/{name}.geojson",
                region_component="region",
            ),
        )

    ## Properties
    # Components
    @property
    def config(self) -> WflowConfigComponent:
        """Return the configurations component."""
        return self.components["config"]

    @property
    def forcing(self) -> ForcingComponent:
        """Return the forcing component."""
        return self.components["forcing"]

    @property
    def lake_tables(self) -> TablesComponent:
        """Return the lake tables component."""
        return self.components["lake_tables"]

    @property
    def states(self) -> GridComponent:
        """Return the states component."""
        return self.components["states"]

    @property
    def staticgeoms(self) -> GeomsComponent:
        """Return the static geometries component."""
        return self.components["staticgeoms"]

    @property
    def staticmaps(self) -> GridComponent:
        """Return the staticmaps component."""
        return self.components["staticmaps"]

    # Other properties
    @property
    def basins(self) -> gpd.GeoDataFrame:
        """Return the basins of the WflowModel."""
        pass

    ## I/O
    @hydromt_step
    def read(self):
        """Read the wflow model."""
        Model.read(self)

    @hydromt_step
    def write(self):
        """Write the wflow model."""
        Model.write(self)

    ## Setup methods
    @hydromt_step
    def setup_config(
        self,
        **settings: dict,
    ) -> None:
        """Set config file entries.

        Parameters
        ----------
        settings : dict
            Settings for the configuration provided as keyword arguments
            (KEY=VALUE).

        Returns
        -------
            None
        """
        logger.info("Setting config entries from user input")
        for key, value in settings.items():
            self.config.set(key, value)

    @hydromt_step
    def setup_region(
        self,
        region: Path | str,
    ) -> None:
        """Set the region of the wflow model.

        Parameters
        ----------
        region : Path | str
            Path to the region vector file.

        Returns
        -------
            None
        """
        region = Path(region)
        logger.info(f"Setting region from '{region.as_posix()}'")
        if not region.is_file():
            raise FileNotFoundError(region.as_posix())
        geom = gpd.read_file(region)
        self.components["region"].set(geom)

    @hydromt_step
    def setup_basemaps(
        self,
        hydrography_fname: Path | str,
        basin_index_fname: Path | str | None = None,
        *,
        res: float | int = 1 / 120,
        upscale_method: str = "ihu",
        derive_region: bool = True,
    ) -> None:
        """Set up the wflow base maps.

        Parameters
        ----------
        hydrography_fname : Path | str
            _description_
        basin_index_fname : Path | str | None, optional
            _description_, by default None
        res : float | int, optional
            _description_, by default 1/120
        upscale_method : str, optional
            _description_, by default "ihu"
        derive_region : bool, optional
            _description_, by default True

        Returns
        -------
        None
        """
        logger.info("Preparing base hydrography basemaps.")
        # Get the data from the catalog
        hydro_data = self.data_catalog.get_rasterdataset(hydrography_fname)
        basin_index_data = None
        if basin_index_fname is not None:
            basin_index_data = self.data_catalog.get_geodataframe(basin_index_fname)

        # First workflow function to sort out the data.
        hydro_data, geom, xy = workflows.prep_raw_basemaps_data(
            hydro_data,
            basin_index_data=basin_index_data,
            region=self.region,
            res=res,
            derive_region=derive_region,
        )

        # Derive the hydrography maps from the prepped raw input
        hydro_grid, _ = workflows.hydrography(
            hydro_data=hydro_data,
            res=res,
            xy=xy,
            upscale_method=upscale_method,
        )

        # Convert flow direction from d8 to ldd format
        hydro_grid = workflows.convert_flow_direction(hydro_grid)

        # Derive the topography maps
        topo_grid = workflows.topography(
            ds=hydro_data,
            ds_like=hydro_grid,
            method="average",
        )

        # Derive the basins by vectorizing
        basins_gdf = vectorize(ds=hydro_grid, var="basins")

        # Set the gridded data into the staticmaps component
        self.staticmaps.set(hydro_grid)
        self.staticmaps.set(topo_grid)

        # Set the geometries into the staticgeoms component
        self.staticgeoms.set(geom, name="basins_highres")
        self.staticgeoms.set(basins_gdf, name="basins")

        # Set the config file
        self.config.set("input.path_static", self.staticmaps._filename)
