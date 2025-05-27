import logging
import math
from typing import Optional

import geopandas as gpd
import numpy as np
import xarray as xr
from hydromt.model import Model
from hydromt.model.components import GeomsComponent
from hydromt.model.processes.basin_mask import get_basin_geometry
from hydromt.model.processes.region import (
    _parse_region_value,
    parse_region_bbox,
    parse_region_geom,
)

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
        super().__init__(
            model=model,
            filename=filename,
            region_component=region_component,
            region_filename=region_filename,
        )

    def parse_region(
        self,
        region: dict,
        resolution: float | int = 1 / 120.0,
        hydrography_fn: str | xr.Dataset | None = None,
        basin_index_fn: str | xr.Dataset | None = None,
    ) -> tuple[gpd.GeoDataFrame, Optional[np.ndarray], xr.Dataset]:
        """Parse the region dictionary to get basin geometry and coordinates."""
        ds_org = self.data_catalog.get_rasterdataset(hydrography_fn)
        if ds_org is None:
            raise ValueError(
                f"hydrography_fn {hydrography_fn} not found in data catalog."
            )
        # Check on resolution (degree vs meter) depending on ds_org res/crs
        scale_ratio = int(np.round(resolution / ds_org.raster.res[0]))
        if scale_ratio < 1:
            raise ValueError(
                f"The model resolution {resolution} should be \
larger than the {hydrography_fn} resolution {ds_org.raster.res[0]}"
            )
        if ds_org.raster.crs.is_geographic:
            if resolution > 1:  # 111 km
                raise ValueError(
                    f"The model resolution {resolution} should be smaller than 1 \
degree (111km) for geographic coordinate systems. "
                    "Make sure you provided res in degree rather than in meters."
                )

        # get basin geometry and clip data
        kind = next(iter(region))
        xy = None
        if kind in ["basin", "subbasin"]:
            # parse_region_basin does not return xy, only geom...
            # should be fixed in core
            region_kwargs = _parse_region_value(
                region.pop(kind),
                data_catalog=self.data_catalog,
            )
            region_kwargs.update(region)
            if basin_index_fn is not None:
                bas_index = self.data_catalog.get_source(basin_index_fn)
            else:
                bas_index = None
            geom, xy = get_basin_geometry(
                ds=ds_org,
                kind=kind,
                basin_index=bas_index,
                **region,
            )
        elif kind == "bbox":
            logger.warning(
                "Kind 'bbox' for the region is not recommended as it can lead "
                "to mistakes in the catchment delineation. Use carefully."
            )
            geom = parse_region_bbox(region)
        elif kind == "geom":
            logger.warning(
                "Kind 'geom' for the region is not recommended as it can lead "
                "to mistakes in the catchment delineation. Use carefully."
            )
            geom = parse_region_geom(region)
        else:
            raise ValueError(
                f"wflow region kind not understood or supported: {kind}. "
                "Use 'basin', 'subbasin', 'bbox' or 'geom'."
            )

        if geom is not None and geom.crs is None:
            raise ValueError("wflow region geometry has no CRS")

        # Set the basins geometry
        logger.debug("Adding basins vector to geoms.")
        ds_org = ds_org.raster.clip_geom(geom, align=resolution, buffer=10)
        ds_org.coords["mask"] = ds_org.raster.geometry_mask(geom)

        # Set name based on scale_factor
        if not math.isclose(scale_ratio, 1):
            self.set(geom, name="basins_highres")

        return geom, xy, ds_org

    def get(self, name: str) -> Optional[gpd.GeoDataFrame | gpd.GeoSeries]:
        """Get geometry by name."""
        geom = self.data.get(name, None)
        if geom is not None:
            logger.info(f"Retrieved geometry '{name}' from geoms.")
        else:
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
