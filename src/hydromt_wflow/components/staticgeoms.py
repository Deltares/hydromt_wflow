"""Custom static geometry component."""

import logging

from hydromt.model import Model
from hydromt.model.components import GeomsComponent

__all__ = ["StaticGeomsComponent"]

logger = logging.getLogger(f"hydromt.{__name__}")


class StaticGeomsComponent(GeomsComponent):
    """Custom static geometries component.

    Inherits from the HydroMT-core GeomComponent model-component.

    Parameters
    ----------
    model : Model
        HydroMT model instance
    filename : str
        The path to use for reading and writing of component data by default.
        By default "staticgeoms/{name}.geojson".
    region_component : str, optional
        The name of the region component to use as reference for this component's
        region. If None, the region will be set to the union of all geometries in
        the data dictionary. By default None
    region_filename : str
        The path to use for writing the region data to a file. By default
        "region.geojson".
    """

    def __init__(
        self,
        model: "Model",
        *,
        filename: str = "staticgeoms/{name}.geojson",
        region_component: str | None = None,
        region_filename: str = "region.geojson",
    ):
        super().__init__(
            model,
            filename=filename,
            region_component=region_component,
            region_filename=region_filename,
        )
