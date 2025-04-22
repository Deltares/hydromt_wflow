"""Custom states component."""

import logging

from hydromt.model import Model
from hydromt.model.components import GridComponent

__all__ = ["StatesComponent"]

logger = logging.getLogger(f"hydromt.{__name__}")


class StatesComponent(GridComponent):
    """Custom states component.

    Inherits from the HydroMT-core GridComponent model-component.

    Parameters
    ----------
    model: Model
        HydroMT model instance
    filename: str
        The path to use for reading and writing of component data by default.
        By default "instate/instates.nc".
    region_component: str, optional
        The name of the region component to use as reference
        for this component's region. If None, the region will be set to the grid extent.
        Note that the create method only works if the region_component is None.
        For add_data_from_* methods, the other region_component should be
        a reference to another grid component for correct reprojection, by default None
    region_filename: str
        The path to use for reading and writing of the region data by default.
        By default "region.geojson".
    """

    def __init__(
        self,
        model: Model,
        *,
        filename: str = "instate/instates.nc",
        region_component: str | None = None,
        region_filename: str = "region.geojson",
    ):
        GridComponent.__init__(
            self,
            model,
            filename=filename,
            region_component=region_component,
            region_filename=region_filename,
        )
