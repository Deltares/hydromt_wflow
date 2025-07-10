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
        pass
