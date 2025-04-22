"""Custom lake tables component."""

import logging

from hydromt.model import Model
from hydromt.model.components import TablesComponent

__all__ = ["LakeTablesComponent"]

logger = logging.getLogger(f"hydromt.{__name__}")


class LakeTablesComponent(TablesComponent):
    """Custom Lake tables component.

    Inherits from the HydroMT-core TablesComponent model-component.

    Parameters
    ----------
    model: Model
        HydroMT model instance
    filename: str
        The default place that should be used for reading and writing unless the
        user overrides it. If a relative path is given it will be used as being
        relative to the model root. By default `{name}.csv`
    """

    def __init__(
        self,
        model: Model,
        filename: str = "{name}.csv",
    ):
        TablesComponent.__init__(
            self,
            model,
            filename=filename,
        )
