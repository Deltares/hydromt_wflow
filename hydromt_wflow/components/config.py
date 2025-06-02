"""Custom wflow config component module."""

from typing import Any

import tomlkit
from hydromt.model import Model
from hydromt.model.components.config import ConfigComponent

from hydromt_wflow.utils import get_config, set_config

__all__ = ["WflowConfigComponent"]


class WflowConfigComponent(ConfigComponent):
    """Manage the wflow configurations.

    Parameters
    ----------
    model : Model
        HydroMT model instance
    filename : str
        A path relative to the root where the configuration file will
        be read and written if user does not provide a path themselves.
        By default 'wflow_sbm.toml'
    default_template_filename : str, optional
        A path to a template file that will be used as default in the ``create``
        method to initialize the configuration file if the user does not provide
        their own template file. This can be used by model plugins to provide a
        default configuration template. By default None.
    """

    def __init__(
        self,
        model: Model,
        *,
        filename="wflow_sbm.toml",
        default_template_filename: str | None = None,
    ):
        super().__init__(
            model=model,
            filename=filename,
            default_template_filename=default_template_filename,
        )

    ## Private
    def _initialize(self, skip_read=False) -> None:
        """Initialize the model config."""
        if self._data is None:
            self._data = tomlkit.TOMLDocument()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    ## Modifying methods
    def get(self, *args) -> Any | None:
        """Get config options."""
        self._initialize()
        return get_config(self._data, *args)

    def set(self, *args):
        """Set the config options."""
        self._initialize()
        set_config(self._data, *args)
