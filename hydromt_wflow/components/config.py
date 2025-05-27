"""config component module."""

from typing import Any

import tomlkit
from hydromt.model import Model
from hydromt.model.components.config import ConfigComponent

from hydromt_wflow.utils import get_config, set_config


class WflowConfigComponent(ConfigComponent):
    """A class to manage the configuration of wflow models."""

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

    def set(self, *args):
        """Set the config options."""
        self._initialize()
        set_config(self._data, *args)

    def get(self, *args) -> Any | None:
        """Get config options."""
        self._initialize()
        return get_config(self._data, *args)

    def _initialize(self, skip_read=False) -> None:
        """Initialize the model config."""
        if self._data is None:
            self._data = tomlkit.TOMLDocument()
            if self.root.is_reading_mode() and not skip_read:
                self.read()
