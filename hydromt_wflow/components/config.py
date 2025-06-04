"""Custom wflow config component module."""

import logging
from pathlib import Path
from typing import Any, cast

import tomlkit
from hydromt.model import Model
from hydromt.model.components.base import ModelComponent
from hydromt.model.steps import hydromt_step
from tomli_w import dump as dump_toml
from tomllib import load as load_toml

from hydromt_wflow.utils import get_config, set_config

__all__ = ["WflowConfigComponent"]

logger = logging.getLogger(f"hydromt.{__name__}")


class WflowConfigComponent(ModelComponent):
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
        default_template_filename: Path | str | None = None,
    ):
        self._data: tomlkit.TOMLDocument[str, Any] | None = None
        self._filename: str = filename
        self._default_template_filename: Path | str | None = default_template_filename

        super().__init__(model=model)

    def __eq__(self, other: ModelComponent):
        """Compare components based on content."""
        if not isinstance(other, WflowConfigComponent):
            raise ValueError(
                f"Can't compare {self.__class__.__name__} \
with type {type(other).__name__}"
            )
        other_config = cast(WflowConfigComponent, other)
        return self.data == other_config.data

    ## Private
    def _initialize(self, skip_read=False) -> None:
        """Initialize the model config."""
        if self._data is None:
            self._data = tomlkit.TOMLDocument()
            if self.root.is_reading_mode() and not skip_read:
                self.read()

    ## Properties
    @property
    def data(self) -> tomlkit.TOMLDocument[str, Any]:
        """Model config values."""
        if self._data is None:
            self._initialize()
        return self._data

    ## I/O Methods
    @hydromt_step
    def read(
        self,
        path: Path | str | None = None,
    ):
        """Read the wflow configurations file."""
        self._initialize(skip_read=True)

        # Solve pathing
        p = path or self._filename
        read_path = Path(self.root.path, p)

        # Switch to default if available and supplied config is not found
        if not read_path.is_file() and self._default_template_filename is not None:
            _new_path = Path(self.root.path, self._default_template_filename)
            logger.warning(
                f"No config file found at {read_path.as_posix()} \
defaulting to {_new_path.as_posix()}"
            )
            read_path = _new_path

        # Check if the file exists
        if read_path.is_file():
            logger.info(f"Reading model config file from {read_path.as_posix()}.")
        else:
            logger.warning(
                f"No default model config was found at {read_path.as_posix()}. "
                "It wil be initialized as empty TOMLDocument"
            )
            return

        # Read the data and set it in the document
        with open(read_path, "rb") as f:
            data = load_toml(f)

        self.data.update(data)

    def write(self, path: Path | str | None = None):
        """Write the wflow configurations to a file."""
        self.root._assert_write_mode()
        # If there is data
        if self.data:
            p = path or self._filename

            # Sort the pathing
            write_path = Path(self.root.path, p)
            logger.info(f"Writing model config to {write_path.as_posix()}.")
            write_path.parent.mkdir(parents=True, exist_ok=True)

            # Dump the toml
            with open(write_path, "wb") as f:
                dump_toml(self.data, f)

        # Warn when there is no data being written
        else:
            logger.warning("Model config has no data, skip writing.")

    ## Modifying methods
    def get(self, *args) -> Any | None:
        """Get config options."""
        self._initialize()
        # Refer to utils function of get_config
        return get_config(self._data, *args)

    def set(self, *args):
        """Set the config options."""
        self._initialize()
        # Refer to utils function of set_config
        set_config(self._data, *args)
