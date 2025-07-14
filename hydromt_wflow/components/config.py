"""Custom wflow config component module."""

import logging
from pathlib import Path
from typing import Any

import tomli
from hydromt._io.writers import _write_toml
from hydromt.model import Model
from hydromt.model.components import ConfigComponent

from hydromt_wflow import utils
from hydromt_wflow.components.utils import make_config_paths_relative

__all__ = ["WflowConfigComponent"]

logger = logging.getLogger(f"hydromt.{__name__}")


class WflowConfigComponent(ConfigComponent):
    """Manage the wflow TOML configuration file for model simulations/settings.

    ``WflowConfigComponent`` data is stored in a dictionary. The component
    is used to prepare and update model simulations/settings of the wflow model.
    """

    def __init__(
        self,
        model: Model,
        *,
        filename: str = "wflow_sbm.toml",
        default_template_filename: str | None = None,
    ):
        """Manage configuration files for model simulations/settings.

        Parameters
        ----------
        model: Model
            HydroMT model instance
        filename: str
            A path relative to the root where the configuration file will
            be read and written if user does not provide a path themselves.
            By default 'config.yml'
        default_template_filename: Optional[Path]
            A path to a template file that will be used as default in the ``create``
            method to initialize the configuration file if the user does not provide
            their own template file. This can be used by model plugins to provide a
            default configuration template. By default None.
        """
        super().__init__(
            model,
            filename=filename,
            default_template_filename=default_template_filename,
        )

    def _initialize(self, skip_read=False) -> None:
        """Initialize the model config."""
        if self._data is None:
            self._data = {}
            if not skip_read:
                # no check for read mode here
                # model config is read if in read-mode and it exists
                # default config if in write-mode
                self.read()

    @property
    def data(self) -> dict:
        """Model config values."""
        if self._data is None:
            self._initialize()
        return self._data

    def read(
        self,
        path: Path | str | None = None,
    ):
        """Read the wflow configuration file at <root>/{path}."""
        self._initialize(skip_read=True)

        # Check if user-defined path or template should be used
        p = path or self._filename
        read_path = Path(self.root.path, p)

        # Switch to default if available and supplied config is not found
        if (
            not read_path.is_file()
            and self._default_template_filename is not None
            and not self.root.is_reading_mode()
        ):
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
                "It wil be initialized as empty dict"
            )
            return

        # Read the data and set it in the document
        with open(read_path, "rb") as file:
            data = tomli.load(file)
        self._data = data

    def write(
        self,
        path: Path | str | None = None,
    ):
        """Write the wflow configurations to a file."""
        self.root._assert_write_mode()
        # If there is data
        if self.data:
            p = path or self._filename

            # Sort the pathing
            write_path = Path(self.root.path, p)
            logger.info(f"Writing model config to {write_path.as_posix()}.")
            write_path.parent.mkdir(parents=True, exist_ok=True)

            # Solve the pathing in the data
            # Extra check for dir_input
            rel_path = Path(write_path.parent, self.get_value("dir_input", fallback=""))
            write_data = make_config_paths_relative(self.data, rel_path)
            _write_toml(write_path, write_data)
        else:
            logger.warning("Model config has no data, skip writing.")


def get_value(
    self,
    key: str,
    fallback: Any | None = None,
    abs_path: bool = False,
) -> Any | None:
    """Get config options.

    Parameters
    ----------
    key : str
        Keys are a string with '.' indicating a new level: ('key1.key2')
    fallback : Any, optional
        Fallback value if key(s) not found in config, by default None.
    abs_path: bool, optional
        If True return the absolute path relative to the model root,
        by default False.
    """
    # Refer to utils function of get_config
    return utils.get_config(
        self.data,
        key,
        root=self.root.path,
        fallback=fallback,
        abs_path=abs_path,
    )
