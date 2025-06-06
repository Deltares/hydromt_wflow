"""Custom wflow config component module."""

import logging
from pathlib import Path
from typing import Any, cast

import tomlkit
from hydromt.model import Model
from hydromt.model.components.base import ModelComponent
from hydromt.model.steps import hydromt_step
from tomlkit.toml_file import TOMLFile

from hydromt_wflow.components.utils import make_config_paths_relative
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
        data = TOMLFile(read_path).read()
        # TODO figure out whether .update might be a good alternative
        # Seems to not preserve the structure as well as direct setting
        # But now it overwrites the already existing keys.
        self._data = data

    @hydromt_step
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
            write_data = make_config_paths_relative(self.data, write_path.parent)

            # Dump the toml
            TOMLFile(write_path).write(write_data)

        # Warn when there is no data being written
        else:
            logger.warning("Model config has no data, skip writing.")

    ## Modifying methods
    def get(
        self,
        *args,
        fallback: Any | None = None,
        abs_path: bool = False,
    ) -> Any | None:
        """Get config options.

        Parameters
        ----------
        args : tuple, str
            Keys can given by multiple args: ('key1', 'key2')
            or a string with '.' indicating a new level: ('key1.key2')
        fallback : Any, optional
            Fallback value if key(s) not found in config, by default None.
        abs_path: bool, optional
            If True return the absolute path relative to the model root,
            by default False.
        """
        self._initialize()
        # Refer to utils function of get_config
        return get_config(
            self._data,
            *args,
            root=self.root.path,
            fallback=fallback,
            abs_path=abs_path,
        )

    def set(self, *args):
        """Set the config options.

        Parameters
        ----------
        args : str, tuple, list
            If tuple or list, minimal length of two
            keys can given by multiple args: ('key1', 'key2', 'value')
            or a string with '.' indicating a new level: ('key1.key2', 'value')
        """
        self._initialize()
        # Refer to utils function of set_config
        set_config(self._data, *args)
