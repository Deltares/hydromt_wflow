"""Custom wflow config component module."""

import logging
from pathlib import Path
from typing import Any, cast

import tomlkit
from hydromt.model import Model
from hydromt.model.components.base import ModelComponent
from tomlkit.toml_file import TOMLFile

from hydromt_wflow import utils
from hydromt_wflow.components.utils import make_config_paths_relative

__all__ = ["WflowConfigComponent"]

logger = logging.getLogger(f"hydromt.{__name__}")


class WflowConfigComponent(ModelComponent):
    """Manage the wflow TOML configuration file for model simulations/settings.

    ``WflowConfigComponent`` data is stored in a tomlkit.TOMLDocument. The component
    is used to prepare and update model simulations/settings of the wflow model.
    TOML config files will be read and written using
    `TOMLkit <https://tomlkit.readthedocs.io/en/latest/quickstart/>`__.
    This package will preserve the order and comments in a TOML file. Note, that any
    comments associated with sections that are to be updated will still disappear.
    """

    def __init__(
        self,
        model: Model,
        *,
        filename="wflow_sbm.toml",
        default_template_filename: Path | str | None = None,
    ):
        """
        Initialize a WflowConfigComponent.

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
        self._data: tomlkit.TOMLDocument[str, Any] | None = None
        self._filename: str = filename
        self._default_template_filename: Path | str | None = default_template_filename

        super().__init__(model=model)

    ## Private
    def _initialize(self, skip_read=False) -> None:
        """Initialize the model config."""
        if self._data is None:
            self._data = tomlkit.TOMLDocument()
            if not skip_read:
                # no check for read mode here
                # model config is read if in read-mode and it exists
                # default config if in write-mode
                self.read()

    ## Properties
    @property
    def data(self) -> tomlkit.TOMLDocument[str, Any]:
        """Model config values."""
        if self._data is None:
            self._initialize()
        return self._data

    ## I/O Methods
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
                "It wil be initialized as empty TOMLDocument"
            )
            return

        # Read the data and set it in the document
        data = TOMLFile(read_path).read()
        # TODO figure out whether .update might be a good alternative
        # Seems to not preserve the structure as well as direct setting
        # But now it overwrites the already existing keys.
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

            # Dump the toml
            TOMLFile(write_path).write(write_data)

        # Warn when there is no data being written
        else:
            logger.warning("Model config has no data, skip writing.")

    ## Add data methods
    def update(self, data: dict[str, Any]):
        """Set the config dictionary at key(s) with values.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary with the values to be set. keys can be dotted like in
            :py:meth:`~hydromt_wflow.components.config.WflowConfigComponent.set`

        Examples
        --------
        Setting data as a nested dictionary::


            >> self.update({'a': 1, 'b': {'c': {'d': 2}}})
            >> self.data
            {'a': 1, 'b': {'c': {'d': 2}}}

        Setting data using dotted notation::

            >> self.update({'a.d.f.g': 1, 'b': {'c': {'d': 2}}})
            >> self.data
            {'a': {'d':{'f':{'g': 1}}}, 'b': {'c': {'d': 2}}}

        """
        if len(data) > 0:
            logger.debug("Setting model config options.")
        for k, v in data.items():
            self.set(k, v)

    ## Modifying methods
    def get_value(
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
        # Refer to utils function of get_config
        return utils.get_config(
            self.data,
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
        utils.set_config(self.data, *args)

    # Testing
    def test_equal(self, other: ModelComponent) -> tuple[bool, dict[str, str]]:
        """Compare components based on content.

        Parameters
        ----------
        other : ModelComponent
            The component to compare against.

        Returns
        -------
        tuple[bool, dict[str, str]]
            True if the components are equal, and a dict with the associated errors per
            property checked.
        """
        eq, errors = super().test_equal(other)
        if not eq:
            return eq, errors
        other_config = cast(WflowConfigComponent, other)

        # check on data equality
        if self.data == other_config.data:
            return True, {}
        else:
            return False, {"config": "Configs are not equal"}
