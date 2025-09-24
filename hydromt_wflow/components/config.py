"""Custom wflow config component module."""

import logging
from pathlib import Path
from typing import Any

from hydromt.io.readers import read_toml
from hydromt.io.writers import write_toml
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
        filename: str | None = None,
    ):
        """
        Read the wflow configuration file from <root/filename>.

        If filename is not provided, will default to <root>/{self._filename} or default
        template configuration file if the model is in write only mode (ie build).

        If filename is provided, it will check if the path is absolute and else will
        assume the given path is relative to the model root.

        """
        self._initialize(skip_read=True)

        # Check if user-defined path or template should be used
        if not filename:
            # Write only mode > read default config
            if (
                not self.root.is_reading_mode()
                and self._default_template_filename is not None
            ):
                prefix = "default"
                read_path = Path(self._default_template_filename)
            else:
                prefix = "model"
                read_path = Path(self.root.path, self._filename)
        else:
            prefix = "user defined"
            # Check if user-defined file is absolute (ie file exists)
            if Path(filename).is_file():
                read_path = Path(filename)
            else:
                read_path = Path(self.root.path, filename)

        # Check if the file exists
        if read_path.is_file():
            logger.info(f"Reading {prefix} config file from {read_path.as_posix()}.")
        else:
            logger.warning(
                f"No config was found at {read_path.as_posix()}. "
                "It wil be initialized as empty dict"
            )
            return

        # Read the data and set it in the document
        self._data = read_toml(read_path)

    def write(
        self,
        filename: str | None = None,
        config_root: Path | str | None = None,
    ):
        """
        Write the configuration to a file.

        The file is written to ``<root>/<filename>`` by default, or to
        ``<config_root>/<filename>`` if a ``config_root`` is provided.

        Parameters
        ----------
        filename : str, optional
            Name of the config file. By default None to use the default name
            self._filename.
        config_root : str, optional
            Root folder to write the config file if different from model root (default).
            Can be absolute or relative to model root.
        """
        # Check for config_root
        path = filename or self._filename
        if config_root is not None:
            path = Path(config_root, path)

        self.root._assert_write_mode()
        # If there is data
        if self.data:
            p = path or self._filename

            # Sort the path
            write_path = Path(self.root.path, p)
            logger.info(f"Writing model config to {write_path.as_posix()}.")
            write_path.parent.mkdir(parents=True, exist_ok=True)

            # Solve the path in the data
            # Extra check for dir_input
            rel_path = Path(write_path.parent, self.get_value("dir_input", fallback=""))
            write_data = make_config_paths_relative(self.data, rel_path)
            write_toml(write_path, write_data)
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

    def remove(self, *args: str, errors: str = "raise") -> Any:
        """
        Remove a config key and return its value.

        Parameters
        ----------
        key: str, tuple[str, ...]
            Key to remove from the config.
            Can be a dotted toml string when providing a list of strings.
        errors: str, optional
            What to do if the key is not found. Can be "raise" (default) or "ignore".

        Returns
        -------
        The popped value, or raises a KeyError if the key is not found.
        """
        args = list(args)
        if len(args) == 1 and "." in args[0]:
            args = args[0].split(".") + args[1:]

        current = self.data
        for index, key in enumerate(args):
            if current is None:
                if errors == "ignore":
                    return None
                else:
                    raise KeyError(f"Key {'.'.join(args)} not found in config.")

            if index == len(args) - 1:
                # Last key, pop it
                if errors == "ignore":
                    current = current.pop(key, None)
                else:
                    current = current.pop(key)
                break

            # Not the last key, go deeper
            current = current.get(key)
        return current

    def remove_reservoirs(
        self, input: list[str | None] = [], state: list[str | None] = []
    ):
        """Remove all reservoir related config options."""
        # a. change reservoir__flag = true to false
        self.set("model.reservoir__flag", False)
        # b. remove reservoir state
        for state_var in state:
            if state_var is not None:
                self.remove(f"state.variables.{state_var}", errors="ignore")
        # c. remove reservoir input
        for input_var in input:
            if input_var is not None:
                # find if variable in input/static/cyclic/forcing
                input_var = utils.get_wflow_var_fullname(input_var, self.data)
                self.remove(input_var, errors="ignore")
