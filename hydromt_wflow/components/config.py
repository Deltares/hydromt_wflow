"""A component to write configuration files for model simulations/kernels."""

from logging import Logger, getLogger
from os import makedirs
from os.path import dirname, isdir, join
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from hydromt.model.components.config import ConfigComponent

if TYPE_CHECKING:
    from hydromt.model import Model

logger: Logger = getLogger(__name__)


class WflowConfigComponent(ConfigComponent):
    """
    A component to manage configuration files for model simulations/settings.

    ``ConfigComponent`` data is stored as a dictionary and can be written to a file
    in yaml or toml format. The component can be used to store model settings
    and parameters that are used in the model simulations or in the model
    settings.
    """

    def __init__(
        self,
        model: "Model",
        *,
        filename: str = "wflow_sbm.toml",
        default_template_filename: Optional[str] = None,
    ):
        """Initialize a ConfigComponent.

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
            model=model,
            filename=filename,
            default_template_filename=default_template_filename,
        )

    def delete_key(self, key: str):
        """Remove the keys from the config dictionary.

        Parameters
        ----------
        key : str
            a string with '.' indicating a new level: 'key1.key2' will translate
            to {"key1":{"key2": value}}

        Examples
        --------
        ::

            >> self.set({'a': 1, 'b': {'c': {'d': 2}}})
            >> self.data
                {'a': 1, 'b': {'c': {'d': 2}}}
            >> self.delete_key('b', 24)
            >> {'a': 99 }
        """
        self._initialize()
        parts = key.split(".")

        num_parts = len(parts)
        current = cast(Dict[str, Any], self._data)
        for i, part in enumerate(parts):
            if part not in current:
                # subtree not found, so nothing to do
                return
            if i < num_parts - 1:
                current = current[part]
            else:
                _ = current.pop(part)

    def write_config(
        self,
        config_name: Optional[str] = None,
        config_root: Optional[str] = None,
    ):
        """
        Write config to <root/config_fn>.

        Parameters
        ----------
        config_name : str, optional
            Name of the config file. By default None to use the default name
            wflow_sbm.toml.
        config_root : str, optional
            Root folder to write the config file if different from model root (default).
        """
        self._assert_write_mode()
        if config_name is not None:
            self._config_fn = config_name
        elif self._config_fn is None:
            self._config_fn = self._CONF
        if config_root is None:
            config_root = self.root
        fn = join(config_root, self._config_fn)
        # Create the folder if it does not exist
        if not isdir(dirname(fn)):
            makedirs(dirname(fn))
        self.logger.info(f"Writing model config to {fn}")
        self._configwrite(fn)
