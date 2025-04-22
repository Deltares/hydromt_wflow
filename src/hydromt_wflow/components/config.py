"""Custom config component."""

from logging import Logger, getLogger

from hydromt.model import Model
from hydromt.model.components import ConfigComponent

__all__ = ["WflowConfigComponent"]

logger: Logger = getLogger(f"hydromt.{__name__}")


class WflowConfigComponent(ConfigComponent):
    """A component to manage configuration files for model simulations/settings.

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

    def __init__(
        self,
        model: Model,
        *,
        filename: str = "wflow_sbm.toml",
        default_template_filename: str | None = None,
    ):
        ConfigComponent.__init__(
            self,
            model,
            filename=filename,
            default_template_filename=default_template_filename,
        )
