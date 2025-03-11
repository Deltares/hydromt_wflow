"""A component to write configuration files for model simulations/kernels."""

from logging import Logger, getLogger
from typing import TYPE_CHECKING, Optional

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
