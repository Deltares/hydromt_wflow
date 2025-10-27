import os
from typing import TYPE_CHECKING

from hydromt.model.components import TablesComponent
from hydromt.model.steps import hydromt_step

if TYPE_CHECKING:
    from hydromt.model.model import Model


class WflowTablesComponent(TablesComponent):
    """Wflow specific tables component."""

    def __init__(
        self, model: "Model", filename: str = "reservoir_{{hq,sh}}_{name}.csv"
    ):
        super().__init__(model, filename=filename)

    @hydromt_step
    def read(self, filename: str | None = None, **kwargs) -> None:
        """Read tables at provided or default file path if none is provided."""
        if filename is None:
            filename = self._generate_filename_from_staticmaps()
        super().read(filename=filename, **kwargs)

    @hydromt_step
    def write(self, filename: str | None = None, **kwargs) -> None:
        """Write tables at provided or default file path if none is provided."""
        if filename is None:
            filename = self._generate_filename_from_staticmaps()
        super().write(filename=filename, **kwargs)

    def _generate_filename_from_staticmaps(self) -> str | None:
        static_maps_path = self.model.config.get_value("input.path_static")
        if static_maps_path is None:
            return None

        static_maps_folder = os.path.dirname(static_maps_path)
        if not static_maps_folder:
            static_maps_folder = "."

        filename = str(static_maps_folder) + "/reservoir_{{hq,sh}}_{name}.csv"
        return filename
