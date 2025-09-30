import os
from typing import TYPE_CHECKING

from hydromt.model.components import TablesComponent

if TYPE_CHECKING:
    from hydromt.model.model import Model


class WflowTablesComponent(TablesComponent):
    """Wflow specific tables component."""

    def __init__(self, model: "Model", filename: str = "{name}.csv"):
        super().__init__(model, filename=filename)

    def read(self, filename: str | None = None, **kwargs) -> None:
        """Read tables at provided or default file path if none is provided."""
        if filename is None:
            filename = self._generate_filename_from_staticmaps()
        super().read(filename=filename, **kwargs)

    def write(self, filename: str | None = None, **kwargs) -> None:
        """Write tables at provided or default file path if none is provided."""
        if filename is None:
            filename = self._generate_filename_from_staticmaps()
        super().write(filename=filename, **kwargs)

    def _generate_filename_from_staticmaps(self) -> str | None:
        static_maps_path = self.model.get_config("input.path_static")
        if static_maps_path is None:
            return None

        static_maps_folder = os.path.dirname(static_maps_path)
        if not static_maps_folder:
            static_maps_folder = "."

        filename = f"{static_maps_folder}/{{name}}.csv"
        return filename
