import glob
import logging
import os
from os.path import basename, join
from typing import TYPE_CHECKING

import pandas as pd
from hydromt.model.components import TablesComponent
from hydromt.model.steps import hydromt_step

if TYPE_CHECKING:
    from hydromt.model.model import Model

logger = logging.getLogger(f"hydromt.{__name__}")


class WflowTablesComponent(TablesComponent):
    """Wflow specific tables component."""

    def __init__(self, model: "Model", filename: str = "{name}.csv"):
        super().__init__(model, filename=filename)

    @hydromt_step
    def read(self, filename: str | None = None) -> None:
        """Read tables at provided or default file path if none is provided."""
        if filename is None:
            filename = self._generate_filename_from_staticmaps()

        self.root._assert_read_mode()
        self._initialize_tables(skip_read=True)
        logger.info("Reading model table files.")
        filenames = glob.glob(join(self.root.path, filename.format(name="*")))
        if len(filenames) > 0:
            for fn in filenames:
                name = basename(fn).split(".")[0]
                if "_hq_" in name:
                    tbl = self._read_hq_table(fn)
                else:
                    tbl = pd.read_csv(fn)
                self.set(tbl, name=name)

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

        filename = f"{static_maps_folder}/{{name}}.csv"
        return filename

    def _read_hq_table(self, filepath: str) -> pd.DataFrame:
        """Read a rating curve table from a CSV file."""
        # For HQ table, the first line should be skipped and manually added after
        df = pd.read_csv(filepath, skiprows=1, header=None)
        df.rename(columns={0: "H"}, inplace=True)

        if len(df.columns) != 366:
            raise ValueError(
                f"HQ table at {filepath} should have 366 columns, 1 for H and 365 for "
                f"DOY Q. Found {len(df.columns)}."
            )

        return df
