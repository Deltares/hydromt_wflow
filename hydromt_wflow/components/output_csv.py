"""Wflow csv output component."""

import logging
from pathlib import Path

import xarray as xr
from hydromt.model import Model
from hydromt.model.components import DatasetsComponent

from hydromt_wflow import utils
from hydromt_wflow.components.staticmaps import WflowStaticmapsComponent

__all__ = ["WflowOutputCsvComponent"]

logger = logging.getLogger(f"hydromt.{__name__}")


class WflowOutputCsvComponent(DatasetsComponent):
    """ModelComponent class for Wflow csv output.

    This class is used for reading the Wflow csv output.

    The overall output csv component data stored in the ``data`` property of
    this class is of the hydromt.gis.vector.GeoDataset type which is an extension of
    xarray.Dataset for vector data.
    """

    def __init__(
        self,
        model: Model,
        *,
        filename: str = "output.csv",
        locations_component: str | None = None,
    ):
        """
        Initialize the Wflow csv output component.

        Parameters
        ----------
        model : Model
            HydroMT model instance.
        filename : str, optional
            Default path relative to the root where the csv output file will be
            read and written. By default 'output.csv'.
        locations_component : str, optional
            Name of the locations component to use for reading the output and mapping
            them to right geo-locations.
        """
        self._locations_component = locations_component

        super().__init__(
            model=model,
            filename=filename,
        )

    ## I/O methods
    def read(self):
        """Read csv model output at root/dir_output/filename.

        Checks the path of the file in the config toml using both
        ``output.csv.path`` and ``dir_output``. If not found uses the default
        path ``output.csv`` in the root folder.
        """
        self.root._assert_read_mode()
        self._initialize(skip_read=True)

        # Sort which path/ filename is actually the one used
        # Hierarchy is: 1: config, 2: default from component
        p = self.model.config.get_value("output.csv.path") or self._filename
        # Check for input dir
        p_input = Path(self.model.config.get_value("dir_output", fallback=""), p)

        # Link with root
        csv_filename = Path(self.root.path, p_input)

        # Check if the file exist
        if not csv_filename.is_file():
            logger.warning(f"CSV output file {csv_filename} not found, skip reading.")
            return

        # Read
        staticmaps = self._get_locations_data()

        # Check if staticmaps data is not empty
        if len(staticmaps) == 0:
            logger.warning("Staticmaps data is empty, skip reading.")
            return

        csv_dict = utils.read_csv_output(
            csv_filename, config=self.model.config.data, maps=staticmaps
        )
        for key in csv_dict:
            # Add to data
            self.set(csv_dict[f"{key}"])

    def write(self):
        """
        Skip writing output files.

        Output files are model results and are therefore not written by HydroMT.
        """
        logger.warning("output_csv is an output of Wflow and will not be written.")

    ## Set methods

    ## Other
    def _get_locations_data(self) -> xr.Dataset:
        """Get locations data to geo-localize the csv outputs."""
        if self._locations_component is not None:
            reference_component = self.model.get_component(self._locations_component)
            if not isinstance(reference_component, WflowStaticmapsComponent):
                raise ValueError(
                    "Unable to find the referenced staticmaps component: "
                    f"'{self._locations_component}'."
                )
            if reference_component.data is None:
                raise ValueError(
                    "Unable to get grid from the referenced locations component: "
                    f"'{self._locations_component}'."
                )
            return reference_component.data
