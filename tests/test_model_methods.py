"""Unit tests for hydromt_wflow methods and workflows"""

import pytest
from os.path import join, dirname, abspath
import warnings
import pdb
from hydromt_wflow.wflow import WflowModel

import logging

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")


def test_setup_staticmaps():
    logger = logging.getLogger(__name__)
    # read model from examples folder
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")

    # Initialize model and read results
    mod = WflowModel(root=root, mode="r", data_libs="artifact_data", logger=logger)

    # Tests on setup_staticmaps_from_raster
    mod.setup_staticmaps_from_raster(
        raster_fn="merit_hydro",
        reproject_method="average",
        variables=["elevtn"],
        wflow_variables=["input.vertical.altitude"],
        fill_method="nearest",
    )
    assert "elevtn" in mod.staticmaps
    assert mod.get_config("input.vertical.altitude") == "elevtn"

    mod.setup_staticmaps_from_raster(
        raster_fn="globcover",
        reproject_method="mode",
        wflow_variables=["input.vertical.landuse"],
    )
    assert "globcover" in mod.staticmaps
    assert mod.get_config("input.vertical.landuse") == "globcover"

    # Test on exceptions
    with pytest.raises(ValueError, match="Length of variables"):
        mod.setup_staticmaps_from_raster(
            raster_fn="merit_hydro",
            reproject_method="average",
            variables=["elevtn", "lndslp"],
            wflow_variables=["input.vertical.altitude"],
        )
    with pytest.raises(ValueError, match="variables list is not provided"):
        mod.setup_staticmaps_from_raster(
            raster_fn="merit_hydro",
            reproject_method="average",
            wflow_variables=["input.vertical.altitude"],
        )
