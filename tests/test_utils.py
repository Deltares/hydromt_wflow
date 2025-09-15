"""Tests for the utils module."""

from os.path import abspath, dirname, join
from pathlib import Path

import numpy as np

from hydromt_wflow import WflowModel, WflowSedimentModel
from hydromt_wflow.utils import get_grid_from_config

TESTDATADIR = Path(dirname(abspath(__file__)), "data")
EXAMPLEDIR = Path(dirname(abspath(__file__)), "..", "examples", "data")


def test_grid_from_config(demda):
    # Create a couple of variables in grid
    grid = demda.to_dataset(name="dem")
    grid["slope"] = demda * 0.1
    grid["mask"] = demda > 0

    # Create config with all options
    config = {
        "input": {
            "dem": "dem",
            "static": {
                "slope": "slope",
                "altitude": {
                    "netcdf": {"variable": {"name": "slope"}},
                    "scale": 10,
                },
            },
            "cyclic": {
                "subsurface_ksat_horizontal_ratio": {"value": 500},
                "ksathorfrac2": {
                    "netcdf": {"variable": {"name": "dem"}},
                    "scale": 0,
                    "offset": 500,
                },
            },
        },
    }

    # Tests
    dem = get_grid_from_config("dem", config=config, grid=grid)
    assert dem.equals(demda)

    slope = get_grid_from_config("slope", config=config, grid=grid)
    assert slope.equals(grid["slope"])

    altitude = get_grid_from_config("altitude", config=config, grid=grid)
    assert altitude.equals(grid["slope"] * 10)

    subsurface_ksat_horizontal_ratio = get_grid_from_config(
        "subsurface_ksat_horizontal_ratio",
        config=config,
        grid=grid,
    )
    assert np.unique(subsurface_ksat_horizontal_ratio.raster.mask_nodata()) == [500]

    ksathorfrac2 = get_grid_from_config(
        "ksathorfrac2",
        config=config,
        grid=grid,
    )
    assert ksathorfrac2.equals(subsurface_ksat_horizontal_ratio)


def test_convert_to_wflow_v1_sbm(tmp_path):
    # Initialize wflow model
    root = join(EXAMPLEDIR, "wflow_upgrade", "sbm")
    config_fn = "wflow_sbm_v0x.toml"

    wflow = WflowModel(root, config_fn=config_fn, mode="r")
    # Convert to v1
    wflow.upgrade_to_v1_wflow()

    # Check with a test config
    config_fn_v1 = join(TESTDATADIR, "wflow_v0x", "sbm", "wflow_sbm_v1.toml")
    wflow_v1 = WflowModel(root, config_fn=config_fn_v1, mode="r")

    # Set kinematic_wave__adaptive_time_step_flag to false to mirror settings in wflow
    #  v1 config
    wflow.config["model"]["kinematic_wave__adaptive_time_step_flag"] = False
    assert wflow.config == wflow_v1.config, "Config files are not equal"

    # Checks on extra data in staticmaps
    res_ids = np.unique(wflow.grid["reservoir_outlet_id"].raster.mask_nodata())
    assert np.all(np.isin([3349.0, 3367.0, 169986.0], res_ids))
    assert np.all(
        np.isin([3.0, 4.0], wflow.grid["reservoir_rating_curve"].raster.mask_nodata())
    )


def test_convert_to_wflow_v1_sediment():
    # Initialize wflow model
    root = join(EXAMPLEDIR, "wflow_upgrade", "sediment")
    config_fn = "wflow_sediment_v0x.toml"

    wflow = WflowSedimentModel(
        root, config_fn=config_fn, data_libs=["artifact_data"], mode="r"
    )
    # Convert to v1
    wflow.upgrade_to_v1_wflow(
        soil_fn="soilgrids",
    )

    # Check with a test config
    config_fn_v1 = join(TESTDATADIR, "wflow_v0x", "sediment", "wflow_sediment_v1.toml")
    wflow_v1 = WflowSedimentModel(root, config_fn=config_fn_v1, mode="r")

    assert wflow.config == wflow_v1.config, "Config files are not equal"

    # Checks on extra data in staticmaps
    assert "soil_sagg_fraction" in wflow.grid
    assert "land_govers_c" in wflow.grid
    assert "river_kodatie_a" in wflow.grid
    assert "reservoir_outlet_id" in wflow.grid
    res_ids = np.unique(wflow.grid["reservoir_outlet_id"].raster.mask_nodata())
    assert np.all(np.isin([3349.0, 3367.0, 169986.0], res_ids))
    assert np.all(
        np.isin(
            [0.0, 1.0], wflow.grid["reservoir_trapping_efficiency"].raster.mask_nodata()
        )
    )


def test_config_toml_overwrite(tmpdir):
    dummy_model = WflowModel(root=tmpdir, mode="w")
    dummy_model.read_config()
    dummy_model.set_config(
        "input.forcing.khorfrac.value",
        100,
    )
    dummy_model.set_config(
        "input.forcing.khorfrac.value",
        200,
    )
    assert dummy_model.get_config("input.forcing.khorfrac.value") == 200

    # Test overwriting top-level key
    dummy_model.set_config("path_log", "log_file.log")
    dummy_model.set_config("path_log", "log_file2.log")
    assert dummy_model.get_config("path_log") == "log_file2.log"
