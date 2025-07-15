"""Tests for the utils module."""

from os.path import abspath, dirname, join
from pathlib import Path

import numpy as np
from tomlkit import load

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


def test_convert_to_wflow_v1_sbm():
    # Initialize wflow model
    root = join(TESTDATADIR, "wflow_v0x", "sbm")
    config_fn = "wflow_sbm_v0x.toml"

    wflow = WflowModel(root, config_filename=config_fn, mode="r")

    # Convert to v1
    wflow.upgrade_to_v1_wflow()

    # Check with a test config
    config_fn_v1 = join(TESTDATADIR, "wflow_v0x", "sbm", "wflow_sbm_v1.toml")
    wflow_v1 = WflowModel(root, config_filename=config_fn_v1, mode="r")

    assert wflow.config == wflow_v1.config, "Config files are not equal"


def test_convert_to_wflow_v1_sediment():
    # Initialize wflow model
    root = join(EXAMPLEDIR, "wflow_upgrade", "sediment")
    config_fn = "wflow_sediment_v0x.toml"

    wflow = WflowSedimentModel(
        root, config_filename=config_fn, data_libs=["artifact_data"], mode="r"
    )
    # Convert to v1
    wflow.upgrade_to_v1_wflow(
        soil_fn="soilgrids",
    )

    # Check with a test config
    config_fn_v1 = join(TESTDATADIR, "wflow_v0x", "sediment", "wflow_sediment_v1.toml")
    wflow_v1 = WflowSedimentModel(root, config_filename=config_fn_v1, mode="r")

    assert wflow.config == wflow_v1.config, "Config files are not equal"

    # Checks on extra data in staticmaps
    assert "soil_sagg_fraction" in wflow.staticmaps.data
    assert "land_govers_c" in wflow.staticmaps.data
    assert "river_kodatie_a" in wflow.staticmaps.data


def test_config_toml_grouping(tmpdir: Path):
    dummy_model = WflowModel(root=tmpdir, mode="w")
    dummy_model.read_config()

    dummy_model.set_config(
        "input",
        "forcing",
        "netcdf.name",
        "blah.nc",
    )
    dummy_model.config.set(
        "input",
        "forcing",
        "scale",
        1,
    )
    dummy_model.config.set(
        "input",
        "static",
        "staticsoil~compacted_surface_water__infiltration_capacity",
        "value",
        5,
    )
    dummy_model.config.set(
        "input",
        "static",
        "soil_root~wet__sigmoid_function_shape_parameter",
        "value",
        -500,
    )
    dummy_model.config.set(
        "input.static.soil_water_sat-zone_bottom__max_leakage_volume_flux.value", 0
    )

    dummy_model.write()

    with open(tmpdir / "wflow_sbm.toml", "r") as file:
        written_config = load(file)

    with open(TESTDATADIR / "grouped_model_config.toml") as file:
        expected_config = load(file)

    assert written_config == expected_config


def test_config_toml_overwrite(tmpdir: Path):
    dummy_model = WflowModel(root=tmpdir, mode="w")
    dummy_model.config.read()
    dummy_model.config.set(
        "input.forcing.khorfrac.value",
        100,
    )
    dummy_model.config.set(
        "input.forcing.khorfrac.value",
        200,
    )
    assert dummy_model.config.get_value("input.forcing.khorfrac.value") == 200
