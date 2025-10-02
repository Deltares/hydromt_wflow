"""Tests for the utils module."""

from os.path import abspath, dirname, join
from pathlib import Path

import numpy as np

from hydromt_wflow import WflowSbmModel, WflowSedimentModel
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
                    "netcdf_variable_name": "slope",
                    "scale": 10,
                },
            },
            "cyclic": {
                "subsurface_ksat_horizontal_ratio": {"value": 500},
                "ksathorfrac2": {
                    "netcdf_variable_name": "dem",
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
    root = join(EXAMPLEDIR, "wflow_upgrade", "sbm")
    config_fn = "wflow_sbm_v0x.toml"

    wflow = WflowSbmModel(root, config_filename=config_fn, mode="r")

    # Convert to v1
    wflow.upgrade_to_v1_wflow()

    # Check with a test config
    config_fn_v1 = join(TESTDATADIR, "wflow_v0x", "sbm", "wflow_sbm_v1.toml")
    wflow_v1 = WflowSbmModel(root, config_filename=config_fn_v1, mode="r")

    # Set kinematic_wave__adaptive_time_step_flag to false to mirror settings in wflow
    #  v1 config
    wflow.config.data["model"]["kinematic_wave__adaptive_time_step_flag"] = False
    assert wflow.config.test_equal(wflow_v1.config)[0]

    # Checks on extra data in staticmaps
    res_ids = np.unique(
        wflow.staticmaps.data["reservoir_outlet_id"].raster.mask_nodata()
    )
    assert np.all(np.isin([3349.0, 3367.0, 169986.0], res_ids))
    assert np.all(
        np.isin(
            [3.0, 4.0],
            wflow.staticmaps.data["reservoir_rating_curve"].raster.mask_nodata(),
        )
    )


def test_convert_to_wflow_v1_sbm_with_exceptions():
    # Initialize wflow model
    root = join(EXAMPLEDIR, "wflow_upgrade", "sbm")
    config_fn = "wflow_sbm_v0x.toml"

    wflow = WflowSbmModel(root, config_filename=config_fn, mode="r")
    theta_s = wflow.config.remove("input.vertical.theta_s")
    theta_r = wflow.config.remove("input.vertical.theta_r")
    g_ttm = wflow.config.remove("input.vertical.g_ttm")
    kv = wflow.config.remove("input.vertical.kv_0")

    wflow.config.set("input.vertical.θₛ", theta_s)
    wflow.config.set("input.vertical.θᵣ", theta_r)
    wflow.config.set("input.vertical.g_tt", g_ttm)
    wflow.config.set("input.vertical.kv₀", kv)

    # Convert to v1
    wflow.upgrade_to_v1_wflow()

    # Check with a test config
    config_fn_v1 = join(TESTDATADIR, "wflow_v0x", "sbm", "wflow_sbm_v1.toml")
    wflow_v1 = WflowSbmModel(root, config_filename=config_fn_v1, mode="r")

    # Set kinematic_wave__adaptive_time_step_flag to false to mirror settings in wflow
    #  v1 config
    wflow.config.data["model"]["kinematic_wave__adaptive_time_step_flag"] = False
    assert wflow.config.test_equal(wflow_v1.config)[0]


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

    assert wflow.config.test_equal(wflow_v1.config)[0]

    # Checks on extra data in staticmaps
    assert "soil_sagg_fraction" in wflow.staticmaps.data
    assert "land_govers_c" in wflow.staticmaps.data
    assert "river_kodatie_a" in wflow.staticmaps.data
    assert "reservoir_outlet_id" in wflow.staticmaps.data
    res_ids = np.unique(
        wflow.staticmaps.data["reservoir_outlet_id"].raster.mask_nodata()
    )
    assert np.all(np.isin([3349.0, 3367.0, 169986.0], res_ids))
    assert np.all(
        np.isin(
            [0.0, 1.0],
            wflow.staticmaps.data["reservoir_trapping_efficiency"].raster.mask_nodata(),
        )
    )


def test_config_toml_overwrite(tmp_path: Path):
    dummy_model = WflowSbmModel(root=tmp_path, mode="w")
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

    # Test overwriting top-level key
    dummy_model.config.set("path_log", "log_file.log")
    dummy_model.config.set("path_log", "log_file2.log")
    assert dummy_model.config.get_value("path_log") == "log_file2.log"


def test_convert_to_wflow_v1_with_lake_files(tmp_path: Path):
    # Initialize wflow model
    root = TESTDATADIR / "wflow_v0x" / "sbm_with_lake_files"
    config_fn = "wflow_sbm_v0x.toml"

    wflow = WflowSbmModel(root, config_filename=config_fn, mode="r")

    # Convert to v1
    wflow.upgrade_to_v1_wflow()
    wflow.set_root(tmp_path, mode="w")
    wflow.write()

    assert (tmp_path / "staticmaps" / "reservoir_hq_1.csv").is_file()
    assert (tmp_path / "staticmaps" / "reservoir_hq_2.csv").is_file()
